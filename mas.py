import autogen
from autogen import AssistantAgent, UserProxyAgent
import pandas as pd

config_list = [
  {
    "model": "llama3:8b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "price": [0.0, 0.0],
  }
]

# llm_config = {"config_list": config_list, "cache_seed": 42}
llm_config = {"config_list": config_list}

user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="Helper agent that only initiates the conversation",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    # human_input_mode="TERMINATE", #NOTE: human inputs allowed
    human_input_mode = "NEVER",
)

document_reader = AssistantAgent(
    name="Document_Reader",
    system_message="""Expert at reading and parsing CSV files and other documents. You extract relevant information and provide structured data for analysis.
                    Do not just repeat the data or wait for further instructions. Perform this analysis immediately when receiving CSV data.
                    After your analysis, explicitly pass the conversation to the Strategist for their insights.""",
    llm_config=llm_config,
)

strategist = AssistantAgent(
    name="Strategist",
    system_message="Strategic planner who analyzes information and develops comprehensive strategies. You provide strategic insights and recommendations based on available data.",
    llm_config=llm_config,
)

summarizer = AssistantAgent(
    name="Summarizer",
    system_message="Expert at condensing complex information into clear, concise summaries. You create executive summaries and highlight key points from discussions and documents.",
    llm_config=llm_config,
)

editor = AssistantAgent(
    name="Editor",
    system_message="Professional editor who reviews and refines content for clarity, accuracy, and style. You improve the quality and readability of written materials.",
    llm_config=llm_config,
)

writer = AssistantAgent(
    name="Writer",
    system_message="Creative writer who produces high-quality content. You craft engaging narratives and clear explanations based on provided information and strategic direction.",
    llm_config=llm_config,
)


# groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm, evaluator, ethics_checker], messages=[], max_round=12)
groupchat = autogen.GroupChat(
    agents=[document_reader, strategist, summarizer, editor, writer],
    messages=[],
    max_round=6,
    speaker_selection_method="round_robin"
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


def start_analysis(csv_path):
    """parses csv using pandas and passes them to Document Reader Agent"""
    
    # parse csv
    csv_content = str(pd.read_csv(csv_path))

    initial_message = f"""Here is the CSV content to analyze:

    {csv_content}

    Please analyze this data and share your key findings with the team."""

    user_proxy.initiate_chat(
        manager,
        message=initial_message
    )

    

    # initial_message = f"Please analyze the CSV file at {csv_path} and share your findings with the team."
    # # start with csv
    # document_reader.initiate_chat(
    #     manager,
    #     message=initial_message
    # )


if __name__ == "__main__":
    CSV_PATH = "nvda_stocks.csv"
    start_analysis(CSV_PATH)