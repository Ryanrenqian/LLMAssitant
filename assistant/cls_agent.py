from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import HuggingFaceHub

# repo_id = "tiiuae/falcon-40b-instruct" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = "google/flan-t5-xl"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_length":2048})
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("What is EGFR?")