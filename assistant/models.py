from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests,httpx
class UrlLLM(LLM):
    url: str
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with httpx.Client() as client:
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "stop": stop + ["Observation:"]
            }
            response = client.post(
                self.url,
                json=json,
                timeout=600
            )
        # response = requests.post(
        #     self.url,
        #     json=json
        # )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
        }

if __name__ =='__main__':
    from langchain.agents import load_tools
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from tools.vetorstore_search import PubMedSearchTool
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    # from langchain import  SerpAPIWrapper
    # from langchain.tools import PubmedQueryRun
    # from dotenv import find_dotenv,load_dotenv
    import os
    os.environ['SERPAPI_API_KEY']="f37fdf8418be72fdfbb5ad3ca36129f1b2638487d646bdc294bff4cb50bc1db0"
    # load_dotenv(find_dotenv())
    # llm = UrlLLM(url="http://region-3.seetacloud.com:54504/prompt")
    llm = UrlLLM(url="http://localhost:6006/prompt")
    embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
)
    vectorstore = FAISS.load_local('/root/autodl-tmp/pubmeds/embeddings2_merge',embeddings=embeddings)
    tools = [
        PubMedSearchTool(
            llm = llm,
            vertorstore = vectorstore
            ),
        ]
    tools += load_tools(["arxiv"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False,return_intermediate_steps=True)
    while True:
        query = input('请提出你的问题：')
        response = agent({"input":query})
        print(response["intermediate_steps"])
        print(response["output"])
