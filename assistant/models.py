from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests

class UrlLLM(LLM):
    url: str
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            self.url,
            json={
                "prompt": prompt,
                "temperature": 0.75,
                "max_new_tokens": 2048,
                "stop": stop + ["Observation:"]
            }
        )
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
    # from langchain import  SerpAPIWrapper
    from langchain.tools import PubmedQueryRun
    # from dotenv import find_dotenv,load_dotenv
    import os
    os.environ['SERPAPI_API_KEY']="f37fdf8418be72fdfbb5ad3ca36129f1b2638487d646bdc294bff4cb50bc1db0"
    # load_dotenv(find_dotenv())
    llm = UrlLLM(url="http://region-3.seetacloud.com:54504/prompt")
    tools = [
        PubmedQueryRun(),
        ]
    tools += load_tools(["arxiv"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    while True:
        query = input('请提出你的问题：')
        print(agent.run(query))
