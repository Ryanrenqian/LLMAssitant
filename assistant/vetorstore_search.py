from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type,Any
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from models import models
class PubMedSearchTool(BaseTool):
    name = "pubmed search"
    description = "useful for when you need to answer questions about biology-related reaseaches,like EGFR."
    vectorstore: VectorStore
    llm: LLM
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        qa_chain = load_qa_chain(self.llm, chain_type="map_reduce")
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=self.vectorstore.as_retriever())
        return qa.run(query)    
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

# 基于vertorstore和llm agent的搜索引擎
vectorsearches = {}
def register_vectorsearches(vectorsearch: PubMedSearchTool,name:str,override:bool=False):
    if not override:
        assert name not in vectorsearches.keys(), f"{name} has been registered."
    vectorsearches[name] = vectorsearch

vectorstores = {}
def register_storevectors(vectorstore: VectorStore,name:str,override:bool=False):
    if not override:
        assert name not in vectorstores.keys(), f"{name} has been registered."
    vectorstores[name] = vectorstore
    
## 加载vectorstore
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
)
test_vectorstore = FAISS.load_local('/root/autodl-tmp/pubmed_embedding/0',embeddings=embeddings)
test_vectorsearch = PubMedSearchTool(
            llm = list(models.values())[0],
            vectorstore = test_vectorstore,
        )
register_storevectors(test_vectorstore,'test_sv')
register_vectorsearches(test_vectorsearch,'test_vs')





if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from assistant.models import UrlLLM
    from langchain.llms import HuggingFacePipeline
    from langchain.agents import load_tools, initialize_agent, AgentType
    from langchain.tools import PubmedQueryRun

    llm = UrlLLM(url="http://localhost:6006/prompt")
    embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
)
    vectorstore = FAISS.load_local('/root/autodl-tmp/pubmed_embedding/0',embeddings=embeddings)
    tools = [
        PubMedSearchTool(
            llm = llm,
            vectorstore = vectorstore),
        ]
    query = 'what is EGFR?'
    print(tools[0].run(query))