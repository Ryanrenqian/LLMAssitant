from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type,Any
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
from accelerate import infer_auto_device_map
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM


class PubMedSearchTool(BaseTool):
    name = "pubmed search"
    description = "useful for when you need to answer questions about biology-related reaseaches,like EGFR."
    vertorstore:VectorStore
    llm:LLM

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        qa_chain = load_qa_chain(self.llm, chain_type="map_reduce")
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=self.vertorstore.as_retriever())
        return qa.run(query)    
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

if __name__ == '__main__':
    from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
    from langchain.llms import HuggingFacePipeline
    from langchain.agents import load_tools, initialize_agent, AgentType
    from fastchat.model.model_adapter import load_model
    from langchain.tools import PubmedQueryRun
    device = 'cuda'
    model_path = '/root/autodl-tmp/cache/transformers/guanaco-33b-merged/'
    model, tokenizer = load_model(model_path=model_path,device='cuda',num_gpus=4,max_gpu_memory='30GiB')
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     low_cpu_mem_usage=True,
    # )
    # device_map = infer_auto_device_map(model)
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer,
        device_map="auto",
        max_length=2048,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1
    )
    embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
)
    vectorstore = FAISS.load_local('/root/autodl-tmp/pubmeds/embeddings2_merge',embeddings=embeddings)
    llm = HuggingFacePipeline(pipeline=pipe)
    tools = [
        PubMedSearchTool(
            llm = llm,
            vertorstore = vectorstore),
        PubmedQueryRun(),
        ]
    tools += load_tools(["arxiv"])
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    while True:
        query = input('请提出你的问题：')
        print(agent.run(query))
