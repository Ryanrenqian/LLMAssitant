import sys
sys.path.append('..')
from assistant.models import UrlLLM
from assistant.vetorstore_search import PubMedSearchTool
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from fastapi import FastAPI
from models import SearchRequest
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
llm = UrlLLM(url="http://localhost:6006/prompt")
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
    )
vectorstore = FAISS.load_local('/root/autodl-tmp/pubmeds/embeddings2_merge',embeddings=embeddings)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())

app = FastAPI()
@app.post("/pubmedsearch")
def pubmedsearch(input: SearchRequest):
    query = input.query
    return {'output': qa.run(input)}
    
