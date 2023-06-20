import sys
sys.append('..')
from assistant.models import UrlLLM
from assistant.tools.vetorstore_search import PubMedSearchTool
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from fastapi import FastAPI
from models import SearchRequest
llm = UrlLLM(url="http://localhost:6006/prompt")
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
    )
vectorstore = FAISS.load_local('/root/autodl-tmp/pubmeds/embeddings2_merge',embeddings=embeddings)
app = FastAPI()
tools = PubMedSearchTool(
    llm=llm,
    vectorstore=vectorstore
)
@app.post("/pubmedsearch")
def pubmedsearch(input: SearchRequest):
    query = input.query
    return {'output': tools.run(query)}
    
