from langchain.vectorstores import FAISS
from langchain.embeddings import *
import dataclasses
from enum import auto,Enum
from typing import List,Any,Dict
import os

@dataclasses.dataclass
class Storevector:
    name: str
    vector: Any = None
    llm: Any = None
    
    def init_cfg(self,embedding_model: str = None,
                 embedding_device: str = 'gpu',
                 llm_model: str = None,
                ):
        pass
    
    def copy(self):
        return Storevector(
            name = self.name,
            vector = self.vector,
        )
    
    def search_knowlegde(self,query,vector_search_top_k=10):
        return self.vector.similarity_search(query, k=vector_search_top_k)
    
    def get_knowledge_based_answer(self, query, vector_search_top_k=10, chat_history=[]):
        related_docs = self.vector.similarity_search(query, k=vector_search_top_k)
        prompt = "\n".join([doc.page_content for doc in related_docs])
        chat_history.append(["",""])
        chat_history[-1][0] = query
        response = {"query": query,
                    "source_documents": related_docs}
        yield response, chat_history

        

# A global registry for all storevector
storevectors: Dict[str,Storevector] = {}

def register_storevectors(store_vector: Storevector,override: bool = False):
    """Register a new store_vector
    """
    if not override:
        assert store_vector.name not in storevectors, f"{name} has been registered."
    storevectors[store_vector.name] = store_vector

def get_storevectors(name:str) ->Storevector:
    return storevectors[name].copy()


embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Summary the text for retirval: "
)

register_storevectors(
    Storevector(
        name = 'Gene-Interence-Pubmed',
        vector = FAISS.load_local(
            './FAISS/HuggingFaceInstructEmbeddings-2000',
            HuggingFaceInstructEmbeddings(
                query_instruction="Summary the text for retirval: "
                )
            ),
    )
)