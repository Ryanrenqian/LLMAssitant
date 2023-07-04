# 
import pandas as pd
import glob
import os,re
import pickle
import argparse

# langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import *

from tqdm import tqdm
def get_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('--output','-o',type=str,help='output folder')
    paser.add_argument('--input','-i',type=str,help='column as content')
    paser.add_argument('--model_name_or_path','-m',type=str,default='/root/autodl-tmp/cache/instructor-xl',help='instruct for LLM embedding')
    return paser.parse_args()

def main():
    args = get_args()
    with open(args.input,'rb') as f:
        docs = pickle.load(f)
    embeddings = HuggingFaceInstructEmbeddings(
            # query_instruction=args.command,
            model_name = args.model_name_or_path
        )
    db2 = Chroma.from_documents(docs, embeddings, persist_directory=args.output+"/chroma_db")
    db2.persist()
    
if __name__ == '__main__':
    main()
