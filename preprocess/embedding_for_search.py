# 
import pandas as pd
import glob
import os,re
import pickle
import argparse

# langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import *

from tqdm import tqdm
def get_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('--output','-o',type=str,help='output folder')
    paser.add_argument('--field','-f',type=str,default='abstract',help='column as content')
    paser.add_argument('--doc_save','-d',type=str,help='column as content')
    paser.add_argument('--ids',type=int,help='ids  of worker')
    paser.add_argument('--workers','-n',type=int,help='number  of worker')
    return paser.parse_args()

def main():
    args = get_args()
    with open(args.doc_save,'rb') as f:
        docs = pickle.load(f)
    st = len(docs)//args.workers * args.ids
    if args.ids != args.workers - 1:
        ed = len(docs)//args.workers * (args.ids+1)
    else:
        ed = len(docs)
    print('embeding range',st,'----->',ed)
    embeddings = HuggingFaceInstructEmbeddings(
            # query_instruction=args.command,
            model_name = args.model_name_or_path
        )
    
    vector_storage = FAISS.from_documents(docs[st:ed], embeddings)
    vector_storage.save_local(args.save)
    
if __name__ == '__main__':
    main()
