import pandas as pd
import glob
import os,re
import pickle
import argparse
import unicodedata
# langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import *
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,NLTKTextSplitter
from langchain.document_loaders import DataFrameLoader
from tqdm import tqdm
def get_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('--input','-i',type=str,help='Regex path match')
    paser.add_argument('--command','-c',type=str,default='summary the text.',help='instruct for LLM embedding')
    paser.add_argument('--model_name_or_path','-m',type=str,default='/root/autodl-tmp/cache/instructor-xl',help='instruct for LLM embedding')
    paser.add_argument('--parser','-p',choices=['csv','pkl'],default='csv',help='chioce file parser')
    paser.add_argument('--field','-f',type=str,default='abstract',help='column as content')
    paser.add_argument('--doc_save','-d',type=str,help='column as content')
    return paser.parse_args()

def clean_string(text):
    normalized_text = unicodedata.normalize('NFKD', text)
    replaced_text = ''.join(c for c in normalized_text if not unicodedata.combining(c))
    return replaced_text

def main():
    args = get_args()
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    doc_save = args.doc_save
    qbar = tqdm(glob.glob(args.input))
    print(len(qbar))
    docs = []
    for csv_file in qbar:
        basename = os.path.basename(csv_file)[:-4]
        if args.parser == 'csv':
            df = pd.read_csv(csv_file)
        elif args.parser == 'pkl':
            try:
                df = pd.read_pickle(csv_file)
                df.columns = ['pmid','title','2','3','4','5','abstract']
            except:
                continue
        else:
            raise ValueError("please check your parser, only pkl and csv supported")
        field = args.field
        if field not in list(df.columns):
            if field.isnumeric():
                field = int(field)
        df = df[df[field].notna()]
        print("Shape of df",df.shape)
        if len(df)<1:
            continue
        df[field] = df[field].apply(clean_string)
        loader = DataFrameLoader(df,page_content_column=field)
        datas = loader.load()
        print("Number of Documents in",basename,":",len(datas))
        
        split_docs = text_splitter.split_documents(datas)
        print("Split of Documents:",len(split_docs))
        docs+=split_docs
    with open(doc_save,'wb') as f:
        pickle.dump(docs,f)
    for doc in docs:
        if len(doc.page_content)>512:
            print(doc.page_content)
            break
if __name__ == '__main__':
    main()