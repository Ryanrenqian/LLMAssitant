# 
import pandas as pd
import glob
import os
import argparse
# langchain
from langchain.vectorstores import FAISS
from langchain.embeddings import *
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter,NLTKTextSplitter
from langchain.document_loaders import DataFrameLoader
from tqdm import tqdm
def get_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('--input','-i',type=str,help='Regex path match')
    paser.add_argument('--output','-o',type=str,help='output folder')
    paser.add_argument('--command','-c',type=str,default='summary the text.',help='instruct for LLM embedding')
    paser.add_argument('--model_name_or_path','-m',type=str,default='/root/autodl-tmp/cache/instructor-xl',help='instruct for LLM embedding')
    paser.add_argument('--parser','-p',choices=['csv','pkl'],default='csv',help='chioce file parser')
    paser.add_argument('--field','-f',type=str,default='abstract',help='column as content')
    return paser.parse_args()
    
def main():
    args = get_args()
    embeddings = HuggingFaceInstructEmbeddings(
            query_instruction=args.command,
            model_name = args.model_name_or_path
        )
    docs = []
    for csv_file in tqdm(glob.glob(args.input)):
        basename = os.path.basename(csv_file)[:-4]
        save = os.path.join(args.output,basename)
        # if os.path.exists(save):
        #     continue
        if args.parser == 'csv':
            df = pd.read_csv(csv_file)
        elif args.parser == 'pkl':
            df = pd.read_pickle(csv_file)
            df.columns = ['pmid','title','2','3','4','5','abstract']
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
        loader = DataFrameLoader(df,page_content_column=field)
        datas = loader.load()
        print("Number of Documents in",basename,":",len(datas))
        text_splitter = CharacterTextSplitter(        
            separator ='.',
            chunk_overlap=0,
            chunk_size = 512,
        )
        # text_splitter = NLTKTextSplitter(chunk_size = 512,)
        split_docs = text_splitter.split_documents(datas)
        print("Split of Documents:",len(split_docs))
#    docs.append(split_docs)
        vector_storage = FAISS.from_documents(split_docs, embeddings)
        vector_storage.save_local(save)
    
if __name__ == '__main__':
    main()
