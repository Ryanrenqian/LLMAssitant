# 
import pandas as pd
import glob
import os,re
import pickle
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
    paser.add_argument('--doc_save','-d',type=str,help='column as content')
    paser.add_argument('--ids',type=int,help='ids  of worker')
    paser.add_argument('--workers','-n',type=int,help='number  of worker')
    return paser.parse_args()

def clean_string(input_string):
    pattern = re.compile(r'[&@#$]')
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

def main():
    args = get_args()
    docs = []
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024, chunk_overlap=0
    )
    doc_save = args.doc_save
    save = args.output
    if not os.path.exists(doc_save):
        for csv_file in tqdm(glob.glob(args.input)):
            basename = os.path.basename(csv_file)[:-4]
            
            if os.path.exists(save):
                continue
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
    else:
        with open(doc_save,'rb') as f:
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
    vector_storage.save_local(save)
    
if __name__ == '__main__':
    main()
