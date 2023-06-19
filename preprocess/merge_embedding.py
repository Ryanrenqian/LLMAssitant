from langchain.vectorstores import FAISS
from langchain.embeddings import *
import glob,sys,os
import argparse,tqdm
def get_args():
    paser = argparse.ArgumentParser()
    paser.add_argument('--input','-i',type=str,help='Regex path match')
    paser.add_argument('--output','-o',type=str,help='output folder')
    paser.add_argument('--command','-c',type=str,default='summary the text.',help='instruct for LLM embedding')
    paser.add_argument('--model_name_or_path','-m',type=str,default='/root/autodl-tmp/cache/instructor-xl',help='instruct for LLM embedding')
    return paser.parse_args()

if __name__ == '__main__':
    args = get_args()
    merged_local = None
    path_reg = f'{args.input}/*/index.faiss'
    faiss_files = glob.glob(path_reg)
    if len(faiss_files) ==0:
        raise ValueError(f'input should be valid path, not {path_reg}')
    faiss_files = tqdm.tqdm(faiss_files)
    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction=args.command,
        model_name = args.model_name_or_path
    )
    for faiss_file in faiss_files:
        faiss_folder = os.path.dirname(faiss_file)
        if merged_local is None:
            merged_local = FAISS.load_local(faiss_folder,embeddings=embeddings)
        else:
            merged_local.merge_from(FAISS.load_local(faiss_folder,embeddings=embeddings))
    merged_local.save_local(args.output)