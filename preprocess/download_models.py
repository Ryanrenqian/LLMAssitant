from huggingface_hub import snapshot_download
import argparse,os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--model_name',type=str,default='tiiuae/falcon-40b-instruct',help = 'hugging face repo ids')
    parser.add_argument('-s','--save_path',type=str,default='/root/autodl-tmp/cache/transformers',help='save path of model weight')
    return parser.parse_args()
if __name__ =='__main__':
    args = get_args()
    snapshot_download(args.model_name,local_dir=os.path.join(args.save_path,args.model_name),local_dir_use_symlinks=False,resume_download=True)