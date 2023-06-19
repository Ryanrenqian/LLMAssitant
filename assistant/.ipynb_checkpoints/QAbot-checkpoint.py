from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from fastchat.model.model_adapter import load_model
import torch
import args
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()
model, tokenizer = load_model(model_path='/root/autodl-tmp/cache/transformers/vicuna/13B/',device=device,num_gpus=2)
