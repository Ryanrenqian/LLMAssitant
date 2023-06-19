from fastapi import FastAPI
from models import *
from fastchat.model.model_adapter import load_model
import torch

app = FastAPI()
model_path = '/root/autodl-tmp/cache/transformers/vicuna/13B'
device ='cuda'
model, tokenizer = load_model(model_path=model_path,device='cuda',num_gpus=2,max_gpu_memory='30GiB')
@torch.inference_mode()
def compute_until_stop(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_parameter = params.get("stop", None)
    if stop_parameter == tokenizer.eos_token:
        stop_parameter = None
    stop_strings = []
    if isinstance(stop_parameter, str):
        stop_strings.append(stop_parameter)
    elif isinstance(stop_parameter, list):
        stop_strings = stop_parameter
    elif stop_parameter is None:
        pass
    else:
        raise TypeError("Stop parameter must be string or list of strings.")

    input_ids = tokenizer(prompt).input_ids
    output_ids = []

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    stop_word = None
    pos = -1
    past_key_values = None
    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]


        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        # print("Partial output:", output)
        for stop_str in stop_strings:
            # print(f"Looking for '{stop_str}' in '{output[:l_prompt]}'#END")
            pos = output.rfind(stop_str)
            if pos != -1:
                # print("Found stop str: ", output)
                output = output[:pos]
                # print("Trimmed output: ", output)
                stopped = True
                stop_word = stop_str
                break
            else:
                pass
                # print("Not found")

        if stopped:
            break

    del past_key_values
    if pos != -1:
        return output[:pos]
    return output

@app.post("/prompt")
def process_prompt(prompt_request: PromptRequest):
    params = {
        "prompt": prompt_request.prompt,
        "temperature": prompt_request.temperature,
        "max_new_tokens": prompt_request.max_new_tokens,
        "stop": prompt_request.stop
    }
    print("Received prompt: ", params["prompt"])
    # request with params...")
    # pprint(params)
    
    output = compute_until_stop(model, tokenizer, params, device='cuda')
    print("Output: ", output)
    return {"response": output}