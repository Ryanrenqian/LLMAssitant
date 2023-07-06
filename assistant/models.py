from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests,httpx

def _get_token_ids_default_method(text: str) -> List[int]:
    """Encode the text into token IDs."""
    # TODO: this method may not be exact.
    # TODO: this method may differ based on model (eg codex).
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/cache/transformers/timdettmers/guanaco-33b-merged")
    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)
class UrlLLM(LLM):
    url: str
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with httpx.Client() as client:
            if stop is None:
                stop = []
            json={
                "prompt": prompt,
                "temperature": 0.7,
                "max_new_tokens": 1024,
                "stop": stop + ["Observation:"]
            }
            response = client.post(
                self.url+'/prompt',
                json=json,
                timeout=99999
            )
        # response = requests.post(
        #     self.url,
        #     json=json
        # )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
        }
    def get_token_ids(self, text: str) -> List[int]:
        """Get the token present in the text."""
        return _get_token_ids_default_method(text)

models = {}
def register_models(model: LLM,name:str,override:bool=False):
    if not override:
        assert name not in models.keys(), f"{name} has been registered."
    models[name] = model

# 加载模型
register_models(UrlLLM(url="http://127.0.0.1:6060"),name='guanaco-33b')

if __name__ =='__main__':
    from langchain import OpenAI, ConversationChain, LLMChain
    from langchain.memory import ConversationBufferMemory
    import os
    llm = UrlLLM(url="http://localhost:6006/prompt")
    # search = PubmedQueryRun()
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(),
        verbose=True
    )
    print(conversation.run("What is ChatGPT?"))
