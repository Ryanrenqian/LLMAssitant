from pydantic import BaseModel
from typing import Union,List
class PromptRequest(BaseModel):
    prompt: str
    temperature: float
    max_new_tokens: int
    stop: Union[str,List[str]]


    