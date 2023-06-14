from fastchat.conversation import get_conv_template
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
Question: {input}
{agent_scratchpad}"""
# set refine template
def vicuna_template_adjust(template):
    '''利用fastchat api提供的vicuna模板来对模板进行微调
    '''
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], template)
    conv.append_message(conv.roles[1], None)
    return  conv.get_prompt()