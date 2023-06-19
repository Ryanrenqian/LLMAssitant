import gradio as gr
import os
from models import UrlLLM
from langchain.agents import AgentType
models = {
    'guanaco-33b': UrlLLM(url='http://region-3.seetacloud.com:54504/prompt'),
}
def get_tools(selected_tools):
    return []
# 相关操作
def fake_op(*args,**kwargs):
    pass
def answer_with_tools(chatbot, query,temperature,top_p,max_length,selected_tools,selected_llm):
    res = None
    llm = models['selected_llm']
    tools = get_tools(selected_tools)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,return_intermediate_steps=True)
    res = agent.run(query)
    # 返回结果
    chatbot.append([query,res])
    query = ''
    return chatbot, query
def main():
    # web-ui 部分代码
    block_css = """.importantButton {
        background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
        border: none !important;
    }
    .importantButton:hover {
        background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
        border: none !important;
    }"""
    webui_title = """
    # 🎉Vibrant KnowlegeAssitant🎉
    """
    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )
    with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    # with gr.Blocks() as demo:
        gr.Markdown(webui_title)
        with gr.Tab("Ai强化检索"):
            init_message = f"""欢迎使用Ai强化检索，该方案采用最先进的sentence_transformers对文本内容进行向量化表征，能够有效的提升检索的相关性。
            """
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([[None, init_message]],
                                        elem_id="chat-box",
                                        show_label=False).style(height=750)
                    query = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
                with gr.Column(scale=5):
                    select_vs = gr.Dropdown([],
                                    label="请选择要加载的知识库",
                                    interactive=True,
                                    value=fake_op
                                    )
                    query.submit(fake_op,
                                    [query, select_vs, chatbot],
                                    [chatbot, query])
        
        with gr.Tab("Ai推理问答"):
            init_message = f"""欢迎使用Ai推理问答。
            Ai智能问答是基于LLM模型和Re-Act推理框架的人工智能助手。它能够有效利用已有的检索工具，其模式为：act then rethink。
            具体来说：提问：食管癌有哪些治疗方案。
            在第一步： 他会先学习： 什么是食管癌？通过检索获取知识。
            第二步： 再次发问，食管癌有哪些治疗方案？通过检索获取知识。
            如此循环下去，直到能够回答问题。该方法能够有效的减少幻想，减少事实性错误。
            但是推理过程时会受到生成参数的影响，如果遇到了奇怪的推理，可以通过优化自己的问题和模型的推理参数以达到想要的目的。
            """
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([[None, init_message],],
                                        elem_id="chat-box",
                                        show_label=False).style(height=750)
                    query = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
                    
                with gr.Column(scale=5):
                    temperature = gr.Slider(0, 1, value=0.7, label="temperature", info="Choose between 0 and 1")
                    top_p = gr.Slider(0, 1, value=0.95, label="top_p", info="Choose between 0 and 1")
                    max_length  = gr.Slider(0, 2048, value=0.95, label="top_p", info="Choose between 0 and 1")
                    selected_tools = gr.CheckboxGroup(['pubmed','arvix','google_search'],label='Tools',info='available tools for ai')
                    selected_llm = gr.CheckboxGroup(list(models.keys()),label='Model',info='available LLM Model')
                query.submit(answer_with_tools,
                            [chatbot, query,temperature,top_p,max_length,selected_tools,selected_llm],
                            [chatbot, query]
                            )
    demo.queue(concurrency_count=3).launch(server_name='0.0.0.0',
            server_port=6007,
            show_api=False,
            share=False,
            inbrowser=False)

if __name__ == '__main__':
    main()