import gradio as gr
# global value
vectorstore = ''
vetorstore_path = {
    "test":'test',
    'pubmed':'pubmed',
    'patents':'patents',
    'aacr':'aacr',
}
def get_vs_list():
    return list(vetorstore_path.keys())
def get_model_list():
    return ['test','gpt3.5-turbo','vicuna-13B','vicuna-7B']

def init_model():
    return "等待加载模型"
def change_mode(mode, history):
    if mode == "知识库问答":
        return  gr.update(visible=True), [[None, history]]
    else:
        return  gr.update(visible=False), history
def change_vectorstore(name, chatbot):
    return vetorstore_path[vectorstore],chatbot

def get_answer(msg,mode,chatbot):
    if mode == '知识库问答':
        pass
    return msg,chatbot
    
def refresh_vs_list():
    pass
def main():
    block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
    webui_title = """
    # 🎉Vibrant Assitant WebUI🎉
    """
    init_message = """欢迎使用 Vibrant Assitant Web UI！

    请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

    知识库问答模式，选择知识库名称和模型后，即可开始问答。

    """

    # 初始化消息
    model_status = init_model()

    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )

    with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
        model_status = gr.State(model_status)
        gr.Markdown(webui_title)
        with gr.Tab("文档小助理"):
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                        elem_id="chat-box",
                                        show_label=False).style(height=750)
                    msg = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
                    clear = gr.Button("Clear")
                    clear.click(lambda: None, None, chatbot, queue=False)
                    
                with gr.Column(scale=5):
                    mode = gr.Radio(["知识库问答"],
                                    label="请选择使用模式",
                                    value="知识库问答", )
                    model_setting = gr.Accordion("模型设定")
                    with model_setting:
                        select_model = gr.Dropdown(get_model_list(),
                                            label="请选择要加载的模型",
                                            interactive=True,
                                            value=get_model_list()[0] if len(get_model_list()) > 0 else None
                                            )
                        select_model.change(
                            fn=lambda x:None,
                            inputs=[],
                            outputs=[]
                            )
                    vs_setting = gr.Accordion("知识库设定")
                    mode.change(fn=change_mode,
                                inputs=[mode, chatbot],
                                outputs=[mode, chatbot])
                    with vs_setting:
                        select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                        select_vs.change(fn=change_vectorstore,
                                     inputs=[select_vs, chatbot],
                                     outputs=[select_vs,chatbot])
                msg.submit(
                        get_answer,
                        [msg,mode,select_vs,chatbot],
                        [msg,chatbot]
                    )
                

        demo.load(
            fn=refresh_vs_list,
            inputs=None,
            # outputs=[select_vs, select_vs_test],
            queue=True,
            show_progress=False,
        )

    (demo
    .queue(concurrency_count=3)
    .launch(server_name='127.0.0.1',
            server_port=7860,
            show_api=False,
            share=False,
            inbrowser=False))
if __name__ == '__main__':
    main()