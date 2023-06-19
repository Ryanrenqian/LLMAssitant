import gradio as gr
import os
import shutil
from retriaval import storevectors
# 相关操作
def get_answer(query, select_vs, history, mode):
    if mode == "知识库问答":
        vs = storevectors[select_vs]
        for resp, history in vs.get_knowledge_based_answer(
                query=query,  chat_history=history):
            source = "\n\n"
            source += "".join(
                [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["pmid"])[-1]}</summary>\n"""
                 f"""{doc.page_content}\n"""
                 f"""</details>"""
                 for i, doc in enumerate(resp["source_documents"])])
            history[-1][-1] += source
    yield history, ""

# 相关参数配置
def get_vs_list():
    return list(storevectors.keys())

def main():
    default_vs = get_vs_list()[0] if len(get_vs_list()) > 0 else None
    model_status = """模型已成功重新加载，可以开始对话，或从右侧选择模式后开始对话"""
    SENTENCE_SIZE = 512
    knowledge_base_test_mode_info = []
    VECTOR_SEARCH_TOP_K = 10
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

    init_message = f"""欢迎使用 Vibrant KnowlegeAssitant！
    请在右侧切换模式，目前支持直接与 LLM 模型对话或基于本地知识库问答。

    知识库问答模式，选择知识库名称后，即可开始问答，当前知识库{default_vs}，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。

    知识库暂不支持文件删除，该功能将在后续版本中推出。
    """
    default_theme_args = dict(
        font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
        font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
    )
    with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
        vs_path, file_status, model_status = gr.State(
            os.path.join(VS_ROOT_PATH, get_vs_list()[0]) if len(get_vs_list()) > 1 else ""), gr.State(""), gr.State(
            model_status)
        gr.Markdown(webui_title)
        with gr.Tab("知识库检索"):
            with gr.Row():
                with gr.Column(scale=10):
                    chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                        elem_id="chat-box",
                                        show_label=False).style(height=750)
                    query = gr.Textbox(show_label=False,
                                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
                with gr.Column(scale=5):
                    mode = gr.Radio(["LLM 对话", "知识库问答", "Bing搜索问答"],
                                    label="请选择使用模式",
                                    value="知识库问答", )
                    select_vs = gr.Dropdown(get_vs_list(),
                                    label="请选择要加载的知识库",
                                    interactive=True,
                                    value=default_vs
                                    )
                    query.submit(get_answer,
                                    [query, select_vs, chatbot, mode],
                                    [chatbot, query])

    demo.queue(concurrency_count=3).launch(server_name='0.0.0.0',
            server_port=6006,
            show_api=False,
            share=False,
            inbrowser=False)

if __name__ == '__main__':
    main()