import gradio as gr
from dotenv import load_dotenv
from gpt_index import download_loader
from llama_index import GPTSimpleVectorIndex

load_dotenv()

YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")


def load_video(url):
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[url], languages=["pt"])

    return GPTSimpleVectorIndex.from_documents(documents), "Video loaded"


def ask_question(question, index):
    response = index.query(question)

    return response.response, "Question answered"


with gr.Blocks() as video_block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Log")
            log = gr.Textbox(label="Log")
        with gr.Column():
            gr.Markdown("# Youtube Video Question Answering")
            yt_url = gr.Textbox(label="YouTube URL")
            index = gr.State()
            btn_load = gr.Button("Load Video")
            btn_load.click(fn=load_video, inputs=[yt_url], outputs=[index, log])

            question = gr.Textbox(label="Question")
            btn_ask = gr.Button("Ask question")
            response = gr.Textbox(label="Response based on the video")
            btn_ask.click(fn=ask_question, inputs=[question, index], outputs=[response, log])

if __name__ == "__main__":
    video_block.launch(share=False)
