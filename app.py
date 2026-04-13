import gradio as gr


def echo(text: str) -> str:
    return text


with gr.Blocks() as demo:
    gr.Markdown("## FPL Transfer Strategist — Hello World")
    inp = gr.Textbox(label="Input")
    btn = gr.Button("Echo")
    out = gr.Textbox(label="Output")
    btn.click(fn=echo, inputs=inp, outputs=out)

demo.queue()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
