from typing import Any

import gradio as gr
import requests
from enterprise_rag.connection_config import HOST_URL, UVICORN_PORT

FASTAPI_URL = f"http://{HOST_URL}:{UVICORN_PORT}"

def ingest_file(file_obj: Any, chunk_size: int, chunk_overlap: int) -> str:
    """Ask the server to ingest the uploaded file."""

    if file_obj is None:
        return "No file uploaded. Try again."

    # e.g.: v/tmp/gradio/21565fc36bbb54c40a3c71756bdd94a6bc39dc78101bc8506dcd063ea4bedcab/gm2_and_gf.txt
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    try:
        # read the file
        with open(file_path, "rb") as f:
            # pack in dictionary/json format
            files = {
                "file": f,
            }
            payload = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

            # send request: I CAN'T put all together --> data and files MUST BE separated
            # it is due to HTTP protocol: files are *multipart/form-data format*,
            # instead, plain text and numbers are *application/json* payload
            response = requests.post(
                FASTAPI_URL + "/ingest",
                data=payload,
                files=files,
            )
            # ARGS: data --> form, files --> multipart (files), json --> pydantic objects

        # handle response status
        if response.status_code == 200:
            data = response.json()
            # use response.get as for dictionaries to grab the field
            return f"{data.get('response')}"
        else:
            return f"Server Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"An error occurred:\n{e!s}"


def send_query(input_text: str, prefetch_limit: int, rrf_limit: int, top_k: int, mode:str):
    """Receive an LLM-powered answer to a question about the ingested files."""

    if not input_text:
        return ""
    try:
        user_input = {
            "query": input_text,
            "prefetch_limit": prefetch_limit,
            "rrf_limit": rrf_limit,
            "top_k": top_k,
            "mode": mode
        }
        response = requests.post(
            FASTAPI_URL + "/ask", json=user_input
        )  # it expects a pydantic model QueryRequest(BaseModel) so I use json

        if response.status_code == 200:
            data = response.json()
            return data.get("ai_answer")
        else:
            return f"Server Error ({response.status_code}): {response.text}"

    except Exception as e:
        return f"An error occurred:\n{e!s}"

# A simple Gradio UI 
def main():
    with gr.Blocks() as demo:  # ingest section
        gr.Markdown(
            "## Ingest files (one per time) (only *.txt* supported for this demo)"
        )
        with gr.Row():
            with gr.Row():
                chunk_size = gr.Slider(minimum=100, maximum=5000, value=1000)
                chunk_overlap = gr.Slider(minimum=50, maximum=500, value=200)
            file = gr.File()
        ingest_btn = gr.Button("Click to Ingest the file uploaded")
        ingestion_status = gr.Textbox(label="Not ingested yet...")
        ingest_btn.click(
            fn=ingest_file,
            inputs=[file, chunk_size, chunk_overlap],
            outputs=ingestion_status,
        )
        gr.Markdown("## Interrogate the Knowledge Base")
        with gr.Row():
            prefetch_limit = gr.Slider(minimum=5, maximum=50, value=15)
            rrf_limit = gr.Slider(minimum=5, maximum=50, value=10)
            top_k = gr.Slider(minimum=1, maximum=10, value=5)
            mode = gr.Radio(
                choices=[
                    "dense", 
                    "sparse", 
                    "hybrid", 
                    "hybrid_rerank"
                ],
                label="Select method for retrieval",
                value="hybrid_rerank" # Default
            )
        with gr.Row():
            input_text = gr.TextArea("Write here your prompt...")
            answer_text = gr.TextArea(
                placeholder="Here the output will be displayed...", interactive=False
            )
        ask_btn = gr.Button("Retrieve")
        ask_btn.click(
            fn=send_query,
            inputs=[input_text, prefetch_limit, rrf_limit, top_k, mode],
            outputs=answer_text,
        )

    demo.launch(
        server_name="0.0.0.0",  # necessary for Docker
        server_port=7860,
    )


if __name__ == "__main__":
    main()
