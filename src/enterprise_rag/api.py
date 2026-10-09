import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from enterprise_rag.config import ROOT
from enterprise_rag.interlocutor import Interlocutor
from enterprise_rag.pydantic_models import QueryRequest
from enterprise_rag.qdrant_ingestion import HybridKnowledgeBuilder


# The lifespan handles the whole cycle of life of the FastAPI app
@asynccontextmanager
async def lifespan_func(app: FastAPI):

    # STARTUP: all the code here is done before starting
    print("Starting the server...")
    interlocutor = Interlocutor()

    # this is the **DICTIONARY** of the initial state (dictionary or nothing)
    shared_data = {"interlocutor": interlocutor}
    yield shared_data

    # SHUTDOWN: all the code here is the cleanup
    print("Shut down with success.")


# FastAPI app
app = FastAPI(title="Local Hybrid-Search RAG - API", lifespan=lifespan_func)


# "/" static path or 'Route' - Root
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}


# 'ellipses' - ... - in FastAPI means "this field is mandatory and it has no default value"
#
# NO 'async' --> FastAPI opens a separated threadpool to not block anything.
# Due to not being asynchronous functions under the hood
# async requires await and await can only be used with async functions
@app.post("/ingest")
def ingest(
    file: UploadFile = File(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
):
    # creating a temporary file to control location of the new file
    tmp_dir = ROOT / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    filename = Path(file.filename)
    full_path = tmp_dir / filename
    with open(full_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        hkb = HybridKnowledgeBuilder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        hkb.ingest(full_path)
        return {"response": "Ingestion completed successfully"}
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong with ingestion. Check connection. Try later.",
        )
    finally:
        full_path.unlink(missing_ok=True)
        print("Temporary file deleted after ingestion.")


@app.post("/ask")
def ask(user_input: QueryRequest, request: Request):
    interlocutor: Interlocutor = request.state.interlocutor
    ai_answer = interlocutor(user_input.model_dump())
    return {"ai_answer": ai_answer}


# THE FOLLOWING IS LOCAL ONLY. NOT SUGGESTED FOR CONTAINERIZATION WITH DOCKER

# from connection_config import HOST_URL, UVICORN_PORT
# if __name__ == "__main__":
#     # uv uvicorn api:app --reload
#     uvicorn.run(
#         "api:app",    # app=app, if reload=False; "api" <=> <filename> before ".py"
#         host=HOST_URL, # 0.0.0.0 if docker
#         port=UVICORN_PORT, # default
#         reload=True # developmente only
#     )
