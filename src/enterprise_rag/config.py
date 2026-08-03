from pathlib import Path

import os
from dotenv import load_dotenv # not used for now

ROOT = Path(__file__).resolve().parent
CHROMA_DB_PATH = ROOT / "chroma_db"
COLLECTION_NAME = "policies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_LLM_NAME = "qwen2.5:1.5b"

K_CHUNKS = 3
SYSTEM_PROMPT = "Sei un assistente utile. Usa SOLO il seguente contesto per rispondere.\n\n\
Contesto:\n{context}"

# TODO: controllare se questo flag ha senso di esistere, se magari il 
# ChatOllama gestisce automaticamente la connessione con Docker oppure va
# scritto diversamente e allora va splittata l'interfaccia LOCAL or not
LOCAL = True
