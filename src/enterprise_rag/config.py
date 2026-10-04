import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
MY_APP_NAMESPACE = uuid.UUID('fb2a6f4d-0ee1-4c25-95c9-345a40bfe21f') # best practise for uuid5(namespace); string taken from uuid4() run in terminal

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # it takes the env value when in Docker, if local use localhost
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "peps_eval")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
VECTOR_SIZE = 384

# Retrieval
CACHE_DIR = ROOT / "rerank_models"
CROSS_ENCODER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
DEFAULT_PREFETCH_LIMIT = 15
DEFAULT_RRF_LIMIT = 10
DEFAULT_TOP_K = 5

# Generation
OLLAMA_LLM_NAME = os.getenv("OLLAMA_LLM_NAME", "qwen2.5:1.5b")
SYSTEM_PROMPT = (
    "You're a useful assistant. Use ONLY the following context to answer.\n\n\
Context:\n{context}"
)
