from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "war"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
VECTOR_SIZE = 384

# Retrival
CACHE_DIR = ROOT / "rerank_models"
CROSS_ENCORDER_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
DEFUALT_PREFETCH_LIMIT = 15
DEFAULT_RRF_LIMIT = 10
DEFAULT_TOP_K = 5

# Generation
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_LLM_NAME = "qwen2.5:1.5b"
SYSTEM_PROMPT = "You're a useful assistant. Use ONLY the follwoing context to answer.\n\n\
Context:\n{context}"


