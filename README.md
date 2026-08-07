# Enterprise-Grade Local RAG: Hybrid Search & Reranking

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Qdrant%20%7C%20Ollama-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A fully containerized, 100% local Retrieval-Augmented Generation (RAG) pipeline designed with Enterprise constraints in mind. This project moves beyond standard LangChain tutorials by implementing an advanced **Information Retrieval Funnel** (Dense + Sparse Search + Cross-Encoder Reranking) to maximize accuracy while maintaining data privacy.

## 🏗️ Architecture

The system is designed as a decoupled microservices architecture, entirely containerized via Docker and Docker Compose. 

### 1. Frontend UI (Gradio)
A lightweight, interactive web interface.
- Allows raw `.txt` file ingestion (with dynamic chunk size/overlap configuration).
- Exposes retrieval tuning parameters directly to the user (Prefetch limit, RRF limit, Top-K) to interactively evaluate the search accuracy.

### 2. REST API (FastAPI)
The core backend bridging the UI and the AI components. 
- Exposes strict, typed endpoints (`/ask` and `/ingest`) using **Pydantic**.
- Handles asynchronous requests and background thread processing for heavy ingestion tasks without blocking the main event loop.

### 3. Generation (LangChain & Ollama)
- **Local-First:** Uses `Qwen2.5:1.5b` (configurable) served via Ollama, ensuring zero data leakage to external APIs.
- **LCEL Pipeline:** Implements LangChain Expression Language for a clean, modular prompt templating and generation chain.

### 4. Advanced Retrieval (Qdrant & FlashRank)
This is the core engineering highlight of the project. Standard Bi-Encoders (dense vectors) struggle with exact entity matching (IDs, acronyms, specific names). To solve this, the pipeline implements:
- **Hybrid Search:** Qdrant Vector Database stores both Dense Vectors (semantic meaning via `BAAI/bge-small`) and Sparse Vectors (keyword/lexical match via SPLADE).
- **Reciprocal Rank Fusion (RRF):** Merges the results of the dense and sparse prefetch queries to leverage both semantic understanding and exact keyword matching.
- **Cross-Encoder Reranking:** Bi-Encoders are fast but lack deep contextual attention. We funnel the Top-N results from Qdrant into a `ms-marco-MiniLM` Cross-Encoder (via FlashRank) to perform a heavy, highly accurate final reranking before injecting the context into the LLM prompt.


## 🚀 How to Run

The entire stack is isolated in a Docker network.

**0. Prerequisites:** 
Ensure Docker is installed. To enable GPU acceleration for the local LLM, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/enterprise-rag.git](https://github.com/yourusername/enterprise-rag.git)
cd enterprise-rag
```

**2. Start the orchestration:**
```
docker compose up --build
```

**3. Open the application:**
Navigate to http://localhost:7860 in your browser. API documentation (Swagger UI) is available at http://localhost:8000/docs.

## ⚙️ Configuration
The system is highly configurable without altering the core logic:
- `docker-compose.yaml`: Manage exposed ports, volumes, and GPU passthrough.
- `src/enterprise_rag/config.py`: Centralized configuration for Embedding models, Cross-Encoder models, Vector dimensions, and default search limits.

## 🔮 Future Extensions
- Implement multi-format document parsers (PDF, DOCX) via unstructured.io.
- Add batch-ingestion for entire folder directories.
- Expose a CLI tool or a dedicated endpoint for Agentic AI interactions.