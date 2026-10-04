from pathlib import Path

from qdrant_client import QdrantClient
from enterprise_rag.config import QDRANT_URL, COLLECTION_NAME, DENSE_MODEL
from enterprise_rag.qdrant_ingestion import HybridKnowledgeBuilder
from eval.data_script import DATA_FOLDER
import time
from qdrant_client.models import Document as QDocument

TEST_CHUNK_SIZE = 1000
TEST_CHUNK_OVERLAP = 50


if __name__ == "__main__":

    # Remove what's old for test
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME): client.delete_collection(COLLECTION_NAME)

    # Create the Builder 
    hkb = HybridKnowledgeBuilder(TEST_CHUNK_SIZE, TEST_CHUNK_OVERLAP)

    # Fake interaction to ensure everything has been initialized and loaded
    hkb.client.query_points(
        collection_name=hkb.collection_name,
        query=QDocument(text="What is PEP about?", model=DENSE_MODEL),
        using="dense_vector",
    )

    # Ingest all files
    files = DATA_FOLDER.glob("*.rst")
    print("Starting ingestion of knowledge base...")
    start = time.perf_counter()
    for file_path in files: 
        hkb.ingest(file_path)
    end = time.perf_counter()
    seconds = end - start
    print(f"Ingestion completed in {seconds} sec - {seconds / 60:.2f} min")


