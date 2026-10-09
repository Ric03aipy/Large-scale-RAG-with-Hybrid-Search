import time

from qdrant_client import QdrantClient
from qdrant_client.models import Document as QDocument

from enterprise_rag.config import COLLECTION_NAME, DENSE_MODEL, QDRANT_URL
from enterprise_rag.qdrant_ingestion import HybridKnowledgeBuilder
from eval.eval_config import DATA_FOLDER, TEST_CHUNK_OVERLAP, TEST_CHUNK_SIZE

if __name__ == "__main__":
   
    # Ingest all files
    files = DATA_FOLDER.glob("*.rst")
    files = list(files)
    
    # If files is empty we don't want to erase the current collection 
    if not files: 
        raise FileNotFoundError(f"{DATA_FOLDER} doesn't contain any '*.rst' file. Please run data_script.py before.")
   
    # Remove what's old for test
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # Create the Builder
    hkb = HybridKnowledgeBuilder(TEST_CHUNK_SIZE, TEST_CHUNK_OVERLAP)

    # Fake interaction to ensure everything has been initialized and loaded
    hkb.client.query_points(
        collection_name=hkb.collection_name,
        query=QDocument(text="What is PEP about?", model=DENSE_MODEL),
        using="dense_vector",
    )

    
    print("Starting ingestion of knowledge base...")
    num_chunks = 0
    start = time.perf_counter()
    for file_path in files:
        num_chunks += hkb.ingest(file_path)
    end = time.perf_counter()
    seconds = end - start
    print(f"Ingestion of {num_chunks} chunks completed in {seconds:.2f} sec - {seconds / 60:.2f} min")
    print(f"Ingestion speed: {num_chunks / seconds :.2f} chunk/s")
