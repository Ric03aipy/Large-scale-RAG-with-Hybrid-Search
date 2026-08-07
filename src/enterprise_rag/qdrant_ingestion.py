import uuid
from pathlib import Path

from config import COLLECTION_NAME, DENSE_MODEL, QDRANT_URL, SPARSE_MODEL, VECTOR_SIZE
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import Document as QDocument



class HybridKnowledgeBuilder:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initilize the Client to the Qdrant server."""
        self.client = QdrantClient(url=QDRANT_URL)
        self.dense_embedding = DENSE_MODEL
        self.sparse_embedding = SPARSE_MODEL

        self.collection_name = COLLECTION_NAME
        self.vector_size = VECTOR_SIZE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._setup_collection()

    def _setup_collection(self):
        """Connect to the collection if exists otherwise first create it."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            # from Qdrant docs - adapted to my case
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": SparseVectorParams(
                        modifier=Modifier.IDF  # Enable Inverse Document Frequency
                    )
                },
            )
        else:
            self.client.get_collection(collection_name=self.collection_name)

    def _load(self, file_path: str | Path) -> list[Document]:
        """Load a txt file."""
        text_loader = TextLoader(file_path, encoding="utf-8")
        documents = text_loader.load()
        return documents

    def _chunk(self, docs: list[Document]) -> list[Document]:
        """Divide documents into a chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(docs)
        return chunks

    def ingest(self, txt_path: str | Path) -> None:
        """Store the file in the vector DB."""
        load_out = self._load(txt_path)
        chunk_out = self._chunk(load_out)

        # the following is an adaptation to my case of a snippet taken from Qdrant docs
        points = []

        for idx, item in enumerate(chunk_out):
            passage = item.page_content

            point = PointStruct(
                id=uuid.uuid4().hex,
                # intstead of passing a simple string I can pass a dictionary --> transform a LC Document into a dictionary
                payload={"page_content": passage, "metadata": item.metadata},
                vector={
                    "dense_vector": QDocument(text=passage, model=self.dense_embedding),
                    "bm25_sparse_vector": QDocument(
                        text=passage, model=self.sparse_embedding
                    ),
                },
            )
            points.append(point)

        self.client.upload_points(
            collection_name=self.collection_name, points=points, batch_size=8
        )
