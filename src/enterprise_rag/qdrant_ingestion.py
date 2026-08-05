from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier, PointStruct, Document as QDocument

from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import Union
from langchain_core.documents import Document

from config import QDRANT_URL, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, VECTOR_SIZE

class HybridKnowledgeBuilder: 

    def __init__(self, chunk_size:int=1000, chunk_overlap:int=200):
        self.client = QdrantClient(url=QDRANT_URL)
        self.dense_embedding = DENSE_MODEL
        self.sparse_embedding = SPARSE_MODEL

        self.collection_name = COLLECTION_NAME
        self.vector_size = VECTOR_SIZE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._setup_collection()

    def _setup_collection(self): 
        if not self.client.collection_exists(collection_name=self.collection_name): 
            # from Qdrant docs - adapted to my case 
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25_sparse_vector": SparseVectorParams(
                        modifier=Modifier.IDF # Enable Inverse Document Frequency
                    )
                }
            )
        else: 
            self.client.get_collection(collection_name=self.collection_name)

    def _load(self, file_path:Union[str, Path]) -> list[Document]:
            text_loader = TextLoader(file_path, encoding="utf-8")
            documents = text_loader.load()
            return documents
        
    def _chunk(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            add_start_index=True
        )   
        chunks =  splitter.split_documents(docs)
        return chunks

    def ingest(self, txt_path: Union[str,Path]): 
        load_out = self._load(txt_path)
        chunk_out = self._chunk(load_out)

        # from Qdrant docs - adapted to my case 
        points = []
    
        for idx, item in enumerate(chunk_out):
            passage = item.page_content

            point = PointStruct(
                id=uuid.uuid4().hex, 
                # intstead of passing a simple string I can pass a dictionary --> transform a LC Document into a dictionary
                payload={
                    "page_content": passage,
                    "metadata": item.metadata
                },
                vector={
                    "dense_vector": QDocument(
                        text=passage,
                        model=self.dense_embedding
                    ),
                    "bm25_sparse_vector": QDocument(
                        text=passage,
                        model=self.sparse_embedding
                    )
                }
            )
            points.append(point)

        self.client.upload_points(
            collection_name=self.collection_name, 
            points=points, 
            batch_size=8
        )



if __name__ == "__main__":

    # testing on the fly
    current_path = Path(__file__).resolve().parent

    # remove what's old
    client = QdrantClient(url="http://localhost:6333")
    if client.collection_exists("war"): client.delete_collection("war")
    del client

    hkb = HybridKnowledgeBuilder(collection_name="war")

    # DO NOT REPEAT INGESTION
    # hkb.ingest(current_path / "raw_data" / "gm2_and_gf.txt")

    # qeurying
    from qdrant_client.models import Prefetch, Document as QDocument, Fusion, FusionQuery
    from qdrant_ingestion import HybridKnowledgeBuilder

    hkb = HybridKnowledgeBuilder("war")
    query = "When were atomic bombs thrown on Japan? And how many?"
    results = hkb.client.query_points(
        collection_name=hkb.collection_name,
        prefetch=[
            Prefetch(
                query=QDocument(
                    text=query,
                    model=hkb.dense_embedding
                ),
                using="dense_vector",
                limit=5
            ),
            Prefetch(
                query=QDocument(
                    text=query,
                    model=hkb.sparse_embedding
                ),
                using="bm25_sparse_vector",
                limit=5
            )
        ], # score here is COSINE similarity
        query=FusionQuery(fusion=Fusion.RRF),   # score here is ranking
        limit=3,
        with_payload=True
    )
    import pprint
    pprint.pprint(results.points)

    passages = [
        {
            "id": response.id,
            "text": response.payload["page_content"],
            "meta": response.payload["meta"] # this is a dictionary
        }
        for response in results.points
    ]
    pprint.pprint(passages)


