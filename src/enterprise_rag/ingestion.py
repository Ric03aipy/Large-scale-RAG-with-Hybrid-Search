from config import EMBEDDING_MODEL

from pathlib import Path

from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from typing import Union
from langchain_core.documents import Document



class KnowledgeBaseBuilder: 
    """
    Given a text file build your knowledge base handling the follwing steps: 
    - Load
    - Chunk
    - Embed
    - Store
    """

    # TODO: si può estendere a una lista di file anche

    def __init__(self, db_path:Union[str, Path], collection_name:str, chunk_size:int=1000, chunk_overlap:int=200): 
        self.db_path=db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

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

    def _embed_and_store(self, chunks: list[Document]) -> None: 
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        chorma_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_name=self.collection_name,
            persist_directory=self.db_path
        )
        
    def ingest(self, path: Union[str, Path]) -> None:
        """Handle the whole processing from raw text to vector DB storage. """
        load_out = self._load(path)
        chunk_out = self._chunk(load_out)
        self._embed_and_store(chunk_out)




