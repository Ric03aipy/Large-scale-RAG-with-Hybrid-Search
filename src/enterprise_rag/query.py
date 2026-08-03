# interlocutor.py 

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import SYSTEM_PROMPT, COLLECTION_NAME, CHROMA_DB_PATH, EMBEDDING_MODEL, K_CHUNKS

class Interlocutor: 
    """Abstract the logic of generation in RAG."""
    def __init__(self, model_name:str="qwen2.5:1.5b", temperature:float=0.8):
        """Define through LCEL syntax the RAG retrieval and generation passage."""

        # Initialize the vector db
        self.chroma_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            collection_name=COLLECTION_NAME
        )

        # Transform a vector db into an object "Retriever" (a LCEL component)
        retriever = self.chroma_db.as_retriever(search_kwargs={"k": K_CHUNKS})

        # Initialize the model
        self.model = ChatOllama(
            model=model_name,
            temperature=temperature
        )

        # Use templating to inject the question in a structured way
        prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}")
        ])

        # Convert the complex LLM answer into clean string
        parser = StrOutputParser()

        # Pipe "|" syntax works as in Bash: output of left operand is input of the right one
        self.chain = (
            {
                "context": retriever,   
                "question": RunnablePassthrough()
            }               # context is retrieved, question pass unchanges
            | prompt        # context and question are formatted in the template prompt
            | self.model    # prompt -> model -> LLM response (containing the answer)
            | parser        # ---> clean answer
        )
        
    def ask(self, query:str) -> str:
        return self.chain.invoke(query)



# query.py
interlocutor = Interlocutor()
answer = interlocutor.ask("Qual è il rimborso giornaliero per un viaggio di lavoro su Marte?")
print(answer)