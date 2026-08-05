# interlocutor.py 

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import SYSTEM_PROMPT,  DEFUALT_PREFETCH_LIMIT, DEFAULT_RRF_LIMIT, DEFAULT_TOP_K

from retriever import Retriever, CrossEncoderReranker


class Interlocutor: 
    """Abstract the logic of generation in RAG."""
    def __init__(self, model_name:str="qwen2.5:1.5b", temperature:float=0.8):
        """Define through LCEL syntax the RAG retrieval and generation passage."""

        # Initialize the retrieval component
        self.retriever = Retriever()
        self.reranker = CrossEncoderReranker()

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
                "context": self._get_custom_context,    # any Python function is ok provided that "{context}" is a string    
                "question": RunnablePassthrough()
            }               # context is retrieved, question pass unchanges
            | prompt        # context and question are formatted in the template prompt
            | self.model    # prompt -> model -> LLM response (containing the answer)
            | parser        # ---> clean answer
        )

    def _get_custom_context(self, user_input:dict) -> str:
        """Bridge between the underlying custom context retrieval and LC context (pure string).""" 

        # unpacking user/UI parameters
        query = user_input["query"]
        prefetch_limit = user_input.get("prefetch_limit", DEFUALT_PREFETCH_LIMIT)
        rrf_limit = user_input.get("rrf_limit", DEFAULT_RRF_LIMIT)
        top_k = user_input.get("top_k", DEFAULT_TOP_K)

        # information retrieval
        retriever_res = self.retriever.retrieve(query, prefetch_limit, rrf_limit)
        rerank_res = self.reranker.rerank(query, retriever_res, top_k)

        # building custom context
        context = "\n\n---\n\n".join([r["text"] for r in rerank_res])
        return context
    
    def ask(self, user_input:dict) -> str:
        return self.chain.invoke(user_input)

    def __call__(self, user_input:dict) -> str:
        return self.ask(user_input)


