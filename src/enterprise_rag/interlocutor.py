from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from enterprise_rag.config import (
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_RRF_LIMIT,
    DEFAULT_TOP_K,
    OLLAMA_LLM_NAME,
    OLLAMA_URL,
    SYSTEM_PROMPT,
)
from enterprise_rag.pydantic_models import QueryModeEnum
from enterprise_rag.retriever import CrossEncoderReranker, Retriever


class Interlocutor:
    """Abstract the logic of generation in RAG."""

    def __init__(self, model_name: str = OLLAMA_LLM_NAME, temperature: float = 0.2):
        """Define through LCEL syntax the RAG retrieval and generation passage."""

        # Initialize the retrieval component
        self.retriever = Retriever()
        self.reranker = CrossEncoderReranker()

        # Initialize the model
        self.model = ChatOllama(
            model=model_name, temperature=temperature, base_url=OLLAMA_URL
        )

        # Use templating to inject the question in a structured way
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )

        # Convert the complex LLM answer into clean string
        parser = StrOutputParser()

        # Pipe "|" syntax works as in Bash: output of left operand is input of the right one
        self.chain = (
            {
                "context": self._get_custom_context,  # any Python function is ok provided that "{context}" is a string
                "question": RunnablePassthrough(),
            }  # context is retrieved, question pass unchanges
            | prompt  # context and question are formatted in the template prompt
            | self.model  # prompt -> model -> LLM response (containing the answer)
            | parser  # ---> clean answer
        )

    def _get_custom_context(self, user_input: dict) -> str:
        """Bridge between the underlying custom context retrieval and LC context (pure string)."""

        # Unpacking user/UI parameters
        query = user_input["query"]
        prefetch_limit = user_input.get("prefetch_limit", DEFAULT_PREFETCH_LIMIT)
        rrf_limit = user_input.get("rrf_limit", DEFAULT_RRF_LIMIT)
        top_k = user_input.get("top_k", DEFAULT_TOP_K)

        # Information retrieval
        retriever_res = self.retriever.retrieve(
            query, user_input["mode"], prefetch_limit, rrf_limit
        )

        # Apply rerank only if required for hybrid retrieval
        if user_input["mode"] == QueryModeEnum.hybrid_rerank:
            rerank_res = self.reranker.rerank(query, retriever_res, top_k)
            # Building custom context
            context = [r["text"] for r in rerank_res]
        else:
            context = [
                point.payload["page_content"] for point in retriever_res.points[:top_k]
            ]

        return "\n\n---\n\n".join(context)

    def ask(self, user_input: dict) -> str:
        return self.chain.invoke(user_input)

    def __call__(self, user_input: dict) -> str:
        return self.ask(user_input)
