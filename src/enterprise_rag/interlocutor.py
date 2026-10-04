from config import (
    DEFAULT_RRF_LIMIT,
    DEFAULT_TOP_K,
    DEFAULT_PREFETCH_LIMIT,
    OLLAMA_LLM_NAME,
    OLLAMA_URL,
    SYSTEM_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from retriever import CrossEncoderReranker, Retriever
from pydantic_models import QueryModeEnum


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
        retriever_res = self.retriever.retrieve(query, user_input['mode'], prefetch_limit, rrf_limit)

        # Apply rerank only if required for hybrid retrival
        if user_input['mode'] == QueryModeEnum.hybrid_rerank:
            rerank_res = self.reranker.rerank(query, retriever_res, top_k)
            # Building custom context
            context = "\n\n---\n\n".join([r["text"] for r in rerank_res])
            return context
        return "\n\n---\n\n".join([point.payload['page_content'] for point in sorted(retriever_res.points, key=lambda x: -x.score)[:top_k]])

    def ask(self, user_input: dict) -> str:
        return self.chain.invoke(user_input)

    def __call__(self, user_input: dict) -> str:
        return self.ask(user_input)
