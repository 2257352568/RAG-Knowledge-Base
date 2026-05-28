import hashlib, os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from vectordb.cache import QueryCache


RAG_SYSTEM_PROMPT = """\
You are a knowledge base assistant helping users find information from their personal knowledge base.

Rules:
1. Only answer based on the "参考资料" (reference materials) below — do not fabricate.
2. If the reference materials lack relevant information, clearly say so.
3. Cite specific sources (file name, section title) in your answers.
4. If references contain conflicting information, point it out and analyze.
5. Default language: reply in 中文 (Chinese).
   - Only reply in English when the user's question is written entirely in English.
   - If the question is mixed Chinese and English, reply in Chinese."""

QUERY_REWRITE_PROMPT = """\
你是一个查询改写助手。将用户的原始问题改写成 2-3 个更适合检索的查询。

规则：
- 消除代词（"它"、"这个" → 具体对象）
- 扩展同义词和相关术语
- 保留原问题的语义
- 输出格式：每行一个查询，不要编号

原始问题：{question}

改写查询："""


class RAGEngine:
    """RAG query engine with LangChain LCEL + Query Rewrite + context reorganization.

    Architecture:
        User Query → Query Rewrite → Hybrid Retrieval → Rerank
        → Context Reorganization → LangChain LCEL → Answer

    Uses LangChain's ChatOpenAI with DeepSeek base_url (OpenAI-compatible).
    """

    def __init__(
        self,
        retriever,       # Retriever | HybridRetriever
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        reranker=None,   # CrossEncoderReranker | DummyReranker
        cache: QueryCache | None = None,
        enable_rewrite: bool = True,
    ):
        self._retriever = retriever
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._model = model
        self._base_url = base_url
        self._reranker = reranker
        self._cache = cache
        self._enable_rewrite = enable_rewrite

    def _make_llm(self):
        return ChatOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            temperature=0.7,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_mmr: bool = False,
    ) -> dict:
        """Full RAG pipeline: rewrite → retrieve → rerank → reorganize → generate."""
        # Check cache
        if self._cache:
            cached = self._cache.get(question)
            if cached:
                cached["cached"] = True
                return cached

        # Step 1: Query Rewrite
        queries = [question]
        if self._enable_rewrite:
            rewritten = self._rewrite_query(question)
            queries = [question] + rewritten

        # Step 2: Multi-query retrieval + merge
        all_chunks = self._multi_query_retrieve(queries, top_k, use_mmr)

        # Step 3: Rerank
        if self._reranker and all_chunks:
            all_chunks = self._reranker.rerank(question, all_chunks, top_k=top_k)

        # Limit
        all_chunks = all_chunks[:top_k]

        # Step 4: Context reorganization
        context = self._reorganize_context(all_chunks)

        # Step 5: Generate via LangChain LCEL pipeline
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT + "\n\n参考资料：\n\n{context}"),
            ("user", "{question}"),
        ])

        llm = self._make_llm()
        llm.temperature = temperature
        llm.max_tokens = max_tokens

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        result = {
            "answer": answer,
            "sources": all_chunks,
            "cached": False,
        }

        if self._cache:
            self._cache.set(question, result)

        return result

    def query_stream(self, question: str, top_k: int = 5, **kwargs):
        """Streaming RAG query."""
        queries = [question]
        if self._enable_rewrite:
            queries += self._rewrite_query(question)

        all_chunks = self._multi_query_retrieve(queries, top_k)
        if self._reranker and all_chunks:
            all_chunks = self._reranker.rerank(question, all_chunks, top_k=top_k)
        all_chunks = all_chunks[:top_k]

        context = self._reorganize_context(all_chunks)
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT + "\n\n参考资料：\n\n{context}"),
            ("user", "{question}"),
        ])

        llm = self._make_llm()
        llm.temperature = kwargs.get("temperature", 0.7)
        llm.max_tokens = kwargs.get("max_tokens", 2048)

        chain = prompt | llm | StrOutputParser()
        for chunk in chain.stream({"context": context, "question": question}):
            yield chunk

    def _rewrite_query(self, question: str) -> list[str]:
        """LLM rewrites query for better retrieval recall."""
        llm = self._make_llm()
        llm.temperature = 0.3
        llm.max_tokens = 256

        prompt = ChatPromptTemplate.from_messages([
            ("user", QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"question": question})

        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        rewritten = [l for l in lines if l != question]
        return rewritten[:2]

    def _multi_query_retrieve(
        self, queries: list[str], top_k: int, use_mmr: bool = False
    ) -> list[dict]:
        """Retrieve with multiple queries, merge and deduplicate."""
        seen_ids = set()
        merged = []
        for q in queries:
            if use_mmr and hasattr(self._retriever, "retrieve_with_mmr"):
                results = self._retriever.retrieve_with_mmr(q, top_k=top_k)
            else:
                results = self._retriever.retrieve(q, top_k=top_k)
            for r in results:
                doc_id = r.get("id", hashlib.md5(r["text"].encode()).hexdigest()[:12])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    merged.append(r)
        return merged

    def _reorganize_context(self, chunks: list[dict]) -> str:
        """Reorganize retrieved chunks into structured context."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source_path", "未知来源")
            section = meta.get("section_title", "")
            page = meta.get("page_start", "")

            header = f"[{i}] 来源: {source}"
            if section:
                header += f" | 章节: {section}"
            if page:
                header += f" | 页码: {page}"

            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)
