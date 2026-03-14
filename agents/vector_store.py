"""
Thin wrapper around the native ChromaDB HTTP client.

Why not langchain-chroma?
langchain-chroma 1.x requires langchain-core>=1.1.3 which doesn't
exist yet. langchain-chroma 0.2.x caps chromadb<0.7.0. There is
currently no version that bridges chromadb 1.x + langchain-core 0.3.x.
This wrapper gives us full ChromaDB 1.x support with clean LangChain
integration by implementing the retrieval interface ourselves.

Embedding model: Gemini models/gemini-embedding-001
  - GA release, top MTEB leaderboard
  - Accessed via langchain_google_genai.GoogleGenerativeAIEmbeddings

ChromaDB pydantic-settings fix
ChromaDB's Settings class uses pydantic-settings with env_file=".env" but
no extra="ignore", so it rejects all our app-level env vars as "extra inputs"
at class definition time (before we can pass any constructor args).
We patch model_config BEFORE chromadb.api is imported to allow extra env
vars to pass through silently. This is safe — we are only relaxing validation,
not changing any ChromaDB behaviour.
"""

import logging
from typing import Optional

import chromadb.config
chromadb.config.Settings.model_config["extra"] = "ignore"

import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.config import ChromaConfig, LLMConfig

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Native ChromaDB HTTP client wrapper with Gemini embeddings.

    Usage
    store = ChromaVectorStore()
    results = store.similarity_search("Apple earnings", ticker="AAPL", k=5)
    """

    def __init__(self):
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None
        self._embedder: Optional[GoogleGenerativeAIEmbeddings] = None

    def _get_client(self) -> chromadb.HttpClient:
        """Lazy-init ChromaDB HTTP client."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=ChromaConfig.HOST,
                port=ChromaConfig.PORT,
                settings=Settings(
                    anonymized_telemetry=False,
                    # Prevent pydantic-settings from reading our .env file
                    # and rejecting our app-level env vars as "extra inputs"
                    chroma_client_auth_provider=None,
                ),
            )
        return self._client

    def _get_collection(self):
        """Lazy-init ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=ChromaConfig.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedder(self) -> GoogleGenerativeAIEmbeddings:
        """Lazy-init Gemini embedding model."""
        if self._embedder is None:
            self._embedder = GoogleGenerativeAIEmbeddings(
                model=LLMConfig.GEMINI_EMBEDDING_MODEL,
                google_api_key=LLMConfig.GEMINI_API_KEY,
            )
        return self._embedder

    def similarity_search(
        self,
        query: str,
        ticker: Optional[str] = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Embed the query and retrieve the top-k most similar documents.

        Parameters:
        query  : Natural language search string
        ticker : Optional ticker filter (e.g. "AAPL") — uses ChromaDB
                 metadata filtering, NOT post-filter
        k      : Number of results to return

        Returns:
        List of dicts with keys: id, document, metadata, distance
        """
        try:
            embedder    = self._get_embedder()
            collection  = self._get_collection()

            query_embedding = embedder.embed_query(query)

            where = {"ticker": ticker} if ticker else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            docs = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    docs.append({
                        "id":       doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    })

            logger.info(
                f"ChromaDB search: '{query[:50]}' ticker={ticker} → {len(docs)} docs"
            )
            return docs

        except Exception as exc:
            logger.error(f"ChromaDB similarity_search failed: {exc}")
            return []

    def count(self) -> int:
        """Return total document count in the collection."""
        try:
            return self._get_collection().count()
        except Exception as exc:
            logger.error(f"ChromaDB count failed: {exc}")
            return 0

    def health_check(self) -> bool:
        """Ping ChromaDB — returns True if reachable."""
        try:
            self._get_client().heartbeat()
            return True
        except Exception:
            return False


# Module-level singleton — reuse across graph nodes
_store_instance: Optional[ChromaVectorStore] = None

def get_vector_store() -> ChromaVectorStore:
    """Return the module-level ChromaVectorStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance