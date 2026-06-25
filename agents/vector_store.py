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

Hybrid search: BM25 (FTS) + vector similarity, merged via Reciprocal Rank Fusion
  - Vector search: semantic similarity via Gemini embeddings
  - BM25 search:   keyword/exact-term matching via ChromaDB v2 built-in FTS
  - RRF:           merges both ranked lists without needing score normalisation
  - Best of both worlds: catches exact ticker mentions AND semantic context

BM25 implementation note:
  ChromaDB v2 FTS is triggered via where_document={"$contains": query}.
  Do NOT use query_texts=[query] — that triggers ChromaDB's internal embedder
  (384-dim sentence-transformers) which conflicts with our Gemini embeddings
  (3072-dim) and causes a dimension mismatch error.

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

# Reciprocal Rank Fusion constant — 60 is the standard default
RRF_K = 60


def _reciprocal_rank_fusion(
    vector_hits: list[dict],
    bm25_hits:   list[dict],
    k:           int = 5,
) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF score = 1/(RRF_K + rank_in_vector_results)
              + 1/(RRF_K + rank_in_bm25_results)

    Documents that appear in both lists get a combined score boost.
    Documents that appear in only one list still get a partial score.

    Args:
        vector_hits : Results from vector similarity search (ordered by relevance)
        bm25_hits   : Results from BM25 full-text search (ordered by relevance)
        k           : Number of final results to return

    Returns:
        Merged list of result dicts, sorted by RRF score descending.
        Each dict has: id, document, metadata, rrf_score,
                       vector_rank (None if not in vector results),
                       bm25_rank   (None if not in BM25 results)
    """
    scores: dict[str, dict] = {}

    # Score from vector results
    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        if doc_id not in scores:
            scores[doc_id] = {
                "id":           doc_id,
                "document":     hit["document"],
                "metadata":     hit["metadata"],
                "rrf_score":    0.0,
                "vector_rank":  None,
                "bm25_rank":    None,
            }
        scores[doc_id]["rrf_score"]   += 1.0 / (RRF_K + rank + 1)
        scores[doc_id]["vector_rank"]  = rank + 1

    # Score from BM25 results
    for rank, hit in enumerate(bm25_hits):
        doc_id = hit["id"]
        if doc_id not in scores:
            scores[doc_id] = {
                "id":           doc_id,
                "document":     hit["document"],
                "metadata":     hit["metadata"],
                "rrf_score":    0.0,
                "vector_rank":  None,
                "bm25_rank":    None,
            }
        scores[doc_id]["rrf_score"] += 1.0 / (RRF_K + rank + 1)
        scores[doc_id]["bm25_rank"]  = rank + 1

    # Sort by RRF score descending and return top-k
    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:k]


def _extract_bm25_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a query for BM25 FTS.

    ChromaDB's $contains operator does exact substring matching on the
    document text. We run multiple single-keyword searches and merge results
    to approximate BM25 behaviour across the query terms.

    Filters out common stop words to focus on meaningful terms.
    """
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "what", "how",
        "why", "when", "where", "who", "which", "that", "this", "these",
        "those", "about", "after", "before", "during", "it", "its",
        "any", "some", "all", "not", "no", "can", "there",
    }
    words = query.lower().split()
    keywords = [w.strip(".,?!:;\"'()[]") for w in words]
    keywords = [w for w in keywords if w and w not in STOP_WORDS and len(w) > 2]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


class ChromaVectorStore:
    """
    Native ChromaDB HTTP client wrapper with Gemini embeddings.
    Supports both pure vector search and hybrid BM25 + vector search.

    Usage:
        store = ChromaVectorStore()

        # Hybrid search (recommended) — BM25 + vector via RRF
        results = store.hybrid_search("Apple earnings beat", ticker="AAPL", k=5)

        # Pure vector search (legacy)
        results = store.similarity_search("Apple earnings", ticker="AAPL", k=5)
    """

    def __init__(self):
        self._client:   Optional[chromadb.HttpClient]            = None
        self._collection                                          = None
        self._embedder: Optional[GoogleGenerativeAIEmbeddings]   = None

    def _get_client(self) -> chromadb.HttpClient:
        """Lazy-init ChromaDB HTTP client."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=ChromaConfig.HOST,
                port=ChromaConfig.PORT,
                settings=Settings(
                    anonymized_telemetry=False,
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

    #Hybrid search (BM25 + vector via RRF) 

    def hybrid_search(
        self,
        query:  str,
        ticker: Optional[str] = None,
        k:      int = 5,
        fetch_k_multiplier: int = 3,
    ) -> list[dict]:
        """
        Hybrid search: combines BM25 full-text search and vector similarity
        search via Reciprocal Rank Fusion (RRF).

        Fetches fetch_k_multiplier * k candidates from each search method,
        then merges and re-ranks to return the top k results.

        BM25 uses ChromaDB v2's where_document $contains operator for FTS —
        runs per keyword and merges hits, avoiding the embedding dimension
        mismatch that occurs with query_texts=[query].

        Args:
            query               : Natural language search string
            ticker              : Optional ticker filter (e.g. "AAPL")
            k                   : Number of final results to return
            fetch_k_multiplier  : How many extra candidates to fetch per method
                                  before RRF merging (default: 3x)

        Returns:
            List of dicts with keys: id, document, metadata,
                                     rrf_score, vector_rank, bm25_rank
        """
        fetch_k = k * fetch_k_multiplier
        where   = {"ticker": ticker} if ticker else None

        #Vector search 
        vector_hits: list[dict] = []
        try:
            embedder    = self._get_embedder()
            collection  = self._get_collection()

            query_embedding = embedder.embed_query(query)
            vector_results  = collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            if vector_results and vector_results.get("ids") and vector_results["ids"][0]:
                for i, doc_id in enumerate(vector_results["ids"][0]):
                    vector_hits.append({
                        "id":       doc_id,
                        "document": vector_results["documents"][0][i],
                        "metadata": vector_results["metadatas"][0][i],
                        "distance": vector_results["distances"][0][i],
                    })

            logger.info(f"Vector search: '{query[:50]}' → {len(vector_hits)} hits")

        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")

        #BM25 full-text search via $contains 
        # ChromaDB v2 FTS: use where_document={"$contains": keyword}
        # Run per keyword and deduplicate results to approximate BM25 ranking.
        bm25_hits: list[dict] = []
        try:
            collection = self._get_collection()
            keywords   = _extract_bm25_keywords(query)

            seen_ids: set[str] = set()

            for keyword in keywords[:5]:  # cap at 5 keywords to limit API calls
                try:
                    where_doc = {"$contains": keyword}

                    # Combine ticker filter with document filter if needed
                    if where:
                        fts_results = collection.get(
                            where=where,
                            where_document=where_doc,
                            include=["documents", "metadatas"],
                            limit=fetch_k,
                        )
                    else:
                        fts_results = collection.get(
                            where_document=where_doc,
                            include=["documents", "metadatas"],
                            limit=fetch_k,
                        )

                    ids  = fts_results.get("ids", [])
                    docs = fts_results.get("documents", [])
                    metas = fts_results.get("metadatas", [])

                    for i, doc_id in enumerate(ids):
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            bm25_hits.append({
                                "id":       doc_id,
                                "document": docs[i] if i < len(docs) else "",
                                "metadata": metas[i] if i < len(metas) else {},
                            })

                except Exception as kw_exc:
                    logger.debug(f"BM25 keyword '{keyword}' failed: {kw_exc}")
                    continue

            logger.info(f"BM25 search:   '{query[:50]}' → {len(bm25_hits)} hits")

        except Exception as exc:
            logger.error(f"BM25 search failed: {exc}")

        #Reciprocal Rank Fusion 
        if not vector_hits and not bm25_hits:
            logger.warning("Both vector and BM25 search returned no results")
            return []

        merged = _reciprocal_rank_fusion(vector_hits, bm25_hits, k=k)

        logger.info(
            f"Hybrid search: '{query[:50]}' ticker={ticker} → "
            f"{len(merged)} docs (vector={len(vector_hits)}, bm25={len(bm25_hits)})"
        )
        return merged

    #Pure vector search (kept for backwards compatibility) 

    def similarity_search(
        self,
        query:  str,
        ticker: Optional[str] = None,
        k:      int = 5,
    ) -> list[dict]:
        """
        Pure vector similarity search via Gemini embeddings.
        Kept for backwards compatibility — prefer hybrid_search() for new code.

        Returns:
            List of dicts with keys: id, document, metadata, distance
        """
        try:
            embedder   = self._get_embedder()
            collection = self._get_collection()

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
                f"Vector search: '{query[:50]}' ticker={ticker} → {len(docs)} docs"
            )
            return docs

        except Exception as exc:
            logger.error(f"ChromaDB similarity_search failed: {exc}")
            return []

    #Utility 

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


#Module-level singleton 

_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Return the module-level ChromaVectorStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance