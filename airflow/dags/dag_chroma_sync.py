"""
Airflow DAG — reads recent news articles from MongoDB,
generates Gemini embeddings, and upserts them into ChromaDB
for semantic search by the News RAG agent.

Schedule : every 4 hours (runs after news scraper)
Retries  : 2
Owner    : stockpilot

ChromaDB document schema:
    id       : url_hash (dedup key)
    document : headline + " " + summary  (what gets embedded)
    metadata : {ticker, source, published_at, url}
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Default args
DEFAULT_ARGS = {
    "owner":            "stockpilot",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
}

# Constants
EMBED_BATCH_SIZE = 50     # Gemini embedding API batch size
LOOKBACK_HOURS   = 5      # sync articles newer than this (overlaps with schedule)


# Task functions
def fetch_recent_articles(**context) -> list[dict]:
    """
    Pulls articles from MongoDB that arrived in the last LOOKBACK_HOURS.
    Returns a list of dicts pushed to XCom for the embed task.
    """
    from datetime import timezone
    from pymongo import MongoClient
    from config.config import MongoConfig

    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    url = (
        f"mongodb://{MongoConfig.USER}:{MongoConfig.PASSWORD}"
        f"@{MongoConfig.HOST}:{MongoConfig.PORT}/{MongoConfig.DB}"
        "?authSource=admin"
    )
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    collection = client[MongoConfig.DB]["articles"]

    articles = list(collection.find(
        {"scraped_at": {"$gte": cutoff}},
        {
            "url_hash":     1,
            "ticker":       1,
            "headline":     1,
            "summary":      1,
            "source":       1,
            "url":          1,
            "published_at": 1,
            "_id":          0,
        }
    ))

    print(f" Fetched {len(articles)} articles from MongoDB (last {LOOKBACK_HOURS}h)")
    return articles


def embed_and_upsert(**context) -> dict:
    """
    Takes the article list from XCom, generates Gemini embeddings in batches,
    and upserts them into ChromaDB.

    Embedding model : models/gemini-embedding-001
    ChromaDB client : native HTTP client (v2 API, no langchain-chroma)
    """
    import chromadb
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from config.config import ChromaConfig, LLMConfig

    articles: list[dict] = context["ti"].xcom_pull(task_ids="fetch_recent_articles")

    if not articles:
        print("No articles to embed — skipping")
        return {"embedded": 0, "skipped": 0}

    #  ChromaDB client (native, no langchain-chroma wrapper) 
    chroma_client = chromadb.HttpClient(
        host=ChromaConfig.HOST,
        port=ChromaConfig.PORT,
    )
    collection = chroma_client.get_or_create_collection(
        name=ChromaConfig.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    #  Gemini embedding model 
    embedder = GoogleGenerativeAIEmbeddings(
        model=LLMConfig.GEMINI_EMBEDDING_MODEL,
        google_api_key=LLMConfig.GEMINI_API_KEY,
    )

    #  Build text to embed: headline + summary 
    def build_embed_text(article: dict) -> str:
        headline = article.get("headline", "").strip()
        summary  = article.get("summary", "").strip()
        return f"{headline}. {summary}" if summary else headline

    # Batch embed and upsert 
    total_embedded = 0
    total_skipped  = 0

    for i in range(0, len(articles), EMBED_BATCH_SIZE):
        batch = articles[i : i + EMBED_BATCH_SIZE]

        texts = [build_embed_text(a) for a in batch]
        ids   = [a["url_hash"] for a in batch]
        metadatas = [
            {
                "ticker":       a.get("ticker", ""),
                "source":       a.get("source", ""),
                "url":          a.get("url", ""),
                # ChromaDB metadata values must be str/int/float/bool
                "published_at": a["published_at"].isoformat()
                                if hasattr(a.get("published_at"), "isoformat")
                                else str(a.get("published_at", "")),
            }
            for a in batch
        ]

        try:
            embeddings = embedder.embed_documents(texts)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total_embedded += len(batch)
            print(f"  Batch {i // EMBED_BATCH_SIZE + 1}: {len(batch)} articles embedded")

        except Exception as exc:
            print(f"  WARNING: batch {i // EMBED_BATCH_SIZE + 1} failed: {exc}")
            total_skipped += len(batch)

    summary = {"embedded": total_embedded, "skipped": total_skipped}
    print(f" ChromaDB upsert complete — {summary}")
    return summary


def validate_chroma_sync(**context) -> None:
    """
    Confirms ChromaDB collection has documents and the latest
    sync added at least some embeddings (or warns if 0).
    """
    import chromadb
    from config.config import ChromaConfig

    embed_summary = context["ti"].xcom_pull(task_ids="embed_and_upsert")
    articles      = context["ti"].xcom_pull(task_ids="fetch_recent_articles")

    chroma_client = chromadb.HttpClient(
        host=ChromaConfig.HOST,
        port=ChromaConfig.PORT,
    )
    collection = chroma_client.get_or_create_collection(ChromaConfig.COLLECTION_NAME)
    total_in_db = collection.count()

    print(f"ChromaDB collection '{ChromaConfig.COLLECTION_NAME}': {total_in_db} total documents")
    print(f"This run: {embed_summary}")

    if articles and embed_summary and embed_summary.get("embedded", 0) == 0:
        print(
            "WARNING: Articles were fetched but none were embedded. "
            "Check Gemini API key and ChromaDB connectivity."
        )
    else:
        print("ChromaDB sync validation passed")


# DAG definition
with DAG(
    dag_id="chroma_sync_pipeline",
    description="Embed news articles with Gemini -> upsert to ChromaDB",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 */4 * * *",   # every 4 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["stockpilot", "embeddings", "chromadb", "rag"],
) as dag:

    # Task 1 — pull recent articles from MongoDB
    fetch_task = PythonOperator(
        task_id="fetch_recent_articles",
        python_callable=fetch_recent_articles,
    )

    # Task 2 — embed with Gemini + upsert to ChromaDB
    embed_task = PythonOperator(
        task_id="embed_and_upsert",
        python_callable=embed_and_upsert,
    )

    # Task 3 — validate ChromaDB state
    validate_task = PythonOperator(
        task_id="validate_chroma_sync",
        python_callable=validate_chroma_sync,
    )

    # DAG flow: fetch -> embed -> validate
    fetch_task >> embed_task >> validate_task