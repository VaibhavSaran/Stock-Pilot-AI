"""
Airflow DAG — archives data older than 90 days to S3 and removes it
from the hot stores (PostgreSQL, MongoDB, ChromaDB).

Schedule : @daily
Retries  : 3 (exponential backoff, 5 min base)
Owner    : stockpilot

Dry-run control:
    Airflow Variable "S3_ARCHIVE_DRY_RUN" (default: "true").
    "true"  -> log exactly what WOULD be archived/deleted. Zero S3 writes, zero deletes.
    "false" -> actually write to S3 and delete from the source stores.

S3 layout:
    s3://stockpilot-ai-archive-vergil2026/prices/{ticker}/{date}.parquet
    s3://stockpilot-ai-archive-vergil2026/news/{date}.jsonl

Auth: relies on the EC2 instance's attached IAM role via boto3's default
credential chain. Never hardcode access keys.

ChromaDB purge note:
    published_at is stored as an ISO-8601 string in Chroma metadata (see
    dag_chroma_sync.py). Whether Chroma's $lt operator reliably supports
    lexicographic string comparison varies by version, so purge_chroma_vectors
    plays it safe: pull all vectors with a published_at metadata field via
    collection.get(), filter older-than-cutoff in Python, then delete by the
    resulting ID list. No separate S3 export is needed here — the underlying
    article text is already archived by archive_mongo_news.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator


# Default args

DEFAULT_ARGS = {
    "owner":            "stockpilot",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

# Constants
S3_BUCKET       = "stockpilot-ai-archive-vergil2026"
RETENTION_DAYS  = 90


def _is_dry_run() -> bool:
    return Variable.get("S3_ARCHIVE_DRY_RUN", default_var="true").strip().lower() == "true"


def _cutoff_date():
    """Date 90 days before today (UTC) — rows/docs/vectors older than this get archived."""
    return (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).date()


# Task functions

def archive_postgres_prices(**context) -> dict:
    """
    For each ticker, finds price rows older than the cutoff, groups them by
    (ticker, date), writes each group as Parquet to S3 (skipping dates already
    archived so reruns are idempotent), then deletes the archived rows from
    Postgres. In dry run: logs what would happen, writes/deletes nothing.
    """
    import boto3
    import pandas as pd
    from botocore.exceptions import ClientError
    from sqlalchemy import create_engine, text

    from config.config import PostgresConfig, TRACKED_TICKERS

    dry_run = _is_dry_run()
    cutoff  = _cutoff_date()

    engine = create_engine(PostgresConfig.URL, pool_pre_ping=True, future=True)
    s3     = boto3.client("s3")

    summary: dict[str, dict] = {}

    with engine.connect() as conn:
        for ticker in TRACKED_TICKERS:
            rows = conn.execute(
                text("""
                    SELECT ticker, open, high, low, close, volume, price_date
                    FROM stock_prices
                    WHERE ticker = :ticker AND price_date < :cutoff
                    ORDER BY price_date
                """),
                {"ticker": ticker, "cutoff": cutoff},
            ).fetchall()

            if not rows:
                summary[ticker] = {"dates_archived": 0, "rows_archived": 0, "dates_skipped": 0}
                continue

            df = pd.DataFrame(rows, columns=["ticker", "open", "high", "low", "close", "volume", "price_date"])

            dates_archived, dates_skipped, rows_archived = [], [], 0
            archived_dates_for_delete = []

            for price_date, group in df.groupby("price_date"):
                key = f"prices/{ticker}/{price_date}.parquet"

                exists = False
                try:
                    s3.head_object(Bucket=S3_BUCKET, Key=key)
                    exists = True
                except ClientError as exc:
                    if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                        raise

                if exists:
                    dates_skipped.append(str(price_date))
                    continue

                if dry_run:
                    print(f"[DRY RUN] Would write {len(group)} rows -> s3://{S3_BUCKET}/{key}")
                    print(f"[DRY RUN] Would delete {len(group)} rows from stock_prices "
                          f"WHERE ticker='{ticker}' AND price_date='{price_date}'")
                else:
                    buf = group.to_parquet(index=False)
                    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf)
                    conn.execute(
                        text("DELETE FROM stock_prices WHERE ticker = :ticker AND price_date = :price_date"),
                        {"ticker": ticker, "price_date": price_date},
                    )
                    conn.commit()

                dates_archived.append(str(price_date))
                rows_archived += len(group)

            summary[ticker] = {
                "dates_archived": len(dates_archived),
                "rows_archived":  rows_archived,
                "dates_skipped":  len(dates_skipped),
                "dates": dates_archived,
            }

    print(f"[archive_postgres_prices] dry_run={dry_run} cutoff={cutoff} summary={summary}")
    return {"dry_run": dry_run, "cutoff": str(cutoff), "summary": summary}


def archive_mongo_news(**context) -> dict:
    """
    Finds articles older than the cutoff, groups by publish date, writes each
    group as a JSONL file to S3 (skipping dates already archived), then deletes
    the archived docs from MongoDB. In dry run: logs what would happen, writes/
    deletes nothing.
    """
    import boto3
    from botocore.exceptions import ClientError
    from bson import json_util
    from pymongo import MongoClient

    from config.config import MongoConfig

    dry_run = _is_dry_run()
    cutoff  = _cutoff_date()
    cutoff_dt = datetime.combine(cutoff, datetime.min.time()).replace(tzinfo=timezone.utc)

    client     = MongoClient(MongoConfig.URL, serverSelectionTimeoutMS=5000)
    collection = client[MongoConfig.DB]["articles"]
    s3         = boto3.client("s3")

    articles = list(collection.find({"published_at": {"$lt": cutoff_dt}}))

    by_date: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        by_date[article["published_at"].date().isoformat()].append(article)

    dates_archived, dates_skipped, docs_archived = [], [], 0

    for date_str, docs in sorted(by_date.items()):
        key = f"news/{date_str}.jsonl"

        exists = False
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=key)
            exists = True
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise

        if exists:
            dates_skipped.append(date_str)
            continue

        if dry_run:
            print(f"[DRY RUN] Would write {len(docs)} articles -> s3://{S3_BUCKET}/{key}")
            print(f"[DRY RUN] Would delete {len(docs)} articles from MongoDB for date={date_str}")
        else:
            body = "\n".join(json_util.dumps(doc) for doc in docs)
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body.encode("utf-8"))
            collection.delete_many({"_id": {"$in": [doc["_id"] for doc in docs]}})

        dates_archived.append(date_str)
        docs_archived += len(docs)

    summary = {
        "dates_archived": len(dates_archived),
        "docs_archived":  docs_archived,
        "dates_skipped":  len(dates_skipped),
        "dates": dates_archived,
    }
    print(f"[archive_mongo_news] dry_run={dry_run} cutoff={cutoff} summary={summary}")
    return {"dry_run": dry_run, "cutoff": str(cutoff), "summary": summary}


def purge_chroma_vectors(**context) -> dict:
    """
    Deletes vectors from ChromaDB whose "published_at" metadata (an ISO-8601
    string) is older than the cutoff. Pulls all vectors that have a
    published_at field via collection.get(), filters older-than-cutoff in
    Python (safer than relying on $lt string comparison support), then
    deletes by the resulting ID list. No S3 export — text already archived
    by archive_mongo_news.
    """
    import chromadb

    from config.config import ChromaConfig

    dry_run = _is_dry_run()
    cutoff  = _cutoff_date()
    cutoff_iso = cutoff.isoformat()

    client     = chromadb.HttpClient(host=ChromaConfig.HOST, port=ChromaConfig.PORT)
    collection = client.get_or_create_collection(
        name=ChromaConfig.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.get(
        where={"published_at": {"$ne": ""}},
        include=["metadatas"],
    )

    ids_to_delete = []
    by_ticker: dict[str, int] = defaultdict(int)

    for doc_id, metadata in zip(existing.get("ids", []), existing.get("metadatas", [])):
        published_at = metadata.get("published_at", "")
        if published_at and published_at < cutoff_iso:
            ids_to_delete.append(doc_id)
            by_ticker[metadata.get("ticker", "unknown")] += 1

    if dry_run:
        print(f"[DRY RUN] Would delete {len(ids_to_delete)} vectors from ChromaDB "
              f"(cutoff={cutoff_iso}) breakdown by ticker: {dict(by_ticker)}")
    elif ids_to_delete:
        collection.delete(ids=ids_to_delete)

    summary = {"vectors_purged": len(ids_to_delete), "by_ticker": dict(by_ticker)}
    print(f"[purge_chroma_vectors] dry_run={dry_run} cutoff={cutoff_iso} summary={summary}")
    return {"dry_run": dry_run, "cutoff": cutoff_iso, "summary": summary}


def validate_postgres_archive(**context) -> None:
    """Confirms archive_postgres_prices completed and reports totals."""
    result = context["ti"].xcom_pull(task_ids="archive_postgres_prices")
    if not result:
        raise ValueError("No summary returned from archive_postgres_prices — possible import error")

    total_rows = sum(v.get("rows_archived", 0) for v in result["summary"].values())
    total_dates = sum(v.get("dates_archived", 0) for v in result["summary"].values())
    print(
        f"[validate_postgres_archive] dry_run={result['dry_run']} cutoff={result['cutoff']} — "
        f"{total_rows} rows across {total_dates} ticker-dates. Per-ticker: {result['summary']}"
    )


def validate_mongo_archive(**context) -> None:
    """Confirms archive_mongo_news completed and reports totals."""
    result = context["ti"].xcom_pull(task_ids="archive_mongo_news")
    if not result:
        raise ValueError("No summary returned from archive_mongo_news — possible import error")

    summary = result["summary"]
    print(
        f"[validate_mongo_archive] dry_run={result['dry_run']} cutoff={result['cutoff']} — "
        f"{summary['docs_archived']} docs across {summary['dates_archived']} dates "
        f"({summary['dates_skipped']} dates already archived, skipped). Dates: {summary['dates']}"
    )


def validate_chroma_purge(**context) -> None:
    """Confirms purge_chroma_vectors completed and reports totals."""
    result = context["ti"].xcom_pull(task_ids="purge_chroma_vectors")
    if not result:
        raise ValueError("No summary returned from purge_chroma_vectors — possible import error")

    summary = result["summary"]
    print(
        f"[validate_chroma_purge] dry_run={result['dry_run']} cutoff={result['cutoff']} — "
        f"{summary['vectors_purged']} vectors purged. By ticker: {summary['by_ticker']}"
    )


# DAG definition
with DAG(
    dag_id="s3_archive_pipeline",
    description="Archive data older than 90 days to S3 and purge from Postgres/MongoDB/ChromaDB",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["stockpilot", "archival", "s3", "retention"],
) as dag:

    # Task 1 — archive + purge Postgres price rows
    archive_postgres_task = PythonOperator(
        task_id="archive_postgres_prices",
        python_callable=archive_postgres_prices,
    )
    validate_postgres_task = PythonOperator(
        task_id="validate_postgres_archive",
        python_callable=validate_postgres_archive,
    )

    # Task 2 — archive + purge MongoDB news articles
    archive_mongo_task = PythonOperator(
        task_id="archive_mongo_news",
        python_callable=archive_mongo_news,
    )
    validate_mongo_task = PythonOperator(
        task_id="validate_mongo_archive",
        python_callable=validate_mongo_archive,
    )

    # Task 3 — purge ChromaDB vectors (text already archived via task 2)
    purge_chroma_task = PythonOperator(
        task_id="purge_chroma_vectors",
        python_callable=purge_chroma_vectors,
    )
    validate_chroma_task = PythonOperator(
        task_id="validate_chroma_purge",
        python_callable=validate_chroma_purge,
    )

    # DAG flow: three independent archive/validate pairs run in parallel
    archive_postgres_task >> validate_postgres_task
    archive_mongo_task >> validate_mongo_task
    purge_chroma_task >> validate_chroma_task
