"""
Airflow DAG — fetches financial news articles via NewsAPI
and writes to MongoDB, with optional full-text enrichment.

Schedule : every 3 hours
Retries  : 3 (exponential backoff, 5 min base)
Owner    : stockpilot
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
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


# Task functions


def scrape_news(**context) -> dict:
    """
    Fetches news for all tracked tickers — last 3 days window.
    Full-text enrichment is disabled in the scheduled run to keep
    runtime predictable; it can be enabled for manual/backfill runs.
    """
    from scraper.news_scraper import run_scraper
    summary = run_scraper(days=3, enrich_full_text=False)
    return summary


def enrich_full_text(**context) -> dict:
    """
    Re-runs the scraper with full-text enrichment enabled for a
    shorter 1-day window, targeting only articles that landed
    in the previous task (avoids re-fetching the full 3-day window).
    Best-effort — individual URL failures are silently skipped.
    """
    from scraper.news_scraper import run_scraper
    summary = run_scraper(days=1, enrich_full_text=True)
    return summary


def validate_mongo_write(**context) -> None:
    """
    Checks that at least some articles were written.
    Tolerates 0-article runs (e.g. NewsAPI rate limit hit)
    but logs a clear warning so it's visible in the Airflow UI.
    """
    summary = context["ti"].xcom_pull(task_ids="scrape_news")
    if not summary:
        raise ValueError("No summary returned from scrape task")

    total_upserted = sum(
        v.get("upserted", 0)
        for v in summary.values()
        if isinstance(v, dict) and "error" not in v
    )
    failed = [k for k, v in summary.items() if isinstance(v, dict) and "error" in v]

    if failed:
        print(f"WARNING: {len(failed)} tickers errored: {failed}")

    if total_upserted == 0:
        # NewsAPI free tier has rate limits — warn but don't fail
        print(
            f"WARNING: 0 articles upserted across all tickers. "
            f"Possible NewsAPI rate limit or no new articles. Summary: {summary}"
        )
    else:
        print(f" Validation passed — {total_upserted} articles upserted")



# DAG definition
with DAG(
    dag_id="news_scraper_pipeline",
    description="Scrape financial news articles -> MongoDB",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 */3 * * *",   # every 3 hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["stockpilot", "ingestion", "mongodb"],
) as dag:

    # Task 1 — fetch articles from NewsAPI -> MongoDB
    scrape_task = PythonOperator(
        task_id="scrape_news",
        python_callable=scrape_news,
    )

    # Task 2 — enrich with full article body (best-effort)
    enrich_task = PythonOperator(
        task_id="enrich_full_text",
        python_callable=enrich_full_text,
    )

    # Task 3 — validate something landed
    validate_task = PythonOperator(
        task_id="validate_mongo_write",
        python_callable=validate_mongo_write,
    )

    # DAG flow: scrape -> enrich -> validate
    scrape_task >> enrich_task >> validate_task