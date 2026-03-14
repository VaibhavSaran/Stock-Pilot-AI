"""
Airflow DAG — scrapes OHLCV stock price data and company metadata
via yfinance and writes to PostgreSQL.

Schedule : @hourly
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
# All heavy logic lives in scraper/ DAG is just the orchestration shell.
def scrape_all_tickers(**context) -> dict:
    """
    Calls the stock scraper for all tracked tickers.
    Fetches the last 2 days to catch any missed trading day on retries.
    XComs the summary dict so downstream tasks / logs can inspect it.
    """
    from scraper.stock_scraper import run_scraper
    summary = run_scraper(days=2)
    return summary


def scrape_single_ticker(ticker: str, **context) -> dict:
    """
    Calls the stock scraper for a single ticker.
    Used by the per-ticker task fan-out pattern.
    """
    from scraper.stock_scraper import run_scraper
    summary = run_scraper(tickers=[ticker], days=2)
    return summary


def validate_postgres_write(**context) -> None:
    """
    Spot-checks that rows were actually written to PostgreSQL.
    Pulls the XCom summary from the upstream scrape task and
    raises an error if every ticker returned 0 rows (signals a
    systemic failure rather than a single-ticker blip).
    """
    summary = context["ti"].xcom_pull(task_ids="scrape_stock_prices")
    if not summary:
        raise ValueError("No summary returned from scrape task — possible import error")

    total_rows = sum(
        v.get("price_rows", 0)
        for v in summary.values()
        if isinstance(v, dict) and "error" not in v
    )
    failed = [k for k, v in summary.items() if isinstance(v, dict) and "error" in v]

    if failed:
        # Log failures but don't fail the DAG — partial success is acceptable
        print(f"WARNING: {len(failed)} tickers failed: {failed}")

    if total_rows == 0 and len(summary) > 0:
        raise ValueError(
            f"Zero price rows written across all tickers. "
            f"Summary: {summary}. Possible yfinance or DB issue."
        )

    print(f"Validation passed — {total_rows} rows written across {len(summary)} tickers")



# DAG definition
with DAG(
    dag_id="stock_prices_pipeline",
    description="Scrape OHLCV stock prices + metadata -> PostgreSQL",
    default_args=DEFAULT_ARGS,
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,          # prevent overlap if a run takes > 1 hour
    tags=["stockpilot", "ingestion", "postgres"],
) as dag:

    # Task 1 — scrape all tickers in one call (batching is faster + fewer 429s)
    scrape_task = PythonOperator(
        task_id="scrape_stock_prices",
        python_callable=scrape_all_tickers,
    )

    # Task 2 — validate something actually landed in Postgres
    validate_task = PythonOperator(
        task_id="validate_postgres_write",
        python_callable=validate_postgres_write,
    )

    # DAG flow: scrape -> validate
    scrape_task >> validate_task