"""
Fetches OHLCV stock price data and company metadata via yfinance
and persists them to PostgreSQL.

Tables written:
  - stock_prices   (ticker, open, high, low, close, volume, price_date)
  - stock_metadata (ticker, company_name, sector, industry, market_cap)

Usage:
    python -m scraper.stock_scraper                  # scrape all tickers, last 7 days
    python -m scraper.stock_scraper --days 30        # last 30 days
    python -m scraper.stock_scraper --tickers AAPL MSFT  # specific tickers

Note: on yfinance 429 rate limiting:
    yfinance 1.x uses curl_cffi to impersonate a real browser internally.
    Do NOT pass a custom session — let yfinance manage authentication.
    Requires: pip install "yfinance>=1.0.0"
"""

import argparse
import logging
import time
from datetime import date, timedelta
from typing import Optional

import yfinance as yf
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.config import PostgresConfig, TRACKED_TICKERS

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | stock_scraper | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Database helper functions
def get_engine():
    """Create and return a SQLAlchemy engine using config."""
    url = (
        f"postgresql://{PostgresConfig.USER}:{PostgresConfig.PASSWORD}"
        f"@{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
    )
    return create_engine(url, pool_pre_ping=True)


def upsert_stock_prices(conn, ticker: str, rows: list[dict]) -> int:
    """
    Insert OHLCV rows, skip duplicates on (ticker, price_date).
    Returns number of rows inserted.
    """
    sql = text("""
        INSERT INTO stock_prices (ticker, open, high, low, close, volume, price_date)
        VALUES (:ticker, :open, :high, :low, :close, :volume, :price_date)
        ON CONFLICT (ticker, price_date) DO UPDATE SET
            open       = EXCLUDED.open,
            high       = EXCLUDED.high,
            low        = EXCLUDED.low,
            close      = EXCLUDED.close,
            volume     = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
    """)
    result = conn.execute(sql, rows)
    return result.rowcount


def upsert_stock_metadata(conn, ticker: str, info: dict) -> None:
    """Insert or update company metadata for a ticker."""
    sql = text("""
        INSERT INTO stock_metadata (ticker, company_name, sector, industry, market_cap, updated_at)
        VALUES (:ticker, :company_name, :sector, :industry, :market_cap, CURRENT_TIMESTAMP)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector       = EXCLUDED.sector,
            industry     = EXCLUDED.industry,
            market_cap   = EXCLUDED.market_cap,
            updated_at   = CURRENT_TIMESTAMP
    """)
    conn.execute(sql, {
        "ticker":       ticker,
        "company_name": info.get("longName") or info.get("shortName", ""),
        "sector":       info.get("sector", ""),
        "industry":     info.get("industry", ""),
        "market_cap":   info.get("marketCap"),
    })


# Core scraping logic

def scrape_prices(ticker: str, start: date, end: date) -> list[dict]:
    """
    Download OHLCV data for a ticker between start and end dates.
    Returns a list of row dicts ready for upsert.

    yfinance 1.x uses curl_cffi to impersonate a real browser, which
    resolves Yahoo Finance 429 rate-limiting automatically.
    Do NOT pass a custom requests session — let yfinance handle auth.
    """
    logger.info(f"Fetching price data: {ticker} {start} → {end}")
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,       # adjusts for splits/dividends
            progress=False,
            multi_level_index=False,  # flatten MultiIndex — yfinance 1.x
        )
    except Exception as exc:
        logger.error(f"yfinance download failed for {ticker}: {exc}")
        return []

    if df.empty:
        logger.warning(f"No price data returned for {ticker}")
        return []

    # Defensive flatten in case multi_level_index kwarg unavailable
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    rows = []
    for price_date, row in df.iterrows():
        rows.append({
            "ticker":     ticker,
            "open":       float(row["Open"])   if row.get("Open")   is not None else None,
            "high":       float(row["High"])   if row.get("High")   is not None else None,
            "low":        float(row["Low"])    if row.get("Low")    is not None else None,
            "close":      float(row["Close"])  if row.get("Close")  is not None else None,
            "volume":     int(row["Volume"])   if row.get("Volume") is not None else None,
            "price_date": price_date.date() if hasattr(price_date, "date") else price_date,
        })

    logger.info(f"  → {len(rows)} price rows fetched for {ticker}")
    return rows


def scrape_metadata(ticker: str) -> dict:
    """Fetch company info dict from yfinance."""
    logger.info(f"Fetching metadata: {ticker}")
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception as exc:
        logger.error(f"Metadata fetch failed for {ticker}: {exc}")
        return {}


# Orchestrator
def run_scraper(
    tickers: Optional[list[str]] = None,
    days: int = 7,
) -> dict:
    """
    Main entry point. Scrapes prices + metadata for all given tickers
    and writes them to PostgreSQL.

    Returns a summary dict with counts per ticker.
    """
    tickers = tickers or TRACKED_TICKERS
    end_date   = date.today()
    start_date = end_date - timedelta(days=days)

    logger.info(
        f"Starting stock scraper | tickers={tickers} | "
        f"range={start_date} → {end_date}"
    )

    engine  = get_engine()
    summary = {}

    with engine.begin() as conn:
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper()

            # Small delay between tickers to be respectful of rate limits
            if i > 0:
                time.sleep(1)

            try:
                # prices 
                rows = scrape_prices(ticker, start_date, end_date)
                inserted = upsert_stock_prices(conn, ticker, rows) if rows else 0

                # metadata 
                info = scrape_metadata(ticker)
                if info:
                    upsert_stock_metadata(conn, ticker, info)

                summary[ticker] = {
                    "price_rows": len(rows),
                    "inserted":   inserted,
                    "metadata":   bool(info),
                }
                logger.info(
                    f"{ticker}: {len(rows)} rows fetched, "
                    f"{inserted} upserted, metadata={'ok' if info else 'failed'}"
                )

            except SQLAlchemyError as exc:
                logger.error(f"DB error for {ticker}: {exc}")
                summary[ticker] = {"error": str(exc)}
            except Exception as exc:
                logger.error(f"Unexpected error for {ticker}: {exc}")
                summary[ticker] = {"error": str(exc)}

    logger.info(f"Stock scraper complete. Summary: {summary}")
    return summary


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPilot stock price scraper")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated tickers (default: all tracked tickers)"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days of history to fetch (default: 7)"
    )
    args = parser.parse_args()
    run_scraper(tickers=args.tickers, days=args.days)