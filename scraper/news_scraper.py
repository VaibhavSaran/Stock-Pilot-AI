"""
Fetches financial news articles for tracked tickers via NewsAPI,
with BeautifulSoup full-text extraction as a fallback/enrichment step,
and persists documents to MongoDB.

MongoDB collection: stockpilot_news.articles
Document schema:
    {
        ticker:       str,
        headline:     str,
        source:       str,
        url:          str,
        published_at: datetime,
        summary:      str,       # NewsAPI description
        full_text:    str|None,  # scraped body (best-effort)
        scraped_at:   datetime,
        metadata: {
            author:   str|None,
            language: str,
        }
    }

Usage:
    python -m scraper.news_scraper                    # all tickers, last 3 days
    python -m scraper.news_scraper --days 7           # last 7 days
    python -m scraper.news_scraper --tickers AAPL NVDA
"""

import argparse
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from config.config import MongoConfig, StockAPIConfig, TRACKED_TICKERS


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | news_scraper | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Constants
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT  = 10       # seconds per HTTP request
MAX_FULL_TEXT_CHARS = 5000  # truncate scraped body to avoid huge docs
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# MongoDB helpers
def get_mongo_collection():
    """Return the articles collection from MongoDB."""
    url = (
        f"mongodb://{MongoConfig.USER}:{MongoConfig.PASSWORD}"
        f"@{MongoConfig.HOST}:{MongoConfig.PORT}/{MongoConfig.DB}"
        "?authSource=admin"
    )
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    db     = client[MongoConfig.DB]
    collection = db["articles"]

    # Ensure indexes (idempotent)
    collection.create_index("url_hash", unique=True)
    collection.create_index("ticker")
    collection.create_index("published_at")

    return collection


def url_hash(url: str) -> str:
    """SHA-256 of URL used as dedup key."""
    return hashlib.sha256(url.encode()).hexdigest()


def bulk_upsert(collection, docs: list[dict]) -> dict:
    """
    Upsert documents by url_hash. Returns a counts dict.
    Uses ordered=False so one bad doc doesn't block the rest.
    """
    if not docs:
        return {"upserted": 0, "matched": 0}

    ops = [
        UpdateOne(
            {"url_hash": doc["url_hash"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in docs
    ]
    result = collection.bulk_write(ops, ordered=False)
    return {
        "upserted": result.upserted_count,
        "matched":  result.matched_count,
    }



# NewsAPI fetch
def fetch_newsapi_articles(ticker: str, from_date: datetime, to_date: datetime) -> list[dict]:
    """
    Query NewsAPI for articles mentioning the ticker.
    Returns raw NewsAPI article dicts.
    """
    api_key = StockAPIConfig.NEWS_API_KEY
    if not api_key:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI fetch")
        return []

    params = {
        "q":        f'"{ticker}" stock',
        "from":     from_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "to":       to_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 20,
        "apiKey":   api_key,
    }

    try:
        resp = requests.get(NEWSAPI_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI error for {ticker}: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        logger.info(f"  NewsAPI → {len(articles)} articles for {ticker}")
        return articles

    except requests.RequestException as exc:
        logger.error(f"NewsAPI request failed for {ticker}: {exc}")
        return []



# Full-text extraction (best-effort)
def extract_full_text(url: str) -> Optional[str]:
    """
    Attempt to scrape and extract the article body from a URL.
    Returns None on any failure — this is purely additive/optional.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try article tag first, fall back to <p> tags
        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)

        text = " ".join(text.split())  # normalise whitespace
        return text[:MAX_FULL_TEXT_CHARS] if text else None

    except Exception:
        # Silent failure — full_text is optional enrichment
        return None



# Document builder
def build_document(ticker: str, raw: dict) -> Optional[dict]:
    """
    Convert a raw NewsAPI article dict into a MongoDB document.
    Returns None if the article has no URL (can't dedup it).
    """
    url = raw.get("url", "")
    if not url or url == "https://removed.com":
        return None

    # Parse published_at to a proper datetime
    published_at_str = raw.get("publishedAt", "")
    try:
        published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        published_at = datetime.now(timezone.utc)

    return {
        "ticker":       ticker,
        "headline":     raw.get("title", ""),
        "source":       raw.get("source", {}).get("name", ""),
        "url":          url,
        "url_hash":     url_hash(url),
        "published_at": published_at,
        "summary":      raw.get("description", "") or raw.get("content", ""),
        "full_text":    None,   # populated in enrichment step
        "scraped_at":   datetime.now(timezone.utc),
        "metadata": {
            "author":   raw.get("author"),
            "language": "en",
        },
    }



# Orchestrator
def run_scraper(
    tickers: Optional[list[str]] = None,
    days: int = 3,
    enrich_full_text: bool = True,
) -> dict:
    """
    Main entry point. Fetches news for each ticker and writes to MongoDB.

    Args:
        tickers:          List of tickers (default: TRACKED_TICKERS)
        days:             How many days back to fetch
        enrich_full_text: Whether to attempt full-text scraping (slower)

    Returns:
        Summary dict with counts per ticker.
    """
    tickers  = tickers or TRACKED_TICKERS
    to_date  = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)

    logger.info(
        f"Starting news scraper | tickers={tickers} | "
        f"range={from_date.date()} → {to_date.date()} | "
        f"full_text_enrichment={enrich_full_text}"
    )

    try:
        collection = get_mongo_collection()
    except PyMongoError as exc:
        logger.error(f"MongoDB connection failed: {exc}")
        return {"error": str(exc)}

    summary = {}

    for ticker in tickers:
        ticker = ticker.upper()
        logger.info(f"Processing {ticker}...")
        try:
            raw_articles = fetch_newsapi_articles(ticker, from_date, to_date)

            docs = []
            for raw in raw_articles:
                doc = build_document(ticker, raw)
                if doc is None:
                    continue

                # Enrich with full text (best-effort, don't let it fail the doc)
                if enrich_full_text and doc["url"]:
                    doc["full_text"] = extract_full_text(doc["url"])

                docs.append(doc)

            counts = bulk_upsert(collection, docs)
            summary[ticker] = {
                "fetched":  len(raw_articles),
                "docs":     len(docs),
                "upserted": counts["upserted"],
                "matched":  counts["matched"],
            }
            logger.info(
                f"✓ {ticker}: {len(raw_articles)} fetched, "
                f"{len(docs)} valid, {counts['upserted']} upserted, "
                f"{counts['matched']} updated"
            )

        except Exception as exc:
            logger.error(f"Unexpected error for {ticker}: {exc}")
            summary[ticker] = {"error": str(exc)}

    logger.info(f"News scraper complete. Summary: {summary}")
    return summary

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPilot news scraper")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated tickers (default: all tracked tickers)"
    )
    parser.add_argument(
        "--days", type=int, default=3,
        help="Number of days of history to fetch (default: 3)"
    )
    parser.add_argument(
        "--no-full-text", action="store_true",
        help="Skip full-text article scraping (faster)"
    )
    args = parser.parse_args()
    run_scraper(
        tickers=args.tickers,
        days=args.days,
        enrich_full_text=not args.no_full_text,
    )