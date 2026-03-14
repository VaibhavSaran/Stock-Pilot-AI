"""
test_scrapers.py
Unit tests for stock_scraper and news_scraper.

Uses unittest.mock throughout — no live DB or API calls.
Run with: pytest tests/test_scrapers.py -v
"""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Stock scraper tests

class TestScrapeStockPrices:
    """Tests for scraper.stock_scraper.scrape_prices"""

    def _make_df(self):
        """Build a minimal yfinance-style DataFrame."""
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame(
            {
                "Open":   [180.0, 182.0],
                "High":   [185.0, 186.0],
                "Low":    [179.0, 181.0],
                "Close":  [184.0, 185.0],
                "Volume": [50_000_000, 55_000_000],
            },
            index=idx,
        )

    @patch("scraper.stock_scraper.yf.download")
    def test_returns_correct_row_count(self, mock_download):
        from scraper.stock_scraper import scrape_prices
        mock_download.return_value = self._make_df()
        rows = scrape_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert len(rows) == 2

    @patch("scraper.stock_scraper.yf.download")
    def test_row_has_required_fields(self, mock_download):
        from scraper.stock_scraper import scrape_prices
        mock_download.return_value = self._make_df()
        rows = scrape_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        required = {"ticker", "open", "high", "low", "close", "volume", "price_date"}
        assert required.issubset(rows[0].keys())

    @patch("scraper.stock_scraper.yf.download")
    def test_ticker_attached_to_rows(self, mock_download):
        from scraper.stock_scraper import scrape_prices
        mock_download.return_value = self._make_df()
        rows = scrape_prices("MSFT", date(2024, 1, 1), date(2024, 1, 5))
        assert all(r["ticker"] == "MSFT" for r in rows)

    @patch("scraper.stock_scraper.yf.download")
    def test_empty_df_returns_empty_list(self, mock_download):
        from scraper.stock_scraper import scrape_prices
        mock_download.return_value = pd.DataFrame()
        rows = scrape_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert rows == []

    @patch("scraper.stock_scraper.yf.download", side_effect=Exception("network error"))
    def test_yfinance_exception_returns_empty_list(self, mock_download):
        from scraper.stock_scraper import scrape_prices
        rows = scrape_prices("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert rows == []


class TestScrapeStockMetadata:
    """Tests for scraper.stock_scraper.scrape_metadata"""

    @patch("scraper.stock_scraper.yf.Ticker")
    def test_returns_info_dict(self, mock_ticker_cls):
        from scraper.stock_scraper import scrape_metadata
        mock_ticker_cls.return_value.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 3_000_000_000_000,
        }
        info = scrape_metadata("AAPL")
        assert info["longName"] == "Apple Inc."
        assert info["sector"] == "Technology"

    @patch("scraper.stock_scraper.yf.Ticker", side_effect=Exception("timeout"))
    def test_exception_returns_empty_dict(self, mock_ticker_cls):
        from scraper.stock_scraper import scrape_metadata
        info = scrape_metadata("AAPL")
        assert info == {}


class TestRunStockScraper:
    """Integration-level tests for scraper.stock_scraper.run_scraper (all mocked)."""

    @patch("scraper.stock_scraper.scrape_metadata")
    @patch("scraper.stock_scraper.scrape_prices")
    @patch("scraper.stock_scraper.get_engine")
    def test_returns_summary_for_all_tickers(
        self, mock_engine, mock_prices, mock_metadata
    ):
        from scraper.stock_scraper import run_scraper

        mock_prices.return_value = [
            {
                "ticker": "AAPL", "open": 180.0, "high": 185.0,
                "low": 179.0, "close": 184.0, "volume": 50_000_000,
                "price_date": date(2024, 1, 2),
            }
        ]
        mock_metadata.return_value = {"longName": "Apple Inc.", "sector": "Technology"}

        # Mock DB connection context manager
        mock_conn = MagicMock()
        mock_conn.execute.return_value.rowcount = 1
        mock_engine.return_value.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.return_value.begin.return_value.__exit__ = MagicMock(return_value=False)

        summary = run_scraper(tickers=["AAPL"], days=7)

        assert "AAPL" in summary
        assert summary["AAPL"]["price_rows"] == 1

    @patch("scraper.stock_scraper.scrape_metadata")
    @patch("scraper.stock_scraper.scrape_prices")
    @patch("scraper.stock_scraper.get_engine")
    def test_handles_per_ticker_errors_gracefully(
        self, mock_engine, mock_prices, mock_metadata
    ):
        from scraper.stock_scraper import run_scraper

        mock_prices.side_effect = Exception("yfinance down")

        mock_conn = MagicMock()
        mock_engine.return_value.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.return_value.begin.return_value.__exit__ = MagicMock(return_value=False)

        summary = run_scraper(tickers=["AAPL"], days=7)

        assert "AAPL" in summary
        assert "error" in summary["AAPL"]


# News scraper tests

class TestBuildDocument:
    """Tests for scraper.news_scraper.build_document"""

    def _raw(self, **overrides):
        base = {
            "title":       "Apple hits record high",
            "source":      {"name": "Reuters"},
            "url":         "https://reuters.com/article/apple-123",
            "publishedAt": "2024-01-02T10:00:00Z",
            "description": "Apple stock surged today...",
            "author":      "Jane Smith",
        }
        base.update(overrides)
        return base

    def test_basic_document_structure(self):
        from scraper.news_scraper import build_document
        doc = build_document("AAPL", self._raw())
        assert doc["ticker"]   == "AAPL"
        assert doc["headline"] == "Apple hits record high"
        assert doc["source"]   == "Reuters"
        assert "url_hash" in doc
        assert isinstance(doc["published_at"], datetime)

    def test_url_hash_is_deterministic(self):
        from scraper.news_scraper import build_document, url_hash
        raw = self._raw()
        doc = build_document("AAPL", raw)
        assert doc["url_hash"] == url_hash(raw["url"])

    def test_missing_url_returns_none(self):
        from scraper.news_scraper import build_document
        assert build_document("AAPL", self._raw(url="")) is None

    def test_removed_url_returns_none(self):
        from scraper.news_scraper import build_document
        assert build_document("AAPL", self._raw(url="https://removed.com")) is None

    def test_bad_date_falls_back_gracefully(self):
        from scraper.news_scraper import build_document
        doc = build_document("AAPL", self._raw(publishedAt="not-a-date"))
        assert isinstance(doc["published_at"], datetime)

    def test_full_text_initialised_to_none(self):
        from scraper.news_scraper import build_document
        doc = build_document("AAPL", self._raw())
        assert doc["full_text"] is None


class TestExtractFullText:
    """Tests for scraper.news_scraper.extract_full_text"""

    @patch("scraper.news_scraper.requests.get")
    def test_extracts_paragraph_text(self, mock_get):
        from scraper.news_scraper import extract_full_text
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = (
            "<html><body><p>Apple reported earnings.</p>"
            "<p>Stock rose 5 percent.</p></body></html>"
        )
        mock_get.return_value.raise_for_status = MagicMock()
        text = extract_full_text("https://example.com/article")
        assert "Apple reported earnings" in text
        assert "Stock rose 5 percent" in text

    @patch("scraper.news_scraper.requests.get", side_effect=Exception("timeout"))
    def test_exception_returns_none(self, mock_get):
        from scraper.news_scraper import extract_full_text
        result = extract_full_text("https://example.com/article")
        assert result is None

    @patch("scraper.news_scraper.requests.get")
    def test_respects_max_length(self, mock_get):
        from scraper.news_scraper import extract_full_text, MAX_FULL_TEXT_CHARS
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = "<html><body>" + "<p>" + "x" * 10_000 + "</p></body></html>"
        mock_get.return_value.raise_for_status = MagicMock()
        text = extract_full_text("https://example.com/article")
        assert len(text) <= MAX_FULL_TEXT_CHARS


class TestRunNewsScraper:
    """Integration-level tests for scraper.news_scraper.run_scraper (all mocked)."""

    @patch("scraper.news_scraper.bulk_upsert")
    @patch("scraper.news_scraper.extract_full_text", return_value="full article body")
    @patch("scraper.news_scraper.fetch_newsapi_articles")
    @patch("scraper.news_scraper.get_mongo_collection")
    def test_returns_summary_per_ticker(
        self, mock_col, mock_fetch, mock_full_text, mock_upsert
    ):
        from scraper.news_scraper import run_scraper
        mock_fetch.return_value = [
            {
                "title":       "NVDA soars",
                "source":      {"name": "Bloomberg"},
                "url":         "https://bloomberg.com/nvda-1",
                "publishedAt": "2024-01-02T10:00:00Z",
                "description": "Nvidia stock up 10%",
                "author":      None,
            }
        ]
        mock_upsert.return_value = {"upserted": 1, "matched": 0}

        summary = run_scraper(tickers=["NVDA"], days=3)
        assert "NVDA" in summary
        assert summary["NVDA"]["fetched"] == 1
        assert summary["NVDA"]["upserted"] == 1

    @patch("scraper.news_scraper.get_mongo_collection")
    def test_mongo_connection_failure_returns_error(self, mock_col):
        from pymongo.errors import PyMongoError
        from scraper.news_scraper import run_scraper
        mock_col.side_effect = PyMongoError("connection refused")
        summary = run_scraper(tickers=["AAPL"], days=3)
        assert "error" in summary

    @patch("scraper.news_scraper.bulk_upsert")
    @patch("scraper.news_scraper.extract_full_text", return_value=None)
    @patch("scraper.news_scraper.fetch_newsapi_articles")
    @patch("scraper.news_scraper.get_mongo_collection")
    def test_removed_urls_are_skipped(
        self, mock_col, mock_fetch, mock_full_text, mock_upsert
    ):
        from scraper.news_scraper import run_scraper
        mock_fetch.return_value = [
            {
                "title":       "Removed article",
                "source":      {"name": "NewsAPI"},
                "url":         "https://removed.com",
                "publishedAt": "2024-01-02T10:00:00Z",
                "description": "",
                "author":      None,
            }
        ]
        mock_upsert.return_value = {"upserted": 0, "matched": 0}

        summary = run_scraper(tickers=["AAPL"], days=1, enrich_full_text=False)
        assert summary["AAPL"]["docs"] == 0