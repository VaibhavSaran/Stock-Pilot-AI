"""
test_api.py
Unit tests for Phase 5 — FastAPI endpoints.
All agent calls and DB connections are mocked.

Run with: pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture — patch the lifespan graph compilation and get a test client
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    # Patch get_graph inside the lifespan coroutine's import path
    with patch("agents.graph.get_graph"), \
         patch("agents.graph.build_graph"):
        from api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Root + Health
# ---------------------------------------------------------------------------
class TestRoot:
    def test_root_returns_api_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "StockPilot AI"
        assert "endpoints" in data

    def test_health_liveness(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestHealthServices:
    @patch("agents.vector_store.get_vector_store")
    @patch("api.routes.health.create_engine")
    def test_all_services_ok(self, mock_engine, mock_get_store, client):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.return_value.connect.return_value = mock_conn

        with patch("pymongo.MongoClient") as mock_mongo:
            mock_mongo.return_value.admin.command.return_value = {"ok": 1}
            mock_store = MagicMock()
            mock_store.health_check.return_value = True
            mock_store.count.return_value = 42
            mock_get_store.return_value = mock_store
            resp = client.get("/health/services")

        assert resp.status_code == 200
        data = resp.json()
        assert data["services"]["chromadb"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
class TestChatEndpoint:

    @patch("api.routes.chat.run_query")
    def test_news_rag_route(self, mock_run, client):
        mock_run.return_value = {
            "final_answer": "Apple reported strong Q1 earnings.",
            "route": "news_rag", "ticker": "AAPL", "sql_query": None,
            "retrieved_docs": [{"id": "1"}, {"id": "2"}],
            "web_search_results": [], "error": None,
        }
        resp = client.post("/api/chat", json={"query": "What is the latest AAPL news?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"] == "news_rag"
        assert data["ticker"] == "AAPL"
        assert data["sources_used"] == 2

    @patch("api.routes.chat.run_query")
    def test_stock_data_rag_route(self, mock_run, client):
        mock_run.return_value = {
            "final_answer": "AAPL closed at $250.12.",
            "route": "stock_data_rag", "ticker": "AAPL",
            "sql_query": "SELECT close FROM stock_prices LIMIT 5;",
            "retrieved_docs": [], "web_search_results": [],
            "sql_results": [{"ticker": "AAPL", "close": 250.12}], "error": None,
        }
        resp = client.post("/api/chat", json={"query": "AAPL closing price last week?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"] == "stock_data_rag"
        assert data["sql_query"] is not None
        assert data["sources_used"] == 1

    @patch("api.routes.chat.run_query")
    def test_general_route(self, mock_run, client):
        mock_run.return_value = {
            "final_answer": "A P/E ratio measures...",
            "route": "general", "ticker": None, "sql_query": None,
            "retrieved_docs": [], "web_search_results": [],
            "sql_results": [], "error": None,
        }
        resp = client.post("/api/chat", json={"query": "What is a P/E ratio?"})
        assert resp.status_code == 200
        assert resp.json()["route"] == "general"

    def test_query_too_short_rejected(self, client):
        resp = client.post("/api/chat", json={"query": "hi"})
        assert resp.status_code == 422

    def test_query_missing_rejected(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    @patch("api.routes.chat.run_query", side_effect=Exception("LLM timeout"))
    def test_agent_error_returns_500(self, mock_run, client):
        resp = client.post("/api/chat", json={"query": "What is the latest news?"})
        assert resp.status_code == 500
        assert "Agent error" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Stocks endpoints
# ---------------------------------------------------------------------------
class TestStocksEndpoints:

    def test_list_tickers(self, client):
        resp = client.get("/api/stocks/")
        assert resp.status_code == 200
        data = resp.json()
        assert "AAPL" in data["tickers"]
        assert data["count"] == 7

    @patch("api.routes.stocks._get_engine")
    def test_get_prices_success(self, mock_engine, client):
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.keys.return_value = ["ticker", "price_date", "open", "high", "low", "close", "volume"]
        mock_result.fetchall.return_value = [
            ("AAPL", "2026-03-13", 248.0, 252.0, 247.0, 250.12, 50_000_000),
        ]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.connect.return_value = mock_conn

        resp = client.get("/api/stocks/AAPL/prices?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert data["count"] == 1
        assert data["prices"][0]["close"] == 250.12

    def test_get_prices_invalid_ticker(self, client):
        resp = client.get("/api/stocks/FAKE/prices")
        assert resp.status_code == 404

    @patch("api.routes.stocks._get_engine")
    def test_get_metadata_success(self, mock_engine, client):
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.keys.return_value = ["ticker", "company_name", "sector", "industry", "market_cap", "updated_at"]
        mock_result.fetchone.return_value = (
            "AAPL", "Apple Inc.", "Technology", "Consumer Electronics",
            3_000_000_000_000, "2026-03-13 10:00:00",
        )
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.connect.return_value = mock_conn

        resp = client.get("/api/stocks/AAPL/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_name"] == "Apple Inc."

    @patch("api.routes.stocks._get_engine")
    def test_get_metadata_not_found(self, mock_engine, client):
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.keys.return_value = ["ticker", "company_name", "sector", "industry", "market_cap", "updated_at"]
        mock_result.fetchone.return_value = None
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value = mock_result
        mock_engine.return_value.connect.return_value = mock_conn

        resp = client.get("/api/stocks/MSFT/metadata")
        assert resp.status_code == 404

    def test_get_metadata_invalid_ticker(self, client):
        resp = client.get("/api/stocks/FAKE/metadata")
        assert resp.status_code == 404