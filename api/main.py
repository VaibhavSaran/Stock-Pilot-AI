"""
StockPilot AI — FastAPI application entry point.

Endpoints:
POST /api/chat                         — main agent query endpoint
GET  /api/stocks/                      — list tracked tickers
GET  /api/stocks/{ticker}/prices       — OHLCV price history
GET  /api/stocks/{ticker}/metadata     — company info
GET  /health                           — liveness probe
GET  /health/services                  — deep service health check
GET  /docs                             — Swagger UI (auto-generated)
GET  /redoc                            — ReDoc UI (auto-generated)

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router
from api.routes.stocks import router as stocks_router
from api.routes.health import router as health_router
from config.config import AppConfig

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lifespan — warm up the graph on startup so the first request isn't slow
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("StockPilot AI starting up...")
    try:
        from agents.graph import get_graph
        get_graph()   # compile and cache the LangGraph graph
        logger.info("LangGraph graph compiled and cached ✓")
    except Exception as exc:
        logger.warning(f"Graph warm-up failed (non-fatal): {exc}")
    yield
    logger.info("StockPilot AI shutting down...")

# App
app = FastAPI(
    title="StockPilot AI",
    description=(
        "AI-powered stock market analysis agent using LangGraph, Claude & Gemini. "
        "Provides news RAG, structured stock data queries, and general market Q&A."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit frontend and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit dev
        "http://localhost:3000",   # any local frontend
        "*",                       # open for now; restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router)
app.include_router(stocks_router)
app.include_router(health_router)

# Root
@app.get("/", tags=["root"], summary="API info")
async def root():
    return {
        "name":        "StockPilot AI",
        "version":     "1.0.0",
        "description": "AI-powered stock market analysis agent",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints": {
            "chat":             "POST /api/chat",
            "stock_prices":     "GET  /api/stocks/{ticker}/prices",
            "stock_metadata":   "GET  /api/stocks/{ticker}/metadata",
            "tracked_tickers":  "GET  /api/stocks/",
            "health":           "GET  /health",
            "health_services":  "GET  /health/services",
        },
    }