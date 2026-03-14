"""
Stock data endpoints — reads directly from PostgreSQL.

GET /api/stocks/ — list all tracked tickers
GET /api/stocks/{ticker}/prices?days=7 — OHLCV price history
GET /api/stocks/{ticker}/metadata — company info
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

from api.models import (
    StockMetadata,
    StockPrice,
    StockPricesResponse,
    TickersResponse,
)
from config.config import PostgresConfig, TRACKED_TICKERS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["stocks"])


# DB engine (module-level singleton)
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        url = (
            f"postgresql://{PostgresConfig.USER}:{PostgresConfig.PASSWORD}"
            f"@{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
        )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


# Routes
@router.get("", response_model=TickersResponse, summary="List tracked tickers")
async def list_tickers():
    """Returns all tickers currently tracked by StockPilot AI."""
    return TickersResponse(tickers=TRACKED_TICKERS, count=len(TRACKED_TICKERS))


@router.get(
    "/{ticker}/prices",
    response_model=StockPricesResponse,
    summary="Get OHLCV price history for a ticker",
)
async def get_stock_prices(
    ticker: str,
    days: int = Query(default=7, ge=1, le=365, description="Number of days of history"),
):
    """
    Returns daily OHLCV price data for a given ticker.
    Data comes from PostgreSQL, populated by the Airflow stock_prices_pipeline DAG.
    """
    ticker = ticker.upper()

    if ticker not in TRACKED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not tracked. Available: {TRACKED_TICKERS}",
        )

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT ticker, price_date, open, high, low, close, volume
                    FROM stock_prices
                    WHERE ticker = :ticker
                      AND price_date >= CURRENT_DATE - INTERVAL '1 day' * :days
                    ORDER BY price_date DESC
                    LIMIT 365
                """),
                {"ticker": ticker, "days": days},
            )
            rows = result.fetchall()
            keys = result.keys()

        prices = [
            StockPrice(**dict(zip(keys, row)))
            for row in rows
        ]

        return StockPricesResponse(
            ticker=ticker,
            days=days,
            count=len(prices),
            prices=prices,
        )

    except Exception as exc:
        logger.error(f"Error fetching prices for {ticker}: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)[:200]}")


@router.get(
    "/{ticker}/metadata",
    response_model=StockMetadata,
    summary="Get company metadata for a ticker",
)
async def get_stock_metadata(ticker: str):
    """
    Returns company info (name, sector, industry, market cap) for a ticker.
    Data is populated by the Airflow stock_prices_pipeline DAG.
    """
    ticker = ticker.upper()

    if ticker not in TRACKED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not tracked. Available: {TRACKED_TICKERS}",
        )

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT ticker, company_name, sector, industry, market_cap, updated_at
                    FROM stock_metadata
                    WHERE ticker = :ticker
                """),
                {"ticker": ticker},
            )
            row = result.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No metadata found for '{ticker}'. Run the stock scraper first.",
            )

        return StockMetadata(**dict(zip(result.keys(), row)))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching metadata for {ticker}: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)[:200]}")