"""
Stock data endpoints — reads directly from PostgreSQL.

GET /api/stocks/                        — list all tracked tickers
GET /api/stocks/{ticker}/prices?days=7  — OHLCV price history
GET /api/stocks/{ticker}/metadata       — company info
GET /api/stocks/{ticker}/forecast?days=7 — Prophet price forecast
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
    ForecastResponse,
    ForecastPoint,
)
from config.config import PostgresConfig, TRACKED_TICKERS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["stocks"])


#DB engine (module-level singleton) 

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


#Existing routes 

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

        prices = [StockPrice(**dict(zip(keys, row))) for row in rows]

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


#Forecast endpoint 

@router.get(
    "/{ticker}/forecast",
    response_model=ForecastResponse,
    summary="Forecast stock closing price using FB Prophet",
)
async def get_stock_forecast(
    ticker: str,
    days: int = Query(default=7, ge=1, le=7, description="Number of days to forecast (1-7)"),
):
    """
    Generates a short-term closing price forecast using Facebook Prophet.

    Uses all available historical daily closing prices from PostgreSQL as
    training data. Returns predicted price with upper/lower confidence bounds
    for each forecast day.

    Note: Financial forecasts are for informational purposes only and should
    not be used as investment advice.
    """
    ticker = ticker.upper()

    if ticker not in TRACKED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' is not tracked. Available: {TRACKED_TICKERS}",
        )

    #Fetch all historical closing prices 
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT price_date, close
                    FROM stock_prices
                    WHERE ticker = :ticker
                      AND close IS NOT NULL
                    ORDER BY price_date ASC
                """),
                {"ticker": ticker},
            )
            rows = result.fetchall()

    except Exception as exc:
        logger.error(f"Error fetching prices for forecast ({ticker}): {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)[:200]}")

    if len(rows) < 10:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient data for {ticker}: need at least 10 data points, "
                f"got {len(rows)}. Run the stock scraper first."
            ),
        )

    #Run Prophet 
    try:
        import pandas as pd
        from prophet import Prophet

        # Prophet requires columns named 'ds' (datestamp) and 'y' (value)
        df = pd.DataFrame(rows, columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"])

        # Fit model
        # - daily_seasonality=False: we have ~90 days, not enough for daily patterns
        # - weekly_seasonality=True: stock market has weekly patterns (weekends)
        # - yearly_seasonality=False: only 90 days of data, yearly doesn't apply
        # - changepoint_prior_scale=0.05: conservative, avoids overfitting on short data
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80,  # 80% confidence interval
        )
        model.fit(df)

        # Generate future dates (Prophet includes weekends; we filter to trading days)
        future = model.make_future_dataframe(periods=days, freq="B")  # "B" = business days
        forecast = model.predict(future)

        # Extract only the forecast rows (not historical)
        forecast_tail = forecast.tail(days)

        forecast_points = [
            ForecastPoint(
                date=str(row["ds"].date()),
                predicted=round(float(row["yhat"]), 2),
                lower=round(float(row["yhat_lower"]), 2),
                upper=round(float(row["yhat_upper"]), 2),
            )
            for _, row in forecast_tail.iterrows()
        ]

        # Last known price for reference
        last_known_price = round(float(df["y"].iloc[-1]), 2)
        last_known_date  = str(df["ds"].iloc[-1].date())

        logger.info(
            f"[forecast] {ticker} | {len(df)} training points | "
            f"{days}-day forecast generated"
        )

        return ForecastResponse(
            ticker=ticker,
            forecast_days=days,
            training_points=len(df),
            last_known_date=last_known_date,
            last_known_price=last_known_price,
            forecast=forecast_points,
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Prophet is not installed. Add 'prophet' to requirements.txt.",
        )
    except Exception as exc:
        logger.error(f"[forecast] Prophet error for {ticker}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Forecast error: {str(exc)[:200]}",
        )