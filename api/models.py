"""
Pydantic request and response schemas for the StockPilot AI API.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field

# Chat
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about stocks or markets",
        examples=["What is the latest news on AAPL?"],
    )

class ChatResponse(BaseModel):
    query:        str
    answer:       str
    route:        Optional[str] = None   # news_rag | stock_data_rag | general
    ticker:       Optional[str] = None
    sql_query:    Optional[str] = None   # populated for stock_data_rag route
    sources_used: int = 0                # number of docs/results used
    error:        Optional[str] = None


# Stock prices
class StockPrice(BaseModel):
    ticker:     str
    price_date: date
    open:       Optional[float] = None
    high:       Optional[float] = None
    low:        Optional[float] = None
    close:      Optional[float] = None
    volume:     Optional[int]   = None

class StockPricesResponse(BaseModel):
    ticker: str
    days:   int
    count:  int
    prices: list[StockPrice]

# Stock metadata
class StockMetadata(BaseModel):
    ticker:       str
    company_name: Optional[str] = None
    sector:       Optional[str] = None
    industry:     Optional[str] = None
    market_cap:   Optional[int] = None
    updated_at:   Optional[datetime] = None

# Tickers list
class TickersResponse(BaseModel):
    tickers: list[str]
    count:   int

# Health
class ServiceStatus(BaseModel):
    status:  str   # "ok" | "error"
    message: str


class HealthResponse(BaseModel):
    status:   str   # "ok" | "degraded" | "error"
    services: dict[str, ServiceStatus]
    version:  str = "1.0.0"