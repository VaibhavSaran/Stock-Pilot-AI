-- StockPilot AI - PostgreSQL Init Script

-- Create airflow database (used by Airflow scheduler)
CREATE DATABASE airflow;

-- Connect to stockpilot DB
\c stockpilot;

-- Stock Prices Table
CREATE TABLE IF NOT EXISTS stock_prices (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(10) NOT NULL,
    open        NUMERIC(12, 4),
    high        NUMERIC(12, 4),
    low         NUMERIC(12, 4),
    close       NUMERIC(12, 4),
    volume      BIGINT,
    price_date  DATE NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (ticker, price_date)
);

-- Stock Metadata Table
CREATE TABLE IF NOT EXISTS stock_metadata (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) UNIQUE NOT NULL,
    company_name    VARCHAR(255),
    sector          VARCHAR(100),
    industry        VARCHAR(100),
    market_cap      BIGINT,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(price_date);
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices(ticker, price_date);