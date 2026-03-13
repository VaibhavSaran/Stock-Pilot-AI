"""
Centralized configuration management for StockPilot AI.
Loads all environment variables with validation.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Primary LLM - Claude for reasoning and generation
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # Gemini - Embeddings (GA, top MTEB leaderboard, text-optimized for RAG)
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # Gemini - Generation (fast, cost-efficient, used as secondary/fallback)
    GEMINI_GENERATION_MODEL: str = "models/gemini-2.0-flash"


class LangSmithConfig:
    TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "stockpilot-ai")


class PostgresConfig:
    HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    DB: str = os.getenv("POSTGRES_DB", "stockpilot")
    USER: str = os.getenv("POSTGRES_USER", "stockpilot_user")
    PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    URL: str = os.getenv(
        "POSTGRES_URL",
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}"
        f"/{os.getenv('POSTGRES_DB', 'stockpilot')}",
    )


class MongoConfig:
    HOST: str = os.getenv("MONGO_HOST", "localhost")
    PORT: int = int(os.getenv("MONGO_PORT", 27017))
    DB: str = os.getenv("MONGO_DB", "stockpilot_news")
    USER: str = os.getenv("MONGO_USER", "stockpilot_user")
    PASSWORD: str = os.getenv("MONGO_PASSWORD", "")
    URL: str = os.getenv(
        "MONGO_URL",
        f"mongodb://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASSWORD')}"
        f"@{os.getenv('MONGO_HOST', 'localhost')}:{os.getenv('MONGO_PORT', 27017)}"
        f"/{os.getenv('MONGO_DB', 'stockpilot_news')}",
    )


class ChromaConfig:
    HOST: str = os.getenv("CHROMA_HOST", "localhost")
    PORT: int = int(os.getenv("CHROMA_PORT", 8001))
    COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "stock_news")
    API_VERSION: str = "v2"  # ChromaDB 1.x uses v2 API


class StockAPIConfig:
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


class AppConfig:
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_ENV: str = os.getenv("API_ENV", "development")
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", 8501))


class AWSConfig:
    REGION: str = os.getenv("AWS_REGION", "us-east-1")
    ACCOUNT_ID: str = os.getenv("AWS_ACCOUNT_ID", "")
    ECR_REPOSITORY: str = os.getenv("ECR_REPOSITORY", "stockpilot-ai")


# Stocks to track
TRACKED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

# Airflow schedule intervals
STOCK_SCRAPE_INTERVAL = "@hourly"
NEWS_SCRAPE_INTERVAL = "0 */3 * * *"   # Every 3 hours
CHROMA_SYNC_INTERVAL = "0 */4 * * *"   # Every 4 hours