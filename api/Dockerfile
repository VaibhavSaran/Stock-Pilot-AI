# StockPilot AI — FastAPI Backend
FROM python:3.10-slim

# Install system dependencies
# - gcc, libpq-dev: required by psycopg2
# - curl: healthcheck
# - build-essential, libpython3-dev: required by Prophet (compiles PyStan)
# - python3-dev: header files for Prophet's C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agents/     ./agents/
COPY api/        ./api/
COPY config/     ./config/
COPY scraper/    ./scraper/

# Patch pydantic-settings before chromadb loads (same fix as conftest.py)
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]