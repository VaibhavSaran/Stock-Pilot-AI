# StockPilot AI — Streamlit Frontend
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only install what the frontend needs
COPY requirements.txt .
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    requests==2.31.0 \
    pandas==2.2.2 \
    plotly==5.22.0

COPY frontend/ ./frontend/

ENV PYTHONPATH=/app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]