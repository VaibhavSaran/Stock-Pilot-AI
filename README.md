# 🚀 StockPilot AI

> AI-powered stock market analysis agent using LangGraph, Claude AI & Apache Airflow — with real-time data pipelines, agentic RAG workflows, and full AWS deployment.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.5-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)
![Airflow](https://img.shields.io/badge/Airflow-2.9.1-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 What is StockPilot AI?

StockPilot AI is a full-stack agentic RAG application that lets you query stock market data and financial news using natural language. It combines real-time data pipelines (Apache Airflow), vector search (ChromaDB), structured data retrieval (PostgreSQL), and a multi-agent LangGraph workflow powered by Claude AI.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  APACHE AIRFLOW                         │
│  DAG 1: Stock Price Scraper  (hourly → PostgreSQL)      │
│  DAG 2: News Scraper         (3hr    → MongoDB)         │
│  DAG 3: ChromaDB Sync        (4hr    → ChromaDB)        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              LANGGRAPH SUPERVISOR AGENT                  │
│                                                          │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐  │
│  │ News RAG     │ │ Stock Data  │ │ Stock Charts     │  │
│  │ Graph        │ │ RAG Graph   │ │ RAG Graph        │  │
│  │ (ChromaDB +  │ │ (NL → SQL   │ │ (NL → SQL →      │  │
│  │  Web Search) │ │  → Postgres)│ │  Visualization)  │  │
│  └──────────────┘ └─────────────┘ └──────────────────┘  │
│                                                          │
│  LLM: Claude (claude-sonnet-4-6)                        │
│  Embeddings: Gemini (embedding-001)                     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   FASTAPI BACKEND                        │
│  /query  /stock/{ticker}/chart  /news/{ticker}           │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                STREAMLIT FRONTEND                        │
│  Chat Interface | Stock Charts | News Feed               │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **AI Orchestration** | LangGraph, LangChain |
| **Primary LLM** | Anthropic Claude (claude-sonnet-4-6) |
| **Embeddings** | Google Gemini (embedding-001) |
| **Vector Store** | ChromaDB |
| **Structured DB** | PostgreSQL |
| **Document Store** | MongoDB |
| **Data Pipelines** | Apache Airflow |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Deployment** | AWS ECS Fargate, AWS ECR |
| **Observability** | LangSmith |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Anthropic API Key
- Google Gemini API Key

### 1. Clone the repo
```bash
git clone https://github.com/your-username/stockpilot-ai.git
cd stockpilot-ai
```

### 2. Setup environment
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Start all services
```bash
docker-compose up -d
```

### 4. Access the apps
| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| Airflow UI | http://localhost:8080 |
| ChromaDB | http://localhost:8001 |

---

## 📁 Project Structure

```
stockpilot-ai/
├── airflow/
│   └── dags/               # Airflow DAGs for data pipelines
├── agents/                 # LangGraph agent graphs
│   ├── supervisor.py
│   ├── news_rag_graph.py
│   ├── stock_data_rag_graph.py
│   └── stock_charts_rag_graph.py
├── api/                    # FastAPI backend
│   └── main.py
├── frontend/               # Streamlit frontend
│   └── app.py
├── scraper/                # Data scrapers
├── db/                     # DB init scripts
├── config/                 # Config management
├── tests/                  # Unit tests
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.