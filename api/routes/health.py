"""
Health check endpoints.

GET /health — basic liveness probe
GET /health/services — deep check of all downstream services
"""

import logging

from fastapi import APIRouter
from sqlalchemy import create_engine, text

from api.models import HealthResponse, ServiceStatus
from config.config import ChromaConfig, MongoConfig, PostgresConfig

logger = APIRouter()
router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Basic health check")
async def health():
    """Liveness probe — returns 200 if the API process is running."""
    return HealthResponse(
        status="ok",
        services={"api": ServiceStatus(status="ok", message="running")},
    )

@router.get(
    "/services",
    response_model=HealthResponse,
    summary="Deep health check of all services",
)
async def health_services():
    """
    Checks connectivity to PostgreSQL, MongoDB, and ChromaDB.
    Returns overall status: 'ok' if all pass, 'degraded' if some fail.
    """
    services: dict[str, ServiceStatus] = {}

    # PostgreSQL
    try:
        url = (
            f"postgresql://{PostgresConfig.USER}:{PostgresConfig.PASSWORD}"
            f"@{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
        )
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        services["postgres"] = ServiceStatus(status="ok", message="connected")
    except Exception as exc:
        services["postgres"] = ServiceStatus(status="error", message=str(exc)[:100])

    # MongoDB
    try:
        from pymongo import MongoClient
        url = (
            f"mongodb://{MongoConfig.USER}:{MongoConfig.PASSWORD}"
            f"@{MongoConfig.HOST}:{MongoConfig.PORT}/{MongoConfig.DB}"
            "?authSource=admin"
        )
        client = MongoClient(url, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        services["mongodb"] = ServiceStatus(status="ok", message="connected")
    except Exception as exc:
        services["mongodb"] = ServiceStatus(status="error", message=str(exc)[:100])

    # ChromaDB 
    try:
        from agents.vector_store import get_vector_store
        store = get_vector_store()
        ok    = store.health_check()
        if ok:
            count = store.count()
            services["chromadb"] = ServiceStatus(
                status="ok", message=f"connected — {count} documents"
            )
        else:
            services["chromadb"] = ServiceStatus(status="error", message="unreachable")
    except Exception as exc:
        services["chromadb"] = ServiceStatus(status="error", message=str(exc)[:100])

    all_ok = all(s.status == "ok" for s in services.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        services=services,
    )