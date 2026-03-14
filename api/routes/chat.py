"""
Main chat endpoint — routes queries through the LangGraph agent.

POST /api/chat
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from api.models import ChatRequest, ChatResponse
from agents.graph import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Query the StockPilot AI agent")
async def chat(request: ChatRequest):
    """
    Routes the user query through the full LangGraph multi-agent system:
    - Supervisor classifies the query and extracts ticker
    - News RAG: ChromaDB retrieval + Tavily web search fallback + Claude generation
    - Stock Data RAG: NL→SQL→PostgreSQL→Claude generation
    - General: Direct Claude response

    Returns the final answer along with routing metadata for transparency.
    """
    start = time.time()
    logger.info(f"[chat] Query received: '{request.query[:80]}'")

    try:
        result = run_query(request.query)

        # Count sources used depending on route
        route = result.get("route")
        if route == "news_rag":
            sources_used = len(result.get("retrieved_docs", [])) + len(
                result.get("web_search_results", [])
            )
        elif route == "stock_data_rag":
            sources_used = len(result.get("sql_results", []))
        else:
            sources_used = 0

        elapsed = round(time.time() - start, 2)
        logger.info(
            f"[chat] Completed in {elapsed}s | route={route} | "
            f"ticker={result.get('ticker')} | sources={sources_used}"
        )

        return ChatResponse(
            query=request.query,
            answer=result.get("final_answer") or "No answer generated.",
            route=route,
            ticker=result.get("ticker"),
            sql_query=result.get("sql_query"),
            sources_used=sources_used,
            error=result.get("error"),
        )

    except Exception as exc:
        logger.error(f"[chat] Unhandled error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(exc)[:300]}",
        )