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

        # Count sources
        if route == "news_rag":
            sources_used = len(result.get("retrieved_docs", [])) + len(
                result.get("web_search_results", [])
            )
        elif route == "stock_data_rag":
            sources_used = len(result.get("sql_results", []))
        else:
            sources_used = 0

        # Build sources list
        sources = []
        if route == "news_rag":
            for doc in result.get("retrieved_docs", []):
                meta = doc.get("metadata", {})
                url = meta.get("url", "")
                if url:
                    sources.append({
                        "title": meta.get("source", "News article"),
                        "url":   url,
                    })
            for doc in result.get("web_search_results", []):
                meta = doc.get("metadata", {})
                # Tavily stores URL in "source" key
                url   = meta.get("source", "")
                title = meta.get("title", "Web source")
                if url:
                    sources.append({
                        "title": title,
                        "url":   url,
                    })

        return ChatResponse(
            query=request.query,
            answer=result.get("final_answer") or "No answer generated.",
            route=route,
            ticker=result.get("ticker"),
            sql_query=result.get("sql_query"),
            sources_used=sources_used,
            sources=sources,
            error=result.get("error"),
        )

    except Exception as exc:
        logger.error(f"[chat] Unhandled error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(exc)[:300]}",
        )