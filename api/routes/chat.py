"""
Main chat endpoint — routes queries through the LangGraph agent.

POST /api/chat
"""

import logging
import re
import time

from fastapi import APIRouter, HTTPException

from api.models import ChatRequest, ChatResponse
from agents.graph import run_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


# Prompt injection guardrail 

# Patterns that indicate prompt injection or jailbreak attempts
_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(previous|all|prior|above)\s+(instructions?|prompts?|context|rules?)|"
    r"you\s+are\s+now\s+(a\s+)?(different|new|another|an?)\s+|"
    r"forget\s+(everything|all|your|previous|prior)|"
    r"disregard\s+(all|previous|prior|your|the)\s+|"
    r"(pretend|act|behave)\s+(you\s+are|as\s+if|like\s+you)|"
    r"new\s+(persona|role|identity|character|instructions?)|"
    r"system\s*prompt|"
    r"jailbreak|"
    r"(do\s+anything\s+now|dan\s+mode)|"
    r"bypass\s+(your\s+)?(safety|filter|restriction|guard|rule)|"
    r"override\s+(your\s+)?(instruction|setting|rule|limit)|"
    r"<\s*(system|instruction|prompt)\s*>|"    # XML-style injection
    r"\[\s*system\s*\]|\[\s*inst\s*\]",        # bracket-style injection
    re.IGNORECASE,
)

# Characters that could be used to smuggle instructions
_SUSPICIOUS_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate a user query for prompt injection attempts.

    Returns:
        (is_safe, reason_if_rejected)
    """
    # Check for control characters / null bytes
    if _SUSPICIOUS_CHAR_PATTERN.search(query):
        return False, "Query contains invalid characters"

    # Check for injection patterns
    if _INJECTION_PATTERNS.search(query):
        return False, "Query contains disallowed instructions"

    # Check for excessive repetition (common in some attack patterns)
    words = query.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False, "Query contains excessive repetition"

    return True, ""


# Chat endpoint 

@router.post("/chat", response_model=ChatResponse, summary="Query the StockPilot AI agent")
async def chat(request: ChatRequest):
    """
    Routes the user query through the full LangGraph multi-agent system:
    - Prompt injection guardrail validates input before processing
    - Supervisor classifies the query and extracts ticker
    - News RAG: ChromaDB hybrid search + Tavily web search fallback + Claude generation
    - Stock Data RAG: NL→SQL→PostgreSQL→Claude generation
    - General: Direct Claude response

    Returns the final answer along with routing metadata for transparency.
    """
    start = time.time()
    logger.info(f"[chat] Query received: '{request.query[:80]}'")

    # Guardrail check 
    is_safe, reason = validate_query(request.query)
    if not is_safe:
        logger.warning(f"[chat] Prompt injection blocked: {reason} | query='{request.query[:80]}'")
        raise HTTPException(
            status_code=400,
            detail="Your query could not be processed. Please ask a question about stocks, news, or markets.",
        )

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
                url  = meta.get("url", "")
                if url:
                    sources.append({
                        "title": meta.get("source", "News article"),
                        "url":   url,
                    })
            for doc in result.get("web_search_results", []):
                meta  = doc.get("metadata", {})
                url   = meta.get("source", "")
                title = meta.get("title", "Web source")
                if url:
                    sources.append({
                        "title": title,
                        "url":   url,
                    })

        elapsed = time.time() - start
        logger.info(f"[chat] Completed in {elapsed:.2f}s | route={route}")

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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[chat] Unhandled error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(exc)[:300]}",
        )