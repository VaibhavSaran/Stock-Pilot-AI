"""
Supervisor Agent — the top-level router in the StockPilot AI architecture.

Responsibilities:
1. Extract ticker symbol from the user query (if any)
2. Classify the query into one of three routes:
   - "news_rag"       -> questions about news, sentiment, recent events
   - "stock_data_rag" -> questions about prices, volume, performance, charts
   - "general"        -> general market questions (answered directly)
3. Dispatch to the appropriate subgraph or answer directly

The Supervisor pattern is standard in multi-agent LangGraph systems.
It gives us a clean single entry point while keeping each subgraph
independently testable and maintainable.
"""

import logging
import re
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from config.config import LLMConfig, TRACKED_TICKERS

logger = logging.getLogger(__name__)


# LLM
_llm: ChatAnthropic | None = None

def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=LLMConfig.CLAUDE_MODEL,
            api_key=LLMConfig.ANTHROPIC_API_KEY,
            max_tokens=256,
        )
    return _llm

# Ticker extraction — fast regex before calling the LLM
def _extract_ticker(query: str) -> str | None:
    """
    Extract a stock ticker from the query.
    Checks tracked tickers first (O(1)), then looks for $TICKER pattern.
    """
    query_upper = query.upper()

    # Check known tickers directly
    for ticker in TRACKED_TICKERS:
        # Word-boundary match to avoid false positives (e.g. "AMAZE" ≠ "AMZN")
        if re.search(rf"\b{ticker}\b", query_upper):
            return ticker

    # $TICKER pattern (e.g. "$NVDA")
    match = re.search(r"\$([A-Z]{1,5})\b", query_upper)
    if match:
        return match.group(1)

    return None


# Node: classify_query
def classify_query(state: AgentState) -> dict:
    """
    Supervisor node: extracts ticker and classifies the query route.

    Uses Claude for classification to handle ambiguous phrasing,
    with a structured prompt that constrains output to exactly
    one of: news_rag | stock_data_rag | general
    """
    query = state["query"]

    # Fast ticker extraction
    ticker = _extract_ticker(query)

    system = """You are a query classifier for a stock market AI assistant.
Classify the user's question into exactly one of these categories:

1. news_rag       - Questions about news, recent events, sentiment, analyst opinions,
                    earnings announcements, company developments, market reactions
                    Examples: "What's the latest news on Apple?",
                              "What did analysts say about Tesla earnings?",
                              "Any recent developments for NVDA?"

2. stock_data_rag - Questions about stock prices, price history, trading volume,
                    performance metrics, price comparisons, highs/lows
                    Examples: "What was AAPL's closing price last week?",
                              "Compare MSFT and GOOGL performance this month",
                              "What's the highest TSLA has traded recently?"

3. general        - General market questions not requiring specific data retrieval,
                    conceptual questions, broad market commentary
                    Examples: "What is a P/E ratio?",
                              "How does the Fed affect stock prices?",
                              "What are the best sectors in 2026?"

Respond with ONLY one word: news_rag, stock_data_rag, or general"""

    human = f"Query: {query}"

    try:
        llm      = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])

        route_raw = response.content.strip().lower()

        # Normalise to valid routes
        if "news" in route_raw:
            route = "news_rag"
        elif "stock" in route_raw or "data" in route_raw:
            route = "stock_data_rag"
        else:
            route = "general"

        logger.info(
            f"[supervisor] Query='{query[:60]}' -> ticker={ticker} route={route}"
        )

        return {
            "ticker":   ticker,
            "route":    route,
            "messages": [
                f"[supervisor] Classified as '{route}'"
                + (f" | ticker={ticker}" if ticker else "")
            ],
        }

    except Exception as exc:
        logger.error(f"[supervisor] Classification failed: {exc}")
        # Safe fallback — try news_rag as default
        return {
            "ticker":  ticker,
            "route":  "news_rag",
            "messages": [f"[supervisor] Classification error, defaulting to news_rag: {exc}"],
        }


# Node: general_answer
def general_answer(state: AgentState) -> dict:
    """
    Handle general market/finance questions directly without RAG.
    Claude answers from its training knowledge.
    """
    query = state["query"]

    system = """You are StockPilot AI, a knowledgeable financial assistant.
Answer the user's general question about markets, finance, or investing clearly and concisely.
Base your answer on well-established financial knowledge.
Keep responses under 300 words unless more detail is clearly needed."""

    try:
        llm= _get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=query),
        ])

        logger.info("[supervisor] General answer generated")
        return {
            "final_answer": response.content,
            "messages": ["[general_answer] Answered from general knowledge"],
        }

    except Exception as exc:
        logger.error(f"[supervisor] General answer failed: {exc}")
        return {
            "final_answer": f"I encountered an error: {exc}",
            "messages":[f"[general_answer] Error: {exc}"],
            "error": str(exc),
        }


# Routing function
def route_query(
    state: AgentState,
) -> Literal["news_rag_subgraph", "stock_data_rag_subgraph", "general_answer"]:
    """Route to the appropriate subgraph based on supervisor classification."""
    route = state.get("route", "general")

    if route == "news_rag":
        return "news_rag_subgraph"
    elif route == "stock_data_rag":
        return "stock_data_rag_subgraph"
    else:
        return "general_answer"