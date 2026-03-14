"""
LangGraph subgraph for News RAG.

Flow:
retrieve_news -> grade_docs -> [web_search (fallback)] -> generate_news_answer

Nodes:
retrieve_news      : Query ChromaDB for semantically similar news articles
grade_docs         : Use Claude to filter irrelevant docs (relevance check)
web_search         : Tavily fallback if graded docs are insufficient
generate_news_answer: Claude synthesises a final answer from context docs

Routing:
After grading: if enough relevant docs -> generate directly
               if too few -> web_search first, then generate
"""

import logging
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.vector_store import get_vector_store
from config.config import LLMConfig, StockAPIConfig

logger = logging.getLogger(__name__)

# LLM — lazy init

_llm: ChatAnthropic | None = None

def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=LLMConfig.CLAUDE_MODEL,
            api_key=LLMConfig.ANTHROPIC_API_KEY,
            max_tokens=1024,
        )
    return _llm

# Node: retrieve_news
def retrieve_news(state: AgentState) -> dict:
    """
    Retrieve top-k news articles from ChromaDB.
    Filters by ticker if one was extracted by the supervisor.
    """
    query  = state["query"]
    ticker = state.get("ticker")

    store = get_vector_store()
    docs  = store.similarity_search(query, ticker=ticker, k=6)

    logger.info(f"[news_rag] Retrieved {len(docs)} docs for query='{query[:50]}'")

    return {
        "retrieved_docs": docs,
        "messages": [f"[retrieve_news] Found {len(docs)} candidate documents"],
    }

# Node: grade_docs
def grade_docs(state: AgentState) -> dict:
    """
    Ask Claude to grade each retrieved document for relevance.
    Keeps only docs that directly address the query.
    Relevance threshold: at least 2 relevant docs needed to skip web search.
    """
    query = state["query"]
    docs  = state.get("retrieved_docs", [])

    if not docs:
        return {
            "retrieved_docs": [],
            "messages": ["[grade_docs] No docs to grade"],
        }

    llm = _get_llm()
    relevant_docs = []

    for doc in docs:
        system = (
            "You are a relevance grader. Given a user question and a news article snippet, "
            "respond with ONLY 'yes' if the article is relevant to the question, or 'no' if not."
        )
        human = f"Question: {query}\n\nArticle snippet: {doc['document'][:500]}"

        try:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=human),
            ])
            if "yes" in response.content.lower():
                relevant_docs.append(doc)
        except Exception as exc:
            logger.warning(f"[grade_docs] Grading failed for doc {doc['id']}: {exc}")
            # On error, keep the doc (fail open)
            relevant_docs.append(doc)

    logger.info(
        f"[news_rag] Grading: {len(docs)} retrieved -> {len(relevant_docs)} relevant"
    )

    return {
        "retrieved_docs": relevant_docs,
        "messages": [
            f"[grade_docs] {len(relevant_docs)}/{len(docs)} docs passed relevance check"
        ],
    }

# Node: web_search
def web_search(state: AgentState) -> dict:
    """
    Tavily web search fallback when ChromaDB docs are insufficient.
    Appends results to web_search_results for use in generation.
    """
    query  = state["query"]
    ticker = state.get("ticker")

    search_query = f"{ticker} {query}" if ticker else query

    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=StockAPIConfig.TAVILY_API_KEY)
        results = client.search(query=search_query, max_results=4)

        web_docs = [
            {
                "document": r.get("content", ""),
                "metadata": {
                    "source": r.get("url", ""),
                    "title":  r.get("title", ""),
                },
            }
            for r in results.get("results", [])
        ]

        logger.info(f"[news_rag] Web search returned {len(web_docs)} results")
        return {
            "web_search_results": web_docs,
            "messages": [f"[web_search] Tavily returned {len(web_docs)} results"],
        }

    except Exception as exc:
        logger.error(f"[news_rag] Web search failed: {exc}")
        return {
            "web_search_results": [],
            "messages": [f"[web_search] Failed: {exc}"],
        }

# Node: generate_news_answer
def generate_news_answer(state: AgentState) -> dict:
    """
    Claude synthesises a final answer from retrieved + web search docs.
    """
    query    = state["query"]
    ticker   = state.get("ticker", "")
    rag_docs = state.get("retrieved_docs", [])
    web_docs = state.get("web_search_results", [])

    # Combine all available context
    all_docs = rag_docs + web_docs

    if not all_docs:
        return {
            "final_answer": (
                f"I couldn't find relevant news articles for your query about "
                f"{ticker or 'this topic'}. Please try rephrasing or ask about "
                f"a specific ticker."
            ),
            "messages": ["[generate_news_answer] No context available — fallback answer"],
        }

    # Build context string
    context_parts = []
    for i, doc in enumerate(all_docs[:6], 1):
        source = doc.get("metadata", {}).get("source", "")
        text   = doc.get("document", "")[:600]
        context_parts.append(f"[Source {i}] {source}\n{text}")

    context = "\n\n".join(context_parts)

    system = """You are StockPilot AI, a financial analysis assistant.
Answer the user's question using ONLY the provided news context.
Be concise, factual, and cite source numbers where relevant.
If the context doesn't fully answer the question, say so clearly."""

    human = f"""Question: {query}

News Context:
{context}

Please provide a clear, well-structured answer based on the above context."""

    try:
        llm      = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        answer = response.content

        logger.info(f"[news_rag] Generated answer ({len(answer)} chars)")
        return {
            "final_answer": answer,
            "messages": ["[generate_news_answer] Answer generated successfully"],
        }

    except Exception as exc:
        logger.error(f"[news_rag] Generation failed: {exc}")
        return {
            "final_answer": f"I encountered an error generating the answer: {exc}",
            "messages": [f"[generate_news_answer] Error: {exc}"],
            "error": str(exc),
        }


# Routing function
def route_after_grading(
    state: AgentState,
) -> Literal["web_search", "generate_news_answer"]:
    """
    Route after doc grading:
    - 2+ relevant docs -> generate directly
    - Fewer -> web search first
    """
    relevant_docs = state.get("retrieved_docs", [])
    if len(relevant_docs) >= 2:
        return "generate_news_answer"
    return "web_search"



# Build the subgraph
def build_news_rag_graph() -> StateGraph:
    """Compile and return the News RAG subgraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("retrieve_news", retrieve_news)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_news_answer", generate_news_answer)

    # Entry point
    graph.set_entry_point("retrieve_news")

    # Edges
    graph.add_edge("retrieve_news", "grade_docs")

    graph.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "web_search":"web_search",
            "generate_news_answer": "generate_news_answer",
        },
    )

    graph.add_edge("web_search", "generate_news_answer")
    graph.add_edge("generate_news_answer", END)

    return graph.compile()