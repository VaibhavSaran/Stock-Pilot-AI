"""
Shared TypedDict state that flows through all LangGraph subgraphs.

Every node reads from and writes to this state. LangGraph merges
updates using the Annotated reducers — lists use operator.add
(append-only) while plain fields are last-write-wins.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Central state object passed between all nodes in the graph.

    Fields
    ------
    query : str
        The original user question, never mutated.
    ticker : str | None
        Ticker extracted from the query (e.g. "AAPL").
        Set by the supervisor, used by downstream subgraphs.
    route : str | None
        Which subgraph the supervisor selected:
        "news_rag" | "stock_data_rag" | "general"
    messages : list[str]
        Append-only log of intermediate outputs from each node.
        Useful for debugging and LangSmith traces.
    retrieved_docs : list[dict]
        Documents retrieved from ChromaDB (news RAG path).
    sql_query : str | None
        Generated SQL query (stock data RAG path).
    sql_results : list[dict]
        Rows returned from executing the SQL query.
    web_search_results : list[dict]
        Results from Tavily web search fallback (news RAG path).
    final_answer : str | None
        The final generated response returned to the user.
    error : str | None
        Error message if any node fails gracefully.
    """

    query:               str
    ticker:              Optional[str]
    route:               Optional[str]
    messages:            Annotated[list[str], operator.add]
    retrieved_docs:      list[dict]
    sql_query:           Optional[str]
    sql_results:         list[dict]
    web_search_results:  list[dict]
    final_answer:        Optional[str]
    error:               Optional[str]


def initial_state(query: str) -> AgentState:
    """Return a clean initial state for a new query."""
    return AgentState(
        query=query,
        ticker=None,
        route=None,
        messages=[],
        retrieved_docs=[],
        sql_query=None,
        sql_results=[],
        web_search_results=[],
        final_answer=None,
        error=None,
    )