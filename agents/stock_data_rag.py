"""
LangGraph subgraph for Stock Data RAG (NL -> SQL -> Execute -> Generate).

Flow:
generate_sql -> execute_sql -> generate_stock_answer

Nodes:
generate_sql        : Claude translates natural language to SQL
execute_sql         : Runs the SQL against PostgreSQL, returns rows
generate_stock_answer: Claude narrates the query results in plain English

Schema context injected into the SQL generation prompt:
  stock_prices(ticker, open, high, low, close, volume, price_date)
  stock_metadata(ticker, company_name, sector, industry, market_cap)
"""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from agents.state import AgentState
from config.config import LLMConfig, PostgresConfig

logger = logging.getLogger(__name__)


# DB engine — lazy init
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        url = (
            f"postgresql://{PostgresConfig.USER}:{PostgresConfig.PASSWORD}"
            f"@{PostgresConfig.HOST}:{PostgresConfig.PORT}/{PostgresConfig.DB}"
        )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine

# LLM — lazy init
_llm: ChatAnthropic | None = None

def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model=LLMConfig.CLAUDE_MODEL,
            api_key=LLMConfig.ANTHROPIC_API_KEY,
            max_tokens=512,
        )
    return _llm

# DB schema — injected into SQL generation prompt
DB_SCHEMA = """
PostgreSQL Database Schema:

TABLE stock_prices (
    ticker      VARCHAR(10),   -- e.g. 'AAPL', 'MSFT'
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      BIGINT,
    price_date  DATE,          -- trading date
    UNIQUE(ticker, price_date)
);

TABLE stock_metadata (
    ticker       VARCHAR(10),
    company_name VARCHAR(255),
    sector       VARCHAR(100),
    industry     VARCHAR(100),
    market_cap   BIGINT
);

Notes:
- All tickers are uppercase: 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META'
- price_date is a DATE column. For "last week" or "recent" queries always use:
  price_date >= CURRENT_DATE - INTERVAL '7 days'
  Never use DATE_TRUNC - it creates overly narrow windows that miss data.
- Always add LIMIT 20 unless the user asks for aggregations
- Use ROUND() for numeric display
- Join tables on ticker when company info is needed
"""


# Node: generate_sql
def generate_sql(state: AgentState) -> dict:
    """
    Translate the user's natural language question into a safe PostgreSQL query.
    Claude is constrained to SELECT only and given the exact schema.
    """
    query  = state["query"]
    ticker = state.get("ticker")

    system = f"""You are a PostgreSQL expert. Convert the user's question into a valid SQL query.

{DB_SCHEMA}

RULES:
1. ONLY generate SELECT statements — never INSERT, UPDATE, DELETE, DROP, or any DDL
2. Always use parameterised-style queries (no user input interpolation)
3. If a ticker is mentioned in the question, filter by that ticker
4. Return ONLY the raw SQL query — no explanation, no markdown, no backticks
5. If the question cannot be answered with the available schema, return: SELECT 'unsupported query' AS message;"""

    ticker_hint = f"\nNote: The user is asking about ticker: {ticker}" if ticker else ""
    human = f"Question: {query}{ticker_hint}"

    try:
        llm      = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])

        sql = response.content.strip()

        # Safety guardrail — reject non-SELECT queries
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            logger.warning(f"[stock_data_rag] Non-SELECT SQL generated — rejecting: {sql[:100]}")
            sql = "SELECT 'Query not supported for safety reasons' AS message;"

        logger.info(f"[stock_data_rag] Generated SQL: {sql[:120]}")
        return {
            "sql_query": sql,
            "messages":  [f"[generate_sql] SQL: {sql[:100]}"],
        }

    except Exception as exc:
        logger.error(f"[stock_data_rag] SQL generation failed: {exc}")
        return {
            "sql_query": None,
            "messages":  [f"[generate_sql] Error: {exc}"],
            "error":     str(exc),
        }

# Node: execute_sql
def execute_sql(state: AgentState) -> dict:
    """
    Execute the generated SQL against PostgreSQL.
    Returns rows as a list of dicts.
    Limits execution to 20 rows max (safety cap).
    """
    sql = state.get("sql_query")

    if not sql:
        return {
            "sql_results": [],
            "messages":    ["[execute_sql] No SQL to execute"],
        }

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows    = result.fetchmany(20)   # hard cap at 20 rows

            sql_results: list[dict[str, Any]] = [
                dict(zip(columns, row)) for row in rows
            ]

        logger.info(f"[stock_data_rag] SQL returned {len(sql_results)} rows")
        return {
            "sql_results": sql_results,
            "messages":    [f"[execute_sql] {len(sql_results)} rows returned"],
        }

    except SQLAlchemyError as exc:
        logger.error(f"[stock_data_rag] SQL execution error: {exc}")
        return {
            "sql_results": [],
            "messages": [f"[execute_sql] DB error: {exc}"],
            "error":str(exc),
        }

# Node: generate_stock_answer
def generate_stock_answer(state: AgentState) -> dict:
    """
    Claude narrates the SQL query results in clear, readable English.
    Includes the SQL for transparency.
    """
    query       = state["query"]
    sql         = state.get("sql_query", "")
    sql_results = state.get("sql_results", [])
    error       = state.get("error")

    if error and not sql_results:
        return {
            "final_answer": (
                f"I encountered a database error while retrieving stock data: {error}. "
                "Please try rephrasing your question."
            ),
            "messages": [f"[generate_stock_answer] Error path: {error}"],
        }

    if not sql_results:
        return {
            "final_answer": (
                "No data found in the database for your query. "
                "The stock data may not have been ingested yet. "
                "Please ensure the Airflow stock_prices_pipeline DAG has run successfully."
            ),
            "messages": ["[generate_stock_answer] Empty results"],
        }

    # Format results as a readable table string
    results_str = _format_results(sql_results)

    system = """You are StockPilot AI, a financial analysis assistant.
The user asked a question about stock data, and you retrieved the following results from the database.
Provide a clear, concise answer that directly addresses the question.
Include specific numbers and dates from the data.
Be professional but accessible — avoid excessive financial jargon."""

    human = f"""Question: {query}

SQL Query Used:
{sql}

Query Results:
{results_str}

Please provide a clear answer based on these results."""

    try:
        llm      = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        answer = response.content

        logger.info(f"[stock_data_rag] Generated answer ({len(answer)} chars)")
        return {
            "final_answer": answer,
            "messages":["[generate_stock_answer] Answer generated successfully"],
        }

    except Exception as exc:
        logger.error(f"[stock_data_rag] Generation failed: {exc}")
        return {
            "final_answer": f"Retrieved {len(sql_results)} rows but failed to generate narrative: {exc}",
            "messages":[f"[generate_stock_answer] Error: {exc}"],
            "error":str(exc),
        }

def _format_results(results: list[dict]) -> str:
    """Format SQL results as a readable string for the LLM prompt."""
    if not results:
        return "No results"

    # Header
    columns = list(results[0].keys())
    header  = " | ".join(columns)
    sep     = "-" * len(header)
    rows    = [" | ".join(str(v) for v in row.values()) for row in results[:10]]

    return "\n".join([header, sep] + rows)

# Build the subgraph
def build_stock_data_rag_graph() -> StateGraph:
    """Compile and return the Stock Data RAG subgraph."""
    graph = StateGraph(AgentState)

    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("generate_stock_answer", generate_stock_answer)

    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql","execute_sql")
    graph.add_edge("execute_sql", "generate_stock_answer")
    graph.add_edge("generate_stock_answer", END)

    return graph.compile()