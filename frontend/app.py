"""
StockPilot AI — Streamlit Frontend

Run with:
    streamlit run frontend/app.py

Requires the FastAPI backend running at http://localhost:8000 for local development.
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


# Config
import os
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

st.set_page_config(
    page_title="StockPilot AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-news     { background: #1e3a5f; color: #90caf9; }
    .badge-stock    { background: #1b3a2f; color: #80cbc4; }
    .badge-general  { background: #3e2723; color: #ffcc80; }
</style>
""", unsafe_allow_html=True)

# API helper functions
def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure `uvicorn api.main:app` is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure `uvicorn api.main:app` is running.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def render_assistant_message(msg: dict):
    """
    Render a single assistant message from history.
    Called from the history loop — expanders work correctly here
    because this runs on a clean rerun, not mid-stream.
    """
    route = msg.get("route")
    if route == "news_rag":
        st.markdown('<span class="route-badge badge-news">NEWS RAG</span>', unsafe_allow_html=True)
    elif route == "stock_data_rag":
        st.markdown('<span class="route-badge badge-stock">STOCK DATA RAG</span>', unsafe_allow_html=True)
    elif route == "general":
        st.markdown('<span class="route-badge badge-general">GENERAL</span>', unsafe_allow_html=True)

    st.markdown(msg["content"])

    if msg.get("sql_query"):
        with st.expander("SQL Query"):
            st.code(msg["sql_query"], language="sql")

    sources = msg.get("sources", [])
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for i, src in enumerate(sources, 1):
                title = src.get("title", f"Source {i}")
                url   = src.get("url", "")
                if url:
                    st.markdown(f"**{i}.** [{title}]({url})")
                else:
                    st.markdown(f"**{i}.** {title}")

    meta_parts = []
    if msg.get("ticker"):
        meta_parts.append(f"Ticker: **{msg['ticker']}**")
    if msg.get("sources_used"):
        meta_parts.append(f"Sources: **{msg['sources_used']}**")
    if meta_parts:
        st.caption(" · ".join(meta_parts))

# Sidebar
with st.sidebar:
    st.title("📈 StockPilot AI")
    st.caption("AI-powered stock market analysis")
    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Chat", "📊 Stock Data", "🔧 System Status"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by Claude · Gemini · LangGraph")

# Page: Chat
if page == "💬 Chat":
    st.header("Chat")
    st.caption("Ask anything about stocks, news, or markets.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render full history — expanders work correctly here
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    # New query input
    if query := st.chat_input("Ask about a stock, news, or market..."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            result = api_post("/api/chat", {"query": query})

        if result:
            st.session_state.messages.append({
                "role":         "assistant",
                "content":      result.get("answer", "No answer returned."),
                "route":        result.get("route"),
                "ticker":       result.get("ticker"),
                "sql_query":    result.get("sql_query"),
                "sources":      result.get("sources", []),
                "sources_used": result.get("sources_used", 0),
            })
            # Rerun so the new message renders from the history loop
            # where expanders work correctly
            st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# Page: Stock Data
elif page == "📊 Stock Data":
    st.header("Stock Data")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Ticker", TICKERS)
    with col2:
        days = st.selectbox("Period", [7, 14, 30, 90], index=0, format_func=lambda x: f"{x} days")

    data = api_get(f"/api/stocks/{ticker}/prices", {"days": days})

    if data and data.get("prices"):
        prices = data["prices"]
        df = pd.DataFrame(prices)
        df["price_date"] = pd.to_datetime(df["price_date"])
        df = df.sort_values("price_date")

        meta = api_get(f"/api/stocks/{ticker}/metadata")

        if meta:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Company", meta.get("company_name", ticker))
            c2.metric("Sector", meta.get("sector", "—"))
            cap = meta.get("market_cap")
            c3.metric("Market Cap", f"${cap/1e12:.2f}T" if cap and cap >= 1e12 else (f"${cap/1e9:.1f}B" if cap else "—"))
            latest = df.iloc[-1]
            prev   = df.iloc[-2] if len(df) > 1 else None
            change = ((latest["close"] - prev["close"]) / prev["close"] * 100) if prev is not None else 0
            c4.metric("Latest Close", f"${latest['close']:.2f}", f"{change:+.2f}%")

        st.divider()

        fig = go.Figure(data=[go.Candlestick(
            x=df["price_date"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )])
        fig.update_layout(
            title=f"{ticker} — {days}-Day Price History",
            xaxis_title="Date", yaxis_title="Price (USD)",
            height=400, xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(data=[go.Bar(
            x=df["price_date"], y=df["volume"],
            name="Volume", marker_color="#546e7a",
        )])
        fig2.update_layout(
            title="Volume", height=200,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Raw Data"):
            display_df = df[["price_date", "open", "high", "low", "close", "volume"]].copy()
            display_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    elif data and data.get("count") == 0:
        st.info(f"No price data for {ticker}. Run the stock scraper first.")

# Page: System Status
elif page == "🔧 System Status":
    st.header("System Status")

    if st.button("Refresh", type="secondary"):
        st.rerun()

    data = api_get("/health/services")

    if data:
        if data.get("status") == "ok":
            st.success("All systems operational")
        else:
            st.warning("One or more services degraded")

        st.divider()

        for name, info in data.get("services", {}).items():
            col1, col2 = st.columns([1, 4])
            with col1:
                if info["status"] == "ok":
                    st.success(name.upper())
                else:
                    st.error(name.upper())
            with col2:
                st.caption(info["message"])

        st.divider()
        st.caption(f"Version: {data.get('version', '—')}  ·  Checked: {datetime.now().strftime('%H:%M:%S')}")