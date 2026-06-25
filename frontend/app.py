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
import os

#Config ────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]

COMPANY_NAMES = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN":  "Amazon",
    "TSLA":  "Tesla",
    "NVDA":  "NVIDIA",
    "META":  "Meta",
}

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
    .forecast-disclaimer {
        font-size: 11px;
        color: #888;
        font-style: italic;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


#API helper functions 

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

    # Sources shown as hyperlinks only — SQL query hidden from users
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


#Sidebar 

with st.sidebar:
    st.title("📈 StockPilot AI")
    st.caption("AI-powered stock market analysis")
    st.divider()

    page = st.radio(
        "Navigation",
        ["💬 Chat With StockPilot AI", "📊 Stock Data Analysis"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Powered by Claude · Gemini · LangGraph")


#Page: Chat 

if page == "💬 Chat With StockPilot AI":
    st.header("Chat")

    # Company names in subtitle instead of ticker symbols
    company_list = ", ".join(COMPANY_NAMES.values())
    st.caption(
        f"Powered by Claude Sonnet 4.6 · Specialized in {company_list}"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render full history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    # New query input
    company_names_short = ", ".join(list(COMPANY_NAMES.values())[:4]) + ", and more..."
    if query := st.chat_input(f"Ask about {company_names_short}"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            result = api_post("/api/chat", {"query": query})

        if result:
            st.session_state.messages.append({
                "role":         "assistant",
                "content":      result.get("answer", "No answer returned."),
                "route":        result.get("route"),
                "ticker":       result.get("ticker"),
                "sources":      result.get("sources", []),
                "sources_used": result.get("sources_used", 0),
            })
            st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()


#Page: Stock Data Analysis 

elif page == "📊 Stock Data Analysis":
    st.header("Stock Data Analysis")

    # Ticker + period selectors shared across tabs
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Ticker", TICKERS, format_func=lambda t: f"{t} — {COMPANY_NAMES[t]}")
    with col2:
        days = st.selectbox("Period", [7, 14, 30, 90], index=2, format_func=lambda x: f"{x} days")

    # Tabs
    tab_prices, tab_forecast = st.tabs(["📈 Price History", "🔮 Price Forecast"])

    #Tab: Price History 
    with tab_prices:
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
                c3.metric(
                    "Market Cap",
                    f"${cap/1e12:.2f}T" if cap and cap >= 1e12
                    else (f"${cap/1e9:.1f}B" if cap else "—")
                )
                latest = df.iloc[-1]
                prev   = df.iloc[-2] if len(df) > 1 else None
                change = (
                    (latest["close"] - prev["close"]) / prev["close"] * 100
                    if prev is not None else 0
                )
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

    #Tab: Price Forecast 
    with tab_forecast:
        st.subheader(f"{COMPANY_NAMES[ticker]} ({ticker}) — Price Forecast")
        st.markdown(
            '<p class="forecast-disclaimer">⚠️ Forecasts are generated by FB Prophet using historical closing prices. '
            'This is for informational purposes only and should not be used as investment advice.</p>',
            unsafe_allow_html=True,
        )

        forecast_days = st.slider(
            "Forecast horizon",
            min_value=1,
            max_value=7,
            value=5,
            step=1,
            format="%d days",
        )

        if st.button("Generate Forecast", type="primary"):
            with st.spinner(f"Running Prophet model for {ticker}..."):
                forecast_data = api_get(
                    f"/api/stocks/{ticker}/forecast",
                    {"days": forecast_days},
                )

            if forecast_data and forecast_data.get("forecast"):
                f = forecast_data

                # Summary metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Last Known Price", f"${f['last_known_price']:.2f}", f['last_known_date'])
                m2.metric("Training Data", f"{f['training_points']} trading days")

                last_forecast = f["forecast"][-1]
                price_change  = last_forecast["predicted"] - f["last_known_price"]
                pct_change    = (price_change / f["last_known_price"]) * 100
                m3.metric(
                    f"Day {forecast_days} Forecast",
                    f"${last_forecast['predicted']:.2f}",
                    f"{pct_change:+.2f}%",
                )

                st.divider()

                # Build forecast chart
                forecast_df = pd.DataFrame(f["forecast"])
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])

                # Fetch historical prices for context (last 30 days)
                hist_data = api_get(f"/api/stocks/{ticker}/prices", {"days": 30})
                fig = go.Figure()

                if hist_data and hist_data.get("prices"):
                    hist_df = pd.DataFrame(hist_data["prices"])
                    hist_df["price_date"] = pd.to_datetime(hist_df["price_date"])
                    hist_df = hist_df.sort_values("price_date")

                    # Historical closing prices
                    fig.add_trace(go.Scatter(
                        x=hist_df["price_date"],
                        y=hist_df["close"],
                        mode="lines",
                        name="Historical Close",
                        line=dict(color="#26a69a", width=2),
                    ))

                # Confidence interval band
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                    y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(99, 102, 241, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Confidence Interval",
                    showlegend=True,
                ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#818cf8", width=2, dash="dash"),
                    marker=dict(size=8, color="#818cf8"),
                ))

                fig.update_layout(
                    title=f"{ticker} — {forecast_days}-Day Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=420,
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                st.subheader("Forecast Details")
                table_df = forecast_df.copy()
                table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
                table_df.columns = ["Date", "Predicted ($)", "Lower ($)", "Upper ($)"]
                st.dataframe(table_df, use_container_width=True, hide_index=True)

            elif forecast_data:
                st.error(forecast_data.get("detail", "Forecast failed. Not enough data."))