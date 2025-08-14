# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go

from src.portfolio.streamlit_utils import (
    load_price_df, filter_by_date, latest_price_and_change, last_updated_timestamp
)

# --- Page config ---
st.set_page_config(
    page_title="GMF | Forecast & Portfolio Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Header / Summary Card ---
col_logo, col_title, col_kpis = st.columns([0.12, 0.58, 0.30])
with col_logo:
    st.markdown("### 🟦")  # placeholder for your logo (replace later)
with col_title:
    st.markdown("## GMF Investments — Forecast & Portfolio Dashboard")
    st.caption("Forecasts are for informational purposes only.")
with col_kpis:
    st.caption(f"Last updated: **{last_updated_timestamp()}**")

st.divider()

# --- Controls row (asset & date range) ---
ctrl1, ctrl2, ctrl3 = st.columns([0.20, 0.32, 0.48])

with ctrl1:
    ticker = st.selectbox("Asset", ["TSLA", "SPY", "BND"], index=0)

# Load data early to compute date bounds
df_all = load_price_df(ticker)
min_d, max_d = df_all.index.min().date(), df_all.index.max().date()

with ctrl2:
    rng = st.date_input(
        "Date Range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )
    # Ensure tuple (start, end)
    if isinstance(rng, tuple):
        start_d, end_d = rng
    else:
        start_d, end_d = min_d, max_d

with ctrl3:
    last_px, pct = latest_price_and_change(df_all)
    k1, k2, k3 = st.columns(3)
    k1.metric("Last Close", f"{last_px:,.2f}" if last_px is not None else "—")
    k2.metric("% Change (d/d)", f"{pct*100:.2f}%" if pct is not None else "—")
    k3.metric("Records", f"{len(df_all):,}")

st.divider()

# --- Main chart (candlestick with date filter) ---
df = filter_by_date(df_all, start_d, end_d)

st.markdown("### Price Chart")
if {"Open","High","Low","Close"}.issubset(df.columns):
    fig = go.Figure(
        data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker
        )]
    )
else:
    # Fallback to line plot if OHLC not present
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker)])

fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=True,
    template="plotly_white",
    title=f"{ticker} — Historical Prices"
)
st.plotly_chart(fig, use_container_width=True)

# --- Sidebar placeholder (we'll add Finnhub news later) ---
with st.sidebar:
    st.header("News (Coming Next)")
    st.caption("TSLA / SPY / BND — live headlines here.")
    st.info("We’ll integrate Finnhub 'company-news' with a refresh button.")

# --- Footer ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© GMF Investments — Forecasts are for informational purposes only.")
