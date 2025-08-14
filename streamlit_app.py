# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import os

from src.portfolio.streamlit_utils import (
    load_price_df, filter_by_date, latest_price_and_change, last_updated_timestamp
)

# --- Page config ---
st.set_page_config(
    page_title="GMF | Forecast & Portfolio Dashboard",
    #page_icon="📈",
    page_icon="../data/processed/GMF.png",
    layout="wide"
)

# --- Header / Summary Card ---
col_logo, col_title, col_kpis = st.columns([0.12, 0.58, 0.30])
with col_logo:
    st.image("Data/raw/GMF.png", width=500) # placeholder for your logo (replace later)
with col_title:
    st.markdown("## GMF Investments — Forecast & Portfolio Dashboard")
    #st.caption("Forecasts are for informational purposes only.")
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


# Always supply a tuple as default so Streamlit renders a range picker
with ctrl2:
 rng = st.date_input(
    "Date Range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

# Robustly interpret the return value
if isinstance(rng, tuple) and len(rng) == 2:
    start_d, end_d = rng
else:
    # User picked a single date or cleared one side; fall back gracefully
    start_d = rng if not isinstance(rng, tuple) else rng[0]
    end_d = max_d

#############################



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

# --- Main chart (built earlier as fig) is already created above this block ---

# 🔽 Only show forecast controls + overlay for TSLA
if ticker == "TSLA":
    st.markdown("### Forecast Controls")

    fc_cols = st.columns([0.25, 0.25, 0.5])
    with fc_cols[0]:
        months = st.selectbox("Forecast Horizon (months)", [3, 6, 9, 12, 18, 24], index=3)
    with fc_cols[1]:
        lookback = st.number_input("LSTM Lookback (days)", min_value=30, max_value=180, value=60, step=5)
    with fc_cols[2]:
        st.caption("TSLA is forecast with the saved LSTM model. SPY/BND use historical returns.")

    # ---- Run LSTM forecast & overlay ----
    from src.lstm_model import load_lstm_and_scaler, forecast_n_steps_from_window, get_last_close_window
    from src.portfolio.streamlit_utils import to_trading_days, build_ci_band, annualized_return_from_path

    tsla_df_full = load_price_df("TSLA")
    try:
        last_win = get_last_close_window(tsla_df_full["Close"], lookback)
        model, scaler = load_lstm_and_scaler()
        n_steps = to_trading_days(months)
        preds = forecast_n_steps_from_window(last_win, n_steps, model=model, scaler=scaler)

        # future index (business days)
        last_dt = tsla_df_full.index.max()
        future_index = pd.bdate_range(start=last_dt + pd.Timedelta(days=1), periods=n_steps, freq="B")
        forecast_series = pd.Series(preds, index=future_index, name="LSTM Forecast")

        # ±5% soft band
        band = build_ci_band(forecast_series, pct=0.05)

        # Add traces to existing fig
        fig.add_trace(go.Scatter(
            x=forecast_series.index, y=forecast_series.values,
            mode="lines", name="LSTM Forecast",
            line=dict(color="#00A676", width=3, shape="spline")
        ))
        fig.add_trace(go.Scatter(
            x=band["upper"].index, y=band["upper"].values,
            mode="lines", name="+5% band", line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=band["lower"].index, y=band["lower"].values,
            mode="lines", name="-5% band",
            fill="tonexty", fillcolor="rgba(244,211,94,0.25)",
            line=dict(width=0), showlegend=True
        ))

        # Forecast KPIs
        mu_annual = annualized_return_from_path(forecast_series.values)
        kf1, kf2, kf3 = st.columns(3)
        kf1.metric("Forecast Horizon", f"{months} months")
        kf2.metric("Annualized μ (TSLA)", f"{mu_annual*100:.2f}%")
        kf3.metric("Forecast Last Price", f"{forecast_series.iloc[-1]:,.2f}")

        # Optional: Save button
        save_col = st.columns(1)[0]
        if save_col.button(f"Save TSLA forecast CSV ({months}m)"):
            import os
            out_dir = "results/forecasts"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"lstm_future_forecast_{months}m.csv")
            forecast_series.to_csv(out_path, header=True)
            st.success(f"Saved: {out_path}")

    except Exception as e:
        st.warning(f"Forecast could not be generated: {e}")

else:
    # For SPY / BND, hide forecast controls and show a small note
    st.info("Forecast horizon is available for TSLA only. SPY and BND use historical returns in the portfolio model.")

# ✅ Render the figure once (with or without forecast overlay)
st.plotly_chart(fig, use_container_width=True)


# --- Sidebar placeholder (we'll add Finnhub news later) ---
with st.sidebar:
    st.header("News (Coming Next)")
    st.caption("TSLA / SPY / BND — live headlines here.")
    st.info("We’ll integrate Finnhub 'company-news' with a refresh button.")

# --- Footer ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© GMF Investments — Forecasts are for informational purposes only.")
