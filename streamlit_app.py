# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import os
import numpy as np
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



st.divider()
st.markdown("## 🎛️ Portfolio Playground")

# --- TSLA μ (annualized) source ---
# If we computed a forecast earlier, store μ in session for reuse
if "tsla_mu_annual" not in st.session_state:
    st.session_state.tsla_mu_annual = None

# If TSLA forecast was just computed above, drop it into session:
# (You already compute mu_annual in the TSLA block. Add this there:)
#   st.session_state.tsla_mu_annual = mu_annual

from src.portfolio.streamlit_utils import (
    build_mu_sigma, portfolio_metrics, normalize_weights, normal_var, load_daily_returns
)

mu, cov = build_mu_sigma(st.session_state.tsla_mu_annual)  # μ uses LSTM for TSLA if available
tickers = list(mu.index)  # ["TSLA","SPY","BND"]

# --- Controls: weights + presets/optimize ---
c1, c2, c3, c4 = st.columns([0.25, 0.25, 0.25, 0.25])

with c1:
    w_tsla = st.slider("TSLA %", 0, 100, 20, step=1)
with c2:
    w_spy = st.slider("SPY %", 0, 100, 50, step=1)
with c3:
    w_bnd = st.slider("BND %", 0, 100, 30, step=1)
with c4:
    auto_norm = st.checkbox("Auto-normalize", value=True)

w = np.array([w_tsla, w_spy, w_bnd], dtype=float) / 100.0
if auto_norm:
    w = normalize_weights(w)

# Presets & optimize buttons
b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    if st.button("60 / 40"):
        w = np.array([0.0, 0.60, 0.40])
with b2:
    if st.button("Max Sharpe"):
        # simple Monte Carlo search (fast & good enough)
        rng = np.random.default_rng(42)
        W = rng.random((8000, 3)); W = W / W.sum(axis=1, keepdims=True)
        rets = W @ mu.values
        vols = np.sqrt(np.sum(W @ cov.values * W, axis=1))
        sharpe = (rets - 0.01) / vols
        w = W[np.argmax(sharpe)]
with b3:
    if st.button("Min Vol"):
        rng = np.random.default_rng(7)
        W = rng.random((8000, 3)); W = W / W.sum(axis=1, keepdims=True)
        vols = np.sqrt(np.sum(W @ cov.values * W, axis=1))
        w = W[np.argmin(vols)]
with b4:
    if st.button("Max Return"):
        rng = np.random.default_rng(21)
        W = rng.random((8000, 3)); W = W / W.sum(axis=1, keepdims=True)
        rets = W @ mu.values
        w = W[np.argmax(rets)]
with b5:
    if st.button("Risk Parity"):
        # inverse vol heuristic on historical daily returns
        rets_df = load_daily_returns(tuple(tickers))
        iv = 1.0 / rets_df.std().values
        w = iv / iv.sum()

# Show current weights
st.caption(f"Current Weights → TSLA: **{w[0]*100:.2f}%**, SPY: **{w[1]*100:.2f}%**, BND: **{w[2]*100:.2f}%**")

# --- Metrics & VaR ---
rf = 0.01
ann_ret, ann_vol, sharpe = portfolio_metrics(w, mu, cov, rf=rf)

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Expected Annual Return", f"{ann_ret*100:.2f}%")
mc2.metric("Annual Volatility", f"{ann_vol*100:.2f}%")
mc3.metric("Sharpe (rf=1%)", f"{sharpe:.2f}")

amt_col, hor_col, alpha_col = st.columns([0.4, 0.3, 0.3])
with amt_col:
    invest_amount = st.number_input("Investment Amount ($)", min_value=1000, value=10000, step=500)
with hor_col:
    var_months = st.selectbox("VaR Horizon (months)", [3, 6, 9, 12, 18, 24], index=1)
with alpha_col:
    alpha = st.selectbox("VaR Confidence", ["95%", "99%"], index=0)
alpha_val = 0.95 if alpha == "95%" else 0.99

var_pct, var_usd = normal_var(invest_amount, ann_ret, ann_vol, months=var_months, alpha=alpha_val)
mc4.metric(f"VaR ({alpha} / {var_months}m)", f"{var_pct*100:.2f}% ≈ ${var_usd:,.0f}")

# --- Downloads ---
dl1, dl2 = st.columns(2)
with dl1:
    if st.button("⬇️ Download Weights (CSV)"):
        import io
        buf = io.StringIO()
        pd.Series(w, index=tickers, name="weight").to_csv(buf)
        st.download_button("Save Weights CSV", data=buf.getvalue(), file_name="weights.csv", mime="text/csv")
with dl2:
    if st.button("⬇️ Download Metrics (JSON)"):
        import json
        payload = {
            "weights": dict(zip(tickers, [float(x) for x in w])),
            "expected_return_annual": float(ann_ret),
            "volatility_annual": float(ann_vol),
            "sharpe_rf_1pct": float(sharpe),
            "VaR_horizon_months": int(var_months),
            "VaR_confidence": alpha,
            "VaR_pct": float(var_pct),
            "VaR_usd": float(var_usd),
        }
        st.download_button(
            "Save Metrics JSON",
            data=json.dumps(payload, indent=2),
            file_name="portfolio_metrics.json",
            mime="application/json"
        )




# --- Footer ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© GMF Investments.")
