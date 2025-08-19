# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import os
import numpy as np
import plotly.graph_objects as go
from src.news_client import fetch_news_finnhub



from src.portfolio_utils import compute_max_return_weights
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
        months = st.selectbox("Forecast Horizon (months)", [3, 6, 9, 12, 18, 24], index=3, key="fc_months")
    with fc_cols[1]:
        lookback = st.number_input("LSTM Lookback (days)", min_value=30, max_value=180, value=60, step=5, key="fc_lookback")
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



with st.sidebar:
    st.markdown("### 🗞️ Live News")

    # Symbols you care about
    news_symbols = st.multiselect(
        "Tickers",
        options=["TSLA", "SPY", "BND"],
        default=["TSLA", "SPY", "BND"],
    )

    days_back = st.slider("Lookback (days)", min_value=1, max_value=30, value=7, step=1)
    max_items = st.number_input("Max per feed", 3, 20, 8, step=1, help="Limit how many headlines to show")
    refresh = st.button("Refresh News")

    # Load API key from Streamlit secrets
    finnhub_key = st.secrets.get("FINNHUB_API_KEY")

    # Cache the fetch to avoid hitting API too often
    @st.cache_data(ttl=1800, show_spinner=False)
    def _cached_fetch_news(k: str, syms: tuple, back: int):
        return fetch_news_finnhub(k, list(syms), days_back=back)

    news_items = []
    if not finnhub_key:
        st.info("Add FINNHUB_API_KEY to .streamlit/secrets.toml to enable news.")
    else:
        if refresh:
            _cached_fetch_news.clear()
        if news_symbols:
            news_items = _cached_fetch_news(finnhub_key, tuple(news_symbols), days_back)

    # ---------- helpers (time-ago + snippet) ----------
    import datetime as _dt

    def _time_ago(unix_ts: int) -> str:
        if not unix_ts:
            return ""
        try:
            dt = _dt.datetime.utcfromtimestamp(unix_ts)
            delta = _dt.datetime.utcnow() - dt
            s = int(delta.total_seconds())
            if s < 60:   return f"{s}s ago"
            m = s // 60
            if m < 60:   return f"{m}m ago"
            h = m // 60
            if h < 24:   return f"{h}h ago"
            d = h // 24
            return f"{d}d ago"
        except Exception:
            return ""

    def _snippet(text: str, n: int = 180) -> str:
        if not text:
            return ""
        text = text.strip().replace("\n", " ")
        return text if len(text) <= n else text[: n - 1] + "…"

    # ---------- render compact cards ----------
    if news_items:
        shown = 0
        for item in news_items:
            if shown >= max_items:
                break

            sym   = item.get("symbol", "")
            sent  = item.get("sentiment", "→")          # ↑ / ↓ / →
            src   = (item.get("source") or "").upper()
            hdl   = item.get("headline") or ""
            url   = item.get("url") or "#"
            summ  = item.get("summary") or ""
            img   = item.get("image") or ""
            when  = _time_ago(item.get("time"))

            # sentiment chip (minimal emoji)
            chip = {"↑": "✅", "↓": "🔻", "→": "➜"}.get(sent, "➜")

            with st.container():
                st.markdown(f"**{chip} [{sym}] [{hdl}]({url})**")
                meta = " • ".join(x for x in [src, when] if x)
                if meta:
                    st.caption(meta)
                if summ:
                    st.write(_snippet(summ, 180))
                if img:
                    st.image(img, use_container_width=True, clamp=True)
                st.divider()
            shown += 1
    else:
        st.write("No news to display yet.")




def _queue_weights(wdict: dict[str, float]):
    """Queue preset weights (0..1), then rerun so we can apply them before sliders render."""
    st.session_state._queued_weights = {
        "TSLA": float(wdict.get("TSLA", 0.0)),
        "SPY":  float(wdict.get("SPY",  0.0)),
        "BND":  float(wdict.get("BND",  0.0)),
    }
    st.rerun()


# Apply queued preset weights BEFORE rendering sliders (prevents Streamlit API exception)
if "_queued_weights" in st.session_state:
    q = st.session_state.pop("_queued_weights")  # remove & apply once
    st.session_state.w_tsla = int(round(q["TSLA"] * 100))
    st.session_state.w_spy  = int(round(q["SPY"]  * 100))
    st.session_state.w_bnd  = int(round(q["BND"]  * 100))


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



from src.portfolio.streamlit_utils import load_presets_json
presets_json = load_presets_json()



# --- Preset loader + slider applier ---
from src.portfolio.streamlit_utils import load_presets_json
presets_json = load_presets_json()  # reads results/optimization/presets.json

def _apply_weights_dict(wdict: dict[str, float]):
    """
    Accepts weights in decimals (0..1). Pushes to sliders (percent ints).
    Forces a rerun so UI reflects the change immediately.
    """
    st.session_state.w_tsla = int(round(wdict.get("TSLA", 0.0) * 100))
    st.session_state.w_spy  = int(round(wdict.get("SPY",  0.0) * 100))
    st.session_state.w_bnd  = int(round(wdict.get("BND",  0.0) * 100))
    st.rerun()  # <- important to refresh the slider positions



def _init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

# neutral defaults (any values are fine; presets will override)
_init("w_tsla", 33)
_init("w_spy",  33)
_init("w_bnd",  34)

def _on_tsla_change():
    # Split the remainder equally between SPY and BND
    rem = max(0, 100 - int(st.session_state.w_tsla))
    st.session_state.w_spy = rem // 2
    st.session_state.w_bnd = rem - st.session_state.w_spy  # absorbs rounding

def _on_spy_change():
    # Keep TSLA fixed; SPY can't exceed (100 - TSLA); BND gets the remainder
    max_spy = max(0, 100 - int(st.session_state.w_tsla))
    st.session_state.w_spy = min(int(st.session_state.w_spy), max_spy)
    st.session_state.w_bnd = max(0, 100 - int(st.session_state.w_tsla) - int(st.session_state.w_spy))

def _on_bnd_change():
    # Keep TSLA fixed; BND can't exceed (100 - TSLA); SPY gets the remainder
    max_bnd = max(0, 100 - int(st.session_state.w_tsla))
    st.session_state.w_bnd = min(int(st.session_state.w_bnd), max_bnd)
    st.session_state.w_spy = max(0, 100 - int(st.session_state.w_tsla) - int(st.session_state.w_bnd))

c1, c2, c3 = st.columns(3)
with c1:
    st.session_state.w_tsla = st.slider(
        "TSLA %", 0, 100, int(st.session_state.w_tsla), step=1, on_change=_on_tsla_change
    )
with c2:
    st.session_state.w_spy = st.slider(
        "SPY %", 0, 100, int(st.session_state.w_spy), step=1, on_change=_on_spy_change
    )
with c3:
    st.session_state.w_bnd = st.slider(
        "BND %", 0, 100, int(st.session_state.w_bnd), step=1, on_change=_on_bnd_change
    )

# Build weights for downstream calc
w = np.array([st.session_state.w_tsla, st.session_state.w_spy, st.session_state.w_bnd], dtype=float) / 100.0
st.caption(f"Remaining allocation: **{max(0, 100 - int(round(w.sum()*100))):.0f}%**")

# If you still offer 'Auto-normalize', apply it AFTER the hard-cap logic:
if locals().get('auto_norm', False):
    from src.portfolio.streamlit_utils import normalize_weights
    w = normalize_weights(w)

# Presets & optimize buttons
from src.portfolio.streamlit_utils import load_presets_json
presets_json = load_presets_json()

b1, b2, b3, b4, b5 = st.columns(5)

with b1:
    if st.button("60 / 40"):
        if presets_json and "sixty_forty" in presets_json["weights"]:
            _queue_weights(presets_json["weights"]["sixty_forty"])
        else:
            _queue_weights({"TSLA": 0.0, "SPY": 0.60, "BND": 0.40})

with b2:
    if st.button("Max Sharpe"):
        if presets_json and "max_sharpe" in presets_json["weights"]:
            _queue_weights(presets_json["weights"]["max_sharpe"])
        else:
            st.warning("No Max Sharpe preset in presets.json")

with b3:
    if st.button("Min Vol"):
        if presets_json and "min_vol" in presets_json["weights"]:
            _queue_weights(presets_json["weights"]["min_vol"])
        else:
            st.warning("No Min Vol preset in presets.json")

# assumes: tickers = ["TSLA","SPY","BND"], mu and cov already built above
from src.portfolio.streamlit_utils import load_daily_returns

with b4:
    if st.button("Max Return"):
        # Use notebook preset if present
        if presets_json and "max_return" in presets_json.get("weights", {}):
            _apply_weights_dict(presets_json["weights"]["max_return"])
        else:
            # Deterministic fallback: 100% to asset with highest expected return μ
            best = mu.idxmax()  # "TSLA" or "SPY" or "BND"
            wdict = {t: 0.0 for t in tickers}
            wdict[best] = 1.0
            _apply_weights_dict(wdict)

with b5:
    if st.button("Risk Parity"):
        if presets_json and "risk_parity" in presets_json.get("weights", {}):
            _apply_weights_dict(presets_json["weights"]["risk_parity"])
        else:
            # Fallback: inverse-volatility weights on historical daily returns
            rets_df = load_daily_returns(tuple(tickers))
            inv_vol = 1.0 / rets_df.std()
            w_iv = (inv_vol / inv_vol.sum()).astype(float)
            _apply_weights_dict({t: float(w_iv[t]) for t in tickers})



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




# =======================
# 📈 Backtest: Strategy vs 60/40
# =======================
st.divider()
st.markdown("## 📈 Backtest: Strategy vs 60/40")

from src.portfolio.streamlit_utils import load_daily_returns

# 1) Daily returns and filter to selected range
rets = load_daily_returns(("TSLA","SPY","BND"))
rets_rng = rets.loc[str(start_d):str(end_d)].copy()
if rets_rng.empty:
    st.warning("No data in the selected range. Adjust the date filter above.")
else:
    # 2) Strategy (current weights) vs benchmark (60/40 SPY/BND)
    w_vec = w  # already a numpy array of len 3 in decimal (TSLA, SPY, BND)
    bench_w = np.array([0.0, 0.60, 0.40])

    strat_ret = (rets_rng * w_vec).sum(axis=1)
    bench_ret = (rets_rng * bench_w).sum(axis=1)

    # 3) Cumulative value (start=1.0)
    cum_s = (1 + strat_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    # 4) Annualized Sharpe on the window (rf = 1%)
    def _ann_sharpe(r):
        mu_d = r.mean()
        sd_d = r.std()
        if sd_d == 0 or np.isnan(sd_d):
            return np.nan
        mu_a = mu_d * 252.0
        sd_a = sd_d * np.sqrt(252.0)
        return (mu_a - 0.01) / sd_a

    s_total = float(cum_s.iloc[-1] - 1)
    b_total = float(cum_b.iloc[-1] - 1)
    s_sharpe = _ann_sharpe(strat_ret)
    b_sharpe = _ann_sharpe(bench_ret)

    # 5) Plot
    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(x=cum_s.index, y=cum_s.values, name="Strategy", mode="lines"))
    bt_fig.add_trace(go.Scatter(x=cum_b.index, y=cum_b.values, name="60/40 (SPY/BND)", mode="lines"))
    bt_fig.update_layout(
        height=420, template="plotly_white",
        yaxis_title="Cumulative Value (start = 1.00)"
    )
    st.plotly_chart(bt_fig, use_container_width=True)

    # 6) Metrics
    c1, c2 = st.columns(2)
    c1.metric("Strategy — Total Return", f"{s_total*100:.2f}%")
    c1.metric("Strategy — Sharpe (rf=1%)", f"{s_sharpe:.2f}")
    c2.metric("Benchmark 60/40 — Total Return", f"{b_total*100:.2f}%")
    c2.metric("Benchmark 60/40 — Sharpe (rf=1%)", f"{b_sharpe:.2f}")


# --- Dollar growth based on the investment amount from the Playground ---
# Reuse your existing 'invest_amount' number_input (from VaR panel). If it's defined elsewhere, bring it above or repeat it here.
ending_strategy = float(cum_s.iloc[-1]) * float(invest_amount)
ending_benchmark = float(cum_b.iloc[-1]) * float(invest_amount)

pnl_strategy = ending_strategy - float(invest_amount)
pnl_benchmark = ending_benchmark - float(invest_amount)

# Dollar-value plot (optional but nice)
bt_dollar = go.Figure()
bt_dollar.add_trace(go.Scatter(
    x=cum_s.index, y=cum_s.values * float(invest_amount),
    name="Strategy ($)", mode="lines"
))
bt_dollar.add_trace(go.Scatter(
    x=cum_b.index, y=cum_b.values * float(invest_amount),
    name="60/40 ($)", mode="lines"
))
bt_dollar.update_layout(
    height=420, template="plotly_white",
    yaxis_title=f"Portfolio Value ($, start = ${int(invest_amount):,})"
)
st.plotly_chart(bt_dollar, use_container_width=True)

# Dollar KPIs
dc1, dc2 = st.columns(2)
dc1.metric("Strategy — Final Value", f"${ending_strategy:,.0f}", f"{pnl_strategy:,.0f}")
dc2.metric("Benchmark — Final Value", f"${ending_benchmark:,.0f}", f"{pnl_benchmark:,.0f}")



# =======================
# 🧭 Efficient Frontier
# =======================
st.divider()
st.markdown("## 🧭 Efficient Frontier")

from src.portfolio.streamlit_utils import load_presets_json
_p = load_presets_json()  # loads results/optimization/presets.json if present

rng = np.random.default_rng(123)
N = 12000
W = rng.random((N, 3))
W = W / W.sum(axis=1, keepdims=True)

# Monte Carlo portfolios (annualized ER/Vol using your mu, cov)
rets_mc = W @ mu.values
vols_mc = np.sqrt(np.sum(W @ cov.values * W, axis=1))
sharpe_mc = (rets_mc - 0.01) / vols_mc  # rf = 1%

# Build customdata for tooltips: [TSLA_w, SPY_w, BND_w, Sharpe]
custom = np.column_stack([W[:, 0], W[:, 1], W[:, 2], sharpe_mc])

ef_fig = go.Figure()
ef_fig.add_trace(go.Scatter(
    x=vols_mc, y=rets_mc,
    mode="markers", name="Portfolios",
    opacity=0.35, marker=dict(size=5),
    customdata=custom,
    hovertemplate=(
        "Vol (ann): %{x:.2%}<br>"
        "ER (ann): %{y:.2%}<br>"
        "Sharpe: %{customdata[3]:.2f}<br>"
        "TSLA: %{customdata[0]:.2%} | "
        "SPY: %{customdata[1]:.2%} | "
        "BND: %{customdata[2]:.2%}<extra></extra>"
    )
))

# Current weights marker (uses w, mu, cov defined earlier)
cur_ret = float(w @ mu.values)
cur_vol = float(np.sqrt(w @ cov.values @ w))
ef_fig.add_trace(go.Scatter(
    x=[cur_vol], y=[cur_ret],
    mode="markers+text", name="Current",
    marker=dict(size=12, color="#00A676"),
    text=["Current"], textposition="top center",
    hovertemplate=(
        f"Vol (ann): {cur_vol:.2%}<br>"
        f"ER (ann): {cur_ret:.2%}<br>"
        f"Weights → TSLA: {w[0]:.2%}, SPY: {w[1]:.2%}, BND: {w[2]:.2%}"
        "<extra></extra>"
    )
))

# Notebook presets (Max Sharpe / Min Vol) if available
if _p and "weights" in _p:
    for label in ["max_sharpe", "min_vol"]:
        if label in _p["weights"]:
            # Assumes mu.index order matches ["TSLA","SPY","BND"]
            ww = np.array([_p["weights"][label][t] for t in mu.index], dtype=float)
            r = float(ww @ mu.values)
            v = float(np.sqrt(ww @ cov.values @ ww))
            ef_fig.add_trace(go.Scatter(
                x=[v], y=[r],
                mode="markers+text", name=label.replace("_"," ").title(),
                marker=dict(size=12),
                text=[label.replace("_"," ").title()], textposition="bottom center",
                hovertemplate=(
                    f"Vol (ann): {v:.2%}<br>"
                    f"ER (ann): {r:.2%}<br>"
                    f"Weights → TSLA: {ww[0]:.2%}, SPY: {ww[1]:.2%}, BND: {ww[2]:.2%}"
                    "<extra></extra>"
                )
            ))

ef_fig.update_layout(
    height=420, template="plotly_white",
    xaxis_title="Volatility (σ, annualized)",
    yaxis_title="Expected Return (annualized)"
)
st.plotly_chart(ef_fig, use_container_width=True)



st.divider()

# =========================
# 🗞️ Market News (center)
# =========================
st.markdown("## 🗞️ Market News")

# Imports
from collections import defaultdict
import datetime as _dt

from src.news_client import fetch_news_finnhub
from src.summarizer import summarize_with_groq         # returns SummaryResult
from src.article_fetcher import fetch_article_text, choose_text_for_summary

# --- Read settings coming from sidebar controls ---
news_symbols = st.session_state.get("news_symbols", ["TSLA", "SPY", "BND"])
days_back    = st.session_state.get("news_days_back", 7)
use_llm      = st.session_state.get("use_llm_news", True)
refresh_news = st.session_state.get("refresh_news", False)

# --- Helpers ---
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_fetch_news(k: str, syms: tuple, back: int):
    return fetch_news_finnhub(k, list(syms), days_back=back)

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_summarize(text: str, sym: str, enabled: bool):
    if not enabled:
        t = (text or "").strip()
        t = (t[:500] + "…") if len(t) > 500 else t
        return {"summary": t, "sentiment": "Neutral"}
    res = summarize_with_groq(text, sym)  # SummaryResult
    return {"summary": res.summary, "sentiment": res.sentiment}

def _time_ago(unix_ts: int) -> str:
    if not unix_ts:
        return ""
    dt = _dt.datetime.utcfromtimestamp(unix_ts)
    delta = _dt.datetime.utcnow() - dt
    s = int(delta.total_seconds())
    if s < 60: return f"{s}s ago"
    m = s // 60
    if m < 60: return f"{m}m ago"
    h = m // 60
    if h < 24: return f"{h}h ago"
    return f"{h // 24}d ago"

def _sent_prefix(sentiment: str) -> str:
    s = (sentiment or "").lower()
    if s.startswith(("good", "pos")): return "+ve"
    if s.startswith(("bad", "neg")):  return "-ve"
    return "N"

# --- Fetch + render ---
finnhub_key = st.secrets.get("FINNHUB_API_KEY")
if not finnhub_key:
    st.info("Add FINNHUB_API_KEY to `.streamlit/secrets.toml` to enable live news.")
else:
    if refresh_news:
        _cached_fetch_news.clear()
        _cached_summarize.clear()

    if not news_symbols:
        st.warning("Select at least one ticker in the sidebar to view news.")
    else:
        items = _cached_fetch_news(finnhub_key, tuple(news_symbols), days_back) or []
        by_sym = defaultdict(list)
        for it in items:
            by_sym[it.get("symbol", "")].append(it)

        if not by_sym:
            st.write("No news to display yet.")
        else:
            tabs = st.tabs(list(by_sym.keys()))
            for sym, tab in zip(by_sym.keys(), tabs):
                with tab:
                    for it in by_sym[sym]:
                        hdl     = it.get("headline") or ""
                        url     = it.get("url") or "#"
                        fin_sum = it.get("summary") or ""    # short publisher blurb
                        img     = it.get("image") or ""
                        src     = (it.get("source") or "").upper()
                        when    = _time_ago(it.get("time"))

                        # NEW: try to fetch full article body (free)
                        article_text = fetch_article_text(url)

                        # Pick best text to summarize: article > finnhub summary > headline
                        news_text = choose_text_for_summary(hdl, fin_sum, article_text)

                        # Summarize & classify impact (cached)
                        sres = _cached_summarize(news_text, sym, use_llm)
                        prefix = _sent_prefix(sres["sentiment"])

                        # Card layout
                        col_img, col_txt = st.columns([0.26, 0.74])
                        with col_img:
                            if img:
                                st.image(img, use_container_width=True, clamp=True)
                        with col_txt:
                            st.markdown(f"**{prefix} [{sym}] [{hdl}]({url})**")
                            st.caption(" • ".join([s for s in [src, when] if s]))
                            # sres["summary"] now contains 3–5 bullets + takeaway (from Groq)
                            if sres["summary"]:
                                st.markdown(sres["summary"])
                        st.markdown("---")

st.caption("News & summaries are informational only and not investment advice.")



# --- Footer ---
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© GMF Investments.")







