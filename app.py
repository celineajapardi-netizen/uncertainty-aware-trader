"""
Streamlit interface for the uncertainty-aware trading simulator.

    streamlit run app.py
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

from uncertain_trader import (  # noqa: E402
    EnsemblePredictor,
    calibration_table,
    default_strategies,
    load_prices,
    run_all,
    simulate_prices,
    threshold_sweep,
)
from uncertain_trader.backtest import build_signals  # noqa: E402
from uncertain_trader.metrics import format_table  # noqa: E402
from uncertain_trader.plots import (  # noqa: E402
    plot_calibration,
    plot_confidence_hist,
    plot_decisions,
    plot_equity,
    plot_threshold_sweep,
)

st.set_page_config(page_title="When should an algorithm trade?", page_icon="🎯", layout="wide")

st.title("When should an algorithm trade?")
st.caption(
    "Investigating the role of prediction confidence in quantitative trading decisions. "
    "The model predicts tomorrow's return *and* says how confident it is; "
    "the strategies differ only in what they do with that confidence."
)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Market")
    source = st.radio("Data source", ["Simulated", "Real (Yahoo Finance)"], horizontal=True)
    if source == "Simulated":
        seed = st.number_input("Random seed", 0, 999, 0)
        signal_strength = st.slider(
            "Signal strength", 0.0, 0.8, 0.30, 0.05,
            help="0 = pure random walk (nothing is predictable). 0.3 ≈ model right ~57 % of days.",
        )
        n_days = st.slider("Days", 1000, 5000, 2500, 250)
        ticker, start = None, None
    else:
        ticker = st.text_input("Ticker", "SPY").strip().upper()
        start = st.date_input("Start", pd.Timestamp("2010-01-01")).isoformat()
        seed, signal_strength, n_days = None, None, None

    st.header("Model")
    train_window = st.slider("Training window (days)", 250, 1000, 500, 50)
    n_models = st.slider("Models in ensemble", 10, 100, 50, 10)

    st.header("Strategies")
    threshold = st.slider("Confidence threshold (Strategy B)", 0.50, 1.00, 0.70, 0.05, format="%.2f")
    cost_bps = st.slider("Transaction cost (bps per unit traded)", 0.0, 25.0, 5.0, 1.0)
    long_only = st.checkbox("Long only (no short selling)", value=False)


# ----------------------------------------------------------------------------- data + model
@st.cache_data(show_spinner="Loading prices…")
def get_prices(source, ticker, start, seed, signal_strength, n_days):
    if source == "Simulated":
        return simulate_prices(n_days=n_days, seed=seed, signal_strength=signal_strength)
    return load_prices(ticker, start=start)


@st.cache_data(show_spinner="Training the walk-forward ensemble…")
def get_signals(prices, train_window, n_models):
    model = EnsemblePredictor(train_window=train_window, n_models=n_models)
    return build_signals(prices, model)


try:
    prices = get_prices(source, ticker, start, seed, signal_strength, n_days)
except Exception as exc:  # bad ticker, no internet, …
    st.error(f"Could not load data: {exc}")
    st.stop()

signals = get_signals(prices, train_window, n_models)
if len(signals) < 250:
    st.error("Not enough history after the warm-up period. Use an earlier start date or a shorter training window.")
    st.stop()

strategies = default_strategies(threshold=threshold, long_only=long_only)
results, table = run_all(signals, strategies, cost_bps=cost_bps)
cal = calibration_table(signals)

market_name = ticker if source != "Simulated" else f"Simulated market (seed {seed}, signal {signal_strength:.2f})"
st.subheader(market_name)
st.caption(
    f"Test period {signals.index[0].date()} → {signals.index[-1].date()} "
    f"({len(signals):,} trading days, after a {train_window}-day warm-up). "
    f"Model accuracy on direction: {(signals['prediction'].gt(0) == signals['target'].gt(0)).mean():.1%}."
)

# ----------------------------------------------------------------------------- today's decision
latest = signals.iloc[-1]
with st.container(border=True):
    st.markdown("**Latest signal** — what each strategy would do at the last close")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price trend (20d)", f"{latest['ret_20']:+.1%}")
    c2.metric("Volatility (20d, ann.)", f"{latest['vol_20'] * 252 ** 0.5:.0%}", help="Realised volatility, annualised")
    c3.metric("Volume vs normal", f"{latest['volume_z']:+.1f} σ")
    c4.metric("Prediction (tomorrow)", f"{latest['prediction']:+.2%}")
    c5.metric("Model confidence", f"{latest['confidence']:.0%}")
    emoji = {"BUY": "🟢 BUY", "HOLD": "⚪ HOLD", "SELL": "🔴 SELL"}
    cols = st.columns(len(results))
    for col, (name, bt) in zip(cols, results.items()):
        last = bt.iloc[-1]
        col.markdown(f"**{name}**  \n{emoji[last['label']]} &nbsp; size {abs(last['decision']):.0%}")

# ----------------------------------------------------------------------------- results table
st.subheader("Strategy comparison")
st.dataframe(format_table(table), width="stretch")
best = table["sharpe"].idxmax()
st.markdown(
    f"Highest risk-adjusted return (Sharpe): **{best}** at {table.loc[best, 'sharpe']:.2f}, "
    f"trading on {table.loc[best, 'pct_in_market']:.0%} of days."
)

# ----------------------------------------------------------------------------- charts
st.pyplot(plot_equity(results, "Growth of 1.0 invested"), width="stretch")

st.subheader("The core experiment: demand more confidence, trade less often")
st.caption(
    "Strategy B re-run at every threshold from 50 % (= Always Trade) to 100 % (only when every model agrees)."
)
with st.spinner("Sweeping thresholds…"):
    sweep = threshold_sweep(signals, cost_bps=cost_bps, long_only=long_only)
st.pyplot(plot_threshold_sweep(sweep, table.loc["Always Trade", "sharpe"]), width="stretch")

left, right = st.columns(2)
with left:
    st.pyplot(plot_calibration(cal), width="stretch")
    st.caption(
        "If the dots sit on the dashed line, confidence is *honest*: a 'sure' prediction really is right more often. "
        "If the line is flat, confidence carries no information — and filtering on it cannot help."
    )
with right:
    st.pyplot(plot_confidence_hist(signals), width="stretch")

st.subheader("Decision timeline")
c1, c2 = st.columns([2, 1])
chosen = c1.selectbox("Strategy", [s.name for s in strategies if s.name != "Buy & Hold"], index=2)
window = c2.slider("Show last N days", 60, min(1000, len(signals)), 250, 10)
st.pyplot(plot_decisions(results[chosen], signals["close"], last_n=window, title=f"{chosen}: last {window} days"), width="stretch")

with st.expander("Daily data (decisions, predictions, confidence)"):
    show = results[chosen][["label", "decision", "prediction", "confidence", "market_ret", "strategy_ret", "equity"]].copy()
    st.dataframe(show.tail(500).iloc[::-1], width="stretch")

st.divider()
st.caption(
    "Educational project using public / simulated data. Nothing here is investment advice, and past "
    "backtest performance says little about the future — that is rather the point."
)
