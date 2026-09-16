"""
Step 6 - Measuring the outcome.

"Which strategy made the most money?" is the wrong question - a strategy that
bets everything on every coin-flip will sometimes win big. We care about
risk-ADJUSTED results, plus how often the strategy actually acted.

    total_return   growth over the whole test, e.g. 0.35 = +35 %
    annual_return  the same, compounded to a yearly rate
    annual_vol     how bumpy the ride was (std of daily returns * sqrt(252))
    sharpe         annual_return / annual_vol  - return per unit of risk (the headline metric)
    max_drawdown   worst peak-to-trough loss, e.g. -0.20 = lost 20 % from a high
    n_trades       how many times the position changed
    pct_in_market  fraction of days holding a position (1 - how often it refused)
    accuracy       when it DID trade, how often was the direction right?
    total_cost     how much was paid in transaction costs (as a fraction of capital)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def summarise(bt: pd.DataFrame) -> pd.Series:
    r = bt["strategy_ret"]
    n_years = len(r) / TRADING_DAYS
    equity_end = float(bt["equity"].iloc[-1])
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ann_ret = equity_end ** (1.0 / n_years) - 1.0 if n_years > 0 and equity_end > 0 else np.nan

    in_market = bt["position"].abs() > 1e-9
    correct = np.sign(bt["position"]) == np.sign(bt["market_ret"])
    accuracy = float(correct[in_market].mean()) if in_market.any() else np.nan

    return pd.Series(
        {
            "total_return": equity_end - 1.0,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "max_drawdown": max_drawdown(bt["equity"]),
            "n_trades": int((bt["traded"] > 1e-9).sum()),
            "pct_in_market": float(in_market.mean()),
            "accuracy": accuracy,
            "total_cost": float(bt["cost"].sum()),
        }
    )


def calibration_table(signals: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Is a "70 % confident" prediction actually right 70 % of the time?

    Groups days by the model's stated confidence and measures the real hit-rate
    in each group. A well-calibrated model lies on the diagonal. This is the
    single most important diagnostic for the whole project: confidence filtering
    can only work if confidence *means* something.
    """
    s = signals.dropna(subset=["prediction", "target"])
    s = s[s["prediction"] != 0]
    correct = np.sign(s["prediction"]) == np.sign(s["target"])
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = pd.cut(s["confidence"], bins=edges, include_lowest=True)
    out = pd.DataFrame({"bin": bins, "correct": correct, "confidence": s["confidence"]})
    grouped = out.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "stated_confidence": grouped["confidence"].mean(),
            "actual_accuracy": grouped["correct"].mean(),
            "n_days": grouped["correct"].size(),
        }
    )


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-friendly version of the summary table (strings, for printing)."""
    fmt = table.copy()
    for c in ["total_return", "annual_return", "annual_vol", "max_drawdown", "pct_in_market", "accuracy", "total_cost"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:+.1%}" if c in ("total_return", "annual_return", "max_drawdown") else f"{v:.1%}")
    fmt["sharpe"] = fmt["sharpe"].map(lambda v: f"{v:.2f}")
    fmt["n_trades"] = fmt["n_trades"].astype(int)
    return fmt.rename(
        columns={
            "total_return": "Return",
            "annual_return": "Return / yr",
            "annual_vol": "Risk (vol)",
            "sharpe": "Sharpe",
            "max_drawdown": "Max DD",
            "n_trades": "Trades",
            "pct_in_market": "In market",
            "accuracy": "Accuracy",
            "total_cost": "Costs",
        }
    )
