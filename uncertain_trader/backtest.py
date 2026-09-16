"""
Step 5 - Backtesting.

Replay each strategy's decisions through history:

    * A decision made at the close of day t is held during day t+1.
    * Changing position costs money: ``cost_bps`` basis points of the amount
      traded (10 bps = 0.10 %, a reasonable all-in figure for a retail account:
      spread + commission + slippage). This is what makes "refusing to trade"
      a real choice rather than a free one.
    * Nothing else is modelled - no leverage, no margin, no dividends. The goal
      is to compare decision rules fairly, not to reproduce a broker statement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_features
from .model import EnsemblePredictor
from .strategies import ConfidenceFilter, Strategy
from .metrics import summarise


def build_signals(prices: pd.DataFrame, model: EnsemblePredictor | None = None) -> pd.DataFrame:
    """Prices -> features -> model outputs. One row per day with everything a strategy needs."""
    model = model or EnsemblePredictor()
    feats = build_features(prices)
    preds = model.predict(feats)
    signals = feats.join(preds)
    signals["close"] = prices["Close"].reindex(signals.index)
    # Drop the warm-up period so every strategy is evaluated on identical days.
    return signals[signals["prediction"].notna()].copy()


def run_backtest(signals: pd.DataFrame, strategy: Strategy, cost_bps: float = 10.0) -> pd.DataFrame:
    """
    Return a daily frame for one strategy:

        decision      position chosen at the close of this day (for tomorrow)
        position      position actually held during this day
        market_ret    the asset's return this day
        traded        |change in position| executed at the start of this day
        cost          transaction cost paid this day
        strategy_ret  net return this day
        equity        growth of 1.0 invested on day one
        label         BUY / HOLD / SELL for the decision column
    """
    decision = strategy.positions(signals)
    held = decision.shift(1).fillna(0.0)
    traded = held.diff().abs()
    traded.iloc[0] = abs(held.iloc[0])
    cost = traded * cost_bps / 10_000.0
    strategy_ret = held * signals["ret"] - cost

    out = pd.DataFrame(
        {
            "decision": decision,
            "position": held,
            "market_ret": signals["ret"],
            "traded": traded,
            "cost": cost,
            "strategy_ret": strategy_ret,
            "equity": (1.0 + strategy_ret).cumprod(),
            "prediction": signals["prediction"],
            "confidence": signals["confidence"],
        },
        index=signals.index,
    )
    out["label"] = np.select([decision > 1e-9, decision < -1e-9], ["BUY", "SELL"], default="HOLD")
    return out


def run_all(
    signals: pd.DataFrame, strategies: list[Strategy], cost_bps: float = 10.0
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Backtest every strategy. Returns ({name: daily frame}, summary table)."""
    results = {s.name: run_backtest(signals, s, cost_bps) for s in strategies}
    table = pd.DataFrame({name: summarise(df) for name, df in results.items()}).T
    return results, table


def threshold_sweep(
    signals: pd.DataFrame,
    thresholds: np.ndarray | list[float] | None = None,
    cost_bps: float = 10.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    The core experiment: how do the numbers change as we demand more confidence
    before trading? threshold = 0.5 is 'Always Trade'; 1.0 is 'only when unanimous'.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 1.0001, 0.05), 2)
    rows = []
    for thr in thresholds:
        strat = ConfidenceFilter(threshold=float(thr), long_only=long_only)
        stats = summarise(run_backtest(signals, strat, cost_bps))
        stats["threshold"] = float(thr)
        rows.append(stats)
    return pd.DataFrame(rows).set_index("threshold")
