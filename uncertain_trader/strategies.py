"""
Step 4 - Decision rules.

A strategy looks at today's (prediction, confidence, volatility) and returns a
*position* for tomorrow, as a fraction of capital in [-1, +1]:

    +1.0  fully long   (BUY)
     0.0  flat         (HOLD / refuse to trade)
    -1.0  fully short  (SELL)

Fractions in between mean "trade smaller". The strategies are deliberately
simple so that the only thing that differs between them is HOW they use
uncertainty:

    BuyAndHold        ignores the model entirely                       (benchmark)
    AlwaysTrade       acts on every prediction, however unsure         (Strategy A)
    ConfidenceFilter  acts only when confidence >= threshold           (Strategy B)
    RiskAware         scales the bet by confidence AND by volatility   (Strategy C)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    name: str = "Base"
    long_only: bool = False

    def positions(self, signals: pd.DataFrame) -> pd.Series:
        """
        ``signals`` has columns prediction, confidence, vol_20 (one row per day).
        Return a Series of positions in [-1, 1] aligned to the same index.
        """
        pos = self._raw_positions(signals)
        pos = pos.clip(lower=0.0 if self.long_only else -1.0, upper=1.0)
        # Warm-up rows (no prediction yet) -> stay flat.
        return pos.where(signals["prediction"].notna(), 0.0).fillna(0.0)

    def _raw_positions(self, s: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


@dataclass
class BuyAndHold(Strategy):
    name: str = "Buy & Hold"

    def _raw_positions(self, s: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=s.index)

    def positions(self, signals: pd.DataFrame) -> pd.Series:
        # Benchmark is invested from the first day the others could trade,
        # so all strategies are compared over the same period.
        pos = pd.Series(1.0, index=signals.index)
        return pos.where(signals["prediction"].notna(), 0.0)


@dataclass
class AlwaysTrade(Strategy):
    """Strategy A: prediction > 0 -> BUY, prediction < 0 -> SELL. Confidence ignored."""

    name: str = "Always Trade"

    def _raw_positions(self, s: pd.DataFrame) -> pd.Series:
        return np.sign(s["prediction"])


@dataclass
class ConfidenceFilter(Strategy):
    """Strategy B: same as A, but HOLD unless confidence >= threshold."""

    name: str = "Confidence Filter"
    threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.name == "Confidence Filter":
            self.name = f"Confidence Filter ({self.threshold:.0%})"

    def _raw_positions(self, s: pd.DataFrame) -> pd.Series:
        confident = s["confidence"] >= self.threshold
        return np.sign(s["prediction"]).where(confident, 0.0)


@dataclass
class RiskAware(Strategy):
    """
    Strategy C: size the bet by how sure we are and how rough the market is.

        size = direction * conviction * vol_scaler

        conviction = (confidence - 0.5) / 0.5        -> 0 at a coin-flip, 1 at unanimity
        vol_scaler = min(1, target_vol / realised_vol) -> shrink positions in turbulent markets

    ``min_confidence`` is a floor below which the strategy refuses to trade at all
    (a tiny position would be all cost and no edge).

    ``rebalance_band``: because the ideal size changes a little every day, a
    naive implementation would trade every day and bleed costs. Instead we keep
    the current position until the ideal one differs from it by more than the
    band. This "no-trade zone" is standard practice in systematic trading.
    """

    name: str = "Risk-Aware"
    target_daily_vol: float = 0.01  # ~16 % annualised
    min_confidence: float = 0.55
    rebalance_band: float = 0.25

    def _raw_positions(self, s: pd.DataFrame) -> pd.Series:
        direction = np.sign(s["prediction"])
        conviction = ((s["confidence"] - 0.5) / 0.5).clip(0.0, 1.0)
        vol_scaler = (self.target_daily_vol / s["vol_20"]).clip(upper=1.0)
        ideal = (direction * conviction * vol_scaler).where(
            s["confidence"] >= self.min_confidence, 0.0
        )
        ideal = ideal.fillna(0.0).to_numpy()

        # Only move when the ideal position has drifted outside the band.
        held = np.zeros_like(ideal)
        current = 0.0
        for i, target in enumerate(ideal):
            if abs(target - current) > self.rebalance_band or (target == 0.0 and current != 0.0):
                current = target
            held[i] = current
        return pd.Series(held, index=s.index)


def default_strategies(threshold: float = 0.70, long_only: bool = False) -> list[Strategy]:
    return [
        BuyAndHold(long_only=long_only),
        AlwaysTrade(long_only=long_only),
        ConfidenceFilter(threshold=threshold, long_only=long_only),
        RiskAware(long_only=long_only),
    ]
