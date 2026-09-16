"""
Step 2 - Features.

Turn a price history into a small table describing "what the market looks like
today". Every feature at row ``t`` uses ONLY prices up to and including day ``t``.
That rule (no look-ahead) is what makes the backtest honest; ``tests/`` checks it.

The target column ``target`` is the *next* day's return - the thing we try to
predict. It is the only column allowed to peek forward, and the model must never
see it at prediction time (only at training time, for past rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",  # yesterday -> today return
    "ret_5",  # 1-week momentum
    "ret_20",  # 1-month momentum
    "ret_60",  # 3-month momentum
    "vol_20",  # realised volatility (std of daily returns, 20 days)
    "vol_ratio",  # short-term vol / long-term vol  (>1 means "getting rougher")
    "ma_gap",  # how far price sits above/below its 50-day average
    "rsi_14",  # classic overbought/oversold oscillator, rescaled to [-1, 1]
    "volume_z",  # today's volume vs its 20-day average, in standard deviations
]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with FEATURE_COLUMNS + ``ret`` (today's return) + ``target``."""
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float)
    ret = close.pct_change()

    f = pd.DataFrame(index=prices.index)
    f["ret"] = ret
    f["ret_1"] = ret
    f["ret_5"] = close.pct_change(5)
    f["ret_20"] = close.pct_change(20)
    f["ret_60"] = close.pct_change(60)
    f["vol_20"] = ret.rolling(20).std()
    f["vol_ratio"] = f["vol_20"] / ret.rolling(100).std()
    f["ma_gap"] = close / close.rolling(50).mean() - 1.0

    gain = ret.clip(lower=0).rolling(14).mean()
    loss = (-ret.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["rsi_14"] = (rsi - 50) / 50

    log_vol = np.log(volume.replace(0, np.nan))
    f["volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()

    # The only forward-looking column: what happens tomorrow.
    f["target"] = ret.shift(-1)

    return f.dropna(subset=FEATURE_COLUMNS + ["ret"])
