"""
Step 1 - Market data.

Two sources, same output shape (a DataFrame indexed by date with columns
Open, High, Low, Close, Volume):

* ``load_prices``     - real, public daily data from Yahoo Finance, cached to CSV
                        so the project runs offline and results are reproducible.
* ``simulate_prices`` - synthetic data where WE decide how predictable the market
                        is. This is the control experiment: if a strategy can't
                        beat "always trade" on data with a known signal, it never will.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"


def load_prices(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for ``ticker``. Downloads once, then reads the CSV cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker.upper()}.csv"

    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # imported lazily so the simulator works without it

        raw = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker!r}")
        # yfinance >= 0.2 returns a two-level column index (field, ticker); flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[COLUMNS].copy()
        df.index.name = "Date"
        df.to_csv(path)

    df = df.loc[start:end] if end else df.loc[start:]
    return df.dropna()


def simulate_prices(
    n_days: int = 2500,
    seed: int = 0,
    signal_strength: float = 0.30,
    daily_vol: float = 0.012,
    annual_drift: float = 0.04,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Simulate a market with a *known* amount of predictability.

    The daily return is::

        r_t = mu_t + sigma_t * noise_t

    * ``mu_t`` is a slowly drifting "hidden trend" (an AR(1) process). Because it
      persists for weeks, recent returns carry information about the next one -
      that is what momentum-style features can pick up.
    * ``sigma_t`` switches between a calm regime and a turbulent regime (a simple
      two-state Markov chain), so volatility is clustered like in real markets.
    * ``signal_strength`` controls how large the trend is compared with the noise.
      0.0 gives a pure random walk (nothing is predictable). 0.3 gives a market
      where our model is right about 57 % of the time - already generous compared
      with real markets, where 52-53 % is a good day.
    * ``annual_drift`` is a small constant upward drift, like a stock index.

    Returns an OHLCV frame so it can be used exactly like real data.
    """
    rng = np.random.default_rng(seed)

    # Hidden trend: AR(1) with high persistence.
    phi = 0.97
    trend = np.zeros(n_days)
    innov_sd = daily_vol * signal_strength * np.sqrt(1 - phi**2)
    for t in range(1, n_days):
        trend[t] = phi * trend[t - 1] + rng.normal(0.0, innov_sd)

    # Volatility regime: 0 = calm, 1 = turbulent. Sticky transitions.
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        stay = 0.985 if regime[t - 1] == 0 else 0.95
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    sigma = np.where(regime == 0, daily_vol * 0.7, daily_vol * 2.0)

    # Student-t noise (fat tails), rescaled so its standard deviation is 1.
    noise = rng.standard_t(df=5, size=n_days) / np.sqrt(5 / 3)
    returns = annual_drift / 252 + trend + sigma * noise
    close = start_price * np.cumprod(1 + returns)

    # Cosmetic OHLC around the close path, plus volume that rises with turbulence.
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, sigma)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.lognormal(mean=np.log(1e6) + regime * 0.5, sigma=0.3, size=n_days)

    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    df.index.name = "Date"
    # Keep the hidden state so experiments can check whether the model found it.
    df.attrs["hidden_trend"] = trend
    df.attrs["regime"] = regime
    return df
