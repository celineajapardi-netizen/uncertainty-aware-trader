# When should an algorithm trade?

*Investigating the role of prediction confidence in quantitative trading decisions.*

Most "AI trading" projects ask *can a model predict whether a stock goes up?*
This one asks a different question:

> **Should a trading algorithm act when it is not confident in its own prediction?**

The model here outputs two numbers every day — a **prediction** (expected return
tomorrow) and a **confidence** (how sure it is about the direction). Three
decision rules then use those numbers differently, and we backtest them side by
side on simulated and real markets:

| Strategy | Rule |
|---|---|
| **A — Always Trade** | prediction > 0 → BUY, < 0 → SELL. Confidence ignored. |
| **B — Confidence Filter** | same, but **HOLD unless confidence ≥ threshold** (default 70 %) |
| **C — Risk-Aware** | bet size = direction × conviction × volatility scaler; refuses tiny-edge trades |
| *Buy & Hold* | benchmark: ignores the model entirely |

We do **not** ask which one makes the most money. We ask whether *refusing to
trade when uncertain* produces better **risk-adjusted** decisions (Sharpe ratio,
drawdown), how often each strategy acts, and — crucially — whether the model's
confidence is *honest* (calibrated) in the first place.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # interactive dashboard
python run_experiment.py        # reproducible tables + figures -> results/
python -m pytest tests -q       # 16 tests: no look-ahead, honest accounting
```

The first real-data run downloads daily prices from Yahoo Finance and caches
them as CSV in `data/`; everything after that works offline. The simulated
market needs no internet at all.

---

## How it works

```
prices ──▶ features ──▶ ensemble model ──▶ (prediction, confidence) ──▶ strategy ──▶ position ──▶ backtest
```

Each stage is one small file in `uncertain_trader/`:

| File | What it does |
|---|---|
| `data.py` | Real daily OHLCV (cached) **or** a simulated market with a *known* amount of predictability |
| `features.py` | 9 technical features (momentum, volatility, moving-average gap, RSI, volume). Every feature on day *t* uses only prices up to day *t*. |
| `model.py` | Walk-forward **ensemble of 50 ridge regressions**, each trained on a bootstrap resample of the last 500 days. Prediction = the average opinion. **Confidence = the fraction of models that agree on the direction.** |
| `strategies.py` | The decision rules above. Positions are fractions of capital in [-1, +1]. |
| `backtest.py` | A decision at the close of day *t* is held during day *t+1*; changing position costs `cost_bps` (default 5 bps). |
| `metrics.py` | Return, volatility, Sharpe, max drawdown, trade count, % of days in market, accuracy, and a **calibration table**. |
| `plots.py` | Equity curves, the threshold sweep, calibration plot, BUY / HOLD / SELL timeline. |

### Why "fraction of models that agree" is a good confidence measure

If you train many models on slightly different slices of history and they all
say *up*, the pattern is stable in the data. If they split 26 / 24, today's
inputs are ones the data does not explain — the prediction is closer to a coin
flip. This is the "deep ensemble" idea from modern machine learning, in the
simplest form that can be explained line by line.

### The simulated market — the control experiment

`simulate_prices()` generates returns as

```
r_t = drift + trend_t + sigma_t × noise_t
```

where `trend_t` is a slowly drifting hidden signal (AR(1), persistence 0.97),
`sigma_t` switches between calm and turbulent regimes, and the noise has fat
tails. `signal_strength` sets how big the trend is relative to the noise:
**0 = pure random walk** (nothing is predictable, so no strategy *should* help),
**0.3 = the model is right ~57 % of the time**. Because we know the truth, we can
check whether the model's confidence tracks reality before trusting it on real data.
(The simulated market has no reliable upward drift, so Buy & Hold is not a
meaningful benchmark there — compare the three model-driven strategies with each other.)

---

## What we found (default settings: 5 bps costs, 70 % threshold)

Run `python run_experiment.py` to regenerate; numbers below come from `results/REPORT.md`.

**1. In a market with a genuine, learnable signal (simulated), confidence matters.**
Across 3 seeds, the Confidence Filter matches or beats Always Trade on Sharpe while
trading 20 % fewer days, and the Risk-Aware rule is best in every seed — mainly by
cutting volatility and drawdown, not by earning more. The threshold sweep shows
Sharpe *rising* as the required confidence goes from 50 % to ~90 %, while the
fraction of days traded falls from 100 % to ~55 %. Refusing to trade when unsure
is a good decision *when confidence is honest*.

**2. On real daily data (SPY, AAPL, BTC-USD 2012-2026), it does not — and the calibration plot says why.**
The model's direction accuracy is ~51 % (a coin flip), and the calibration curve is
flat: days the model calls "95 % confident" are right no more often than days it
calls "55 % confident". Confidence that carries no information cannot be filtered
on, so none of the model-driven strategies comes close to Buy & Hold, and
transaction costs (40-80 % of capital over the period for the daily traders) do
most of the damage.

**3. The uncertainty-aware rules still lose *less*.** Even on real data, Strategy C
runs at roughly half the volatility of Strategy A with smaller drawdowns and a
fraction of the costs, because refusing tiny-edge trades and shrinking positions
in turbulent periods cuts both risk and turnover.

The take-away that links to the AI-trust theme: *an algorithm's confidence is only
worth acting on if it is calibrated — and checking calibration is a separate, prior
step from checking accuracy.*

---

## Things to try (ideas for extending the project)

* **Cost sensitivity.** Slide costs from 0 to 25 bps in the app. At 0 bps the filter
  loses its main advantage; at 25 bps almost nothing survives. Where is the crossover?
* **Long-only.** Tick the box — does removing short selling change the ranking?
* **Contradictory signals.** `signals` contains every feature; write a strategy that
  holds when momentum and RSI disagree.
* **Regime awareness.** `vol_ratio` > 1 flags turbulent periods; use a different
  threshold in each regime.
* **A different confidence measure.** Replace ensemble agreement with
  |prediction| / (spread of model opinions) and re-check the calibration plot.
* **Weekly horizon.** Predict 5-day returns instead of 1-day; costs bite less.

---

## Honesty notes

* All data is public (Yahoo Finance) or simulated. The code is written from scratch
  for this project and contains no proprietary strategy.
* The backtest is deliberately simple: no leverage, no margin, no dividends, a flat
  cost per unit traded. It is built to compare *decision rules fairly*, not to
  reproduce a broker statement.
* Nothing here is investment advice. The real-data result — that a simple daily
  model is a coin flip and costs eat everything — is the normal outcome, and the
  project is more useful for showing that clearly than for hiding it.
