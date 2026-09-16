# When Should an Algorithm Trade?

**Uncertainty Aware Trading** is a trading simulation I built in Python to investigate whether an algorithm should act on a prediction when it is not confident in it — and how to tell whether its confidence means anything at all.

Most AI trading projects try to predict whether a price goes up and report how much money the model would have made. I wanted to ask a different question. My model produces two numbers every day: a **prediction** (expected return tomorrow) and a **confidence** (how sure it is about the direction). I then compare three decision rules that treat that confidence differently, on both a simulated market and real market data.

## Live Demo
**Open the interactive Streamlit dashboard : https://uncertainty-aware-trader-fvy7vrbdgz9uxxkjfgyami.streamlit.app**

## Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard lets you switch between simulated and real data (SPY, AAPL, BTC-USD), change the confidence threshold, transaction cost, training window and ensemble size, and see the equity curves, calibration plot and trading decisions update live.

## Features

- **Prediction + Confidence** — an ensemble of 50 small models; confidence is the share of models that agree on the direction
- **Three decision rules** — Always Trade, Confidence Filter (hold when unsure), and Risk-Aware (scale the bet by confidence and volatility)
- **Buy & Hold benchmark** — the baseline every active strategy has to beat
- **Simulated market** — a market with a known, adjustable amount of predictability, used as a control experiment
- **Real market data** — daily prices for SPY, AAPL and BTC-USD from 2012 to 2026
- **Calibration check** — does "90 % confident" actually mean right 90 % of the time?
- **Honest backtest** — walk-forward only, one-day holding lag, transaction costs on every trade
- **Test suite** — 16 tests that check the model cannot see the future and that costs are accounted for correctly
- **Reproducible results** — one script regenerates every table and figure in `results/`

## Concept

The project treats confidence as something to be tested, not assumed.

In the simulated market I control exactly how predictable prices are, so I can check whether the model's confidence is *calibrated* before trusting it. Then I run the same code on real data, where the truth is unknown, and see whether the same rules still help.

The aim is to find out which rule makes better **risk-adjusted** decisions — Sharpe ratio, drawdown, how often the strategy is in the market — not which one ends with the biggest number.

## What I found

The two markets give opposite answers, and the reason is the point of the project.

- **Simulated market (confidence is honest):** refusing to trade when unsure improves the Sharpe ratio and cuts drawdowns. The Risk-Aware rule was best in every run, mainly by reducing volatility rather than earning more.
- **Real market (confidence is not):** the model's direction accuracy is about 51 % — a coin flip — and its confidence carries no information. A "95 % confident" day is right no more often than a "55 %" one. Nothing built on that confidence can work, and transaction costs do the rest.

Same code, same rules — the only difference is calibration. **A model's confidence is only worth acting on if it has been shown to be right.**

This connects to my other research project on human trust in AI, which asks the mirror question: whether *people* trust a confident-sounding AI answer more, regardless of whether it is correct.

## Development

I built the whole pipeline from scratch in Python — data loading, features, the ensemble model, the strategies, the backtest, the metrics and the plots — so that I could explain every line without relying on a machine-learning library.

The main challenge was making the backtest honest. It is easy to write a backtest that accidentally uses tomorrow's prices, or that ignores costs, and then reports results that look far better than they are. Most of the test suite exists to guard against exactly that.

## Technologies

- **Python**
- **NumPy / pandas** — data handling and a closed-form ridge regression
- **Streamlit** — interactive dashboard
- **Matplotlib** — figures
- **yfinance** — public market data
- **pytest** — test suite
- Quantitative finance, backtesting, model calibration

## Future Development

If I continue the project, I would like to try a weekly prediction horizon (where costs matter less), test alternative confidence measures, use different thresholds in calm and turbulent market regimes, and explore where the crossover point is between transaction cost and the benefit of filtering.

---

*Developed by Celine Angelica Japardi*
