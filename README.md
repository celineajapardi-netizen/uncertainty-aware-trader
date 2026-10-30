
# When Should an Algorithm Trade?
### Exploring Uncertainty in Algorithmic Trading

**Uncertainty-Aware Trading** is an independent Python project exploring how prediction confidence can influence algorithmic trading decisions.

Instead of focusing solely on predicting market movements, I wanted to investigate a different question: *Does knowing when a model is uncertain lead to better trading decisions?*

I built an ensemble-based prediction model that estimates next-day returns and measures confidence through model agreement. I then compared different trading strategies across simulated and real financial markets.

## Live Demo

**[Explore the Interactive Streamlit Dashboard](https://uncertainty-aware-trader-fvy7vrbdgz9uxxkjfgyami.streamlit.app)**

## Dashboard

The interactive dashboard brings the entire experiment together, allowing users to adjust model parameters, explore market conditions and compare trading strategies in real time.

### 1. Market Overview

Displays the latest market signals, predicted returns, model confidence and the decisions each trading strategy would make.

![Market Overview](screenshots/01-dashboard-overview.png)

### 2. Strategy Performance

Visualises the growth of an initial investment across four strategies, making it easier to compare returns, volatility and risk-adjusted performance.

![Strategy Performance](screenshots/02-strategy-performance.png)

### 3. Confidence Analysis

Explores how changing the minimum confidence threshold affects the Sharpe ratio, trading frequency and prediction accuracy. It also examines whether the model's confidence is properly calibrated.

![Confidence Analysis](screenshots/03-confidence-analysis.png)

### 4. Decision Timeline

Tracks the algorithm's trading decisions over time, showing when it chooses to enter, exit or hold positions based on its predictions and confidence.

![Decision Timeline](screenshots/04-decision-timeline.png)

### Interactive Controls

Users can experiment with:
- Simulated markets or real market data (SPY, AAPL and BTC-USD).
- Market signal strength and simulation duration.
- Model training window and ensemble size.
- Confidence thresholds and transaction costs.
- Trading constraints and strategy configurations.

## What It Does

- **Ensemble Modelling:** Combines 50 models to predict returns and estimate directional confidence.
- **Trading Strategies:** Compares Buy & Hold, Always Trade, Confidence Filter and Risk-Aware approaches.
- **Market Simulation:** Tests confidence-based decisions under different levels of predictability.
- **Real Market Data:** Evaluates strategies using historical financial data.
- **Confidence Calibration:** Tests whether higher model confidence corresponds to greater accuracy.
- **Risk Analysis:** Evaluates Sharpe ratio, maximum drawdown, volatility and cumulative returns.
- **Backtesting & Testing:** Implements walk-forward backtesting, transaction costs and 16 automated tests.

## Key Findings

The experiments revealed an interesting contrast between simulated and real markets.

- **Simulated Market:** Confidence-aware strategies improved risk-adjusted performance, with the Risk-Aware strategy achieving a Sharpe ratio of 3.42 in the displayed experiment.
- **Real Market:** Directional accuracy was approximately 51%, while model confidence showed limited ability to distinguish correct predictions from incorrect ones.

**The main takeaway:** A model being confident does not necessarily mean it is correct. Confidence is only useful when it reflects genuine predictive ability.

## Built With

Python · NumPy · pandas · Streamlit · Matplotlib · yfinance · pytest

## Getting Started

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/uncertainty-trader.git
cd uncertainty-trader
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Future Improvements

Explore alternative uncertainty measures, adaptive confidence thresholds, different prediction horizons and improved model calibration.

## Author

**Celine Angelica Japardi**

Independent Project | Quantitative Finance & Machine Learning
