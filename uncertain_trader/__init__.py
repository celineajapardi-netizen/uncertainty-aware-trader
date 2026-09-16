"""
uncertain_trader
================

An uncertainty-aware trading simulator.

Research question:
    "When should an algorithm trade? Does refusing to act when a prediction
    is not confident produce better risk-adjusted decisions?"

Pipeline (each module is one step):

    data.py       -> daily prices (real, cached from Yahoo Finance, or simulated)
    features.py   -> technical features that describe "what the market looks like today"
    model.py      -> an ensemble model that outputs a PREDICTION and a CONFIDENCE
    strategies.py -> decision rules that turn (prediction, confidence, volatility) into BUY/HOLD/SELL
    backtest.py   -> replay the decisions through history and measure the outcome
    metrics.py    -> return, risk, Sharpe, drawdown, trade count, accuracy, calibration
"""

from .data import load_prices, simulate_prices
from .features import build_features, FEATURE_COLUMNS
from .model import EnsemblePredictor
from .strategies import (
    AlwaysTrade,
    BuyAndHold,
    ConfidenceFilter,
    RiskAware,
    Strategy,
    default_strategies,
)
from .backtest import run_backtest, run_all, threshold_sweep
from .metrics import summarise, calibration_table

__all__ = [
    "load_prices",
    "simulate_prices",
    "build_features",
    "FEATURE_COLUMNS",
    "EnsemblePredictor",
    "Strategy",
    "AlwaysTrade",
    "BuyAndHold",
    "ConfidenceFilter",
    "RiskAware",
    "default_strategies",
    "run_backtest",
    "run_all",
    "threshold_sweep",
    "summarise",
    "calibration_table",
]
