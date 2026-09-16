"""
Run with:  python -m pytest tests -q

These tests protect the two things a backtest must get right:
  1. no look-ahead  - a decision on day t never uses information from day t+1
  2. honest accounting - costs and returns add up the way they should
"""

import numpy as np
import pandas as pd
import pytest

from uncertain_trader import (
    AlwaysTrade,
    ConfidenceFilter,
    EnsemblePredictor,
    FEATURE_COLUMNS,
    RiskAware,
    build_features,
    calibration_table,
    run_backtest,
    simulate_prices,
    summarise,
    threshold_sweep,
)
from uncertain_trader.backtest import build_signals

SMALL_MODEL = dict(train_window=200, n_models=10, refit_every=10)


@pytest.fixture(scope="module")
def prices():
    return simulate_prices(n_days=900, seed=3)


@pytest.fixture(scope="module")
def signals(prices):
    return build_signals(prices, EnsemblePredictor(**SMALL_MODEL))


# ------------------------------------------------------------------ look-ahead
def test_features_do_not_use_the_future(prices):
    """Changing prices AFTER day t must not change any feature ON day t."""
    cut = 600
    tampered = prices.copy()
    tampered.iloc[cut + 1 :, :] *= 1.5  # wreck the future
    a = build_features(prices)[FEATURE_COLUMNS]
    b = build_features(tampered)[FEATURE_COLUMNS]
    common = a.index[a.index <= prices.index[cut]]
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_target_is_next_day_return(prices):
    f = build_features(prices)
    assert np.allclose(f["target"].iloc[:-1].values, f["ret"].iloc[1:].values, equal_nan=True)


def test_model_predictions_do_not_use_the_future(prices):
    """Predictions up to day t are identical whether or not the future is altered."""
    cut = 700
    tampered = prices.copy()
    tampered.iloc[cut + 1 :, :] *= 1.5
    a = EnsemblePredictor(**SMALL_MODEL).predict(build_features(prices))
    b = EnsemblePredictor(**SMALL_MODEL).predict(build_features(tampered))
    common = a.index[a.index <= prices.index[cut]]
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_confidence_is_a_probability(signals):
    c = signals["confidence"]
    assert c.between(0.5, 1.0).all()


# ------------------------------------------------------------------ accounting
def test_decision_is_applied_next_day(signals):
    bt = run_backtest(signals, AlwaysTrade(), cost_bps=0.0)
    assert (bt["position"].iloc[1:].values == bt["decision"].iloc[:-1].values).all()
    assert bt["position"].iloc[0] == 0.0


def test_zero_cost_return_equals_position_times_market(signals):
    bt = run_backtest(signals, AlwaysTrade(), cost_bps=0.0)
    expected = bt["position"] * bt["market_ret"]
    assert np.allclose(bt["strategy_ret"], expected)


def test_costs_are_charged_on_position_changes(signals):
    bt = run_backtest(signals, AlwaysTrade(), cost_bps=10.0)
    flips = bt["position"].diff().abs().fillna(bt["position"].abs())
    assert np.allclose(bt["cost"], flips * 0.001)
    # A full flip long -> short trades 2 units of capital.
    assert bt["traded"].max() == pytest.approx(2.0)


def test_confidence_filter_trades_less_than_always(signals):
    a = summarise(run_backtest(signals, AlwaysTrade()))
    b = summarise(run_backtest(signals, ConfidenceFilter(threshold=0.9)))
    assert b["pct_in_market"] < a["pct_in_market"]
    assert a["pct_in_market"] == pytest.approx(1.0, abs=0.01)


def test_threshold_half_equals_always_trade(signals):
    sweep = threshold_sweep(signals, thresholds=[0.5], cost_bps=5.0)
    always = summarise(run_backtest(signals, AlwaysTrade(), cost_bps=5.0))
    assert sweep.loc[0.5, "sharpe"] == pytest.approx(always["sharpe"])


def test_long_only_never_shorts(signals):
    bt = run_backtest(signals, RiskAware(long_only=True))
    assert (bt["position"] >= 0).all()
    assert bt["position"].abs().max() <= 1.0


def test_risk_aware_positions_are_fractional_and_bounded(signals):
    bt = run_backtest(signals, RiskAware())
    assert bt["position"].abs().max() <= 1.0
    assert (bt["position"].abs().between(0.01, 0.99)).any()


def test_buy_and_hold_matches_market(signals):
    from uncertain_trader import BuyAndHold

    bt = run_backtest(signals, BuyAndHold(), cost_bps=0.0)
    market = (1 + signals["ret"].iloc[1:]).prod()
    assert bt["equity"].iloc[-1] == pytest.approx(market)


# ------------------------------------------------------------------ metrics
def test_max_drawdown_of_monotonic_equity_is_zero():
    bt = pd.DataFrame(
        {
            "strategy_ret": [0.01] * 10,
            "equity": np.cumprod([1.01] * 10),
            "position": [1.0] * 10,
            "market_ret": [0.01] * 10,
            "traded": [1.0] + [0.0] * 9,
            "cost": [0.0] * 10,
        }
    )
    s = summarise(bt)
    assert s["max_drawdown"] == 0.0
    assert s["accuracy"] == 1.0
    assert s["n_trades"] == 1


def test_calibration_table_shape(signals):
    cal = calibration_table(signals, n_bins=5)
    assert cal["n_days"].sum() <= len(signals)
    assert cal["actual_accuracy"].between(0, 1).all()


# ------------------------------------------------------------------ simulation sanity
def test_random_walk_is_unpredictable():
    px = simulate_prices(n_days=1500, seed=11, signal_strength=0.0)
    s = build_signals(px, EnsemblePredictor(**SMALL_MODEL))
    acc = (np.sign(s["prediction"]) == np.sign(s["target"])).mean()
    assert abs(acc - 0.5) < 0.05


def test_signal_is_recoverable():
    px = simulate_prices(n_days=1500, seed=11, signal_strength=0.6)
    s = build_signals(px, EnsemblePredictor(**SMALL_MODEL))
    acc = (np.sign(s["prediction"]) == np.sign(s["target"])).mean()
    assert acc > 0.55
