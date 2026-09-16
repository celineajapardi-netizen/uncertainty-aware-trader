"""
Step 3 - A model that knows when it doesn't know.

Most "AI stock predictors" output a single number. Ours outputs two:

    prediction  - the expected next-day return (e.g. +0.21 %)
    confidence  - how sure the model is about the SIGN of that prediction (50-100 %)

How confidence is produced (bagging / "ensemble disagreement"):

    1. Take the last ``train_window`` days of (features, next-day return).
    2. Train ``n_models`` small ridge-regression models, each on a random
       resample (bootstrap) of those days. Each model has seen slightly
       different history, so each has a slightly different opinion.
    3. prediction = average of the models' opinions.
    4. confidence = fraction of models whose opinion has the same sign as the
       average.  All 50 say "up" -> 100 %.  A 26/24 split -> 52 %.

This is the same idea used by "deep ensembles" in modern machine learning:
if independently-trained models disagree, the input is one the data does not
explain well, and the prediction should not be trusted.

Everything is walk-forward: the model used on day t was fitted only on days
strictly before t. Row t-1's target is the return of day t, which is known by
the close of day t - the moment we decide - so rows up to t-1 are fair game and
row t itself is not. No look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression: w = (X'X + lam*I)^-1 X'y. X has a bias column."""
    n_feat = X.shape[1]
    penalty = lam * np.eye(n_feat)
    penalty[0, 0] = 0.0  # do not shrink the intercept
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


@dataclass
class EnsemblePredictor:
    train_window: int = 500  # ~2 years of trading days
    n_models: int = 50
    ridge_lambda: float = 5.0
    refit_every: int = 5  # refit weekly; predicting daily with a weekly-fit model
    seed: int = 0
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Walk forward through ``features`` and return a frame with columns
        ``prediction`` and ``confidence`` (NaN for the warm-up period).
        """
        cols = list(self.feature_columns)
        X_all = features[cols].to_numpy(dtype=float)
        y_all = features["target"].to_numpy(dtype=float)
        n = len(features)
        rng = np.random.default_rng(self.seed)

        prediction = np.full(n, np.nan)
        confidence = np.full(n, np.nan)

        weights: np.ndarray | None = None  # shape (n_models, n_feat + 1)
        mean = std = None

        for t in range(self.train_window, n):
            # Training rows: [t - window, t). Row t-1's target is the return of
            # day t, which is known by the close of day t - when we decide.
            lo, hi = t - self.train_window, t
            if weights is None or (t - self.train_window) % self.refit_every == 0:
                X_tr, y_tr = X_all[lo:hi], y_all[lo:hi]
                ok = ~np.isnan(y_tr)
                X_tr, y_tr = X_tr[ok], y_tr[ok]
                mean, std = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-12
                Z = np.hstack([np.ones((len(X_tr), 1)), (X_tr - mean) / std])
                weights = np.empty((self.n_models, Z.shape[1]))
                for m in range(self.n_models):
                    idx = rng.integers(0, len(Z), size=len(Z))  # bootstrap resample
                    weights[m] = _ridge_fit(Z[idx], y_tr[idx], self.ridge_lambda)

            z_t = np.concatenate([[1.0], (X_all[t] - mean) / std])
            votes = weights @ z_t  # one opinion per model
            consensus = votes.mean()
            agree = np.mean(np.sign(votes) == np.sign(consensus)) if consensus != 0 else 0.5
            agree = max(float(agree), 0.5)  # a split vote is 50 %, never less
            prediction[t] = consensus
            confidence[t] = agree

        return pd.DataFrame(
            {"prediction": prediction, "confidence": confidence}, index=features.index
        )
