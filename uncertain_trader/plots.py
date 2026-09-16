"""
Charts, shared by the command-line report and the Streamlit app.

Conventions: each strategy always gets the same colour (identity, not rank);
BUY / HOLD / SELL use status colours plus a marker shape, so the decision
timeline still reads in greyscale; one y-axis per chart, never two.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 1.0,
        "axes.edgecolor": "#c3c2b7",
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "font.size": 10,
    }
)

# Fixed categorical order: benchmark, then strategies A, B, C.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
DECISION_STYLE = {
    "BUY": {"color": "#008300", "marker": "^", "label": "BUY"},
    "HOLD": {"color": "#8a897f", "marker": "o", "label": "HOLD"},
    "SELL": {"color": "#e34948", "marker": "v", "label": "SELL"},
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"


def _colour(i: int) -> str:
    return SERIES[i % len(SERIES)]


def plot_equity(results: dict[str, pd.DataFrame], title: str = "Growth of 1.0") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, bt) in enumerate(results.items()):
        eq = bt["equity"]
        ax.plot(
            eq.index, eq.values, lw=2, color=_colour(i), solid_joinstyle="round",
            label=f"{name}: {eq.iloc[-1]:.2f}x",
        )
        ax.scatter([eq.index[-1]], [eq.iloc[-1]], s=30, color=_colour(i), edgecolors="white", linewidths=1.2, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title)
    ax.set_ylabel("equity (log scale)", color=TEXT_2)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_decisions(
    bt: pd.DataFrame, close: pd.Series, last_n: int = 250, title: str = "Decisions"
) -> plt.Figure:
    """Price line with a marker per day showing what the strategy decided at that close."""
    bt = bt.iloc[-last_n:]
    close = close.reindex(bt.index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(close.index, close.values, lw=1.5, color="#9ec5f4", zorder=1)
    for label, style in DECISION_STYLE.items():
        mask = bt["label"] == label
        if not mask.any():
            continue
        size = 16 if label == "HOLD" else 36
        ax.scatter(
            close.index[mask],
            close.values[mask],
            s=size,
            color=style["color"],
            marker=style["marker"],
            label=f'{style["label"]} ({int(mask.sum())} days)',
            edgecolors="white",
            linewidths=0.6,
            zorder=2,
        )
    ax.set_title(title)
    ax.set_ylabel("price", color=TEXT_2)
    ax.legend(loc="upper left", fontsize=8.5, markerscale=1.4)
    fig.tight_layout()
    return fig


def plot_calibration(cal: pd.DataFrame, title: str = "Is confidence honest?") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#c3c2b7", lw=1.5, ls=(0, (4, 3)), label="perfectly calibrated")
    ax.axhline(0.5, color="#e6e5e1", lw=1)
    sizes = 40 + 160 * cal["n_days"] / cal["n_days"].max()
    ax.plot(cal["stated_confidence"], cal["actual_accuracy"], color=_colour(0), lw=2, zorder=2)
    ax.scatter(
        cal["stated_confidence"],
        cal["actual_accuracy"],
        s=sizes,
        color=_colour(0),
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
        label="model (dot size = number of days)",
    )
    for _, row in cal.iterrows():
        ax.annotate(
            f'{row["actual_accuracy"]:.0%}',
            (row["stated_confidence"], row["actual_accuracy"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            color=TEXT_2,
        )
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("what the model said (confidence)", color=TEXT_2)
    ax.set_ylabel("how often it was right", color=TEXT_2)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep: pd.DataFrame, baseline_sharpe: float | None = None) -> plt.Figure:
    """Two small multiples that share an x-axis - never a dual-axis chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = sweep.index.to_numpy()

    ax1.plot(x, sweep["sharpe"], lw=2, color=_colour(2), marker="o", ms=5)
    if baseline_sharpe is not None:
        ax1.axhline(baseline_sharpe, color=_colour(1), lw=1.5, ls=(0, (4, 3)))
        ax1.annotate("Always Trade", (x[0], baseline_sharpe), xytext=(0, 5), textcoords="offset points", fontsize=8.5, color=TEXT_2)
    ax1.axhline(0, color="#c3c2b7", lw=1)
    ax1.set_title("Sharpe vs confidence threshold")
    ax1.set_xlabel("minimum confidence to trade", color=TEXT_2)

    ax2.plot(x, sweep["pct_in_market"], lw=2, color=_colour(2), marker="o", ms=5, label="days in market")
    ax2.plot(x, sweep["accuracy"], lw=2, color=_colour(3), marker="o", ms=5, label="accuracy when trading")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("How often it trades, and how well")
    ax2.set_xlabel("minimum confidence to trade", color=TEXT_2)
    ax2.legend(fontsize=8.5, loc="lower left")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    return fig


def plot_confidence_hist(signals: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(signals["confidence"].dropna(), bins=np.linspace(0.5, 1.0, 26), color=_colour(0), edgecolor="white")
    ax.set_title("How confident is the model, day to day?")
    ax.set_xlabel("confidence", color=TEXT_2)
    ax.set_ylabel("days", color=TEXT_2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    return fig
