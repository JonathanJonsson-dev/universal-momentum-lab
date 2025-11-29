"""Meta-strategy overlays for the ERN momentum allocation and the 3-asset trend model.

This script treats the ERN strategy and the simple 3m>10m trend model as
underlying systems and applies two layers of meta logic:

1) An equity-curve filter per system (moving-average trend plus drawdown stop)
   that moves to cash when the system is cold.
2) A relative-strength allocator that rotates between the two systems (or cash)
   based on lagged risk-adjusted scores.

Outputs: headline metrics for the base systems, their filtered versions, and
the meta allocator, plus recent signals/weights for inspection.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import pandas as pd

from ern_momentum_blog_backtest import (
    EXPENSE_RATIOS,
    OUTPUT_DIR,
    compute_metrics,
    daily_to_monthly_returns,
    load_market_data,
    run_backtest,
    to_wealth,
)
from trend_vs_ern_comparison import (
    align_returns,
    compute_trend_returns,
    format_metrics,
)


def equity_curve_filter(
    base_returns: pd.Series,
    cash_returns: pd.Series,
    ma_window: int = 6,
    max_drawdown: float = 0.12,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple meta filter: stay invested when the equity curve is above a lagged
    moving average and not beyond the drawdown limit; otherwise sit in cash.
    """
    df = pd.concat({"base": base_returns, "cash": cash_returns}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="filtered"), pd.Series(dtype=float, name="signal")

    equity = to_wealth(df["base"])
    equity_lag = equity.shift(1)
    ma = equity_lag.rolling(window=ma_window, min_periods=ma_window).mean()
    peak = equity_lag.cummax()
    dd = equity_lag / peak - 1.0
    signal = (equity_lag > ma) & (dd >= -max_drawdown)
    signal = signal.fillna(False)

    filtered = df["base"].where(signal, df["cash"])
    filtered.name = "filtered"
    return filtered, signal.astype(int).rename("signal")


def strategy_allocator(
    ern_returns: pd.Series,
    trend_returns: pd.Series,
    cash_returns: pd.Series,
    lookback: int = 6,
    min_score: float = 0.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Allocate between ERN and the trend model using lagged risk-adjusted scores.
    Score = trailing excess mean / trailing vol (Sharpe-like), annualized.
    If both scores are below the threshold, capital parks in cash.
    """
    systems = pd.concat({"ERN": ern_returns, "Trend": trend_returns}, axis=1).dropna()
    if systems.empty:
        return pd.Series(dtype=float, name="meta_allocator"), pd.DataFrame()

    cash_aligned = cash_returns.loc[systems.index]
    excess = systems.sub(cash_aligned, axis=0)
    trailing_mean = excess.rolling(window=lookback, min_periods=lookback).mean()
    trailing_vol = systems.rolling(window=lookback, min_periods=lookback).std().replace(0.0, math.nan)
    score = (trailing_mean / trailing_vol) * math.sqrt(12)
    score = score.shift(1)  # trade on prior window information

    weight_rows = []
    for date, row in score.iterrows():
        positives = row[row > min_score].dropna()
        if positives.empty:
            weight_rows.append(pd.Series({"ERN": 0.0, "Trend": 0.0, "CASH": 1.0}, name=date))
            continue
        alloc = positives / positives.sum()
        weights = pd.Series(
            {"ERN": alloc.get("ERN", 0.0), "Trend": alloc.get("Trend", 0.0)},
            name=date,
        )
        weights["CASH"] = 1.0 - weights.sum()
        weight_rows.append(weights)

    weights_df = pd.DataFrame(weight_rows).reindex(systems.index)
    weights_df = weights_df.fillna({"ERN": 0.0, "Trend": 0.0, "CASH": 1.0})
    portfolio = (weights_df[systems.columns] * systems).sum(axis=1) + weights_df["CASH"] * cash_aligned
    portfolio.name = "meta_allocator"
    return portfolio, weights_df


def collect_metrics(
    base: pd.Series,
    filtered: pd.Series,
    cash: pd.Series,
) -> Dict[str, Dict[str, float]]:
    return {
        "base": compute_metrics(base, cash),
        "meta": compute_metrics(filtered, cash),
    }


def plot_meta_performance(
    ern_base: pd.Series,
    ern_filtered: pd.Series,
    trend_base: pd.Series,
    trend_filtered: pd.Series,
    meta_returns: pd.Series,
    outfile: str = "meta_strategy_wealth.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping meta performance plot.")
        return

    labeled = {
        "ERN base": ern_base,
        "ERN filtered": ern_filtered,
        "Trend base": trend_base,
        "Trend filtered": trend_filtered,
        "Meta allocator": meta_returns,
    }
    series_list = [s for s in labeled.values() if s is not None and not s.empty]
    if not series_list:
        print("No data to plot meta performance.")
        return

    common = series_list[0].index
    for series in series_list[1:]:
        common = common.intersection(series.index)
    if common.empty:
        print("No overlapping data to plot meta performance.")
        return

    outfile_path = OUTPUT_DIR / outfile
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, series in labeled.items():
        if series is None or series.empty:
            continue
        wealth = to_wealth(series).loc[common]
        ax.plot(wealth.index, wealth.values, label=label, linewidth=1.5)

    ax.set_title("Meta Strategy: ERN vs. Trend and Allocator")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outfile_path, dpi=150)
    plt.close(fig)
    print(f"Saved meta performance plot to {outfile_path}")


def plot_meta_weights(meta_weights: pd.DataFrame, outfile: str = "meta_allocator_weights.png") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping meta weights plot.")
        return
    if meta_weights.empty:
        print("No meta weights to plot.")
        return

    outfile_path = OUTPUT_DIR / outfile
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = ["ERN", "Trend", "CASH"]
    available = [c for c in cols if c in meta_weights]
    ax.stackplot(meta_weights.index, [meta_weights[c] for c in available], labels=available)
    ax.set_title("Meta Allocator Weights Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outfile_path, dpi=150)
    plt.close(fig)
    print(f"Saved meta weights plot to {outfile_path}")


def main() -> None:
    ern_result = run_backtest()
    prices, cash_daily, _ = load_market_data()
    trend_returns, trend_weights = compute_trend_returns(prices, cash_daily)

    monthly_cash = daily_to_monthly_returns(cash_daily)
    monthly_cash = monthly_cash - EXPENSE_RATIOS["CASH"] / 12.0

    ern_aligned, trend_aligned, cash_aligned = align_returns(
        ern_result.returns, trend_returns, monthly_cash
    )

    ern_filtered, ern_signal = equity_curve_filter(
        ern_aligned, cash_aligned, ma_window=8, max_drawdown=0.12
    )
    trend_filtered, trend_signal = equity_curve_filter(
        trend_aligned, cash_aligned, ma_window=6, max_drawdown=0.10
    )
    meta_allocator_returns, meta_weights = strategy_allocator(
        ern_aligned, trend_aligned, cash_aligned, lookback=6, min_score=0.0
    )

    ern_metrics = collect_metrics(ern_aligned, ern_filtered, cash_aligned)
    trend_metrics = collect_metrics(trend_aligned, trend_filtered, cash_aligned)
    allocator_metrics = compute_metrics(meta_allocator_returns, cash_aligned)

    print("===== Meta Strategy Overlays: ERN vs. 3-Asset Trend =====")
    print(f"Common sample: {ern_aligned.index.min().date()} -> {ern_aligned.index.max().date()}")

    print("\nERN base metrics:")
    print(format_metrics(ern_metrics["base"]))
    print("\nERN equity-curve filter metrics:")
    print(format_metrics(ern_metrics["meta"]))

    print("\nTrend base metrics:")
    print(format_metrics(trend_metrics["base"]))
    print("\nTrend equity-curve filter metrics:")
    print(format_metrics(trend_metrics["meta"]))

    print("\nMeta allocator (switching between ERN / Trend / Cash):")
    print(format_metrics(allocator_metrics))

    print("\nRecent ERN filter signals (1=on):")
    print(ern_signal.tail().to_frame("ERN_on"))
    print("\nRecent Trend filter signals (1=on):")
    print(trend_signal.tail().to_frame("Trend_on"))
    print("\nRecent trend-model weights:")
    print(trend_weights.tail().round(3))
    if not meta_weights.empty:
        print("\nRecent meta allocator weights:")
        print(meta_weights.tail().round(3))
    plot_meta_performance(
        ern_aligned,
        ern_filtered,
        trend_aligned,
        trend_filtered,
        meta_allocator_returns,
    )
    plot_meta_weights(meta_weights)


if __name__ == "__main__":
    main()
