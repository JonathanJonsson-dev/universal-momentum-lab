"""Compare the ERN multi-horizon momentum strategy with a simple trend rule.

Trend rule (from the quoted article):
* Universe: EQUITY, BONDS, GOLD.
* Signal: 3-month SMA > 10-month SMA on monthly closes.
* Allocate equally across assets with a positive signal; leftover goes to cash.
* Same cost assumptions as the ERN script (expense drag + 0.03% per-side t-cost).

Outputs: metrics for both strategies on the common sample, plus latest weights for
the simple trend rule.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Tuple

import pandas as pd

from ern_momentum_blog_backtest import (
    apply_expense_drag,
    apply_transaction_costs,
    compute_metrics,
    daily_to_monthly_returns,
    EXPENSE_RATIOS,
    OUTPUT_DIR,
    load_market_data,
    run_backtest,
    to_wealth,
)


ASSETS = ["EQUITY", "BONDS", "GOLD"]


def build_trend_weights(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight assets with 3m SMA above 10m SMA; leftover to cash."""
    sma_short = monthly_prices.rolling(window=3, min_periods=3).mean()
    sma_long = monthly_prices.rolling(window=10, min_periods=10).mean()
    signal = sma_short > sma_long

    def _row_weights(row: pd.Series) -> pd.Series:
        positives = row[row].index.tolist()
        if len(positives) == 0:
            weights = {asset: 0.0 for asset in ASSETS}
            weights["CASH"] = 1.0
            return pd.Series(weights)
        w = 1.0 / len(positives)
        weights = {asset: (w if asset in positives else 0.0) for asset in ASSETS}
        weights["CASH"] = 1.0 - len(positives) * w
        return pd.Series(weights)

    records = {date: _row_weights(row) for date, row in signal.iterrows()}
    return pd.DataFrame.from_dict(records, orient="index")


def compute_trend_returns(
    prices: pd.DataFrame, cash_daily: pd.Series
) -> Tuple[pd.Series, pd.DataFrame]:
    monthly_prices = prices.resample("ME").last().dropna(how="any")
    monthly_cash = daily_to_monthly_returns(cash_daily)
    common_months = monthly_prices.index.intersection(monthly_cash.index)
    monthly_prices = monthly_prices.loc[common_months]
    monthly_cash = monthly_cash.loc[common_months]

    asset_returns = monthly_prices.pct_change().dropna()
    monthly_cash = monthly_cash.loc[asset_returns.index]

    weights = build_trend_weights(monthly_prices).loc[asset_returns.index]
    asset_returns, monthly_cash = apply_expense_drag(asset_returns, monthly_cash)

    gross = (weights[ASSETS] * asset_returns[ASSETS]).sum(axis=1)
    gross = gross + weights["CASH"] * monthly_cash
    t_cost = apply_transaction_costs(weights)
    net = gross - t_cost
    net.name = "trend_strategy"
    return net, weights


def align_returns(
    ern_returns: pd.Series, trend_returns: pd.Series, cash_returns: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    common = ern_returns.index.intersection(trend_returns.index).intersection(cash_returns.index)
    return ern_returns.loc[common], trend_returns.loc[common], cash_returns.loc[common]


def plot_comparison(
    ern_returns: pd.Series,
    trend_returns: pd.Series,
    combined_returns: pd.Series | None = None,
    equal_risk_returns: pd.Series | None = None,
    static_returns: pd.Series | None = None,
    filtered_returns: pd.Series | None = None,
    outfile: str = "ern_vs_trend_wealth.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping comparison plot.")
        return

    ern_wealth = to_wealth(ern_returns)
    trend_wealth = to_wealth(trend_returns)
    common = ern_wealth.index.intersection(trend_wealth.index)
    ern_wealth = ern_wealth.loc[common]
    trend_wealth = trend_wealth.loc[common]

    outfile_path = OUTPUT_DIR / outfile
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ern_wealth.index, ern_wealth.values, label="ERN momentum", linewidth=1.5)
    ax.plot(trend_wealth.index, trend_wealth.values, label="3m>10m trend (EW)", linewidth=1.5)
    if combined_returns is not None and not combined_returns.empty:
        combo_wealth = to_wealth(combined_returns).loc[common]
        ax.plot(
            combo_wealth.index,
            combo_wealth.values,
            label="Combined (17.6% ERN / 82.4% Trend)",
            linewidth=1.5,
            linestyle="--",
        )
    if equal_risk_returns is not None and not equal_risk_returns.empty:
        er_common = common.intersection(equal_risk_returns.index)
        er_wealth = to_wealth(equal_risk_returns).loc[er_common]
        if er_wealth.empty:
            er_wealth = None
        ax.plot(
            er_wealth.index,
            er_wealth.values,
            label="Equal-risk (30-period)",
            linewidth=1.5,
            linestyle=":",
        ) if er_wealth is not None else None
    if static_returns is not None and not static_returns.empty:
        st_common = common.intersection(static_returns.index)
        st_wealth = to_wealth(static_returns).loc[st_common]
        ax.plot(
            st_wealth.index,
            st_wealth.values,
            label="50/50 static",
            linewidth=1.3,
            linestyle="-.",
        )
    if filtered_returns is not None and not filtered_returns.empty:
        filt_common = common.intersection(filtered_returns.index)
        filt_wealth = to_wealth(filtered_returns).loc[filt_common]
        ax.plot(
            filt_wealth.index,
            filt_wealth.values,
            label="50/50 with 10m off filter",
            linewidth=1.3,
            linestyle="--",
        )
    ax.set_title("Cumulative Wealth: ERN Momentum vs. 3m>10m Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(outfile_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {outfile_path}")


def compute_kelly_allocation(ern_returns: pd.Series, trend_returns: pd.Series) -> pd.Series:
    df = pd.concat({"ERN": ern_returns, "Trend": trend_returns}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="kelly_fraction")
    mean = df.mean().values  # per-period mean returns
    cov = df.cov().values
    inv_cov = np.linalg.pinv(cov)
    fractions = inv_cov @ mean
    series = pd.Series(fractions, index=df.columns, name="kelly_fraction")
    total = series.abs().sum()
    if total > 0:
        series = series / total
    return series


def compute_equal_risk_weights(
    ern_returns: pd.Series, trend_returns: pd.Series, window: int = 30
) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.concat({"ERN": ern_returns, "Trend": trend_returns}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="equal_risk_returns"), pd.DataFrame()
    rolling_vol = df.rolling(window=window).std()
    weights_list = []
    for date, vols in rolling_vol.iterrows():
        if vols.isnull().any():
            weights_list.append(pd.Series(index=df.columns, dtype=float, name=date))
            continue
        inv_vol = 1.0 / vols.replace(0, np.nan)
        weight = inv_vol / inv_vol.sum()
        weights_list.append(weight.rename(date))
    weights = pd.DataFrame(weights_list).dropna()
    common = df.index.intersection(weights.index)
    weights = weights.loc[common]
    aligned_returns = df.loc[common]
    portfolio = (weights * aligned_returns).sum(axis=1)
    portfolio.name = "equal_risk_returns"
    return portfolio, weights


def compute_static_mix(
    ern_returns: pd.Series, trend_returns: pd.Series, ern_weight: float = 0.5
) -> pd.Series:
    aligned = pd.concat({"ERN": ern_returns, "Trend": trend_returns}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="static_mix")
    w = {"ERN": ern_weight, "Trend": 1.0 - ern_weight}
    mix = aligned["ERN"] * w["ERN"] + aligned["Trend"] * w["Trend"]
    mix.name = "static_mix"
    return mix


def compute_off_filter(
    ern_returns: pd.Series,
    trend_returns: pd.Series,
    cash_returns: pd.Series,
    window: int = 10,
    base_weights: Tuple[float, float] = (0.5, 0.5),
) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.concat({"ERN": ern_returns, "Trend": trend_returns}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="filtered_mix"), pd.DataFrame()
    wealth = to_wealth(df)
    sma = wealth.shift(1).rolling(window=window, min_periods=window).mean()
    signals = wealth > sma

    def row_weights(sig_row: pd.Series) -> pd.Series:
        active = sig_row[sig_row].index.tolist()
        weights = {k: 0.0 for k in df.columns}
        if not active:
            weights["CASH"] = 1.0
            return pd.Series(weights)
        raw = {df.columns[0]: base_weights[0], df.columns[1]: base_weights[1]}
        total = sum(raw[a] for a in active)
        for a in active:
            weights[a] = raw[a] / total if total > 0 else 0.0
        weights["CASH"] = 0.0
        return pd.Series(weights)

    records = {date: row_weights(sig_row) for date, sig_row in signals.iterrows()}
    wts = pd.DataFrame.from_dict(records, orient="index").dropna()
    common = df.index.intersection(wts.index).intersection(cash_returns.index)
    wts = wts.loc[common]
    aligned_rets = df.loc[common]
    cash_aligned = cash_returns.loc[common]
    gross = (wts[df.columns] * aligned_rets).sum(axis=1) + wts["CASH"] * cash_aligned
    gross.name = "filtered_mix"
    return gross, wts


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for key, value in metrics.items():
        if math.isnan(value):
            display = "nan"
        elif key in {"CAGR", "Volatility", "Sharpe", "Calmar"}:
            display = f"{value:.3f}"
        elif key == "Max Drawdown":
            display = f"{value:.3f}"
        else:
            display = f"{value:.2f}"
        lines.append(f"{key:>15}: {display}")
    return "\n".join(lines)


def main() -> None:
    ern_result = run_backtest()
    prices, cash_daily, _ = load_market_data()
    trend_returns, trend_weights = compute_trend_returns(prices, cash_daily)

    # Use ERN cash returns (already expense-adjusted inside compute_metrics) for Sharpe.
    # Align to common sample for fair comparison.
    common_cash = ern_result.extras["coverage"]  # dates only, so recompute series
    monthly_cash = daily_to_monthly_returns(cash_daily)
    monthly_cash = monthly_cash - EXPENSE_RATIOS["CASH"] / 12.0
    monthly_cash = monthly_cash.loc[trend_returns.index.intersection(monthly_cash.index)]

    ern_aligned, trend_aligned, cash_aligned = align_returns(
        ern_result.returns, trend_returns, monthly_cash
    )
    combo_weights = {"ERN": 0.176, "Trend": 0.824}
    combined_returns = combo_weights["ERN"] * ern_aligned + combo_weights["Trend"] * trend_aligned
    equal_risk_returns, equal_risk_weights = compute_equal_risk_weights(
        ern_aligned, trend_aligned, window=30
    )
    er_common = equal_risk_returns.index.intersection(cash_aligned.index)
    equal_risk_returns = equal_risk_returns.loc[er_common]
    cash_equal_risk = cash_aligned.loc[er_common]
    static_mix = compute_static_mix(ern_aligned, trend_aligned, ern_weight=0.5)
    filtered_mix, filtered_weights = compute_off_filter(
        ern_aligned,
        trend_aligned,
        cash_aligned,
        window=10,
        base_weights=(0.5, 0.5),
    )

    ern_metrics = compute_metrics(ern_aligned, cash_aligned)
    trend_metrics = compute_metrics(trend_aligned, cash_aligned)
    combined_metrics = compute_metrics(combined_returns, cash_aligned)
    equal_risk_metrics = compute_metrics(equal_risk_returns, cash_equal_risk)
    static_metrics = compute_metrics(static_mix, cash_aligned)
    filtered_metrics = compute_metrics(filtered_mix, cash_aligned)
    corr = ern_aligned.corr(trend_aligned) if not ern_aligned.empty else float("nan")
    kelly = compute_kelly_allocation(ern_aligned, trend_aligned)

    print("===== ERN Momentum vs. 3m>10m Trend (Equal Weight) =====")
    print(f"Common sample: {ern_aligned.index.min().date()} -> {ern_aligned.index.max().date()}")
    print("\nERN momentum metrics:")
    print(format_metrics(ern_metrics))
    print("\nTrend-following metrics:")
    print(format_metrics(trend_metrics))
    print(f"\nMonthly return correlation: {corr:.3f}" if not math.isnan(corr) else "\nMonthly return correlation: nan")
    print("\nCombined (17.6% ERN / 82.4% Trend) metrics:")
    print(format_metrics(combined_metrics))
    print("\nEqual-risk (30-period inverse-vol) metrics:")
    print(format_metrics(equal_risk_metrics))
    print("\n50/50 static mix metrics:")
    print(format_metrics(static_metrics))
    print("\n50/50 with 10-month off filter metrics:")
    print(format_metrics(filtered_metrics))
    if not kelly.empty:
        print("\nContinuous Kelly fractions (per period, raw):")
        print(kelly.apply(lambda x: f"{x:.3f}"))
    print("\nMost recent trend weights:")
    print(trend_weights.tail().round(3))
    if not equal_risk_weights.empty:
        print("\nMost recent equal-risk weights:")
        print(equal_risk_weights.tail().round(3))
    if not filtered_weights.empty:
        print("\nMost recent filtered-mix weights:")
        print(filtered_weights.tail().round(3))
    plot_comparison(
        ern_aligned,
        trend_aligned,
        combined_returns=combined_returns,
        equal_risk_returns=equal_risk_returns,
        static_returns=static_mix,
        filtered_returns=filtered_mix,
    )


if __name__ == "__main__":
    main()
