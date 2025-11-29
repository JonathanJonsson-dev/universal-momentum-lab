"""Backtest a handcrafted Rob Carver style allocation with a diversification multiplier."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2005-01-01"
TICKERS = ["SPY", "GLD", "TLT"]
VOL_WINDOW = 30
CORR_WINDOW = 30
RISK_FREE_ANNUAL = 0.02
BASE_RISK_WEIGHTS = pd.Series(1.0 / len(TICKERS), index=TICKERS)
SHARPE_TARGET = 0.9
SHARPE_WINDOW = 60
SHARPE_MAX_LEVERAGE = 5.0
VOL_TARGET = 0.40
VOL_TARGET_WINDOW = 20
VOL_TARGET_MAX_LEVERAGE = 5.0


@dataclass
class BacktestResult:
    returns: pd.Series
    wealth: pd.Series
    metrics: Dict[str, float]
    weights: pd.DataFrame


def download_prices(tickers: List[str], start: str) -> pd.DataFrame:
    prices = yf.download(
        tickers, start=start, auto_adjust=True, progress=False, threads=False
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    return prices.dropna()


def daily_rate_from_annual(annual_rate: float) -> float:
    return math.exp(annual_rate / 252) - 1.0


def diversification_multiplier(corr_matrix: pd.DataFrame, weights: pd.Series) -> float:
    corr_values = corr_matrix.values.astype(float)
    np.fill_diagonal(corr_values, 1.0)
    weight_vec = weights.values.astype(float).reshape(1, -1)
    quad = float(weight_vec @ corr_values @ weight_vec.T)
    quad = max(quad, 1e-9)
    return 1.0 / math.sqrt(quad)


def build_groups(corr_matrix: pd.DataFrame, ordered_tickers: List[str]) -> List[List[str]]:
    """Pair the most correlated assets together (if correlation > 0) and leave the rest solo."""
    best_pair = None
    best_corr = float("-inf")
    for i, left in enumerate(ordered_tickers):
        for right in ordered_tickers[i + 1 :]:
            value = corr_matrix.loc[left, right]
            if pd.notna(value) and value > best_corr:
                best_corr = value
                best_pair = (left, right)
    if best_pair is None or best_corr <= 0.0:
        return [[ticker] for ticker in ordered_tickers]
    grouped = [list(best_pair)]
    grouped.extend([[ticker] for ticker in ordered_tickers if ticker not in best_pair])
    return grouped


def compute_adjusted_risk_weights(
    returns: pd.DataFrame,
    base_risk: pd.Series,
    vol_window: int,
    corr_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lagged_returns = returns.shift(1)
    rolling_vol = lagged_returns.rolling(window=vol_window).std()
    rolling_corr = lagged_returns.rolling(window=corr_window).corr()

    records: List[pd.Series] = []
    for date in returns.index:
        vol_row = rolling_vol.loc[date]
        if vol_row.isnull().any():
            records.append(pd.Series(np.nan, index=returns.columns))
            continue
        try:
            corr_matrix = rolling_corr.loc[date]
        except KeyError:
            records.append(pd.Series(np.nan, index=returns.columns))
            continue

        corr_matrix = corr_matrix.reindex(index=returns.columns, columns=returns.columns)
        if corr_matrix.isnull().values.all():
            records.append(pd.Series(np.nan, index=returns.columns))
            continue

        groups = build_groups(corr_matrix, returns.columns.tolist())
        group_info = []
        for members in groups:
            member_weights = base_risk.loc[members]
            base_sum = member_weights.sum()
            if base_sum <= 0:
                continue
            within = member_weights / base_sum
            if len(members) == 1:
                dm = 1.0
            else:
                corr_slice = corr_matrix.loc[members, members].fillna(0.0)
                np.fill_diagonal(corr_slice.values, 1.0)
                dm = diversification_multiplier(corr_slice, within)
            group_info.append((members, base_sum, dm, within))

        total = sum(weight * dm for _, weight, dm, _ in group_info)
        if total <= 0:
            records.append(pd.Series(np.nan, index=returns.columns))
            continue

        day_weights = pd.Series(0.0, index=returns.columns, dtype=float)
        for members, base_sum, dm, within in group_info:
            group_allocation = (base_sum * dm) / total
            for member in members:
                day_weights.loc[member] = group_allocation * within.loc[member]

        records.append(day_weights)

    adjusted = pd.DataFrame(records, index=returns.index)
    return adjusted, rolling_vol


def convert_risk_to_cash(
    risk_weights: pd.DataFrame, rolling_vol: pd.DataFrame
) -> pd.DataFrame:
    inv_vol = risk_weights.div(rolling_vol)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
    cash = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return cash.dropna(how="any")


def apply_sharpe_target_overlay(
    base_returns: pd.Series,
    target_sharpe: float,
    window: int,
    max_leverage: float,
    rf_rate: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    rolling_mean = base_returns.rolling(window=window).mean()
    rolling_std = base_returns.rolling(window=window).std().replace(0.0, np.nan)
    rolling_sharpe = (rolling_mean / rolling_std) * math.sqrt(252)
    lagged_sharpe = rolling_sharpe.shift(1)

    scaling = pd.Series(0.0, index=base_returns.index)
    valid = lagged_sharpe > 0
    scaling.loc[valid] = target_sharpe / lagged_sharpe.loc[valid]
    scaling = scaling.replace([np.inf, -np.inf], np.nan)
    scaling = scaling.clip(lower=0.0, upper=max_leverage)
    scaling = scaling.fillna(0.0)

    rf_daily = daily_rate_from_annual(rf_rate)
    overlay_returns = scaling * base_returns + (1.0 - scaling) * rf_daily
    return overlay_returns, scaling, lagged_sharpe


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).std() * math.sqrt(252)


def apply_vol_target_overlay(
    base_returns: pd.Series,
    target_vol: float,
    window: int,
    max_leverage: float,
    rf_rate: float,
) -> tuple[pd.Series, pd.Series]:
    realized = realized_vol(base_returns, window).shift(1)
    scaling = target_vol / realized
    scaling = scaling.replace([np.inf, -np.inf], np.nan)
    scaling = scaling.clip(lower=0.0, upper=max_leverage)
    scaling = scaling.fillna(0.0)
    rf_daily = daily_rate_from_annual(rf_rate)
    overlay_returns = scaling * base_returns + (1.0 - scaling) * rf_daily
    return overlay_returns, scaling


def max_drawdown(wealth: pd.Series) -> float:
    rolling_peak = wealth.cummax()
    drawdown = wealth / rolling_peak - 1.0
    return float(drawdown.min())


def compute_metrics(returns: pd.Series, rf_rate: float) -> Dict[str, float]:
    wealth = (1.0 + returns).cumprod()
    total_days = len(returns)
    cagr = wealth.iloc[-1] ** (252.0 / total_days) - 1.0
    vol = returns.std() * math.sqrt(252)
    excess_return = returns.mean() * 252 - rf_rate
    sharpe = excess_return / vol if vol > 0 else float("nan")
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown(wealth),
        "Terminal Wealth": wealth.iloc[-1],
    }


def run_backtest() -> BacktestResult:
    prices = download_prices(TICKERS, START_DATE)
    returns = prices.pct_change().dropna()

    risk_weights, rolling_vol = compute_adjusted_risk_weights(
        returns=returns,
        base_risk=BASE_RISK_WEIGHTS,
        vol_window=VOL_WINDOW,
        corr_window=CORR_WINDOW,
    )
    cash_weights = convert_risk_to_cash(risk_weights, rolling_vol)
    aligned_returns = returns.loc[cash_weights.index]
    portfolio_returns = (cash_weights * aligned_returns).sum(axis=1)
    wealth = (1.0 + portfolio_returns).cumprod()
    metrics = compute_metrics(portfolio_returns, RISK_FREE_ANNUAL)
    return BacktestResult(
        returns=portfolio_returns,
        wealth=wealth,
        metrics=metrics,
        weights=cash_weights,
    )


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
    for key, value in metrics.items():
        if math.isnan(value):
            display = "nan"
        elif key in {"CAGR", "Volatility", "Sharpe"}:
            display = f"{value:.3f}"
        elif key == "Max Drawdown":
            display = f"{value:.3f}"
        else:
            display = f"{value:.2f}"
        lines.append(f"{key:>15}: {display}")
    return "\n".join(lines)


def main() -> None:
    result = run_backtest()
    print("===== Handcrafted Diversified Backtest =====")
    print(format_metrics(result.metrics))
    print("\nLast available weights:")
    print(result.weights.tail().round(4))
    print(f"\nSample covers {result.returns.index.min().date()} -> {result.returns.index.max().date()}")
    overlay_returns, overlay_scaling, rolling_sharpe = apply_sharpe_target_overlay(
        base_returns=result.returns,
        target_sharpe=SHARPE_TARGET,
        window=SHARPE_WINDOW,
        max_leverage=SHARPE_MAX_LEVERAGE,
        rf_rate=RISK_FREE_ANNUAL,
    )
    overlay_metrics = compute_metrics(overlay_returns, RISK_FREE_ANNUAL)
    print("\n===== Rolling Sharpe Target Overlay =====")
    print(format_metrics(overlay_metrics))
    print(
        "\nLeverage stats "
        f"(mean {overlay_scaling.mean():.2f}, "
        f"median {overlay_scaling.median():.2f}, "
        f"95th pct {overlay_scaling.quantile(0.95):.2f})"
    )
    latest_signal = rolling_sharpe.dropna().tail(5)
    if not latest_signal.empty:
        print("\nRecent rolling Sharpe estimates:")
        print(latest_signal.round(3))

    vt_returns, vt_scaling = apply_vol_target_overlay(
        base_returns=result.returns,
        target_vol=VOL_TARGET,
        window=VOL_TARGET_WINDOW,
        max_leverage=VOL_TARGET_MAX_LEVERAGE,
        rf_rate=RISK_FREE_ANNUAL,
    )
    vt_metrics = compute_metrics(vt_returns, RISK_FREE_ANNUAL)
    print("\n===== Fixed 40% Vol Target Overlay =====")
    print(format_metrics(vt_metrics))
    print(
        "\nTarget leverage stats "
        f"(mean {vt_scaling.mean():.2f}, "
        f"median {vt_scaling.median():.2f}, "
        f"95th pct {vt_scaling.quantile(0.95):.2f})"
    )


if __name__ == "__main__":
    main()
