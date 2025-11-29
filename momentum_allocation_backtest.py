"""Multi-horizon momentum backtest approximating the Early Retirement Now setup.

The original blog post relies on an offline dataset that begins in 1871.
Yahoo Finance cannot deliver that entire history, so this script fetches the
longest proxy series it can find for each asset class and documents which
tickers were used. The implementation follows the described logic:

* Monthly data
* 3 momentum horizons (8/9/10 months)
* 2 momentum formulas (level vs. rolling average and level vs. t-N)
* 2 index variants (raw index and index in excess of cash)
* Sequential re-allocation of unused weights: Gold -> Equity -> Bonds -> Cash
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "1871-01-01"
END_DATE = "2025-11-12"
HORIZONS = [8, 9, 10]
BASE_WEIGHTS = {"EQUITY": 0.70, "BONDS": 0.20, "GOLD": 0.10}
VOL_TARGET = 0.90
VOL_WINDOW_DAYS = 30
VOL_MAX_LEVERAGE = 99

# Proxies first try to use total-return style indices, then fall back to ETFs.
ASSET_CONFIG: Dict[str, Dict[str, Iterable[str]]] = {
    "EQUITY": {"tickers": ["^SP500TR", "^GSPC", "SPY"]},
    "BONDS": {"tickers": ["IEF", "GOVT", "TLT"]},
    "GOLD": {"tickers": ["GLD", "IAU"]},
}

CASH_PROXIES: List[Tuple[str, str]] = [
    ("BIL", "price"),  # iShares 1-3 month T-Bill ETF
    ("SHV", "price"),  # Short Treasury ETF
    ("^IRX", "yield"),  # 13-week T-Bill yield (annualized percent)
]


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    wealth: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]
    extras: Dict[str, object]


def download_single_ticker(
    ticker: str, start: str, end: str, auto_adjust: bool = True
) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer extracting the field level so column names remain OHLC labels.
        try:
            data = data.xs(ticker, axis=1, level=-1)
        except KeyError:
            data = data.droplevel(0, axis=1)

    preferred = ["Adj Close", "Close", "close", "adjclose"]
    column = next((col for col in preferred if col in data.columns), None)
    if column is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(dtype=float, name=ticker)
        column = numeric_cols[0]

    series = data[column].astype(float).dropna()
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")
    series.name = ticker
    return series


def fetch_price_series(asset: str, config: Dict[str, Iterable[str]]) -> Tuple[pd.Series, str]:
    for ticker in config["tickers"]:
        series = download_single_ticker(ticker, START_DATE, END_DATE, auto_adjust=True)
        if series.empty:
            continue
        series = series.dropna()
        if series.empty:
            continue
        series.name = asset
        return series, ticker
    raise ValueError(f"Could not download any usable data for {asset}")


def fetch_cash_returns() -> Tuple[pd.Series, str]:
    for ticker, kind in CASH_PROXIES:
        auto_adjust = kind == "price"
        series = download_single_ticker(ticker, START_DATE, END_DATE, auto_adjust=auto_adjust)
        if series.empty:
            continue
        series = series.dropna()
        if series.empty:
            continue
        if kind == "price":
            returns = series.pct_change().dropna()
        else:
            # Convert annualized percent yield to forward-looking daily return.
            returns = ((1.0 + series / 100.0) ** (1.0 / 252.0)) - 1.0
            returns = returns.dropna()
        if returns.empty:
            continue
        returns.name = "CASH"
        return returns, ticker
    raise ValueError("Unable to build cash return series from configured proxies.")


def daily_to_monthly_returns(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return returns
    monthly = (1.0 + returns).resample("M").prod() - 1.0
    return monthly.dropna()


def normalize_price(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series
    first_valid = clean.iloc[0]
    if first_valid == 0:
        return series
    return series / first_valid


def momentum_vs_average(
    index_series: pd.Series, horizon: int, use_two_month_avg: bool
) -> pd.Series:
    rolling = index_series.rolling(window=horizon, min_periods=horizon).mean()
    if use_two_month_avg:
        front = (index_series + index_series.shift(1)) / 2.0
    else:
        front = index_series
    return front - rolling


def momentum_n_month_change(index_series: pd.Series, horizon: int) -> pd.Series:
    previous = index_series.shift(horizon)
    return index_series - previous


def compute_asset_signal(
    price_index: pd.Series,
    cash_index: pd.Series,
    use_two_month_avg: bool,
) -> pd.Series:
    normalized = normalize_price(price_index)
    normalized, cash_index = normalized.align(cash_index, join="inner")
    excess_index = normalized / cash_index
    signal_frames: List[pd.Series] = []
    for index_variant in (normalized, excess_index):
        for horizon in HORIZONS:
            ma_signal = momentum_vs_average(index_variant, horizon, use_two_month_avg)
            n_signal = momentum_n_month_change(index_variant, horizon)
            signal_frames.append((ma_signal > 0).astype(float))
            signal_frames.append((n_signal > 0).astype(float))
    combined = pd.concat(signal_frames, axis=1)
    return combined.mean(axis=1, skipna=True)


def build_momentum_scores(
    price_levels: pd.DataFrame, cash_returns: pd.Series
) -> pd.DataFrame:
    cash_index = (1.0 + cash_returns).cumprod()
    cash_index = cash_index / cash_index.iloc[0]
    scores: Dict[str, pd.Series] = {}
    for asset in price_levels.columns:
        use_two_month_avg = asset == "EQUITY"
        scores[asset] = compute_asset_signal(
            price_index=price_levels[asset],
            cash_index=cash_index,
            use_two_month_avg=use_two_month_avg,
        )
    return pd.DataFrame(scores)


def sequential_allocation(momentum_row: pd.Series) -> pd.Series:
    gold_signal = float(np.clip(momentum_row.get("GOLD", 0.0), 0.0, 1.0))
    equity_signal = float(np.clip(momentum_row.get("EQUITY", 0.0), 0.0, 1.0))
    bond_signal = float(np.clip(momentum_row.get("BONDS", 0.0), 0.0, 1.0))

    gold_weight = BASE_WEIGHTS["GOLD"] * gold_signal
    leftover_gold = BASE_WEIGHTS["GOLD"] - gold_weight

    equity_base = BASE_WEIGHTS["EQUITY"] + leftover_gold
    equity_weight = equity_base * equity_signal
    leftover_equity = equity_base - equity_weight

    bond_base = BASE_WEIGHTS["BONDS"] + leftover_equity
    bond_weight = bond_base * bond_signal
    cash_weight = bond_base - bond_weight

    return pd.Series(
        {
            "EQUITY": equity_weight,
            "BONDS": bond_weight,
            "GOLD": gold_weight,
            "CASH": cash_weight,
        }
    )


def allocate_weights(momentum_scores: pd.DataFrame) -> pd.DataFrame:
    records = {date: sequential_allocation(row) for date, row in momentum_scores.iterrows()}
    weights = pd.DataFrame.from_dict(records, orient="index")
    return weights


def build_daily_strategy_returns(
    monthly_weights: pd.DataFrame,
    daily_asset_returns: pd.DataFrame,
    daily_cash_returns: pd.Series,
) -> pd.Series:
    if monthly_weights.empty or daily_asset_returns.empty:
        return pd.Series(dtype=float)
    weights_daily = monthly_weights.resample("D").ffill()
    common_index = daily_asset_returns.index.intersection(weights_daily.index)
    weights_daily = weights_daily.loc[common_index]
    daily_asset_returns = daily_asset_returns.loc[common_index]
    daily_cash_returns = daily_cash_returns.reindex(common_index).fillna(0.0)
    asset_cols = ["EQUITY", "BONDS", "GOLD"]
    strategy = (weights_daily[asset_cols] * daily_asset_returns[asset_cols]).sum(axis=1)
    strategy = strategy + weights_daily["CASH"] * daily_cash_returns
    strategy.name = "strategy_daily"
    return strategy


def to_wealth(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    wealth.name = "wealth"
    return wealth


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def compute_metrics(
    portfolio_returns: pd.Series, cash_returns: pd.Series
) -> Dict[str, float]:
    if portfolio_returns.empty:
        nan = float("nan")
        return {
            "CAGR": nan,
            "Volatility": nan,
            "Sharpe": nan,
            "Max Drawdown": nan,
            "Calmar": nan,
            "Terminal Wealth": nan,
        }
    wealth = to_wealth(portfolio_returns)
    num_years = len(portfolio_returns) / 12.0
    cagr = wealth.iloc[-1] ** (1.0 / num_years) - 1.0 if num_years > 0 else float("nan")
    vol = portfolio_returns.std() * math.sqrt(12)
    excess = (portfolio_returns - cash_returns).mean() * 12.0
    sharpe = excess / vol if vol > 0 else float("nan")
    mdd = max_drawdown(wealth)
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "Terminal Wealth": wealth.iloc[-1],
    }


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).std() * math.sqrt(252)


def apply_vol_target_overlay(
    base_returns: pd.Series,
    target_vol: float,
    window: int,
    rf_returns: pd.Series,
    max_leverage: float,
) -> Tuple[pd.Series, pd.Series]:
    if base_returns.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    rf_aligned = rf_returns.reindex(base_returns.index).fillna(0.0)
    realized = realized_vol(base_returns, window).shift(1)
    scaling = target_vol / realized
    scaling = scaling.replace([np.inf, -np.inf], np.nan)
    scaling = scaling.clip(lower=0.0, upper=max_leverage)
    scaling = scaling.fillna(1.0)
    overlay = scaling * base_returns + (1.0 - scaling) * rf_aligned
    overlay.name = "vol_target_daily"
    return overlay, scaling


def load_market_data() -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    daily_prices: Dict[str, pd.Series] = {}
    proxies: Dict[str, str] = {}
    for asset, config in ASSET_CONFIG.items():
        series, ticker = fetch_price_series(asset, config)
        daily_prices[asset] = series
        proxies[asset] = ticker
    prices = pd.concat(daily_prices.values(), axis=1)
    prices.columns = list(daily_prices.keys())
    prices = prices.sort_index().dropna(how="any")

    cash_returns, cash_ticker = fetch_cash_returns()
    proxies["CASH"] = cash_ticker
    common_index = prices.index.intersection(cash_returns.index)
    prices = prices.loc[common_index]
    cash_returns = cash_returns.loc[common_index]
    return prices, cash_returns, proxies


def run_backtest() -> BacktestResult:
    daily_prices, daily_cash_returns, proxies = load_market_data()

    monthly_prices = daily_prices.resample("M").last().dropna(how="any")
    monthly_cash_returns = daily_to_monthly_returns(daily_cash_returns)
    common_months = monthly_prices.index.intersection(monthly_cash_returns.index)
    monthly_prices = monthly_prices.loc[common_months]
    monthly_cash_returns = monthly_cash_returns.loc[common_months]

    asset_returns = monthly_prices.pct_change().dropna()
    cash_returns = monthly_cash_returns.loc[asset_returns.index]

    momentum_scores = build_momentum_scores(monthly_prices, monthly_cash_returns)
    momentum_scores = momentum_scores.loc[asset_returns.index]
    momentum_scores = momentum_scores.shift(1).dropna()

    asset_returns = asset_returns.loc[momentum_scores.index]
    cash_returns = cash_returns.loc[momentum_scores.index]

    weights = allocate_weights(momentum_scores).loc[momentum_scores.index]

    asset_cols = ["EQUITY", "BONDS", "GOLD"]
    portfolio_returns = (weights[asset_cols] * asset_returns[asset_cols]).sum(axis=1)
    portfolio_returns += weights["CASH"] * cash_returns
    portfolio_returns.name = "strategy"

    metrics = compute_metrics(portfolio_returns, cash_returns)

    daily_asset_returns = daily_prices.pct_change().dropna()
    daily_asset_returns = daily_asset_returns.loc[
        daily_asset_returns.index.intersection(daily_cash_returns.index)
    ]
    daily_cash_returns = daily_cash_returns.loc[daily_asset_returns.index]

    valid_periods = weights.index.to_period("M")
    daily_periods = daily_asset_returns.index.to_period("M")
    mask = daily_periods.isin(valid_periods)
    daily_asset_returns = daily_asset_returns.loc[mask]
    daily_cash_returns = daily_cash_returns.loc[daily_asset_returns.index]

    daily_strategy_returns = build_daily_strategy_returns(
        weights,
        daily_asset_returns,
        daily_cash_returns,
    )
    vt_daily, vt_scaling = apply_vol_target_overlay(
        daily_strategy_returns,
        target_vol=VOL_TARGET,
        window=VOL_WINDOW_DAYS,
        rf_returns=daily_cash_returns,
        max_leverage=VOL_MAX_LEVERAGE,
    )
    vt_monthly = daily_to_monthly_returns(vt_daily)
    vt_monthly = vt_monthly.loc[portfolio_returns.index]
    vt_metrics = compute_metrics(vt_monthly, cash_returns)

    return BacktestResult(
        name="Multi-Horizon Momentum Allocation",
        returns=portfolio_returns,
        wealth=to_wealth(portfolio_returns),
        weights=weights,
        metrics=metrics,
        extras={
            "proxies": proxies,
            "momentum_scores": momentum_scores,
            "coverage": (
                portfolio_returns.index.min().date(),
                portfolio_returns.index.max().date(),
            ),
            "vol_target": {
                "daily_returns": vt_daily,
                "monthly_returns": vt_monthly,
                "scaling": vt_scaling,
                "metrics": vt_metrics,
            },
        },
    )


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
    result = run_backtest()
    start, end = result.extras["coverage"]
    print("===== Multi-Horizon Momentum Allocation =====")
    print(f"Sample covers {start} -> {end}")
    print("\nProxies used:")
    for asset, ticker in result.extras["proxies"].items():
        print(f"  {asset:<6}: {ticker}")
    print("\nPerformance metrics:")
    print(format_metrics(result.metrics))
    print("\nMost recent weights:")
    print(result.weights.tail().round(3))
    vt_info = result.extras.get("vol_target")
    if vt_info:
        print("\n===== 30-Day Vol Target Overlay =====")
        print(format_metrics(vt_info["metrics"]))
        scaling = vt_info["scaling"].dropna()
        if not scaling.empty:
            print(
                "\nOverlay leverage stats "
                f"(mean {scaling.mean():.2f}, "
                f"median {scaling.median():.2f}, "
                f"95th pct {scaling.quantile(0.95):.2f})"
            )


if __name__ == "__main__":
    main()
