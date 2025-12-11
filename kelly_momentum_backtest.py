"""Kelly-style multi-horizon momentum backtest with optional cash sleeve.

Strategy spec (both universes):
* Assets v1: SP500, Gold, TLT, and Cash. Assets v2: SP500, Gold, TLT.
* Signals and covariance from daily data; rebalanced monthly.
* For each lookback horizon (1/3/6/12 months), compute Kelly fraction
  F = C^-1 * M where M is the mean daily return vector over the horizon
  and C is the daily return covariance over the same window.
* Average F across the four horizons, apply a half-Kelly safety factor,
  clamp negative exposures to zero, and cap exposures at 10x.
* Long-only interpretation (no shorting); zero exposures stay at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Start far enough back to cover the longest horizon with ETF history.
START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
HORIZON_MONTHS = [1, 3, 6, 12]
TRADING_DAYS_PER_MONTH = 21
HORIZON_DAYS = [m * TRADING_DAYS_PER_MONTH for m in HORIZON_MONTHS]
MAX_LOOKBACK_DAYS = max(HORIZON_DAYS)
HALF_KELLY_FACTOR = 0.5
MAX_EXPOSURE = 10.0
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Proxy tickers (prefer liquid, long-history series).
ASSET_CONFIG: Dict[str, Dict[str, Iterable[str]]] = {
    "SP500": {"tickers": ["^GSPC", "SPY"]},
    "GOLD": {"tickers": ["GLD", "IAU", "GC=F"]},
    "TLT": {"tickers": ["TLT", "IEF", "GOVT"]},
}

CASH_PROXIES: List[Tuple[str, str]] = [
    ("BIL", "price"),  # 1-3m T-Bills ETF
    ("SHV", "price"),  # Short Treasuries ETF
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
    raise ValueError(f"Could not download usable data for {asset}")


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


def to_wealth(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    wealth.name = "wealth"
    return wealth


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def compute_metrics(portfolio_returns: pd.Series, cash_returns: pd.Series) -> Dict[str, float]:
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
    num_years = len(portfolio_returns) / 252.0
    cagr = wealth.iloc[-1] ** (1.0 / num_years) - 1.0 if num_years > 0 else float("nan")
    vol = portfolio_returns.std() * math.sqrt(252)
    excess = (portfolio_returns - cash_returns).mean() * 252.0
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


def kelly_from_window(returns_window: pd.DataFrame) -> pd.Series:
    """Compute averaged, capped half-Kelly allocations across horizons."""
    fractions: List[pd.Series] = []
    for days in HORIZON_DAYS:
        if len(returns_window) < days:
            continue
        lookback = returns_window.tail(days)
        lookback = lookback.dropna(how="any")
        if len(lookback) < days:
            continue
        mean_returns = lookback.mean()
        cov = lookback.cov()
        if cov.isnull().values.any() or mean_returns.isnull().any():
            continue
        # Use the same-length window for both M (cumulative returns) and C (covariance).
        inv = np.linalg.pinv(cov.values)
        f_raw = inv.dot(mean_returns.values)
        fractions.append(pd.Series(f_raw, index=lookback.columns))

    if not fractions:
        return pd.Series(0.0, index=returns_window.columns)

    averaged = pd.concat(fractions, axis=1).mean(axis=1)
    averaged = averaged.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    averaged = averaged * HALF_KELLY_FACTOR
    averaged = averaged.clip(lower=0.0, upper=MAX_EXPOSURE)
    averaged[averaged == 0] = 0.0
    return averaged


def build_monthly_weights(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Return monthly Kelly allocations computed from daily data (no look-ahead)."""
    month_ends = daily_returns.resample("ME").last().index
    records: Dict[pd.Timestamp, pd.Series] = {}
    for date in month_ends:
        window = daily_returns.loc[:date]
        if len(window) < MAX_LOOKBACK_DAYS:
            continue
        weights = kelly_from_window(window)
        weights.name = date
        records[date] = weights
    if not records:
        return pd.DataFrame(columns=daily_returns.columns)
    weights_df = pd.DataFrame.from_dict(records, orient="index").sort_index()
    weights_df = weights_df.reindex(columns=daily_returns.columns).fillna(0.0)
    return weights_df


def build_daily_strategy_returns(
    monthly_weights: pd.DataFrame, daily_returns: pd.DataFrame
) -> pd.Series:
    if monthly_weights.empty:
        return pd.Series(dtype=float)
    # Shift weights to start the day after the month-end calculation to avoid look-ahead.
    shifted_weights = monthly_weights.copy()
    shifted_weights.index = shifted_weights.index + pd.offsets.BDay(1)
    weights_daily = shifted_weights.resample("D").ffill()
    start = weights_daily.index.min()
    aligned_returns = daily_returns.loc[start:]
    weights_daily = weights_daily.reindex(aligned_returns.index).ffill().fillna(0.0)
    aligned_returns = aligned_returns[weights_daily.columns]
    strategy = (weights_daily * aligned_returns).sum(axis=1)
    strategy.name = "strategy_daily"
    return strategy


def load_market_data(include_cash: bool) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    prices: Dict[str, pd.Series] = {}
    proxies: Dict[str, str] = {}
    for asset, config in ASSET_CONFIG.items():
        series, ticker = fetch_price_series(asset, config)
        prices[asset] = series
        proxies[asset] = ticker

    price_df = pd.concat(prices.values(), axis=1)
    price_df.columns = list(prices.keys())
    price_df = price_df.sort_index().dropna(how="any")
    asset_returns = price_df.pct_change().dropna()

    cash_returns, cash_ticker = fetch_cash_returns()
    proxies["CASH"] = cash_ticker

    common_index = asset_returns.index.intersection(cash_returns.index)
    asset_returns = asset_returns.loc[common_index]
    cash_returns = cash_returns.loc[common_index]

    if include_cash:
        asset_returns = pd.concat([asset_returns, cash_returns], axis=1)

    return asset_returns, cash_returns, proxies


def run_backtest(include_cash: bool) -> BacktestResult:
    daily_returns, cash_returns, proxies = load_market_data(include_cash=include_cash)
    monthly_weights = build_monthly_weights(daily_returns)
    strategy_daily = build_daily_strategy_returns(monthly_weights, daily_returns)

    rf_aligned = cash_returns.reindex(strategy_daily.index).fillna(0.0)
    metrics = compute_metrics(strategy_daily, rf_aligned)
    coverage = (
        strategy_daily.index.min().date() if not strategy_daily.empty else None,
        strategy_daily.index.max().date() if not strategy_daily.empty else None,
    )

    return BacktestResult(
        name="Kelly Momentum (with cash)" if include_cash else "Kelly Momentum (no cash)",
        returns=strategy_daily,
        wealth=to_wealth(strategy_daily),
        weights=monthly_weights,
        metrics=metrics,
        extras={"proxies": proxies, "coverage": coverage},
    )


def plot_wealth(results: List[BacktestResult], output_path: Path) -> None:
    curves = []
    for res in results:
        wealth = res.wealth
        if wealth.empty:
            continue
        curves.append(wealth.rename(res.name))
    if not curves:
        return
    wealth_df = pd.concat(curves, axis=1).dropna(how="all")
    ax = wealth_df.plot(figsize=(10, 6))
    ax.set_title("Kelly Momentum Wealth (with vs without cash)")
    ax.set_ylabel("Cumulative wealth (gross)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
    results: List[BacktestResult] = []
    for include_cash in (True, False):
        result = run_backtest(include_cash=include_cash)
        results.append(result)
        coverage = result.extras["coverage"]
        label = "WITH CASH" if include_cash else "NO CASH"
        print(f"\n===== {result.name} [{label}] =====")
        if all(coverage):
            print(f"Sample covers {coverage[0]} -> {coverage[1]}")
        print("\nProxies used:")
        for asset, ticker in result.extras["proxies"].items():
            print(f"  {asset:<6}: {ticker}")
        print("\nPerformance metrics:")
        print(format_metrics(result.metrics))
        print("\nMost recent weights (monthly, half-Kelly capped at 10x):")
        print(result.weights.tail().round(3))
    plot_path = OUTPUT_DIR / "kelly_momentum_wealth.png"
    plot_wealth(results, plot_path)
    print(f"\nSaved wealth plot to {plot_path}")


if __name__ == "__main__":
    main()
