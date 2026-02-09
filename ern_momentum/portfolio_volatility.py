"""Compute annualized volatility for static SP500/Gold/TLT portfolios."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, Iterable, Tuple

import pandas as pd
import yfinance as yf


START_DATE = "1990-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
TRADING_DAYS = 252

TICKER_CANDIDATES: Dict[str, Iterable[str]] = {
    "SP500": ["^SP500TR", "^GSPC", "SPY"],
    "GOLD": ["GLD", "IAU", "GC=F"],
    "TLT": ["TLT"],
}

PORTFOLIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Portfolio 1": {"SP500": 0.1836, "GOLD": 0.0931, "TLT": 0.7233},
    "Portfolio 2": {"SP500": 0.33, "GOLD": 0.33, "TLT": 0.33},
    "Portfolio 3": {"SP500": 0.70, "GOLD": 0.10, "TLT": 0.20},
}


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
        try:
            data = data.xs(ticker, axis=1, level=-1)
        except KeyError:
            data = data.droplevel(0, axis=1)

    preferred = ["Adj Close", "Close", "close", "adjclose"]
    column = next((col for col in preferred if col in data.columns), None)
    if column is None:
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) == 0:
            return pd.Series(dtype=float, name=ticker)
        column = numeric_cols[0]

    series = data[column].astype(float).dropna()
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")
    series.name = ticker
    return series


def select_longest_series(
    name: str, candidates: Iterable[str], start: str, end: str
) -> Tuple[pd.Series, str]:
    best_series = pd.Series(dtype=float)
    best_ticker = ""
    best_span = pd.Timedelta(0)

    for ticker in candidates:
        series = download_single_ticker(ticker, start, end, auto_adjust=True)
        if series.empty:
            continue
        span = series.index[-1] - series.index[0]
        if best_series.empty or span > best_span:
            best_series = series
            best_ticker = ticker
            best_span = span

    if best_series.empty:
        raise ValueError(f"No usable data found for {name} with tickers {list(candidates)}")

    return best_series.rename(name), best_ticker


def load_price_history() -> Tuple[pd.DataFrame, Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {}
    proxies: Dict[str, str] = {}
    for asset, candidates in TICKER_CANDIDATES.items():
        series, ticker = select_longest_series(asset, candidates, START_DATE, END_DATE)
        series_map[asset] = series
        proxies[asset] = ticker

    prices = pd.concat(series_map.values(), axis=1)
    prices = prices.dropna(how="any")
    return prices, proxies


def to_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def normalize_weights(weights: Dict[str, float]) -> pd.Series:
    series = pd.Series(weights, dtype=float)
    total = float(series.sum())
    if total == 0.0:
        raise ValueError("Weights must sum to a non-zero value.")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        series = series / total
    return series


def align_weights(weights: Dict[str, float], assets: Iterable[str]) -> pd.Series:
    asset_list = list(assets)
    extra = set(weights) - set(asset_list)
    if extra:
        raise ValueError(f"Weights include unknown assets: {sorted(extra)}")
    aligned = normalize_weights(weights).reindex(asset_list)
    if aligned.isnull().any():
        missing = aligned[aligned.isnull()].index.tolist()
        raise ValueError(f"Missing weights for assets: {missing}")
    return aligned


def portfolio_volatility(returns: pd.DataFrame, weights: Dict[str, float]) -> float:
    aligned_weights = align_weights(weights, returns.columns)
    portfolio_returns = returns.dot(aligned_weights)
    return float(portfolio_returns.std(ddof=0) * math.sqrt(TRADING_DAYS))


def format_weights(weights: pd.Series) -> str:
    return ", ".join(f"{asset} {weight * 100.0:.2f}%" for asset, weight in weights.items())


def main() -> None:
    prices, proxies = load_price_history()
    returns = to_daily_returns(prices)
    if returns.empty:
        raise ValueError("No overlapping return history available.")

    start = returns.index.min().date()
    end = returns.index.max().date()
    proxy_line = ", ".join(f"{asset}={ticker}" for asset, ticker in proxies.items())

    print("Using proxies:", proxy_line)
    print(f"Sample window: {start} to {end} ({len(returns)} observations)")
    print(f"Annualized volatility from daily returns (sqrt({TRADING_DAYS}))")

    for name, weights in PORTFOLIO_WEIGHTS.items():
        aligned_weights = align_weights(weights, returns.columns)
        vol = portfolio_volatility(returns, weights)
        print(f"{name}: {vol:.2%} | {format_weights(aligned_weights)}")


if __name__ == "__main__":
    main()
