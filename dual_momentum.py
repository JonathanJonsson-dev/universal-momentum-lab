from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf


START_DATE = "1970-01-01"
LOOKBACK_MONTHS = 12
ABS_MOM_THRESHOLD = 0.0

# Proxies chosen to mirror Antonacci's data hierarchy while constrained to Yahoo Finance.
ASSET_PROXIES: Dict[str, List[str]] = {
    "US": ["^GSPC", "SPY"],
    "INTL": ["^MSCIACWI", "^MSEAFE", "^MSCIWORLD", "ACWX", "VEU"],
    "BONDS": ["^LBUSTRUU", "AGG", "BND"],
}


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    wealth: pd.Series
    weights: pd.DataFrame
    metrics: Dict[str, float]
    extras: Dict[str, object] | None = None


def download_single_ticker(ticker: str, start: str) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        return pd.Series(dtype=float, name=ticker)
    series = data["Close"].dropna()
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")
    series.name = ticker
    return series


def select_price_series(
    name: str, candidates: Iterable[str], start: str
) -> Tuple[pd.Series, str | None]:
    best_series = pd.Series(dtype=float)
    selected_ticker: str | None = None
    best_span = pd.Timedelta(0)

    for ticker in candidates:
        series = download_single_ticker(ticker, start)
        if series.empty:
            continue
        span = series.index[-1] - series.index[0]
        if best_series.empty or span > best_span:
            best_series = series
            selected_ticker = ticker
            best_span = span

    return best_series.rename(name), selected_ticker


def get_asset_prices() -> Tuple[pd.DataFrame, Dict[str, str]]:
    series_map: Dict[str, pd.Series] = {}
    proxy_used: Dict[str, str] = {}

    for asset, proxies in ASSET_PROXIES.items():
        series, ticker = select_price_series(asset, proxies, START_DATE)
        if series.empty or not ticker:
            raise ValueError(f"No usable price data found for {asset} using proxies {proxies}")
        series_map[asset] = series
        proxy_used[asset] = ticker

    prices = pd.concat(series_map.values(), axis=1).dropna(how="all")
    prices = prices.dropna(subset=list(series_map.keys()), how="any")
    return prices, proxy_used


def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last().dropna(how="any")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def dual_momentum_signals(
    monthly_prices: pd.DataFrame,
    lookback: int,
    abs_threshold: float,
) -> pd.Series:
    momentum = monthly_prices.pct_change(lookback).shift(1)
    relative = momentum[["US", "INTL"]]
    best_asset = relative.idxmax(axis=1)
    best_score = relative.max(axis=1)
    signals = best_asset.where(best_score > abs_threshold, "BONDS")
    return signals.dropna()


def build_weights(signals: pd.Series, assets: List[str]) -> pd.DataFrame:
    weights = pd.get_dummies(signals)
    weights = weights.reindex(columns=assets, fill_value=0.0)
    return weights.astype(float)


def to_wealth(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    wealth.name = "wealth"
    return wealth


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def compute_metrics(monthly_returns: pd.Series) -> Dict[str, float]:
    if monthly_returns.empty:
        nan = float("nan")
        return {
            "CAGR": nan,
            "Volatility": nan,
            "Sharpe": nan,
            "Max Drawdown": nan,
            "Calmar": nan,
            "Terminal Wealth": nan,
        }

    wealth = to_wealth(monthly_returns)
    num_years = len(monthly_returns) / 12.0
    cagr = wealth.iloc[-1] ** (1.0 / num_years) - 1.0 if num_years > 0 else float("nan")
    vol = monthly_returns.std() * math.sqrt(12)
    avg_return = monthly_returns.mean() * 12
    sharpe = avg_return / vol if vol > 0 else float("nan")
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


def run_backtest() -> Tuple[Dict[str, BacktestResult], Dict[str, str]]:
    daily_prices, proxies = get_asset_prices()
    monthly_prices = to_monthly(daily_prices)
    monthly_returns = to_returns(monthly_prices)

    signals = dual_momentum_signals(
        monthly_prices=monthly_prices,
        lookback=LOOKBACK_MONTHS,
        abs_threshold=ABS_MOM_THRESHOLD,
    )
    signals = signals.loc[monthly_returns.index]

    assets = ["US", "INTL", "BONDS"]
    weights = build_weights(signals, assets)

    aligned_returns = monthly_returns.loc[weights.index, assets]
    strategy_returns = (weights * aligned_returns).sum(axis=1)
    strategy_returns = strategy_returns.dropna()

    strategy = BacktestResult(
        name="Dual Momentum Strategy",
        returns=strategy_returns,
        wealth=to_wealth(strategy_returns),
        weights=weights.loc[strategy_returns.index],
        metrics=compute_metrics(strategy_returns),
        extras={"signals": signals.loc[strategy_returns.index]},
    )

    results: Dict[str, BacktestResult] = {"strategy": strategy}
    for asset in assets:
        asset_returns = monthly_returns[asset].loc[strategy_returns.index]
        results[asset] = BacktestResult(
            name=asset,
            returns=asset_returns,
            wealth=to_wealth(asset_returns),
            weights=pd.DataFrame(),
            metrics=compute_metrics(asset_returns),
        )

    return results, proxies


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    lines = [name]
    for key, value in metrics.items():
        if math.isnan(value):
            display = "nan"
        else:
            if key in {"CAGR", "Volatility", "Sharpe", "Calmar"}:
                display = f"{value:.3f}"
            elif key == "Max Drawdown":
                display = f"{value:.3f}"
            else:
                display = f"{value:.2f}"
        lines.append(f"    {key:>14}: {display}")
    return "\n".join(lines)


def main() -> None:
    results, proxies = run_backtest()
    strat = results["strategy"]
    start_date = strat.returns.index.min().date()
    end_date = strat.returns.index.max().date()

    print("===== Gary Antonacci Dual Momentum Backtest =====")
    print("Universe: US equities vs International equities; safety = US Aggregate bonds.")
    print(f"Proxy mapping (Yahoo symbols): {proxies}")
    print(f"Sample window: {start_date} -> {end_date}")
    print(f"Lookback: {LOOKBACK_MONTHS} months, absolute threshold {ABS_MOM_THRESHOLD:.2%}\n")

    for label in ["strategy", "US", "INTL", "BONDS"]:
        print(format_metrics(results[label].name, results[label].metrics))
        print()

    allocation = strat.weights.idxmax(axis=1)
    freq = allocation.value_counts(normalize=True).sort_index()
    print("Allocation frequency:")
    for asset, share in freq.items():
        print(f"    {asset}: {share:.1%}")


if __name__ == "__main__":
    main()
