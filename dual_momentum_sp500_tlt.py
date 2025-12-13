from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


START_DATE = "1990-01-01"
LOOKBACK_MONTHS = 12
VOL_TARGET = 0.50  # annualized
VOL_LOOKBACK_MONTHS = 6
MAX_LEVERAGE = 10.0
PLOT_PATH = Path("plots/dual_momentum_sp500_tlt.png")

# Prefer total-return style S&P data when available; fall back to price index/ETF.
TICKER_CANDIDATES: Dict[str, List[str]] = {
    "SP500": ["^SP500TR", "^GSPC", "SPY"],
    "TLT": ["TLT"],
}


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    wealth: pd.Series
    leverage: pd.Series
    metrics: Dict[str, float]
    extras: Dict[str, object]


def download_single_ticker(
    ticker: str, start: str, auto_adjust: bool = True
) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
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
    name: str, candidates: Iterable[str], start: str
) -> Tuple[pd.Series, str]:
    best_series = pd.Series(dtype=float)
    best_ticker = ""
    best_span = pd.Timedelta(0)

    for ticker in candidates:
        series = download_single_ticker(ticker, start)
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
        series, ticker = select_longest_series(asset, candidates, START_DATE)
        series_map[asset] = series
        proxies[asset] = ticker

    prices = pd.concat(series_map.values(), axis=1)
    prices = prices.dropna(how="any")
    return prices, proxies


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = prices.resample("ME").last().dropna(how="any")
    monthly_returns = monthly_prices.pct_change().dropna(how="any")
    return monthly_returns


def momentum_signals(monthly_prices: pd.DataFrame, lookback: int) -> pd.Series:
    momentum = monthly_prices.pct_change(lookback)
    momentum = momentum.dropna(how="all")
    signals = momentum.idxmax(axis=1, skipna=True)
    signals = signals.shift(1)  # trade on next month to avoid look-ahead
    return signals.dropna()


def rolling_vol(monthly_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = monthly_returns.rolling(window=window).std(ddof=0) * math.sqrt(12.0)
    return vol.shift(1)  # only use information available at rebalance


def build_strategy_returns(
    signals: pd.Series,
    monthly_returns: pd.DataFrame,
    vol_estimates: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    assets = list(monthly_returns.columns)
    aligned_index = signals.index.intersection(monthly_returns.index)
    signals = signals.loc[aligned_index]
    monthly_returns = monthly_returns.loc[aligned_index]
    vol_estimates = vol_estimates.reindex(aligned_index)

    strat_returns: List[float] = []
    leverage_values: List[float] = []
    weight_rows: List[pd.Series] = []
    for date, asset in signals.items():
        asset_return = monthly_returns.at[date, asset]
        asset_vol = vol_estimates.at[date, asset]
        if not math.isfinite(asset_return):
            continue

        if math.isfinite(asset_vol) and asset_vol > 0.0:
            lev = min(MAX_LEVERAGE, VOL_TARGET / asset_vol)
        else:
            lev = 1.0

        strat_returns.append(lev * asset_return)
        leverage_values.append(lev)
        row = pd.Series(0.0, index=assets, name=date)
        row[asset] = lev
        weight_rows.append(row)

    strategy = pd.Series(strat_returns, index=signals.index, name="dual_momentum_vol_target")
    leverage = pd.Series(leverage_values, index=signals.index, name="leverage")
    weights = pd.DataFrame(weight_rows)
    return strategy, leverage, weights


def to_wealth(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    wealth.name = "wealth"
    return wealth


def max_drawdown(wealth: pd.Series) -> float:
    rolling_max = wealth.cummax()
    drawdowns = wealth / rolling_max - 1.0
    return float(drawdowns.min()) if not drawdowns.empty else float("nan")


def compute_metrics(monthly_returns: pd.Series) -> Dict[str, float]:
    if monthly_returns.empty:
        nan = float("nan")
        return {
            "CAGR": nan,
            "Volatility": nan,
            "Sharpe": nan,
            "Max Drawdown": nan,
            "Terminal Wealth": nan,
        }

    wealth = to_wealth(monthly_returns)
    years = len(monthly_returns) / 12.0
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol = monthly_returns.std(ddof=0) * math.sqrt(12.0)
    sharpe = (monthly_returns.mean() * 12.0) / vol if vol > 0 else float("nan")
    mdd = max_drawdown(wealth)
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Terminal Wealth": wealth.iloc[-1],
    }


def plot_equity(wealth_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in wealth_frame.columns:
        ax.plot(wealth_frame.index, wealth_frame[column], label=column)
    ax.set_title("Dual Momentum: S&P 500 vs. TLT (50% vol target)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_backtest() -> Tuple[BacktestResult, Dict[str, BacktestResult], Dict[str, str]]:
    prices, proxies = load_price_history()
    monthly_prices = prices.resample("ME").last().dropna(how="any")
    monthly_returns = monthly_prices.pct_change().dropna(how="any")

    signals = momentum_signals(monthly_prices, LOOKBACK_MONTHS)
    signals = signals.reindex(monthly_returns.index).dropna()
    vol_estimates = rolling_vol(monthly_returns, VOL_LOOKBACK_MONTHS)

    strategy_returns, leverage, weights = build_strategy_returns(
        signals, monthly_returns, vol_estimates
    )
    strategy_returns = strategy_returns.dropna()
    leverage = leverage.reindex(strategy_returns.index)
    weights = weights.reindex(strategy_returns.index)

    strat_result = BacktestResult(
        name="Dual Momentum SP500 vs TLT (50% vol target)",
        returns=strategy_returns,
        wealth=to_wealth(strategy_returns),
        leverage=leverage,
        metrics=compute_metrics(strategy_returns),
        extras={"signals": signals, "weights": weights},
    )

    asset_results: Dict[str, BacktestResult] = {}
    for asset in monthly_returns.columns:
        asset_series = monthly_returns[asset].reindex(strategy_returns.index)
        asset_results[asset] = BacktestResult(
            name=asset,
            returns=asset_series,
            wealth=to_wealth(asset_series),
            leverage=pd.Series(dtype=float),
            metrics=compute_metrics(asset_series),
            extras={},
        )

    return strat_result, asset_results, proxies


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(label)
    for key, value in metrics.items():
        display = "nan" if math.isnan(value) else f"{value:.3f}"
        if key == "Terminal Wealth":
            display = f"{value:.2f}" if not math.isnan(value) else "nan"
        print(f"  {key:>14}: {display}")
    print()


def main() -> None:
    strategy, assets, proxies = run_backtest()

    wealth_frame = pd.DataFrame(
        {
            "Strategy (vol targeted)": strategy.wealth,
            "SP500": assets["SP500"].wealth,
            "TLT": assets["TLT"].wealth,
        }
    ).dropna()
    plot_equity(wealth_frame, PLOT_PATH)

    start = strategy.returns.index.min().date()
    end = strategy.returns.index.max().date()
    print("==== Dual Momentum: SP500 vs TLT ====")
    print(f"Data sources: {proxies}")
    print(f"Sample window: {start} -> {end}")
    print(f"Lookback: {LOOKBACK_MONTHS} months | Vol target: {VOL_TARGET:.0%}")
    print(f"Vol lookback: {VOL_LOOKBACK_MONTHS} months | Max leverage: {MAX_LEVERAGE:.1f}\n")

    print_metrics(strategy.name, strategy.metrics)
    for asset_name, result in assets.items():
        print_metrics(result.name, result.metrics)

    print(f"Equity curve saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
