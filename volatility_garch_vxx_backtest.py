"""Backtest a GARCH vs VIX signal for trading VXX.

Strategy from the slides (long-only variant):
* RV(t+1) is the GARCH-predicted realized volatility for the next period.
* IV(t) is the current implied volatility (VIX).
* If RV(t+1) - VIX(t) > 0, go long VXX; otherwise hold cash.
* Position size scales with signal strength, capped at full exposure.

This implementation uses a rolling GARCH(1,1) fit on SPX log returns to
forecast the next 21 trading days of volatility, compares it to VIX, and
trades VXX on the next day to avoid look-ahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"
VXX_TICKER = "VXX"
ESTIMATION_WINDOW = 756  # ~3 years of daily data
FORECAST_HORIZON = 21  # ~1 month
UPDATE_FREQUENCY = 21  # re-estimate monthly
TRANSACTION_COST = 0.0005  # 5 bps per unit traded
SIGNAL_SCALE = 10.0  # signal points for full exposure
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestResult:
    returns: pd.Series
    wealth: pd.Series
    signal: pd.Series
    position: pd.Series
    metrics: Dict[str, float]
    vxx_returns: pd.Series


@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float


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
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(dtype=float, name=ticker)
        column = numeric_cols[0]

    series = data[column].astype(float).dropna()
    if isinstance(series, pd.DataFrame):
        series = series.squeeze("columns")
    series.name = ticker
    return series


def log_returns(prices: pd.Series) -> pd.Series:
    returns = np.log(prices).diff()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna()


def garch_loglikelihood(
    returns: np.ndarray, omega: float, alpha: float, beta: float
) -> float:
    if len(returns) < 2:
        return float("-inf")
    var = float(np.var(returns))
    if var <= 0:
        return float("-inf")
    h = var
    ll = 0.0
    for t in range(1, len(returns)):
        h = omega + alpha * returns[t - 1] ** 2 + beta * h
        if h <= 0:
            return float("-inf")
        ll += -0.5 * (math.log(2.0 * math.pi) + math.log(h) + returns[t] ** 2 / h)
    return ll


def estimate_garch_params(returns: np.ndarray) -> GarchParams:
    variance = float(np.var(returns))
    if variance <= 0:
        return GarchParams(omega=1e-8, alpha=0.05, beta=0.9)

    alpha_grid = np.linspace(0.02, 0.20, 10)
    beta_grid = np.linspace(0.70, 0.98, 15)
    best_ll = float("-inf")
    best_params: Tuple[float, float, float] | None = None

    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha + beta >= 0.995:
                continue
            omega = variance * (1.0 - alpha - beta)
            if omega <= 0:
                continue
            ll = garch_loglikelihood(returns, omega, alpha, beta)
            if ll > best_ll:
                best_ll = ll
                best_params = (omega, alpha, beta)

    if best_params is None:
        alpha = 0.05
        beta = 0.9
        omega = variance * (1.0 - alpha - beta)
        return GarchParams(omega=omega, alpha=alpha, beta=beta)

    omega, alpha, beta = best_params
    return GarchParams(omega=omega, alpha=alpha, beta=beta)


def garch_filter(returns: np.ndarray, params: GarchParams) -> np.ndarray:
    variance = float(np.var(returns))
    if variance <= 0:
        variance = 1e-8
    filtered = np.empty(len(returns))
    filtered[0] = variance
    for t in range(1, len(returns)):
        filtered[t] = (
            params.omega
            + params.alpha * returns[t - 1] ** 2
            + params.beta * filtered[t - 1]
        )
        if filtered[t] <= 0:
            filtered[t] = 1e-12
    return filtered


def expected_variance_path(
    h_next: float, params: GarchParams, horizon: int
) -> np.ndarray:
    if horizon <= 0:
        return np.array([], dtype=float)
    alpha_beta = params.alpha + params.beta
    if alpha_beta >= 1.0 or params.omega <= 0:
        return np.full(horizon, h_next)
    long_run = params.omega / (1.0 - alpha_beta)
    powers = np.power(alpha_beta, np.arange(horizon))
    return long_run + (h_next - long_run) * powers


def forecast_garch_vol(
    returns: pd.Series,
    estimation_window: int,
    forecast_horizon: int,
    update_frequency: int,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    values = returns.values
    index = returns.index
    forecast = pd.Series(index=index, dtype=float)

    start_idx = estimation_window
    if start_idx >= len(values):
        return forecast

    last_params: GarchParams | None = None
    last_h: float | None = None

    for i in range(start_idx, len(values)):
        if last_params is None or (i - start_idx) % update_frequency == 0:
            window = values[i - estimation_window : i]
            last_params = estimate_garch_params(window)
            last_h = float(garch_filter(window, last_params)[-1])

        params = last_params
        if params is None or last_h is None:
            continue

        prev_return = values[i - 1]
        h_t = params.omega + params.alpha * prev_return ** 2 + params.beta * last_h
        if h_t <= 0:
            h_t = 1e-12

        current_return = values[i]
        h_next = params.omega + params.alpha * current_return ** 2 + params.beta * h_t
        if h_next <= 0:
            h_next = 1e-12

        expected_variances = expected_variance_path(h_next, params, forecast_horizon)
        if expected_variances.size == 0:
            continue
        annualized_vol = math.sqrt(expected_variances.mean() * 252.0)
        forecast.iloc[i] = annualized_vol * 100.0
        last_h = h_t

    return forecast


def to_wealth(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    wealth.name = "wealth"
    return wealth


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        nan = float("nan")
        return {
            "CAGR": nan,
            "Volatility": nan,
            "Sharpe": nan,
            "Max Drawdown": nan,
            "Calmar": nan,
            "Hit Rate": nan,
            "Total Return": nan,
        }
    wealth = to_wealth(returns)
    num_years = len(returns) / 252.0
    cagr = wealth.iloc[-1] ** (1.0 / num_years) - 1.0 if num_years > 0 else float("nan")
    vol = returns.std() * math.sqrt(252)
    sharpe = returns.mean() / returns.std() * math.sqrt(252) if vol > 0 else float("nan")
    mdd = max_drawdown(wealth)
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    hit_rate = float((returns > 0).mean())
    total_return = wealth.iloc[-1] - 1.0
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "Hit Rate": hit_rate,
        "Total Return": total_return,
    }


def plot_results(
    signal: pd.Series,
    strategy_wealth: pd.Series,
    vxx_wealth: pd.Series,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(signal.index, signal, color="tab:blue", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("RV(t+1) - VIX(t) (pct)")
    axes[0].set_title("GARCH Forecast minus VIX")

    axes[1].plot(strategy_wealth.index, strategy_wealth, label="Strategy", linewidth=1.2)
    axes[1].plot(vxx_wealth.index, vxx_wealth, label="VXX Buy & Hold", linewidth=1.0)
    axes[1].set_ylabel("Wealth")
    axes[1].set_title("Wealth Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vxx_garch_signal_backtest.png", dpi=150)
    plt.close(fig)


def run_backtest() -> BacktestResult:
    spx_prices = download_single_ticker(SPX_TICKER, START_DATE, END_DATE, auto_adjust=True)
    vix = download_single_ticker(VIX_TICKER, START_DATE, END_DATE, auto_adjust=False)
    vxx_prices = download_single_ticker(VXX_TICKER, START_DATE, END_DATE, auto_adjust=True)

    if spx_prices.empty or vix.empty or vxx_prices.empty:
        raise ValueError("Missing data for SPX, VIX, or VXX. Check tickers and data source.")

    spx_returns = log_returns(spx_prices)
    vxx_returns = vxx_prices.pct_change().dropna()

    forecast = forecast_garch_vol(
        spx_returns, ESTIMATION_WINDOW, FORECAST_HORIZON, UPDATE_FREQUENCY
    )

    aligned = pd.concat([forecast, vix, vxx_returns], axis=1, join="inner")
    aligned.columns = ["predicted_vol", "vix", "vxx_returns"]
    aligned = aligned.dropna()

    signal = aligned["predicted_vol"] - aligned["vix"]
    position = (signal / SIGNAL_SCALE).clip(lower=0.0, upper=1.0)
    position = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(0.0)
    strategy_returns = position * aligned["vxx_returns"] - turnover * TRANSACTION_COST

    strategy_returns = strategy_returns.dropna()
    signal = signal.reindex(strategy_returns.index)
    position = position.reindex(strategy_returns.index)
    metrics = compute_metrics(strategy_returns)

    return BacktestResult(
        returns=strategy_returns,
        wealth=to_wealth(strategy_returns),
        signal=signal,
        position=position,
        metrics=metrics,
        vxx_returns=aligned["vxx_returns"],
    )


def main() -> None:
    result = run_backtest()

    vxx_returns = result.vxx_returns.reindex(result.returns.index).dropna()
    vxx_wealth = to_wealth(vxx_returns)

    plot_results(result.signal, result.wealth, vxx_wealth)

    metrics_df = pd.DataFrame(
        {"Strategy": result.metrics, "VXX Buy & Hold": compute_metrics(vxx_returns)}
    )
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(metrics_df)
        print()
        print(f"Plot saved to: {OUTPUT_DIR / 'vxx_garch_signal_backtest.png'}")


if __name__ == "__main__":
    main()
