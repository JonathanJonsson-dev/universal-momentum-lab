"""Backtest non-parametric allocation schemes using the universal_portfolios library.

The script retrieves daily prices for a diversified ETF basket, runs several
online portfolio-selection algorithms, and reports common performance metrics.

Algorithms covered:
* Cover's Universal Portfolio (UP)
* Follow-the-Leader (FTL)
* Follow-the-Regularized-Leader (FTRL) with an L2-style shrink
* Online Gradient Descent (OGD)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# The repo ships with a Windows virtual environment. When running the script
# from WSL or another interpreter, explicitly add its site-packages so we can
# import universal_portfolios and other dependencies without activating the venv.
CURRENT_DIR = Path(__file__).resolve().parent
POSSIBLE_SITE_PACKAGES = [
    CURRENT_DIR / ".venv" / "Lib" / "site-packages",
    CURRENT_DIR / ".venv" / "lib" / "python3.12" / "site-packages",
]
for path in POSSIBLE_SITE_PACKAGES:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import yfinance as yf
from universal.algo import Algo
from universal.algos.up import UP


START_DATE = "2005-01-01"
TICKERS = ["SPY", "GLD", "TLT"]
RISK_FREE_ANNUAL = 0.02  # 2% cash proxy for Sharpe
UP_EVAL_POINTS = 4000
UP_LEVERAGE = 1.25


@dataclass
class StrategyResult:
    name: str
    returns: pd.Series
    wealth: pd.Series
    metrics: Dict[str, float]
    weights: pd.DataFrame


def download_prices(tickers: Sequence[str], start: str) -> pd.DataFrame:
    """Fetch adjusted close prices for the supplied tickers."""
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    data = data.dropna(how="any")
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def to_wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    if returns.empty:
        return {
            "CAGR": float("nan"),
            "Volatility": float("nan"),
            "Sharpe": float("nan"),
            "Max Drawdown": float("nan"),
            "Terminal Wealth": float("nan"),
        }
    wealth = to_wealth(returns)
    total_days = len(returns)
    cagr = wealth.iloc[-1] ** (252.0 / total_days) - 1.0
    vol = returns.std(ddof=0) * math.sqrt(252.0)
    excess_return = returns.mean() * 252.0 - RISK_FREE_ANNUAL
    sharpe = excess_return / vol if vol > 0 else float("nan")
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown(wealth),
        "Terminal Wealth": wealth.iloc[-1],
    }


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project vector onto the probability simplex."""
    v = np.asarray(vector, dtype=float).ravel()
    if np.all(np.isfinite(v)) is False:
        v = np.nan_to_num(v, copy=False)
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        theta = cssv[-1] / n
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    total = w.sum()
    if total <= 0:
        w = np.ones_like(v) / float(n)
    else:
        w /= total
    return w


def _to_numpy(series_or_array: Iterable[float]) -> np.ndarray:
    if isinstance(series_or_array, pd.Series):
        return series_or_array.to_numpy(dtype=float, copy=False)
    return np.asarray(series_or_array, dtype=float)


def _log_loss_gradient(weights: np.ndarray, rel_returns: np.ndarray) -> np.ndarray:
    denom = float(np.dot(weights, rel_returns))
    denom = max(denom, 1e-9)
    return -rel_returns / denom


class FollowTheLeader(Algo):
    """Choose the asset with the highest trailing compounded return."""

    PRICE_TYPE = "ratio"

    def __init__(self, lookback: int | None = 63):
        super().__init__()
        self.lookback = lookback

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        window = history if self.lookback is None else history.iloc[-self.lookback :]
        compounded = window.cumprod().iloc[-1]
        leaders = (compounded == compounded.max()).astype(float)
        total = leaders.sum()
        if total <= 0:
            leaders[:] = 1.0 / float(len(leaders))
            return leaders.to_numpy(dtype=float)
        return (leaders / total).to_numpy(dtype=float)


class FollowTheRegularizedLeader(Algo):
    """FTRL with cumulative log-loss gradients & L2-style shrink toward uniform."""

    PRICE_TYPE = "ratio"

    def __init__(self, eta: float = 0.2, ridge: float = 0.05):
        super().__init__()
        self.eta = eta
        self.ridge = ridge

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        self.grad_sum = np.zeros(X.shape[1])

    def step(self, x, last_b, history):
        weights = _to_numpy(last_b)
        rel_returns = _to_numpy(x)
        grad = _log_loss_gradient(weights, rel_returns)
        self.grad_sum += grad
        raw = -self.eta * self.grad_sum
        updated = project_to_simplex(raw)
        if self.ridge > 0:
            uniform = np.ones_like(updated) / updated.size
            updated = (1 - self.ridge) * updated + self.ridge * uniform
        return updated


class OnlineGradientDescent(Algo):
    """Plain Online Gradient Descent on the log-loss surface."""

    PRICE_TYPE = "ratio"

    def __init__(self, eta: float = 0.05):
        super().__init__()
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        weights = _to_numpy(last_b)
        rel_returns = _to_numpy(x)
        grad = _log_loss_gradient(weights, rel_returns)
        raw = weights - self.eta * grad
        return project_to_simplex(raw)


AlgorithmFactory = Tuple[str, Callable[[], Algo]]

ALGORITHM_FACTORIES: List[AlgorithmFactory] = [
    (
        "Cover Universal Portfolio",
        lambda: UP(eval_points=UP_EVAL_POINTS, leverage=UP_LEVERAGE),
    ),
    ("Follow the Leader", lambda: FollowTheLeader(lookback=63)),
    ("Follow-the-Regularized-Leader", lambda: FollowTheRegularizedLeader(eta=0.25, ridge=0.1)),
    ("Online Gradient Descent", lambda: OnlineGradientDescent(eta=0.07)),
]


def run_algorithms(prices: pd.DataFrame) -> List[StrategyResult]:
    """Run each algorithm on the given price history."""
    results: List[StrategyResult] = []
    for name, factory in ALGORITHM_FACTORIES:
        algo = factory()
        algo_result = algo.run(prices)
        returns = (algo_result.r - 1.0).astype(float)
        returns.name = name
        strategy_result = StrategyResult(
            name=name,
            returns=returns,
            wealth=returns.add(1.0).cumprod(),
            metrics=compute_metrics(returns),
            weights=algo_result.weights,
        )
        results.append(strategy_result)
    return results


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered = ["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Terminal Wealth"]
    lines: List[str] = []
    for key in ordered:
        value = metrics.get(key, float("nan"))
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
    prices = download_prices(TICKERS, START_DATE)
    if prices.empty:
        raise RuntimeError("No pricing data downloaded.")

    print(f"Loaded {len(prices)} observations spanning {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"Universe: {', '.join(TICKERS)}")

    strategy_results = run_algorithms(prices)

    print("\n===== Universal Portfolio Backtests =====")
    for result in strategy_results:
        print(f"\n{result.name}")
        print(format_metrics(result.metrics))


if __name__ == "__main__":
    main()
