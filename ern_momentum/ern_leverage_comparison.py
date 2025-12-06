"""Compare leverage estimation algorithms on the ERN momentum strategy.

Implements four online leverage estimators on top of the ERN momentum
strategy's daily returns:
* EWMA Kelly (lambda decay on mean/variance estimates).
* Online gradient descent on log-wealth.
* Thompson sampling for Kelly sizing (normal-inverse-gamma posterior).
* Simple 30-day rolling vol targeting to 50% annualized vol.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Allow running as a script (`python ern_momentum/ern_leverage_comparison.py`)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ern_momentum.ern_momentum_blog_backtest import (
    OUTPUT_DIR,
    allocate_weights,
    apply_expense_drag,
    apply_expense_drag_daily,
    apply_vol_target_overlay,
    build_daily_strategy_returns,
    build_momentum_scores,
    compute_metrics,
    compute_portfolio_returns,
    daily_to_monthly_returns,
    format_metrics,
    load_market_data,
    to_wealth,
)


ANNUALIZATION = 252.0


@dataclass
class StrategyInputs:
    daily_returns: pd.Series
    daily_cash: pd.Series
    monthly_cash: pd.Series
    monthly_strategy: pd.Series
    weights: pd.DataFrame
    proxies: Dict[str, str]


@dataclass
class LeverageResult:
    name: str
    daily_returns: pd.Series
    monthly_returns: pd.Series
    wealth: pd.Series
    leverage: pd.Series
    metrics: Dict[str, float]


def prepare_strategy_inputs() -> StrategyInputs:
    """Load data, build monthly weights, and project to daily strategy returns."""
    daily_prices, daily_cash_returns, proxies = load_market_data()

    monthly_prices = daily_prices.resample("ME").last().dropna(how="any")
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
    asset_returns, cash_net_monthly = apply_expense_drag(asset_returns, cash_returns)
    monthly_strategy = compute_portfolio_returns(weights, asset_returns, cash_net_monthly)

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

    daily_asset_returns, daily_cash_net = apply_expense_drag_daily(
        daily_asset_returns, daily_cash_returns
    )
    daily_strategy = build_daily_strategy_returns(weights, daily_asset_returns, daily_cash_net)
    common_daily = daily_strategy.index.intersection(daily_cash_net.index)
    daily_strategy = daily_strategy.loc[common_daily]
    daily_cash_net = daily_cash_net.loc[common_daily]

    return StrategyInputs(
        daily_returns=daily_strategy,
        daily_cash=daily_cash_net,
        monthly_cash=cash_net_monthly,
        monthly_strategy=monthly_strategy,
        weights=weights,
        proxies=proxies,
    )


def _to_result(
    name: str,
    daily_overlay: pd.Series,
    leverage: pd.Series,
    monthly_cash: pd.Series,
) -> LeverageResult:
    monthly_overlay = daily_to_monthly_returns(daily_overlay)
    monthly_overlay = monthly_overlay.dropna()
    common_months = monthly_overlay.index.intersection(monthly_cash.index)
    monthly_overlay = monthly_overlay.loc[common_months]
    cash_aligned = monthly_cash.loc[common_months]
    metrics = compute_metrics(monthly_overlay, cash_aligned)
    wealth = to_wealth(monthly_overlay)
    leverage = leverage.reindex(daily_overlay.index)
    return LeverageResult(
        name=name,
        daily_returns=daily_overlay,
        monthly_returns=monthly_overlay,
        wealth=wealth,
        leverage=leverage,
        metrics=metrics,
    )


def ewma_kelly_overlay(
    base_returns: pd.Series,
    cash_returns: pd.Series,
    monthly_cash: pd.Series,
    lam: float = 0.97,
    dt: float = 1.0 / ANNUALIZATION,
    mu0: float = 0.0,
    sigma2_0: float = 0.04,
    f_min: float = 0.0,
    f_max: float = 10.0,
) -> LeverageResult:
    mu_hat = mu0
    sigma2_hat = sigma2_0
    leverage_values = []
    overlay_returns = []
    cash_aligned = cash_returns.reindex(base_returns.index).fillna(0.0)
    eps = 1e-8

    f_t = float(np.clip(mu_hat / max(sigma2_hat, eps), f_min, f_max))

    for r_t, rf_t in zip(base_returns, cash_aligned):
        # Use prior-day leverage, then update estimates with the realized return.
        leverage_values.append(f_t)
        overlay_returns.append(f_t * r_t + (1.0 - f_t) * rf_t)

        mu_hat = lam * mu_hat + (1.0 - lam) * (r_t / dt)
        sigma2_hat = lam * sigma2_hat + (1.0 - lam) * ((r_t - mu_hat * dt) ** 2) / dt
        sigma2_hat = max(sigma2_hat, eps)
        f_t = float(np.clip(mu_hat / sigma2_hat, f_min, f_max))

    daily_overlay = pd.Series(overlay_returns, index=base_returns.index, name="ewma_kelly")
    leverage_series = pd.Series(leverage_values, index=base_returns.index, name="ewma_kelly_f")
    return _to_result("EWMA Kelly", daily_overlay, leverage_series, monthly_cash)


def online_gradient_overlay(
    base_returns: pd.Series,
    cash_returns: pd.Series,
    monthly_cash: pd.Series,
    eta0: float = 0.5,
    beta: float = 0.94,
    f0: float = 1.0,
    sigma2_0: float | None = None,
    f_min: float = 0.0,
    f_max: float = 10.0,
) -> LeverageResult:
    sigma2_hat = (
        float(base_returns.var(ddof=0)) if sigma2_0 is None or sigma2_0 <= 0.0 else sigma2_0
    )
    if not math.isfinite(sigma2_hat) or sigma2_hat <= 0.0:
        sigma2_hat = 0.04

    f_t = f0
    leverage_values = []
    overlay_returns = []
    cash_aligned = cash_returns.reindex(base_returns.index).fillna(0.0)
    eps = 1e-8

    for step, (r_t, rf_t) in enumerate(zip(base_returns, cash_aligned), start=1):
        leverage_values.append(f_t)
        overlay_returns.append(f_t * r_t + (1.0 - f_t) * rf_t)

        sigma2_hat = beta * sigma2_hat + (1.0 - beta) * (r_t ** 2)
        sigma2_hat = max(sigma2_hat, eps)
        eta_t = eta0 / math.sqrt(step)
        grad = (r_t / sigma2_hat) - f_t
        f_t = float(np.clip(f_t + eta_t * grad, f_min, f_max))

    daily_overlay = pd.Series(overlay_returns, index=base_returns.index, name="ogd_overlay")
    leverage_series = pd.Series(leverage_values, index=base_returns.index, name="ogd_f")
    return _to_result("Online Gradient", daily_overlay, leverage_series, monthly_cash)


def thompson_sampling_overlay(
    base_returns: pd.Series,
    cash_returns: pd.Series,
    monthly_cash: pd.Series,
    mu0: float = 0.0,
    tau0: float = 0.05,
    alpha0: float = 3.0,
    beta0: float = 1e-4,
    f_max: float = 10.0,
    seed: int | None = 0,
) -> LeverageResult:
    rng = np.random.default_rng(seed)
    kappa0 = 1.0 / max(tau0 ** 2, 1e-8)
    count = 0
    mean = 0.0
    sse = 0.0
    leverage_values = []
    overlay_returns = []
    cash_aligned = cash_returns.reindex(base_returns.index).fillna(0.0)

    # Start from prior before any observations.
    kappa_n = kappa0
    mu_n = mu0
    alpha_n = alpha0
    beta_n = beta0

    for r_t, rf_t in zip(base_returns, cash_aligned):
        # Sample leverage from prior/posterior before observing r_t.
        gamma_sample = rng.gamma(shape=alpha_n, scale=1.0 / max(beta_n, 1e-12))
        sigma2_sample = 1.0 / max(gamma_sample, 1e-12)
        mu_sample = rng.normal(loc=mu_n, scale=math.sqrt(sigma2_sample / max(kappa_n, 1e-12)))

        f_t = float(np.clip(mu_sample / sigma2_sample, 0.0, f_max))
        leverage_values.append(f_t)
        overlay_returns.append(f_t * r_t + (1.0 - f_t) * rf_t)

        # Update posterior with observed return r_t (for next step).
        count += 1
        if count == 1:
            mean = r_t
            sse = 0.0
        else:
            delta = r_t - mean
            mean = mean + delta / count
            sse = sse + delta * (r_t - mean)

        kappa_n = kappa0 + count
        mu_n = (kappa0 * mu0 + count * mean) / kappa_n
        alpha_n = alpha0 + count / 2.0
        beta_n = beta0 + 0.5 * sse + (kappa0 * count * (mean - mu0) ** 2) / (2.0 * kappa_n)
        beta_n = max(beta_n, 1e-12)

    daily_overlay = pd.Series(overlay_returns, index=base_returns.index, name="ts_kelly")
    leverage_series = pd.Series(leverage_values, index=base_returns.index, name="ts_kelly_f")
    return _to_result("Thompson Kelly", daily_overlay, leverage_series, monthly_cash)


def simple_vol_target_overlay(
    base_returns: pd.Series,
    cash_returns: pd.Series,
    monthly_cash: pd.Series,
    target_vol: float = 0.50,
    window: int = 30,
    max_leverage: float = 10.0,
) -> LeverageResult:
    overlay, scaling = apply_vol_target_overlay(
        base_returns,
        target_vol=target_vol,
        window=window,
        rf_returns=cash_returns,
        max_leverage=max_leverage,
    )
    overlay = overlay.rename("vol_target_overlay")
    scaling = scaling.rename("vol_target_scaling")
    return _to_result("30d Vol Target 50%", overlay, scaling, monthly_cash)


def run_leverage_comparison() -> Tuple[Dict[str, LeverageResult], StrategyInputs]:
    inputs = prepare_strategy_inputs()
    base_leverage = pd.Series(1.0, index=inputs.daily_returns.index, name="base_f")
    baseline = _to_result(
        "Baseline (unscaled)",
        inputs.daily_returns,
        base_leverage,
        inputs.monthly_cash,
    )

    ewma = ewma_kelly_overlay(
        inputs.daily_returns,
        inputs.daily_cash,
        inputs.monthly_cash,
        lam=0.97,
        dt=1.0 / ANNUALIZATION,
        mu0=0.0,
        sigma2_0=0.04,
        f_min=0.0,
        f_max=10.0,
    )
    ogd = online_gradient_overlay(
        inputs.daily_returns,
        inputs.daily_cash,
        inputs.monthly_cash,
        eta0=0.5,
        beta=0.94,
        f0=1.0,
        sigma2_0=0.04,
        f_min=0.0,
        f_max=10.0,
    )
    ts = thompson_sampling_overlay(
        inputs.daily_returns,
        inputs.daily_cash,
        inputs.monthly_cash,
        mu0=0.0,
        tau0=0.05,
        alpha0=3.0,
        beta0=1e-4,
        f_max=10.0,
        seed=0,
    )
    vt = simple_vol_target_overlay(
        inputs.daily_returns,
        inputs.daily_cash,
        inputs.monthly_cash,
        target_vol=0.50,
        window=30,
        max_leverage=10.0,
    )

    results = {
        "baseline": baseline,
        "ewma_kelly": ewma,
        "online_gradient": ogd,
        "thompson_kelly": ts,
        "vol_target": vt,
    }
    return results, inputs


def plot_leverage_results(results: Dict[str, LeverageResult], output_dir: Path = OUTPUT_DIR) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping leverage plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    wealth_df = pd.concat({res.name: res.wealth for res in results.values()}, axis=1)
    leverage_df = pd.concat(
        {res.name: res.leverage for key, res in results.items() if key != "baseline"},
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    wealth_df.plot(ax=ax)
    ax.set_title("ERN Momentum: Wealth Paths with Leverage Overlays")
    ax.set_ylabel("Wealth (multiple of start)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ern_leverage_wealth.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    leverage_df.plot(ax=ax2)
    ax2.set_title("Leverage Paths")
    ax2.set_ylabel("Leverage")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "ern_leverage_exposures.png", dpi=150)
    plt.close(fig2)


def print_summary(results: Dict[str, LeverageResult]) -> None:
    for res in results.values():
        print(f"\n=== {res.name} ===")
        print(format_metrics(res.metrics))


if __name__ == "__main__":
    results, inputs = run_leverage_comparison()
    print("Proxy tickers:", inputs.proxies)
    print_summary(results)
    plot_leverage_results(results)
