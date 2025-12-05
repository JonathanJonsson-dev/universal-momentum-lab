"""Compare ERN momentum base strategy with vol-target and Browne goal-seeking overlays.

Uses the ERN momentum pipeline (monthly signals, daily implementation) and applies:
* Vol-target overlay on daily strategy returns.
* Browne absolute-goal overlay on daily strategy returns.

Outputs performance metrics, goal hit dates, and plots of wealth/exposures.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from ern_momentum.ern_momentum_blog_backtest import (
    BacktestResult,
    OUTPUT_DIR,
    apply_expense_drag,
    apply_expense_drag_daily,
    apply_vol_target_overlay,
    build_daily_strategy_returns,
    build_momentum_scores,
    compute_metrics,
    compute_portfolio_returns,
    daily_to_monthly_returns,
    load_market_data,
    to_wealth,
    allocate_weights,
)

VOL_TARGET = 0.50
VOL_WINDOW_DAYS = 30
VOL_MAX_LEVERAGE = 10.0

INITIAL_WEALTH_SEK = 1_000_000.0
TARGET_WEALTH_SEK = 20_000_000.0  # adjust to your FI goal
BROWNE_HORIZON_DAYS = 252 * 7  # 7-year window
BROWNE_SIGMA_WINDOW = 40
BROWNE_MAX_LEV = 10.0
STOP_ON_HIT = True
RISK_FREE_ANNUAL = 0.02


def daily_rate_from_annual(annual_rate: float) -> float:
    return math.exp(annual_rate / 252.0) - 1.0


def apply_browne_absolute_target(
    base_returns: pd.Series,
    rf_rate: float,
    initial_wealth: float,
    target_wealth: float,
    horizon_days: int,
    sigma_window: int,
    max_leverage: float,
    stop_on_hit: bool,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    wealth = 1.0  # multiples of initial_wealth
    target_multiple = target_wealth / initial_wealth
    rf_daily = daily_rate_from_annual(rf_rate)
    fractions = []
    wealth_path = []
    overlay_returns = []
    hit_index = None
    hit_date = None
    locked = False

    returns = base_returns.copy()
    returns_values = returns.values
    index = returns.index

    for idx, date in enumerate(index):
        if locked:
            fraction = 0.0
            daily_return = rf_daily
        else:
            days_left = max(horizon_days - idx, 1)
            tau = days_left / 252.0

            start = max(0, idx - sigma_window)
            window_slice = returns.iloc[start:idx]
            sigma = window_slice.std(ddof=0)
            sigma = float(sigma * math.sqrt(252)) if sigma and sigma > 0 else 0.0

            if sigma == 0.0 or tau <= 0.0:
                fraction = 0.0
            else:
                scaled_ratio = (wealth * math.exp(rf_rate * tau)) / target_multiple
                scaled_ratio = float(np.clip(scaled_ratio, 1e-9, 1 - 1e-9))
                z = norm.ppf(scaled_ratio)
                pdf = norm.pdf(z)
                fraction = (
                    (1.0 / (sigma * math.sqrt(tau)))
                    * (target_multiple * math.exp(-rf_rate * tau) / wealth)
                    * pdf
                )
                fraction = float(np.clip(fraction, -max_leverage, max_leverage))

            daily_return = fraction * returns_values[idx] + (1.0 - fraction) * rf_daily

        wealth *= 1.0 + daily_return
        overlay_returns.append(daily_return)
        fractions.append(fraction)
        wealth_path.append(wealth)

        if hit_date is None and wealth >= target_multiple:
            hit_date = date
            hit_index = idx
            if stop_on_hit:
                locked = True

    returns_series = pd.Series(data=overlay_returns, index=index, name="browne_returns")
    fraction_series = pd.Series(data=fractions, index=index, name="browne_fraction")
    wealth_series = pd.Series(data=wealth_path, index=index, name="browne_wealth")
    extras = {
        "fraction": fraction_series,
        "hit_date": hit_date,
        "hit_index": hit_index,
        "target_multiple": target_multiple,
        "achieved": hit_date is not None,
        "wealth": wealth_series,
    }
    return returns_series, wealth_series, extras


def find_first_hit(wealth: pd.Series, target_multiple: float) -> pd.Timestamp | None:
    hits = wealth[wealth >= target_multiple]
    return hits.index[0] if not hits.empty else None


def plot_results(wealth_df: pd.DataFrame, exposures: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    wealth_df.plot(ax=ax)
    ax.set_title("Wealth Paths (SEK)")
    ax.set_ylabel("Wealth (SEK)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ern_goal_wealth.png", dpi=150)
    plt.close(fig)

    fig_log, ax_log = plt.subplots(figsize=(10, 6))
    wealth_df.apply(np.log).plot(ax=ax_log)
    ax_log.set_title("Log Wealth Paths (SEK)")
    ax_log.set_ylabel("log(Wealth)")
    ax_log.grid(True, alpha=0.3)
    fig_log.tight_layout()
    fig_log.savefig(output_dir / "ern_goal_log_wealth.png", dpi=150)
    plt.close(fig_log)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    exposures.plot(ax=ax2)
    ax2.set_title("Overlay Exposures")
    ax2.set_ylabel("Fraction / Leverage")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "ern_goal_exposures.png", dpi=150)
    plt.close(fig2)


def format_metrics(metrics: Dict[str, float], initial_wealth: float | None = None) -> str:
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
    if initial_wealth is not None and "Terminal Wealth" in metrics:
        terminal = metrics["Terminal Wealth"]
        if not math.isnan(terminal):
            lines.append(f"{'Terminal Wealth (SEK)':>15}: {terminal * initial_wealth:,.0f}")
    return "\n".join(lines)


def run_comparison() -> Dict[str, BacktestResult]:
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
    asset_returns, cash_returns = apply_expense_drag(asset_returns, cash_returns)

    portfolio_returns = compute_portfolio_returns(weights, asset_returns, cash_returns)
    base_metrics = compute_metrics(portfolio_returns, cash_returns)
    base_result = BacktestResult(
        name="ERN Momentum Allocation",
        returns=portfolio_returns,
        wealth=to_wealth(portfolio_returns),
        weights=weights,
        metrics=base_metrics,
        extras={"proxies": proxies, "coverage": (portfolio_returns.index.min(), portfolio_returns.index.max())},
    )

    daily_asset_returns = daily_prices.pct_change().dropna()
    daily_asset_returns = daily_asset_returns.loc[daily_asset_returns.index.intersection(daily_cash_returns.index)]
    daily_cash_returns = daily_cash_returns.loc[daily_asset_returns.index]

    valid_periods = weights.index.to_period("M")
    daily_periods = daily_asset_returns.index.to_period("M")
    mask = daily_periods.isin(valid_periods)
    daily_asset_returns = daily_asset_returns.loc[mask]
    daily_cash_returns = daily_cash_returns.loc[daily_asset_returns.index]

    daily_asset_returns, daily_cash_net = apply_expense_drag_daily(daily_asset_returns, daily_cash_returns)
    daily_strategy_returns = build_daily_strategy_returns(
        weights,
        daily_asset_returns,
        daily_cash_net,
    )

    vt_daily, vt_scaling = apply_vol_target_overlay(
        daily_strategy_returns,
        target_vol=VOL_TARGET,
        window=VOL_WINDOW_DAYS,
        rf_returns=daily_cash_net,
        max_leverage=VOL_MAX_LEVERAGE,
    )
    vt_monthly = daily_to_monthly_returns(vt_daily)
    vt_monthly = vt_monthly.loc[cash_returns.index.intersection(vt_monthly.index)]
    vt_metrics = compute_metrics(vt_monthly, cash_returns)
    vt_result = BacktestResult(
        name="Vol Target Overlay",
        returns=vt_monthly,
        wealth=to_wealth(vt_monthly),
        weights=weights,
        metrics=vt_metrics,
        extras={"scaling": vt_scaling, "daily_returns": vt_daily},
    )

    browne_returns, browne_wealth_path, browne_extras = apply_browne_absolute_target(
        base_returns=daily_strategy_returns,
        rf_rate=RISK_FREE_ANNUAL,
        initial_wealth=INITIAL_WEALTH_SEK,
        target_wealth=TARGET_WEALTH_SEK,
        horizon_days=BROWNE_HORIZON_DAYS,
        sigma_window=BROWNE_SIGMA_WINDOW,
        max_leverage=BROWNE_MAX_LEV,
        stop_on_hit=STOP_ON_HIT,
    )
    browne_monthly = daily_to_monthly_returns(browne_returns)
    browne_monthly = browne_monthly.loc[cash_returns.index.intersection(browne_monthly.index)]
    browne_metrics = compute_metrics(browne_monthly, cash_returns)
    browne_result = BacktestResult(
        name="Browne Absolute Goal Overlay",
        returns=browne_monthly,
        wealth=to_wealth(browne_monthly),
        weights=weights,
        metrics=browne_metrics,
        extras=browne_extras,
    )

    return {
        "base": base_result,
        "vol_target": vt_result,
        "browne": browne_result,
    }


def main() -> None:
    results = run_comparison()
    base = results["base"]
    vt = results["vol_target"]
    browne = results["browne"]

    coverage = base.extras.get("coverage")
    if coverage:
        start, end = coverage
        print(f"Sample covers {start.date()} -> {end.date()}")

    print("\n===== Performance =====")
    for result in (base, vt, browne):
        print(f"\n{result.name}")
        print(format_metrics(result.metrics, INITIAL_WEALTH_SEK))

    vt_scaling = vt.extras.get("scaling") if vt.extras else pd.Series(dtype=float)
    browne_fraction = browne.extras.get("fraction") if browne.extras else pd.Series(dtype=float)

    target_mult = TARGET_WEALTH_SEK / INITIAL_WEALTH_SEK
    hit_dates = {}
    if isinstance(vt_scaling, pd.Series) and not vt_scaling.empty:
        print(
            "\nVol target leverage: "
            f"mean {vt_scaling.mean():.2f}, median {vt_scaling.median():.2f}, "
            f"95th pct {vt_scaling.quantile(0.95):.2f}"
        )
    if isinstance(browne_fraction, pd.Series) and not browne_fraction.empty:
        print(
            "Browne fraction: "
            f"mean {browne_fraction.mean():.2f}, median {browne_fraction.median():.2f}, "
            f"95th pct {browne_fraction.quantile(0.95):.2f}"
        )

    vt_hit = find_first_hit(vt.wealth, target_mult)
    browne_hit = browne.extras.get("hit_date")
    base_hit = find_first_hit(base.wealth, target_mult)
    if base_hit:
        print(f"\nBase strategy hit target on {base_hit.date()}")
    else:
        print("\nBase strategy did NOT hit target in sample.")
    if vt_hit:
        print(f"Vol Target Overlay hit target on {vt_hit.date()}")
    else:
        print("Vol Target Overlay did NOT hit target in sample.")
    if browne_hit:
        print(f"Browne Overlay hit target on {browne_hit.date()}")
    else:
        print("Browne Overlay did NOT hit target in sample.")

    wealth_df = pd.concat(
        {
            "base": base.wealth * INITIAL_WEALTH_SEK,
            "vol_target": vt.wealth * INITIAL_WEALTH_SEK,
            "browne": browne.extras.get("wealth", browne.wealth) * INITIAL_WEALTH_SEK
            if isinstance(browne.extras, dict) and browne.extras.get("wealth") is not None
            else browne.wealth * INITIAL_WEALTH_SEK,
        },
        axis=1,
    )
    exposures = pd.concat(
        {"Vol Target Leverage": vt_scaling, "Browne Fraction": browne_fraction},
        axis=1,
    )
    plot_results(wealth_df, exposures, OUTPUT_DIR)
    print(f"\nSaved plots to {OUTPUT_DIR}")
    print(
        "\nConfig: "
        f"start {INITIAL_WEALTH_SEK:,.0f} SEK -> target {TARGET_WEALTH_SEK:,.0f} SEK, "
        f"horizon {BROWNE_HORIZON_DAYS} days, sigma window {BROWNE_SIGMA_WINDOW}, "
        f"max lev {BROWNE_MAX_LEV}, vol target {VOL_TARGET} (window {VOL_WINDOW_DAYS}, max {VOL_MAX_LEVERAGE})"
    )


if __name__ == "__main__":
    main()
