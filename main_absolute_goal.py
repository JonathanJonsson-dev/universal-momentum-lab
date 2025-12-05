import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from main import (
    BacktestResult,
    apply_vol_target,
    compute_metrics,
    compute_risk_parity_returns,
    daily_rate_from_annual,
    download_prices,
    format_metrics,
    to_wealth,
)

START_DATE = "2005-01-01"
TICKERS = ["SPY", "GLD", "TLT"]
RISK_FREE_ANNUAL = 0.02
RISK_PARITY_WINDOW = 60

VOL_TARGET = 0.45
VOL_TARGET_WINDOW = 20
VOL_TARGET_MAX_LEV = 10.0

KELLY_WINDOW = 252
KELLY_MAX_LEV = 10.0

INITIAL_WEALTH_SEK = 1_000_000.0
TARGET_WEALTH_SEK = 1000_000_000.0
BROWNE_HORIZON_DAYS = 252 * 10 # 10-year window
BROWNE_SIGMA_WINDOW = 40
BROWNE_MAX_LEV = 10.0
STOP_ON_HIT = True


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
    }
    return returns_series, wealth_series, extras


def apply_kelly_overlay(
    base_returns: pd.Series,
    rf_rate: float,
    window: int,
    max_leverage: float,
) -> Tuple[pd.Series, pd.Series]:
    rf_daily = daily_rate_from_annual(rf_rate)
    rolling_mean = base_returns.rolling(window=window, min_periods=window).mean()
    rolling_var = base_returns.rolling(window=window, min_periods=window).var(ddof=0)
    rolling_mean = rolling_mean.shift(1)
    rolling_var = rolling_var.shift(1)

    excess_mean = rolling_mean - rf_daily
    fraction = excess_mean / rolling_var
    fraction = fraction.replace([np.inf, -np.inf], np.nan)
    # Long-only Kelly: clamp at zero and cap leverage.
    fraction = fraction.clip(lower=0.0, upper=max_leverage)
    fraction = fraction.reindex_like(base_returns).fillna(0.0)

    overlay_returns = fraction * base_returns + (1.0 - fraction) * rf_daily
    return overlay_returns, fraction


def plot_money_management(results: Dict[str, BacktestResult], initial_wealth: float) -> None:
    wealth_df = pd.concat(
        {key: result.wealth * initial_wealth for key, result in results.items()}, axis=1
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    wealth_df.plot(ax=ax)
    ax.set_title("Wealth Paths (SEK)")
    ax.set_ylabel("Wealth (SEK)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("money_management_wealth.png", dpi=150)
    plt.close(fig)

    fig_log, ax_log = plt.subplots(figsize=(10, 6))
    wealth_df.apply(np.log).plot(ax=ax_log)
    ax_log.set_title("Log Wealth Paths (SEK)")
    ax_log.set_ylabel("log(Wealth)")
    ax_log.grid(True, alpha=0.3)
    fig_log.tight_layout()
    fig_log.savefig("money_management_log_wealth.png", dpi=150)
    plt.close(fig_log)

    vt_scaling = results["vol_target"].extras.get("scaling")
    kelly_fraction = results["kelly"].extras.get("fraction")
    browne_fraction = results["browne"].extras.get("fraction")
    exposures = pd.concat(
        {
            "Vol Target Leverage": vt_scaling,
            "Kelly Fraction": kelly_fraction,
            "Browne Fraction": browne_fraction,
        },
        axis=1,
    )
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    exposures.plot(ax=ax2)
    ax2.set_title("Overlay Exposures")
    ax2.set_ylabel("Fraction / Leverage")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("money_management_exposures.png", dpi=150)
    plt.close(fig2)


def format_metrics_with_currency(metrics: Dict[str, float], initial_wealth: float) -> str:
    lines = format_metrics(metrics).splitlines()
    terminal = metrics.get("Terminal Wealth")
    if terminal is not None and not math.isnan(terminal):
        lines.append(f"Terminal Wealth (SEK): {terminal * initial_wealth:,.0f}")
    return "\n".join(lines)


def run_backtest() -> Dict[str, BacktestResult]:
    prices = download_prices(TICKERS, START_DATE)
    rp_returns, weights = compute_risk_parity_returns(prices, RISK_PARITY_WINDOW)
    rp_returns = rp_returns.dropna()

    rp_metrics = compute_metrics(rp_returns, RISK_FREE_ANNUAL)
    rp_result = BacktestResult(
        name="Risk Parity (unscaled)",
        returns=rp_returns,
        wealth=to_wealth(rp_returns),
        metrics=rp_metrics,
    )

    vt_returns, vt_scaling = apply_vol_target(
        base_returns=rp_returns,
        target_vol=VOL_TARGET,
        vol_window=VOL_TARGET_WINDOW,
        max_leverage=VOL_TARGET_MAX_LEV,
        rf_rate=RISK_FREE_ANNUAL,
    )
    vt_metrics = compute_metrics(vt_returns, RISK_FREE_ANNUAL)
    vt_result = BacktestResult(
        name="Vol Target Overlay",
        returns=vt_returns,
        wealth=to_wealth(vt_returns),
        metrics=vt_metrics,
        extras={"scaling": vt_scaling},
    )

    kelly_returns, kelly_fraction = apply_kelly_overlay(
        base_returns=rp_returns,
        rf_rate=RISK_FREE_ANNUAL,
        window=KELLY_WINDOW,
        max_leverage=KELLY_MAX_LEV,
    )
    kelly_metrics = compute_metrics(kelly_returns, RISK_FREE_ANNUAL)
    kelly_result = BacktestResult(
        name="Kelly Overlay",
        returns=kelly_returns,
        wealth=to_wealth(kelly_returns),
        metrics=kelly_metrics,
        extras={"fraction": kelly_fraction},
    )

    (
        browne_returns,
        browne_wealth_path,
        browne_extras,
    ) = apply_browne_absolute_target(
        base_returns=rp_returns,
        rf_rate=RISK_FREE_ANNUAL,
        initial_wealth=INITIAL_WEALTH_SEK,
        target_wealth=TARGET_WEALTH_SEK,
        horizon_days=BROWNE_HORIZON_DAYS,
        sigma_window=BROWNE_SIGMA_WINDOW,
        max_leverage=BROWNE_MAX_LEV,
        stop_on_hit=STOP_ON_HIT,
    )
    browne_metrics = compute_metrics(browne_returns, RISK_FREE_ANNUAL)
    browne_result = BacktestResult(
        name="Browne Absolute Goal Overlay",
        returns=browne_returns,
        wealth=browne_wealth_path,
        metrics=browne_metrics,
        extras=browne_extras,
    )

    return {
        "rp": rp_result,
        "vol_target": vt_result,
        "kelly": kelly_result,
        "browne": browne_result,
    }


def main() -> None:
    results = run_backtest()
    rp = results["rp"]
    vt = results["vol_target"]
    kelly = results["kelly"]
    browne = results["browne"]

    print("===== Performance Comparison (Absolute Goal) =====")
    for result in (rp, vt, kelly, browne):
        print(f"\n{result.name}")
        print(format_metrics_with_currency(result.metrics, INITIAL_WEALTH_SEK))

    vt_scaling = vt.extras.get("scaling") if vt.extras else pd.Series(dtype=float)
    browne_fraction = (
        browne.extras.get("fraction") if browne.extras else pd.Series(dtype=float)
    )
    kelly_fraction = (
        kelly.extras.get("fraction") if kelly.extras else pd.Series(dtype=float)
    )

    if isinstance(vt_scaling, pd.Series) and not vt_scaling.empty:
        print(
            "\nVol target leverage: "
            f"mean {vt_scaling.mean():.2f}, median {vt_scaling.median():.2f}, "
            f"95th pct {vt_scaling.quantile(0.95):.2f}"
        )
    if isinstance(kelly_fraction, pd.Series) and not kelly_fraction.empty:
        print(
            "Kelly fraction: "
            f"mean {kelly_fraction.mean():.2f}, median {kelly_fraction.median():.2f}, "
            f"95th pct {kelly_fraction.quantile(0.95):.2f}"
        )
    if isinstance(browne_fraction, pd.Series) and not browne_fraction.empty:
        print(
            "Browne fraction: "
            f"mean {browne_fraction.mean():.2f}, median {browne_fraction.median():.2f}, "
            f"95th pct {browne_fraction.quantile(0.95):.2f}"
        )

    if browne.extras:
        hit_date = browne.extras.get("hit_date")
        hit_index = browne.extras.get("hit_index")
        if hit_date is not None and hit_index is not None:
            print(
                f"Target hit on {hit_date.date()} after {hit_index + 1} trading days "
                f"at ~{TARGET_WEALTH_SEK:,.0f} SEK"
            )
        else:
            print("Target not hit within the sample.")

    print("\nTarget configuration:")
    print(
        f"Start wealth {INITIAL_WEALTH_SEK:,.0f} SEK -> target {TARGET_WEALTH_SEK:,.0f} SEK, "
        f"horizon {BROWNE_HORIZON_DAYS} days, sigma window {BROWNE_SIGMA_WINDOW}, "
        f"max leverage {BROWNE_MAX_LEV}"
    )
    print("\nSample statistics cover period:")
    print(f"{rp.returns.index.min().date()} -> {rp.returns.index.max().date()}")

    plot_money_management(results, INITIAL_WEALTH_SEK)
    
    target_mult = TARGET_WEALTH_SEK / INITIAL_WEALTH_SEK
    for key in ("vol_target", "kelly", "browne"):
        wealth = results[key].wealth
        hit_idx = wealth[wealth >= target_mult].index
        if len(hit_idx):
            first_hit = hit_idx[0]
            print(f"{results[key].name} hit target on {first_hit.date()}")
        else:
            print(f"{results[key].name} did NOT hit target")


if __name__ == "__main__":
    main()
