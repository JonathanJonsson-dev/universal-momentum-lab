import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf


START_DATE = "2005-01-01"
TICKERS = ["SPY", "GLD", "TLT"]
RISK_FREE_ANNUAL = 0.02  # 2% reference cash rate
RISK_PARITY_WINDOW = 60  # trading days for vol estimate
VOL_TARGET = 0.45  # 10% annualized target
VOL_TARGET_WINDOW = 20  # lookback for realized vol of RP portfolio
VOL_TARGET_MAX_LEV = 10.0
STARTING_WEALTH = 850_000
BROWNE_TARGET_MULTIPLE =  100_000_000 / STARTING_WEALTH #2  # aim for +50% per campaign
BROWNE_HORIZON_DAYS = 252 * 10
BROWNE_SIGMA_WINDOW = 60
BROWNE_MAX_LEV = 10.0


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    wealth: pd.Series
    metrics: Dict[str, float]
    extras: Dict[str, object] | None = None


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    data = yf.download(
        tickers, start=start, auto_adjust=True, progress=False, threads=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna()


def compute_risk_parity_returns(
    prices: pd.DataFrame, window: int
) -> Tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std()
    inv_vol = 1.0 / rolling_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.shift(1).dropna()
    aligned_returns = returns.loc[weights.index]
    portfolio_returns = (weights * aligned_returns).sum(axis=1)
    return portfolio_returns, weights


def annualize_rate(daily_rate: float) -> float:
    return (1.0 + daily_rate) ** 252 - 1.0


def daily_rate_from_annual(annual_rate: float) -> float:
    return math.exp(annual_rate / 252) - 1.0


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).std() * math.sqrt(252)


def apply_vol_target(
    base_returns: pd.Series,
    target_vol: float,
    vol_window: int,
    max_leverage: float,
    rf_rate: float,
) -> Tuple[pd.Series, pd.Series]:
    realized = realized_vol(base_returns, vol_window).shift(1)
    scaling = target_vol / realized
    scaling = scaling.replace([np.inf, -np.inf], np.nan)
    scaling = scaling.clip(lower=0.0, upper=max_leverage)
    scaling = scaling.reindex_like(base_returns).fillna(1.0)
    rf_daily = daily_rate_from_annual(rf_rate)
    overlay_returns = scaling * base_returns + (1.0 - scaling) * rf_daily
    return overlay_returns, scaling


def apply_browne_strategy(
    base_returns: pd.Series,
    rf_rate: float,
    target_multiple: float,
    horizon_days: int,
    sigma_window: int,
    max_leverage: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    wealth = 1.0
    rf_daily = daily_rate_from_annual(rf_rate)
    fractions = []
    wealth_path = []
    overlay_returns = []
    cycle_start_index = 0
    target_wealth = wealth * target_multiple
    campaign_hits = 0
    campaign_misses = 0

    returns = base_returns.copy()
    returns_values = returns.values
    index = returns.index

    for idx, date in enumerate(index):
        # reset campaign if target reached or horizon expired
        days_elapsed = idx - cycle_start_index
        if wealth >= target_wealth or days_elapsed >= horizon_days:
            if wealth >= target_wealth:
                campaign_hits += 1
            else:
                campaign_misses += 1
            cycle_start_index = idx
            target_wealth = wealth * target_multiple
            days_elapsed = 0

        days_left = max(horizon_days - days_elapsed, 1)
        tau = days_left / 252.0

        start = max(0, idx - sigma_window)
        window_slice = returns.iloc[start:idx]
        sigma = window_slice.std(ddof=0)
        sigma = float(sigma * math.sqrt(252)) if sigma and sigma > 0 else 0.0

        if sigma == 0.0 or tau <= 0.0:
            fraction = 0.0
        else:
            scaled_ratio = (wealth * math.exp(rf_rate * tau)) / target_wealth
            scaled_ratio = float(np.clip(scaled_ratio, 1e-9, 1 - 1e-9))
            z = norm.ppf(scaled_ratio)
            pdf = norm.pdf(z)
            fraction = (
                (1.0 / (sigma * math.sqrt(tau)))
                * (target_wealth * math.exp(-rf_rate * tau) / wealth)
                * pdf
            )
            fraction = float(np.clip(fraction, -max_leverage, max_leverage))

        daily_return = fraction * returns_values[idx] + (1.0 - fraction) * rf_daily
        wealth = wealth * (1.0 + daily_return)
        overlay_returns.append(daily_return)
        fractions.append(fraction)
        wealth_path.append(wealth)

    returns_series = pd.Series(data=overlay_returns, index=index, name="browne_returns")
    fraction_series = pd.Series(data=fractions, index=index, name="browne_fraction")
    wealth_series = pd.Series(data=wealth_path, index=index, name="browne_wealth")
    extras = {
        "fraction": fraction_series,
        "campaign_hits": campaign_hits,
        "campaign_misses": campaign_misses,
    }
    return returns_series, wealth_series, extras


def to_wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def max_drawdown(wealth: pd.Series) -> float:
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown.min()


def compute_metrics(returns: pd.Series, rf_rate: float) -> Dict[str, float]:
    wealth = to_wealth(returns)
    total_days = len(returns)
    cagr = wealth.iloc[-1] ** (252.0 / total_days) - 1.0
    vol = returns.std() * math.sqrt(252)
    excess_return = returns.mean() * 252 - rf_rate
    sharpe = excess_return / vol if vol > 0 else float("nan")
    calmar = (
        cagr / abs(max_drawdown(wealth)) if max_drawdown(wealth) < 0 else float("nan")
    )
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown(wealth),
        "Calmar": calmar,
        "Terminal Wealth": wealth.iloc[-1],
    }


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

    (
        browne_returns,
        browne_wealth_path,
        browne_extras,
    ) = apply_browne_strategy(
        base_returns=rp_returns,
        rf_rate=RISK_FREE_ANNUAL,
        target_multiple=BROWNE_TARGET_MULTIPLE,
        horizon_days=BROWNE_HORIZON_DAYS,
        sigma_window=BROWNE_SIGMA_WINDOW,
        max_leverage=BROWNE_MAX_LEV,
    )
    browne_metrics = compute_metrics(browne_returns, RISK_FREE_ANNUAL)
    browne_result = BacktestResult(
        name="Browne Goal-Seeking Overlay",
        returns=browne_returns,
        wealth=browne_wealth_path,
        metrics=browne_metrics,
        extras=browne_extras,
    )

    return {"rp": rp_result, "vol_target": vt_result, "browne": browne_result}


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = []
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
        lines.append(f"{key:>15}: {display}")
    return "\n".join(lines)


def main() -> None:
    results = run_backtest()
    rp = results["rp"]
    vt = results["vol_target"]
    browne = results["browne"]

    print("===== Performance Comparison =====")
    for result in (rp, vt, browne):
        print(f"\n{result.name}")
        print(format_metrics(result.metrics))

    vt_scaling = vt.extras.get("scaling") if vt.extras else pd.Series(dtype=float)
    browne_fraction = (
        browne.extras.get("fraction") if browne.extras else pd.Series(dtype=float)
    )

    if isinstance(vt_scaling, pd.Series) and not vt_scaling.empty:
        print("\nExposure Overview")
        print(
            f"Vol target leverage: mean {vt_scaling.mean():.2f}, "
            f"median {vt_scaling.median():.2f}, 95th pct {vt_scaling.quantile(0.95):.2f}"
        )
    if isinstance(browne_fraction, pd.Series) and not browne_fraction.empty:
        print(
            f"Browne fraction: mean {browne_fraction.mean():.2f}, "
            f"median {browne_fraction.median():.2f}, 95th pct {browne_fraction.quantile(0.95):.2f}"
        )

    if browne.extras:
        hits = browne.extras.get("campaign_hits", 0)
        misses = browne.extras.get("campaign_misses", 0)
        total_campaigns = hits + misses
        hit_rate = hits / total_campaigns if total_campaigns else float("nan")
        print(
            f"Browne campaigns: hits {hits}, misses {misses}, "
            f"hit rate {hit_rate:.2%}" if total_campaigns else "Browne campaigns: n/a"
        )

    print("\nSample statistics cover period:")
    print(f"{rp.returns.index.min().date()} -> {rp.returns.index.max().date()}")


if __name__ == "__main__":
    main()
