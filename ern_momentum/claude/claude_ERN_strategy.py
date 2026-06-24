"""
Momentum / Trend-Following asset allocation  +  volatility-target overlay
=========================================================================

Faithful implementation of the long-only momentum strategy from
Early Retirement Now, "SWR Series Part 63"
(Can we increase the Safe Withdrawal Rate with Momentum/Trend-Following?)

Core strategy (per the post)
----------------------------
- 4 assets: Equities, 10y Treasuries (bonds), Gold, Cash (3m T-bills).
- Base weights when all 3 risky signals are positive: 70 / 20 / 10 / 0 (%).
- For each risky asset, build 12 = 3 horizons x 2 formulas x 2 index-versions
  binary momentum signals, then average them to a [0,1] score:
      * horizons:   8, 9, 10 months
      * formulas:   (a) level vs rolling n-month average  ("crossover")
                    (b) level vs level n months ago        ("T-N to T return")
      * versions:   raw total-return index, and index/cash (excess-over-cash)
  Equities use a 2-month average of the "current" level (negative serial corr).
- Translate scores -> weights with a hierarchical reshuffle of unused weight:
      gold-unused -> equity base, equity-unused -> bond base, bond-unused -> cash.
- Costs: per-asset expense ratios (eq .03%, bond .15%, gold .09%, cash .09%)
  and 0.03% per unit of one-way turnover (the "2x" in the post is just both
  sides of a trade summing in total turnover).

Extension (this file)
---------------------
A volatility target. At every monthly rebalance we estimate the portfolio's
realized vol from the **30-day rolling** window of daily asset returns held at
the current momentum weights, then scale the whole risky sleeve by
`target_vol / realized_vol`, moving the residual to/from cash. Long-only and
no-borrow by default (the overlay can only deploy idle cash to lever up, never
borrow), matching the spirit of the original post.

Data
----
Source is **Yahoo Finance** via `yfinance` (`load_yahoo_data`): S&P 500
total-return index, IEF (10y Treasuries), GLD (gold), and BIL (T-bills), all on
a total-return basis (auto_adjust folds in distributions). Common history is
capped by the youngest series (BIL, inception 2007), so the backtest runs
~2007+. Yahoo carries no CPI, so real-return stats are skipped -- splice in FRED
CPIAUCSL yourself if you want them. A live fetch needs internet access to
query{1,2}.finance.yahoo.com.

Run:      `python momentum_strategy.py`
Requires: `pip install yfinance pandas numpy matplotlib`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ASSETS = ["equity", "bond", "gold", "cash"]
RISKY = ["equity", "bond", "gold"]


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    base_weights: dict = field(default_factory=lambda: {
        "equity": 0.70, "bond": 0.20, "gold": 0.10, "cash": 0.00})
    horizons: tuple = (8, 9, 10)               # momentum look-backs, months
    smooth_assets: tuple = ("equity",)         # which assets use the 2m average
    smooth_window: int = 2
    expense_ratios: dict = field(default_factory=lambda: {
        "equity": 0.0003, "bond": 0.0015, "gold": 0.0009, "cash": 0.0009})
    transaction_cost: float = 0.0003           # per unit of one-way turnover
    trading_days: int = 252


@dataclass
class VolTargetConfig:
    enabled: bool = True
    target_vol: float = 0.10      # annualized portfolio vol target
    lookback_days: int = 30       # 30-day rolling realized vol
    max_leverage: float = 1.5     # cap on the risky-sleeve scalar
    min_leverage: float = 0.0
    allow_borrow: bool = False    # if False, can't push cash < 0 (no borrowing)


# ---------------------------------------------------------------------------
# 2. DATA  (Yahoo Finance)
# ---------------------------------------------------------------------------
DEFAULT_YF_TICKERS = {
    "equity": "^SP500TR",   # S&P 500 *total return* index (dividends already in)
    "bond":   "IEF",        # iShares 7-10y Treasury; auto_adjust folds in coupons
    "gold":   "GLD",        # SPDR Gold Trust, spot-price proxy
    "cash":   "BIL",        # SPDR 1-3m T-Bill, cash / T-bill proxy
}


def load_yahoo_data(tickers: dict | None = None,
                    start: str = "2002-01-01", end: str | None = None):
    """Daily total-return indices for the 4 assets, pulled from Yahoo Finance.

    Uses auto_adjust=True so 'Close' is split- and dividend-adjusted: that turns
    each distribution-paying ETF (IEF, BIL) into a total-return series, while
    ^SP500TR and GLD (no distributions) pass through unchanged. All series are
    aligned to their common trading days.

    History is capped by the youngest ticker (BIL starts 2007-05), so the
    default backtest is ~2007+. For the 150-year history in the post you need
    stitched data (Shiller / FRED / the ERN sheet) -- Yahoo can't go that far.

    Drop-in alternatives you can pass via `tickers`:
        equity: ^GSPC, SPY, VOO        bond: SHY, GOVT, TLT
        gold:   IAU, IAUM, GC=F        cash: SGOV, SHV
    Note: Yahoo has no CPI, so this returns cpi=None and real-return stats are
    skipped; add FRED CPIAUCSL yourself if you want them.
    """
    import yfinance as yf
    tickers = tickers or DEFAULT_YF_TICKERS

    closes = {}
    for asset, tk in tickers.items():
        df = yf.download(tk, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(
                f"No data for {asset!r} ({tk!r}). Check the ticker and that "
                f"your network can reach query1.finance.yahoo.com.")
        # single-ticker downloads come back with MultiIndex columns
        # (field, ticker); keep just the field level so df['Close'] works.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        closes[asset] = df["Close"].rename(asset)

    px = pd.concat(closes.values(), axis=1)[ASSETS].dropna()  # align dates
    if px.empty:
        raise RuntimeError("Selected tickers have no overlapping dates.")
    # Save to CSV
    px.to_csv("./ern_momentum/claude/yahoo_asset_data.csv")
    return px, None


# ---------------------------------------------------------------------------
# 3. MOMENTUM SIGNALS  (monthly)
# ---------------------------------------------------------------------------
def _binary_signals(level: pd.Series, cash: pd.Series,
                    cfg: StrategyConfig, smooth: bool) -> pd.DataFrame:
    """The 12 binary (0/1) momentum signals for one risky asset."""
    versions = {"total": level, "excess": level / cash}
    out = {}
    for vname, idx in versions.items():
        current = idx.rolling(cfg.smooth_window).mean() if smooth else idx
        for h in cfg.horizons:
            # (a) crossover: current vs rolling h-month average
            out[f"{vname}_x_{h}"] = (current > idx.rolling(h).mean()).astype(float)
            # (b) return: current vs level h months ago
            out[f"{vname}_r_{h}"] = (current > idx.shift(h)).astype(float)
    df = pd.DataFrame(out)
    # invalidate rows that depend on not-yet-available history
    warmup = max(max(cfg.horizons), cfg.smooth_window)
    df.iloc[:warmup] = np.nan
    return df


def compute_momentum_weights(monthly_px: pd.DataFrame, cfg: StrategyConfig):
    """Return (monthly target weights, raw [0,1] momentum scores)."""
    cash = monthly_px["cash"]
    scores = pd.DataFrame({
        a: _binary_signals(monthly_px[a], cash, cfg,
                           smooth=(a in cfg.smooth_assets)).mean(axis=1)
        for a in RISKY})

    bw = cfg.base_weights
    # hierarchical reshuffle: gold -> equity -> bond -> cash
    gold_w = scores["gold"] * bw["gold"]
    eq_base = bw["equity"] + (bw["gold"] - gold_w)
    eq_w = scores["equity"] * eq_base
    bond_base = bw["bond"] + (eq_base - eq_w)
    bond_w = scores["bond"] * bond_base
    cash_w = 1.0 - (eq_w + bond_w + gold_w)    # equals bond-unused; guards sum=1

    w = pd.DataFrame({"equity": eq_w, "bond": bond_w,
                      "gold": gold_w, "cash": cash_w}).dropna()
    return w, scores.loc[w.index]


# ---------------------------------------------------------------------------
# 4. VOLATILITY-TARGET OVERLAY  (30-day rolling vol)   <-- the extension
# ---------------------------------------------------------------------------
def apply_vol_target(weights: pd.DataFrame, daily_rets: pd.DataFrame,
                     cfg: StrategyConfig, vt: VolTargetConfig):
    """Scale the risky sleeve to hit `target_vol`, using the 30-day rolling
    realized vol of the portfolio at the current momentum weights.

    Returns (vol-targeted monthly weights, leverage series)."""
    td = cfg.trading_days
    rr = daily_rets[RISKY]
    lev = pd.Series(index=weights.index, dtype=float)

    for dt, w in weights.iterrows():
        win = rr.loc[:dt].tail(vt.lookback_days)
        if len(win) < vt.lookback_days:
            lev[dt] = 1.0
            continue
        wr = w[RISKY].to_numpy(float)
        realized = (win.to_numpy() @ wr).std(ddof=1) * np.sqrt(td)
        raw = vt.max_leverage if realized <= 1e-8 else vt.target_vol / realized

        cap = vt.max_leverage
        risky_sum = float(w[RISKY].sum())
        if not vt.allow_borrow and risky_sum > 1e-9:
            cap = min(cap, 1.0 / risky_sum)            # keep cash >= 0
        lev[dt] = float(np.clip(raw, vt.min_leverage, cap))

    w_vt = weights.copy()
    w_vt[RISKY] = weights[RISKY].mul(lev, axis=0)
    w_vt["cash"] = 1.0 - w_vt[RISKY].sum(axis=1)
    return w_vt, lev


# ---------------------------------------------------------------------------
# 5. BACKTEST ENGINE
# ---------------------------------------------------------------------------
def _month_end_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = idx.to_series()
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def backtest(weights: pd.DataFrame, daily_px: pd.DataFrame, cfg: StrategyConfig):
    """Apply monthly target weights to daily returns (weights set at month-end
    t take effect the next trading day -> no look-ahead). Constant-weight
    within the month; expense drag applied daily, turnover cost on rebalance.

    Returns (equity curve [starts at 1.0], realized daily weights, net rets)."""
    td = cfg.trading_days
    er = pd.Series(cfg.expense_ratios)
    rets = daily_px.pct_change().fillna(0.0)

    w_daily = weights.reindex(daily_px.index).ffill().shift(1).dropna(how="all")
    common = w_daily.dropna().index.intersection(rets.index)
    w_daily, rets = w_daily.loc[common], rets.loc[common]

    gross = (w_daily[ASSETS] * rets[ASSETS]).sum(axis=1)
    er_drag = (w_daily[ASSETS] * (er[ASSETS] / td)).sum(axis=1)
    net = gross - er_drag

    # turnover cost at the first trading day after each signal date
    tc = pd.Series(0.0, index=net.index)
    prev = None
    for dt in weights.index:
        future = common[common > dt]
        if len(future) == 0:
            continue
        new = weights.loc[dt, ASSETS].to_numpy(float)
        turnover = np.abs(new).sum() if prev is None else np.abs(new - prev).sum()
        tc.loc[future[0]] += turnover * cfg.transaction_cost
        prev = new

    net = net - tc
    equity = (1.0 + net).cumprod()
    return equity, w_daily, net


def constant_weights(template_index, weights: dict) -> pd.DataFrame:
    return pd.DataFrame({a: weights.get(a, 0.0) for a in ASSETS},
                        index=template_index)


# ---------------------------------------------------------------------------
# 6. STATS
# ---------------------------------------------------------------------------
def perf_stats(equity: pd.Series, cfg: StrategyConfig,
               cpi: pd.Series | None = None, rf: float = 0.03) -> dict:
    td = cfg.trading_days
    r = equity.pct_change().dropna()
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(td)
    ann_mean = r.mean() * td
    sharpe = (ann_mean - rf) / vol if vol > 0 else np.nan
    maxdd = (equity / equity.cummax() - 1).min()
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}
    if cpi is not None:
        infl = (cpi.iloc[-1] / cpi.iloc[0]) ** (1 / yrs) - 1
        out["RealCAGR"] = (1 + cagr) / (1 + infl) - 1
    return out


# ---------------------------------------------------------------------------
# 7. PLOTS
# ---------------------------------------------------------------------------
def make_plots(curves, weights_mom, weights_vt, leverage, roll_vol, vt,
               path="ern_momentum/claude/momentum_results.png"):
    fig, ax = plt.subplots(3, 2, figsize=(15, 13))

    ax[0, 0].set_title("Cumulative growth of $1 (log scale)")
    for name, eq in curves.items():
        ax[0, 0].plot(eq.index, eq.values, label=name, lw=1.4)
    ax[0, 0].set_yscale("log"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

    ax[0, 1].set_title("Drawdowns")
    for name, eq in curves.items():
        dd = eq / eq.cummax() - 1
        ax[0, 1].plot(dd.index, dd.values, label=name, lw=1.1)
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

    ax[1, 0].set_title("Momentum weights (base strategy)")
    ax[1, 0].stackplot(weights_mom.index, *[weights_mom[a] for a in ASSETS],
                       labels=ASSETS, alpha=.85)
    ax[1, 0].set_ylim(0, 1); ax[1, 0].legend(loc="upper left", ncol=4)

    ax[1, 1].set_title("Momentum + vol-target weights")
    ax[1, 1].stackplot(weights_vt.index,
                       *[weights_vt[a].clip(lower=0) for a in ASSETS],
                       labels=ASSETS, alpha=.85)
    ax[1, 1].legend(loc="upper left", ncol=4)

    ax[2, 0].set_title("Vol-target leverage on risky sleeve")
    ax[2, 0].plot(leverage.index, leverage.values, lw=1.0)
    ax[2, 0].axhline(1.0, color="k", ls="--", lw=.8); ax[2, 0].grid(alpha=.3)

    ax[2, 1].set_title("Realized 30-day vol of the momentum portfolio (ann.)")
    ax[2, 1].plot(roll_vol.index, roll_vol.values, lw=.8)
    ax[2, 1].axhline(vt.target_vol, color="r", ls="--", lw=1.0,
                     label=f"target {vt.target_vol:.0%}")
    ax[2, 1].legend(); ax[2, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# 8. DEMO
# ---------------------------------------------------------------------------
def main():
    cfg = StrategyConfig()
    vt = VolTargetConfig()

    daily_px, cpi = load_yahoo_data()

    # ETF Adj-Close already nets fund expense ratios, so charging the configured
    # ratios again would double-count (^SP500TR has no fee; the ~0.03% equity-ETF
    # equivalent is immaterial). Keep transaction costs only.
    cfg.expense_ratios = {a: 0.0 for a in ASSETS}

    print(f"Data: Yahoo Finance | {daily_px.index[0].date()} -> "
          f"{daily_px.index[-1].date()} ({len(daily_px):,} trading days)")

    daily_rets = daily_px.pct_change().fillna(0.0)
    me = _month_end_trading_days(daily_px.index)
    monthly_px = daily_px.loc[me]

    # --- base momentum ---
    w_mom, scores = compute_momentum_weights(monthly_px, cfg)

    # --- momentum + 30-day vol target ---
    w_vt, leverage = apply_vol_target(w_mom, daily_rets, cfg, vt)

    # --- benchmarks ---
    w_static = constant_weights(w_mom.index, {"equity": .70, "bond": .20, "gold": .10})
    w_eq = constant_weights(w_mom.index, {"equity": 1.0})

    curves = {}
    curves["Momentum"], wd_mom, _ = backtest(w_mom, daily_px, cfg)
    curves["Momentum + VolTarget"], wd_vt, _ = backtest(w_vt, daily_px, cfg)
    curves["Static 70/20/10"], _, _ = backtest(w_static, daily_px, cfg)
    curves["Equity buy & hold"], _, _ = backtest(w_eq, daily_px, cfg)
    
    # realized leverage on the risky sleeve (levered risky sum / unlevered risky sum)
    lev = wd_vt[RISKY].sum(axis=1) / wd_mom[RISKY].sum(axis=1)

    print("\nLast 5 days -- UNLEVERED momentum weights:")
    print(wd_mom[ASSETS].tail(5).round(4).to_string())
    print("\nLast 5 days -- leverage + LEVERED (vol-target) weights:")
    print(wd_vt[ASSETS].assign(leverage=lev).tail(5).round(4).to_string())

    # realized 30d vol of the (un-levered) momentum portfolio, for the plot
    roll_vol = (wd_mom[RISKY] * daily_rets.reindex(wd_mom.index)[RISKY]).sum(axis=1) \
        .rolling(vt.lookback_days).std(ddof=1) * np.sqrt(cfg.trading_days)

    stats = pd.DataFrame({k: perf_stats(v, cfg, cpi) for k, v in curves.items()}).T
    cols = [c for c in ["CAGR", "RealCAGR", "Vol", "Sharpe", "MaxDD"]
            if c in stats.columns]
    fmt = stats.copy()
    for c in cols:
        fmt[c] = (fmt[c].map(lambda x: f"{x:6.2f}") if c == "Sharpe"
                  else (fmt[c] * 100).map(lambda x: f"{x:6.2f}%"))
    print("\nAverage momentum weights (base):")
    print(w_mom[ASSETS].mean().map(lambda x: f"{x:5.1%}").to_string())
    print(f"\nAverage vol-target leverage: {leverage.mean():.2f}"
          f"  (min {leverage.min():.2f}, max {leverage.max():.2f})")
    print("\nPerformance (Yahoo data, nominal):\n")
    print(fmt[cols].to_string())

    path = make_plots(curves, w_mom, w_vt, leverage, roll_vol, vt)
    print(f"\nSaved chart -> {path}")
    
    return stats


if __name__ == "__main__":
    main()