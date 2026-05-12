"""
Dual Momentum Backtest — Index Futures
======================================
Strategy:
  Step 1 (Absolute Momentum): Compare 12-month return of SP500 vs Bonds.
          If bonds win OR both are negative → hold bonds (safe harbor).
  Step 2 (Relative Momentum): If SP500 wins → pick the index future with
          the highest 12-month return from the universe.

Rebalancing: Monthly (end of month).

Data: Downloaded automatically via yfinance (free, no API key needed).
      Tickers map to the closest liquid ETF/index proxies for each future.

Requirements:
    pip install yfinance pandas numpy matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
START_DATE        = "1950-01-01"   # backtest start
END_DATE          = datetime.today().strftime("%Y-%m-%d")
LOOKBACK_MONTHS   = 12             # momentum lookback
INITIAL_CAPITAL   = 100_000        # USD
TRANSACTION_COST  = 0.00          # 0.1% per trade (round-trip 0.2%)

# IG Markets index futures → closest liquid proxy tickers
FUTURES_UNIVERSE = {
    "SP500":       "ES=F",
    # "Nasdaq":      "^NDX",
    # "DAX":         "^GDAXI",
    # "FTSE 100":    "^FTSE",
    # "Nikkei 225":  "^N225",
    # "Hang Seng":   "^HSI",
    # "ASX 200":     "^AXJO",
    # "Euro Stoxx":  "^STOXX50E",
    "Non-US":      "IEFA",
}

BONDS_TICKER  = "ZB=F"   # iShares 20+ Year Treasury Bond ETF (safe harbor)
SP500_TICKER  = "ES=F" # absolute momentum benchmark

# ─────────────────────────────────────────────
# 1. DOWNLOAD DATA
# ─────────────────────────────────────────────
def download_data():
    print("Downloading price data via yfinance…")
    all_tickers = list(set(FUTURES_UNIVERSE.values())) + [BONDS_TICKER]
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=False, progress=False)["Close"]

    # Rename columns: ticker → friendly name
    reverse_map = {v: k for k, v in FUTURES_UNIVERSE.items()}
    rename = {}
    for col in raw.columns:
        if col in reverse_map:
            rename[col] = reverse_map[col]
        elif col == BONDS_TICKER:
            rename[col] = "Bonds"
    raw.rename(columns=rename, inplace=True)

    # Monthly end-of-month prices
    monthly = raw.resample("ME").last()
    print(f"  Data: {monthly.index[0].date()} → {monthly.index[-1].date()} "
          f"({len(monthly)} months)\n")
    return monthly

# ─────────────────────────────────────────────
# 2. COMPUTE 12-MONTH MOMENTUM
# ─────────────────────────────────────────────
def momentum(prices: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Return over the past `months` months (skip most recent month)."""
    return prices.pct_change(months)

# ─────────────────────────────────────────────
# 3. SIGNAL GENERATION
# ─────────────────────────────────────────────
def generate_signals(monthly: pd.DataFrame) -> pd.DataFrame:
    mom = momentum(monthly, LOOKBACK_MONTHS)

    signals = []
    index_cols = list(FUTURES_UNIVERSE.keys())

    for date in mom.index[LOOKBACK_MONTHS:]:
        row = mom.loc[date]
        sp500_mom = row.get("SP500", np.nan)
        bonds_mom  = row.get("Bonds",  np.nan)

        # Step 1: Absolute momentum
        if pd.isna(sp500_mom) or pd.isna(bonds_mom):
            holding = "Bonds"
            reason  = "missing data"
        elif sp500_mom <= 0:
            holding = "Bonds"
            reason  = "SP500 negative mom"
        elif bonds_mom >= sp500_mom:
            holding = "Bonds"
            reason  = "bonds > SP500"
        else:
            # Step 2: Relative momentum — best index future
            idx_moms = {k: row[k] for k in index_cols if k in row and not pd.isna(row[k])}
            if not idx_moms:
                holding = "Bonds"
                reason  = "no index data"
            else:
                holding = max(idx_moms, key=idx_moms.get)
                reason  = f"best index ({idx_moms[holding]:.1%})"

        signals.append({
            "date":     date,
            "holding":  holding,
            "sp500_m":  sp500_mom,
            "bonds_m":  bonds_mom,
            "reason":   reason,
        })

    return pd.DataFrame(signals).set_index("date")

# ─────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────
def backtest(monthly: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    # Monthly returns for each asset
    returns = monthly.pct_change()

    equity   = INITIAL_CAPITAL
    records  = []
    prev_holding = None

    for i, (date, sig) in enumerate(signals.iterrows()):
        holding = sig["holding"]

        # Return for this month = next month's return of the chosen asset
        # (signal at end of month t → hold during month t+1)
        next_dates = returns.index[returns.index > date]
        if len(next_dates) == 0:
            break
        next_date = next_dates[0]

        asset_return = returns.loc[next_date, holding] if holding in returns.columns else 0.0
        if pd.isna(asset_return):
            asset_return = 0.0

        # Transaction cost on switch
        cost = TRANSACTION_COST if (holding != prev_holding and prev_holding is not None) else 0.0
        net_return = asset_return - cost

        equity *= (1 + net_return)
        prev_holding = holding

        records.append({
            "date":         next_date,
            "holding":      holding,
            "gross_return": asset_return,
            "cost":         cost,
            "net_return":   net_return,
            "equity":       equity,
        })

    return pd.DataFrame(records).set_index("date")

# ─────────────────────────────────────────────
# 5. BENCHMARKS
# ─────────────────────────────────────────────
def build_benchmark(monthly: pd.DataFrame, col: str) -> pd.Series:
    rets = monthly[col].pct_change().dropna()
    eq   = INITIAL_CAPITAL * (1 + rets).cumprod()
    eq.name = col
    return eq

# ─────────────────────────────────────────────
# 6. PERFORMANCE METRICS
# ─────────────────────────────────────────────
def metrics(equity_series: pd.Series, label: str) -> dict:
    r = equity_series.pct_change().dropna()
    n = len(r)
    years = n / 12

    total_ret  = equity_series.iloc[-1] / INITIAL_CAPITAL - 1
    cagr       = (1 + total_ret) ** (1 / years) - 1
    vol        = r.std() * np.sqrt(12)
    sharpe     = (cagr - 0.02) / vol if vol > 0 else np.nan  # rf = 2%
    roll_max   = equity_series.cummax()
    drawdown   = (equity_series - roll_max) / roll_max
    max_dd     = drawdown.min()
    calmar     = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Strategy":   label,
        "CAGR":       f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe":     f"{sharpe:.2f}",
        "Max DD":     f"{max_dd:.2%}",
        "Calmar":     f"{calmar:.2f}",
        "Total Ret":  f"{total_ret:.2%}",
    }

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_results(results: pd.DataFrame, sp500_eq: pd.Series,
                 bonds_eq: pd.Series, signals: pd.DataFrame):

    strat_eq = results["equity"]
    fig = plt.figure(figsize=(16, 18), facecolor="#0f1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT  = "#00d4aa"
    SP_COL  = "#4f8ef7"
    BND_COL = "#f7a24f"
    RED     = "#f7594f"
    DARK    = "#0f1117"
    PANEL   = "#1a1d27"
    TEXT    = "#e0e0e0"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.yaxis.label.set_color(TEXT)
        ax.grid(axis="y", color="#2a2d3a", linewidth=0.5, linestyle="--")

    fig.text(0.5, 0.97, "Dual Momentum — Index Futures Backtest",
             ha="center", va="top", color=TEXT, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.955,
             f"12-Month Lookback  ·  Monthly Rebalancing  ·  {START_DATE[:4]}–{END_DATE[:4]}",
             ha="center", va="top", color="#888", fontsize=10)

    # ── Equity curves ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "Equity Curves")
    common_start = max(strat_eq.index[0], sp500_eq.index[0], bonds_eq.index[0])
    s = strat_eq[strat_eq.index >= common_start]
    sp = sp500_eq[sp500_eq.index >= common_start]
    bn = bonds_eq[bonds_eq.index >= common_start]
    # Normalise to same start
    s  = s  / s.iloc[0]  * INITIAL_CAPITAL
    sp = sp / sp.iloc[0] * INITIAL_CAPITAL
    bn = bn / bn.iloc[0] * INITIAL_CAPITAL

    ax1.plot(s.index,  s.values,  color=ACCENT,  lw=2,   label="Dual Momentum", zorder=3)
    ax1.plot(sp.index, sp.values, color=SP_COL,  lw=1.5, label="SP500 B&H",     zorder=2, alpha=0.85)
    ax1.plot(bn.index, bn.values, color=BND_COL, lw=1.5, label="Bonds B&H",     zorder=2, alpha=0.85)
    ax1.fill_between(s.index, INITIAL_CAPITAL, s.values,
                     where=s.values >= INITIAL_CAPITAL, alpha=0.07, color=ACCENT)
    ax1.set_ylabel("Portfolio Value ($)", color=TEXT)
    ax1.legend(facecolor=PANEL, edgecolor="#2a2d3a", labelcolor=TEXT, fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # ── Drawdown ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    style_ax(ax2, "Drawdown")
    dd = (s - s.cummax()) / s.cummax() * 100
    ax2.fill_between(dd.index, 0, dd.values, color=RED, alpha=0.6)
    ax2.plot(dd.index, dd.values, color=RED, lw=1)
    ax2.set_ylabel("Drawdown (%)", color=TEXT)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── Holdings pie ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(PANEL)
    ax3.set_title("Holdings Distribution", color=TEXT, fontsize=11, fontweight="bold", pad=8)
    counts = results["holding"].value_counts()
    colors = plt.cm.cool(np.linspace(0, 1, len(counts)))
    wedges, texts, autotexts = ax3.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": TEXT, "fontsize": 8},
        wedgeprops={"edgecolor": DARK, "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color(DARK)
        at.set_fontsize(7)

    # ── Annual returns bar ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    style_ax(ax4, "Annual Returns — Dual Momentum")
    annual = s.resample("YE").last().pct_change().dropna() * 100
    bar_colors = [ACCENT if v >= 0 else RED for v in annual.values]
    ax4.bar(annual.index.year, annual.values, color=bar_colors,
            width=0.7, edgecolor=DARK, linewidth=0.5)
    ax4.axhline(0, color="#555", lw=0.8)
    ax4.set_ylabel("Return (%)", color=TEXT)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax4.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Rolling 12m return ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    style_ax(ax5, "Rolling 12-Month Return")
    roll12 = s.pct_change(12) * 100
    ax5.plot(roll12.index, roll12.values, color=ACCENT, lw=1.5)
    ax5.fill_between(roll12.index, 0, roll12.values,
                     where=roll12.values >= 0, alpha=0.15, color=ACCENT)
    ax5.fill_between(roll12.index, 0, roll12.values,
                     where=roll12.values < 0, alpha=0.2, color=RED)
    ax5.axhline(0, color="#555", lw=0.8)
    ax5.set_ylabel("Return (%)", color=TEXT)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ── Monthly return heatmap ─────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.set_facecolor(PANEL)
    ax6.set_title("Monthly Returns Heatmap", color=TEXT, fontsize=11, fontweight="bold", pad=8)
    monthly_ret = s.pct_change() * 100
    heat = monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month]).first().unstack()
    heat.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    im = ax6.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=-heat.abs().max().max(), vmax=heat.abs().max().max())
    ax6.set_xticks(range(len(heat.columns)))
    ax6.set_xticklabels(heat.columns, fontsize=7)
    ax6.set_yticks(range(len(heat.index)))
    ax6.set_yticklabels(heat.index, fontsize=7)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax6.text(j, i, f"{heat.iloc[i, j]:.1f}", ha="center", va="center", size=6, color="black" if abs(heat.iloc[i, j]) < heat.abs().max().max()/2 else "white")
    plt.colorbar(im, ax=ax6, shrink=0.8)
    ax6.tick_params(colors=TEXT, labelsize=7)
    ax6.xaxis.tick_top()
    ax6.xaxis.set_label_position("top")

    plt.savefig("dual_momentum_results.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    print("  Chart saved → dual_momentum_results.png")

# ─────────────────────────────────────────────
# 8. TRADE LOG
# ─────────────────────────────────────────────
def print_trade_log(results: pd.DataFrame):
    print("\n── TRADE LOG (last 24 months) ──────────────────────────────")
    print(f"{'Date':<12} {'Holding':<14} {'Gross Ret':>10} {'Cost':>7} {'Net Ret':>10} {'Equity':>12}")
    print("─" * 68)
    for date, row in results.tail(24).iterrows():
        switched = "←" if row["cost"] > 0 else ""
        print(f"{str(date.date()):<12} {row['holding']:<14} "
              f"{row['gross_return']:>9.2%} {row['cost']:>6.2%} "
              f"{row['net_return']:>9.2%} ${row['equity']:>10,.0f} {switched}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  DUAL MOMENTUM — INDEX FUTURES BACKTEST")
    print("=" * 60)

    monthly  = download_data()
    signals  = generate_signals(monthly)
    results  = backtest(monthly, signals)

    # Benchmarks
    sp500_eq = build_benchmark(monthly, "SP500")
    bonds_eq = build_benchmark(monthly, "Bonds")

    # Align to strategy start
    t0 = results.index[0]
    sp500_eq = sp500_eq[sp500_eq.index >= t0]
    bonds_eq = bonds_eq[bonds_eq.index >= t0]

    # Metrics table
    strat_m = metrics(results["equity"], "Dual Momentum")
    sp_m    = metrics(sp500_eq,          "SP500 B&H")
    bn_m    = metrics(bonds_eq,          "Bonds B&H")

    print("\n── PERFORMANCE SUMMARY ─────────────────────────────────────")
    header = ["Strategy", "CAGR", "Volatility", "Sharpe", "Max DD", "Calmar", "Total Ret"]
    row_fmt = "{:<20} {:>8} {:>11} {:>8} {:>8} {:>8} {:>10}"
    print(row_fmt.format(*header))
    print("─" * 78)
    for m in [strat_m, sp_m, bn_m]:
        print(row_fmt.format(*[m[h] for h in header]))

    print("\n── SIGNAL HISTORY (last 12 months) ────────────────────────")
    print(f"{'Date':<12} {'Holding':<14} {'SP500 Mom':>10} {'Bonds Mom':>10} {'Reason'}")
    print("─" * 70)
    for date, row in signals.tail(12).iterrows():
        sp_m_val = f"{row['sp500_m']:.2%}" if not pd.isna(row['sp500_m']) else "n/a"
        bn_m_val = f"{row['bonds_m']:.2%}" if not pd.isna(row['bonds_m']) else "n/a"
        print(f"{str(date.date()):<12} {row['holding']:<14} {sp_m_val:>10} {bn_m_val:>10}  {row['reason']}")

    print_trade_log(results)

    print("\nGenerating charts…")
    plot_results(results, sp500_eq, bonds_eq, signals)

    # Export to CSV
    results.to_csv("dual_momentum_trades.csv")
    signals.to_csv("dual_momentum_signals.csv")
    print("  Trades saved → dual_momentum_trades.csv")
    print("  Signals saved → dual_momentum_signals.csv")
    print("\nDone.\n")

if __name__ == "__main__":
    main()