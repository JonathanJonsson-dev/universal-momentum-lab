"""
ERN momentum strategy backtest (fixed)

The prior draft treated Treasury yields as prices. This version downloads price
proxies for Equity/Bonds/Gold and converts T-bill yields to forward returns
before calculating momentum, weights, and P&L.
"""

import numpy as np
import pandas as pd
import yfinance as yf


class ERNMomentumStrategy:
    """
    Implements the ERN Momentum Strategy for tactical asset allocation.
    Asset classes: Equities (S&P 500), Bonds (intermediate Treasuries),
    Gold, and Cash (3-month T-bills).
    """

    def __init__(self, initial_weights=None, expense_ratios=None, transaction_cost=0.0003):
        self.initial_weights = initial_weights or {
            "Equities": 0.70,
            "Bonds": 0.20,
            "Gold": 0.10,
            "Cash": 0.00,
        }
        self.expense_ratios = expense_ratios or {
            "Equities": 0.0003,
            "Bonds": 0.0015,
            "Gold": 0.0009,
            "Cash": 0.0009,
        }
        self.transaction_cost = transaction_cost
        self.momentum_horizons = [8, 9, 10]  # months
        self.proxies = {}

    def _download_single(self, ticker, start_date, end_date):
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            return pd.Series(dtype=float)
        preferred_cols = ["Adj Close", "Close", "adjclose", "close"]
        col = next((c for c in preferred_cols if c in data.columns), None)
        if col is None:
            numeric = data.select_dtypes(include=[np.number]).columns
            if len(numeric) == 0:
                return pd.Series(dtype=float)
            col = numeric[0]
        series = data[col].astype(float).dropna()
        if isinstance(series, pd.DataFrame):
            series = series.squeeze("columns")
        series.name = ticker
        return series

    def _first_available(self, tickers, start_date, end_date):
        for ticker in tickers:
            series = self._download_single(ticker, start_date, end_date)
            if not series.empty:
                return series, ticker
        raise ValueError(f"No data available for tickers: {tickers}")

    def fetch_historical_data(self, start_date, end_date):
        """
        Fetch monthly price data for risk assets and forward cash returns.
        """
        ticker_map = {
            "Equities": ["^SP500TR", "^GSPC", "SPY"],
            "Bonds": ["IEF", "GOVT", "TLT"],
            "Gold": ["GLD", "IAU", "GC=F"],
        }
        price_frames = {}
        for asset, candidates in ticker_map.items():
            series, proxy = self._first_available(candidates, start_date, end_date)
            price_frames[asset] = series
            self.proxies[asset] = proxy

        prices = pd.concat(price_frames, axis=1)
        prices = prices.sort_index().dropna(how="any")

        # Cash: ^IRX is an annualized percent yield. Convert to forward 1-month return.
        cash_series, cash_proxy = self._first_available(["^IRX"], start_date, end_date)
        self.proxies["Cash"] = cash_proxy
        cash_returns = ((1.0 + cash_series / 100.0) ** (1.0 / 12.0)) - 1.0
        cash_returns.name = "Cash"
        cash_returns = cash_returns.dropna()

        common_index = prices.index.intersection(cash_returns.index)
        prices = prices.loc[common_index]
        cash_returns = cash_returns.loc[common_index]

        return prices, cash_returns

    def calculate_momentum_signals(self, prices):
        """
        Momentum = average of price vs. N-month SMA signals across horizons.
        """
        signals = pd.DataFrame(index=prices.index, columns=["Equities", "Bonds", "Gold"], dtype=float)

        for horizon in self.momentum_horizons:
            for asset in ["Equities", "Bonds", "Gold"]:
                rolling_avg = prices[asset].rolling(window=horizon, min_periods=horizon).mean()
                momentum_signal = (prices[asset] > rolling_avg).astype(float)
                signals[f"{asset}_Momentum_{horizon}"] = momentum_signal

        for asset in ["Equities", "Bonds", "Gold"]:
            cols = [f"{asset}_Momentum_{h}" for h in self.momentum_horizons]
            signals[asset] = signals[cols].mean(axis=1)

        return signals[["Equities", "Bonds", "Gold"]].fillna(0.0)

    def calculate_portfolio_weights(self, signals):
        """
        Sequentially allocate unused weight down the stack: Gold → Equity → Bonds → Cash.
        """
        weights = pd.DataFrame(index=signals.index, columns=["Equities", "Bonds", "Gold", "Cash"], dtype=float)

        for i in range(len(signals)):
            gold_signal = float(np.clip(signals["Gold"].iloc[i], 0.0, 1.0))
            equity_signal = float(np.clip(signals["Equities"].iloc[i], 0.0, 1.0))
            bond_signal = float(np.clip(signals["Bonds"].iloc[i], 0.0, 1.0))

            gold_weight = self.initial_weights["Gold"] * gold_signal
            leftover_gold = self.initial_weights["Gold"] - gold_weight

            equity_base = self.initial_weights["Equities"] + leftover_gold
            equity_weight = equity_base * equity_signal
            leftover_equity = equity_base - equity_weight

            bond_base = self.initial_weights["Bonds"] + leftover_equity
            bond_weight = bond_base * bond_signal

            cash_weight = 1.0 - (gold_weight + equity_weight + bond_weight)
            weights.iloc[i] = [equity_weight, bond_weight, gold_weight, cash_weight]

        return weights

    def backtest_strategy(self, start_date, end_date):
        """
        Backtest the momentum strategy over a given period (monthly frequency).
        """
        prices, cash_returns = self.fetch_historical_data(start_date, end_date)
        signals = self.calculate_momentum_signals(prices)
        weights = self.calculate_portfolio_weights(signals)

        asset_returns = prices.pct_change().dropna()
        weights = weights.loc[asset_returns.index]
        weights = weights.shift(1).dropna()  # use previous month weights
        asset_returns = asset_returns.loc[weights.index]

        cash_returns = cash_returns.reindex(weights.index).fillna(0.0)
        returns_with_cash = asset_returns.copy()
        returns_with_cash["Cash"] = cash_returns

        portfolio_returns = (returns_with_cash * weights).sum(axis=1)
        portfolio_returns = portfolio_returns - self._calculate_costs(weights)

        return pd.DataFrame(
            {
                "Returns": portfolio_returns,
                "Equities": weights["Equities"],
                "Bonds": weights["Bonds"],
                "Gold": weights["Gold"],
                "Cash": weights["Cash"],
            }
        )

    def _calculate_costs(self, weights):
        expense_costs = sum(weights[asset] * self.expense_ratios[asset] / 12.0 for asset in weights.columns)
        changes = weights.diff().fillna(0.0).abs()
        transaction_costs = sum(changes[asset] * self.transaction_cost for asset in weights.columns)
        return expense_costs + transaction_costs

    def calculate_performance_metrics(self, returns, risk_free_rate=0.0):
        if returns.empty:
            return {}
        cumulative_returns = (1.0 + returns).cumprod()
        n_years = len(returns) / 12.0
        cagr = cumulative_returns.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
        volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = (
            (returns.mean() * 12.0 - risk_free_rate) / volatility if volatility > 0 else np.nan
        )
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        terminal_wealth = cumulative_returns.iloc[-1]

        return {
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Terminal Wealth": terminal_wealth,
        }


if __name__ == "__main__":
    strategy = ERNMomentumStrategy()
    start_date = "1970-01-01"
    end_date = "2025-12-16"
    results = strategy.backtest_strategy(start_date, end_date)
    metrics = strategy.calculate_performance_metrics(results["Returns"])

    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
