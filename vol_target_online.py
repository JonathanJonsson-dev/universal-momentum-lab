from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


###############################################################################
# Configuration and utility structures
###############################################################################


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2014-01-01"
    end_date: str | None = None
    tickers: Sequence[str] = ("SPY", "GLD", "TLT")
    rf_annual: float = 0.02
    rp_lookback: int = 60
    vol_window: int = 20
    max_leverage: float = 4.0
    min_vol_target: float = 0.05
    max_vol_target: float = 0.35
    transaction_cost_bps: float = 5.0
    momentum_window: int = 63
    realized_window: int = 40

    def transaction_cost(self) -> float:
        return self.transaction_cost_bps / 10_000.0

    def rf_daily(self) -> float:
        return math.exp(self.rf_annual / 252.0) - 1.0


@dataclass
class MarketState:
    date: pd.Timestamp
    rp_return: float
    rp_vol: float
    momentum: float
    drawdown: float
    scaling: float


@dataclass
class ModelTrack:
    name: str
    returns: List[float] = field(default_factory=list)
    targets: List[float] = field(default_factory=list)
    scaling: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    realized_vol: List[float] = field(default_factory=list)
    dates: List[pd.Timestamp] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "return": self.returns,
                "vol_target": self.targets,
                "scaling": self.scaling,
                "transaction_cost": self.costs,
                "realized_vol": self.realized_vol,
            },
            index=pd.Index(self.dates, name="date"),
        )


###############################################################################
# Risk parity utilities
###############################################################################


def download_prices(
    tickers: Sequence[str], start: str, end: str | None
) -> pd.DataFrame:
    clean = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not clean:
        raise ValueError("At least one ticker symbol is required.")
    data = yf.download(
        clean,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna()


def compute_risk_parity_returns(
    prices: pd.DataFrame, window: int
) -> tuple[pd.Series, pd.DataFrame]:
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std()
    inv_vol = 1.0 / rolling_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.shift(1).dropna()
    aligned_returns = returns.loc[weights.index]
    portfolio_returns = (weights * aligned_returns).sum(axis=1)
    return portfolio_returns, weights


def compute_equal_weight_uptrend_returns(
    prices: pd.DataFrame,
    short_window: int = 63,
    long_window: int = 210,
) -> pd.Series:
    """
    Builds an equal-weight portfolio of assets whose shorter moving average
    is above their longer moving average. Signals are evaluated monthly and
    lagged one day to avoid look-ahead bias.
    """
    if prices.shape[1] == 0:
        raise ValueError("Price table must contain at least one asset.")
    returns = prices.pct_change()
    short_ma = prices.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = prices.rolling(window=long_window, min_periods=long_window).mean()
    signals = (short_ma > long_ma).astype(float)
    signals = signals.shift(1)
    monthly_signals = signals.resample("ME").last().fillna(0.0)
    weight_counts = monthly_signals.sum(axis=1)
    monthly_weights = monthly_signals.div(weight_counts.replace(0.0, np.nan), axis=0)
    monthly_weights = monthly_weights.fillna(1.0 / prices.shape[1])
    weights = monthly_weights.reindex(signals.index, method="ffill")
    weights = weights.fillna(1.0 / prices.shape[1])
    aligned_returns = returns.loc[weights.index]
    strategy = (weights * aligned_returns).sum(axis=1)
    return strategy.dropna()


def realized_vol(series: Iterable[float]) -> float:
    arr = np.fromiter(series, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=0) * math.sqrt(252.0))


def max_drawdown(wealth: pd.Series) -> float:
    rolling_max = wealth.cummax()
    drawdown = wealth / rolling_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    returns: pd.Series, rf_rate: float, wealth: pd.Series
) -> Dict[str, float]:
    if returns.empty:
        return {
            "CAGR": float("nan"),
            "Volatility": float("nan"),
            "Sharpe": float("nan"),
            "Max Drawdown": float("nan"),
            "Realized MAE": float("nan"),
            "Target Std": float("nan"),
        }
    cagr = (1.0 + returns).prod() ** (252 / len(returns)) - 1.0
    vol = returns.std(ddof=0) * math.sqrt(252)
    excess = returns.mean() * 252 - rf_rate
    sharpe = excess / vol if vol > 0 else float("nan")
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown(wealth),
    }


###############################################################################
# Online target models
###############################################################################


class OnlineVolTargetModel:
    name: str

    def predict(self, state: MarketState) -> float:
        raise NotImplementedError

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        raise NotImplementedError


class EWMAVolatilityTarget(OnlineVolTargetModel):
    """
    Tracks an exponentially weighted volatility estimate and nudges the
    target up or down depending on the sign and magnitude of the
    trailing Sharpe ratio.
    """

    def __init__(
        self,
        init_target: float,
        alpha_vol: float = 0.05,
        alpha_return: float = 0.1,
        feedback: float = 0.25,
        bounds: tuple[float, float] = (0.05, 0.35),
    ) -> None:
        self.name = "EWMA Feedback"
        self._alpha_vol = alpha_vol
        self._alpha_return = alpha_return
        self._feedback = feedback
        self._bounds = bounds
        self._target = init_target
        self._ewma_vol = init_target
        self._ewma_return = 0.0

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._target, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        self._ewma_vol = (
            self._alpha_vol * state.rp_vol + (1.0 - self._alpha_vol) * self._ewma_vol
        )
        self._ewma_return = (
            self._alpha_return * overlay_return
            + (1.0 - self._alpha_return) * self._ewma_return
        )
        sharpe_est = (
            self._ewma_return * math.sqrt(252) / self._ewma_vol
            if self._ewma_vol > 0
            else 0.0
        )
        adjust = math.exp(self._feedback * np.tanh(sharpe_est))
        raw_target = self._ewma_vol * adjust
        lower, upper = self._bounds
        self._target = float(np.clip(raw_target, lower, upper))


class OnlineGradientDescentTarget(OnlineVolTargetModel):
    """
    Performs a one-dimensional online gradient ascent on a smooth
    utility proxy to learn a suitable annual volatility target.
    """

    def __init__(
        self,
        init_target: float,
        learning_rate: float = 0.5,
        penalty: float = 0.05,
        bounds: tuple[float, float] = (0.05, 0.35),
        max_leverage: float = 4.0,
    ) -> None:
        self.name = "Online Gradient"
        self._theta = init_target
        self._eta = learning_rate
        self._penalty = penalty
        self._bounds = bounds
        self._max_lev = max_leverage

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._theta, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        sigma = state.rp_vol
        if not np.isfinite(sigma) or sigma <= 0.0:
            return
        leverage = np.clip(target / sigma, 0.0, self._max_lev)
        denom = 1.0 + leverage * state.rp_return
        if denom <= 0.0:
            denom = 1e-6
        grad = (state.rp_return / denom) / sigma - self._penalty * (leverage - 1.0) / sigma
        self._theta = float(
            np.clip(self._theta + self._eta * grad, self._bounds[0], self._bounds[1])
        )


class RecursiveLeastSquaresTarget(OnlineVolTargetModel):
    """
    Fits a linear mapping from state features to volatility target with
    recursive least squares. Targets track the realised overlay
    volatility, encouraging stability while still reacting to shifts in
    momentum and drawdown.
    """

    def __init__(
        self,
        init_target: float,
        lam: float = 0.98,
        delta: float = 10.0,
        bounds: tuple[float, float] = (0.05, 0.35),
    ) -> None:
        self.name = "Recursive LS"
        self._lam = lam
        self._bounds = bounds
        self._w = np.zeros(4)
        self._P = (1.0 / delta) * np.eye(4)
        self._last_target = init_target

    @staticmethod
    def _features(state: MarketState) -> np.ndarray:
        return np.array(
            [
                1.0,
                state.rp_vol,
                state.momentum,
                state.drawdown,
            ],
            dtype=float,
        )

    def predict(self, state: MarketState) -> float:
        x = self._features(state)
        target = float(self._w @ x) if np.any(self._w) else self._last_target
        lower, upper = self._bounds
        target = float(np.clip(target, lower, upper))
        self._last_target = target
        return target

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        if not np.isfinite(realized_vol_overlay):
            return
        x = self._features(state)
        Px = self._P @ x
        gain_den = self._lam + x.T @ Px
        if gain_den <= 0.0:
            return
        k = Px / gain_den
        error = realized_vol_overlay - self._w @ x
        self._w = self._w + k * error
        self._P = (self._P - np.outer(k, x) @ self._P) / self._lam


class QuadraticFTRLVolTarget(OnlineVolTargetModel):
    """
    Follow-the-regularized-leader style optimiser that accumulates first-
    and second-order information about a smooth risk/return surrogate to
    choose the next annual volatility target.
    """

    def __init__(
        self,
        init_target: float,
        rf_daily: float,
        transaction_cost: float,
        max_leverage: float,
        bounds: tuple[float, float] = (0.05, 0.35),
        lambda_vol: float = 1.5,
        lambda_drawdown: float = 1.0,
        base_curvature: float = 8.0,
        curvature_floor: float = 1e-3,
        barrier: float = 5e-3,
    ) -> None:
        self.name = "Quadratic FTRL"
        self._rf = rf_daily
        self._tc = transaction_cost
        self._max_leverage = max_leverage
        self._bounds = bounds
        self._lambda_vol = lambda_vol
        self._lambda_drawdown = lambda_drawdown
        self._barrier = barrier
        self._curvature_floor = curvature_floor
        self._grad_sum = -init_target * base_curvature
        self._curv_sum = base_curvature
        self._target = init_target

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._target, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        sigma = state.rp_vol
        if not np.isfinite(sigma) or sigma <= 0.0:
            return

        lower, upper = self._bounds
        leverage = np.clip(target / sigma, 0.0, self._max_leverage)
        # Sensitivity of leverage w.r.t. target is zero when we're at the leverage limits.
        if 0.0 < leverage < self._max_leverage:
            d_leverage = 1.0 / sigma
        else:
            d_leverage = 0.0

        grad_return = -(state.rp_return - self._rf) * d_leverage

        grad_cost = 0.0
        if self._tc > 0.0 and d_leverage != 0.0:
            delta_lev = leverage - state.scaling
            if delta_lev != 0.0:
                grad_cost = self._tc * np.sign(delta_lev) * d_leverage

        if np.isfinite(realized_vol_overlay):
            grad_vol = 2.0 * self._lambda_vol * (target - realized_vol_overlay)
        else:
            grad_vol = 0.0

        grad_drawdown = -self._lambda_drawdown * state.drawdown

        eps = 1e-6
        grad_barrier = 0.0
        if self._barrier > 0.0:
            clipped_target = float(np.clip(target, lower + eps, upper - eps))
            grad_barrier = self._barrier * (
                1.0 / (clipped_target - lower) - 1.0 / (upper - clipped_target)
            )

        grad_total = grad_return + grad_cost + grad_vol + grad_drawdown + grad_barrier
        curvature = max(2.0 * self._lambda_vol + self._curvature_floor, self._curvature_floor)

        self._grad_sum += grad_total
        self._curv_sum += curvature
        proposal = -self._grad_sum / self._curv_sum
        self._target = float(np.clip(proposal, lower + eps, upper - eps))


class EWMAKellyVolTarget(OnlineVolTargetModel):
    """
    Uses exponentially weighted estimates of mean and variance to produce a
    fractional-Kelly volatility target with drawdown-aware scaling.
    """

    def __init__(
        self,
        init_target: float,
        rf_daily: float,
        max_leverage: float,
        bounds: tuple[float, float] = (0.05, 0.35),
        alpha_mean: float = 0.05,
        alpha_var: float = 0.05,
        kelly_fraction: float = 0.5,
        drawdown_scale: float = 3.0,
        smoothing: float = 0.35,
        var_floor: float = 1e-5,
        confidence_horizon: float = 90.0,
    ) -> None:
        self.name = "EWMA Kelly"
        self._rf = rf_daily
        self._max_leverage = max_leverage
        self._bounds = bounds
        self._alpha_mean = alpha_mean
        self._alpha_var = alpha_var
        self._kelly_fraction = kelly_fraction
        self._drawdown_scale = drawdown_scale
        self._smoothing = smoothing
        self._var_floor = var_floor
        self._confidence_horizon = confidence_horizon

        self._mean = 0.0
        self._var = var_floor
        self._target = init_target
        self._observations = 0.0

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._target, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        self._observations += 1.0
        rp_ret = state.rp_return

        delta = rp_ret - self._mean
        self._mean += self._alpha_mean * delta
        centered = rp_ret - self._mean
        self._var = (1.0 - self._alpha_var) * self._var + self._alpha_var * centered**2
        self._var = max(self._var, self._var_floor)

        excess_mean = self._mean - self._rf
        if excess_mean <= 0.0:
            proposal = self._bounds[0]
        else:
            kelly_leverage = self._kelly_fraction * excess_mean / self._var
            leverage = float(np.clip(kelly_leverage, 0.0, self._max_leverage))

            confidence = 1.0 - math.exp(-self._observations / max(self._confidence_horizon, 1.0))
            if state.drawdown < 0.0:
                drawdown_scale = math.exp(self._drawdown_scale * state.drawdown)
            else:
                drawdown_scale = 1.0
            leverage *= confidence * drawdown_scale

            sigma = state.rp_vol
            if not np.isfinite(sigma) or sigma <= 0.0:
                return
            proposal = float(np.clip(leverage * sigma, self._bounds[0], self._bounds[1]))

        self._target = (1.0 - self._smoothing) * self._target + self._smoothing * proposal


class SharpeThresholdVolTarget(OnlineVolTargetModel):
    """
    Maintains exponentially weighted mean/variance estimates and deploys
    volatility only when the forward Sharpe proxy clears a threshold,
    enforcing hysteresis to reduce churn.
    """

    def __init__(
        self,
        target_high: float,
        target_low: float,
        rf_daily: float,
        alpha_mean: float = 0.03,
        alpha_var: float = 0.06,
        threshold: float = 0.75,
        band: float = 0.15,
        max_leverage: float = 4.0,
        bounds: tuple[float, float] = (0.05, 0.35),
        smoothing: float = 0.2,
        var_floor: float = 1e-5,
    ) -> None:
        lower, upper = bounds
        self.name = "Sharpe Gate"
        self._target_high = float(np.clip(target_high, lower, upper))
        self._target_low = float(np.clip(target_low, lower, upper))
        self._rf = rf_daily
        self._alpha_mean = alpha_mean
        self._alpha_var = alpha_var
        self._threshold = threshold
        self._band = band
        self._max_leverage = max_leverage
        self._bounds = bounds
        self._smoothing = smoothing
        self._var_floor = var_floor

        self._mean = 0.0
        self._var = var_floor
        self._state_on = False
        self._target = self._target_low

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._target, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        rp_ret = state.rp_return
        delta = rp_ret - self._mean
        self._mean += self._alpha_mean * delta
        centered = rp_ret - self._mean
        self._var = (1.0 - self._alpha_var) * self._var + self._alpha_var * centered**2
        self._var = max(self._var, self._var_floor)

        sharpe_proxy = 0.0
        if self._var > 0.0:
            sharpe_proxy = (self._mean - self._rf) / math.sqrt(self._var)

        if self._state_on:
            if sharpe_proxy < self._threshold - self._band:
                self._state_on = False
        else:
            if sharpe_proxy > self._threshold + self._band:
                self._state_on = True

        desired = self._target_high if self._state_on else self._target_low
        lower, upper = self._bounds
        desired = float(np.clip(desired, lower, upper))
        self._target = (1.0 - self._smoothing) * self._target + self._smoothing * desired


class SigmoidMomentumVolTarget(OnlineVolTargetModel):
    """
    Maps momentum, realised volatility, and drawdown into a smooth target
    via a logistic surface; aims to scale risk only when carry and momentum
    strongly favour it while volatility is subdued.
    """

    def __init__(
        self,
        bounds: tuple[float, float],
        max_leverage: float,
        momentum_scale: float = 20.0,
        vol_scale: float = 8.0,
        drawdown_scale: float = 6.0,
        intercept: float = -0.2,
        smoothing: float = 0.3,
    ) -> None:
        self.name = "Sigmoid Momentum"
        self._bounds = bounds
        self._max_leverage = max_leverage
        self._momentum_scale = momentum_scale
        self._vol_scale = vol_scale
        self._drawdown_scale = drawdown_scale
        self._intercept = intercept
        self._smoothing = smoothing
        self._target = float(np.mean(bounds))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + math.exp(-np.clip(x, -50.0, 50.0))))

    def predict(self, state: MarketState) -> float:
        lower, upper = self._bounds
        return float(np.clip(self._target, lower, upper))

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        sigma = state.rp_vol
        if not np.isfinite(sigma) or sigma <= 0.0:
            return

        momentum_term = self._momentum_scale * state.momentum
        vol_term = self._vol_scale * (sigma - 0.08)
        drawdown_term = self._drawdown_scale * state.drawdown
        activation = self._intercept + momentum_term - vol_term + drawdown_term
        weight = self._sigmoid(activation)
        lower, upper = self._bounds
        desired = lower + weight * (upper - lower)

        leverage = desired / sigma
        if leverage > self._max_leverage:
            desired = self._max_leverage * sigma

        desired = float(np.clip(desired, lower, upper))
        self._target = (1.0 - self._smoothing) * self._target + self._smoothing * desired


class LogisticSignalVolTarget(OnlineVolTargetModel):
    """
    Uses an online logistic classifier on simple state features to gauge the
    probability of a positive overlay return and scales exposure only when
    that probability is sufficiently high.
    """

    def __init__(
        self,
        high_target: float,
        low_target: float = 0.0,
        threshold: float = 0.65,
        learning_rate: float = 0.3,
        l2: float = 1e-4,
        power: float = 2.0,
    ) -> None:
        if high_target <= low_target:
            raise ValueError("high_target must exceed low_target.")
        self.name = "Logistic Signal"
        self._high = high_target
        self._low = low_target
        self._threshold = threshold
        self._eta = learning_rate
        self._l2 = l2
        self._power = power
        self._w = np.zeros(5)
        self._last_x: Optional[np.ndarray] = None
        self._last_prob: float = 0.5
        self._target = low_target

    @staticmethod
    def _features(state: MarketState) -> np.ndarray:
        return np.array(
            [
                1.0,
                state.momentum * 252.0,
                state.rp_vol,
                state.drawdown,
                state.scaling,
            ],
            dtype=float,
        )

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = np.clip(z, -35.0, 35.0)
        return float(1.0 / (1.0 + math.exp(-z)))

    def predict(self, state: MarketState) -> float:
        x = self._features(state)
        prob = self._sigmoid(float(self._w @ x))
        self._last_x = x
        self._last_prob = prob
        if prob <= self._threshold:
            self._target = self._low
        else:
            score = (prob - self._threshold) / (1.0 - self._threshold)
            scaled = score**self._power
            self._target = self._low + scaled * (self._high - self._low)
        return float(self._target)

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        if self._last_x is None:
            return
        label = 1.0 if state.rp_return > 0.0 else 0.0
        error = label - self._last_prob
        self._w = (1.0 - self._eta * self._l2) * self._w + self._eta * error * self._last_x
class MomentumBreakoutVolTarget(OnlineVolTargetModel):
    """
    Binary allocation: embrace a volatility target when medium-term momentum
    is positive and current risk is contained, otherwise stay flat.
    """

    def __init__(
        self,
        high_target: float,
        threshold: float = 6e-4,
        vol_cap: float = 0.12,
        low_target: float = 0.0,
    ) -> None:
        if high_target <= low_target:
            raise ValueError("high_target must exceed low_target.")
        self.name = "Momentum Breakout"
        self._high = high_target
        self._low = low_target
        self._threshold = threshold
        self._vol_cap = vol_cap
        self._target = low_target

    def predict(self, state: MarketState) -> float:
        positive_momentum = state.momentum > self._threshold
        vol_ok = state.rp_vol <= self._vol_cap
        if np.isfinite(state.scaling):
            _ = state.scaling  # ensures attribute touched for coverage parity
        desired = self._high if (positive_momentum and vol_ok) else self._low
        self._target = desired
        return float(desired)

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        # no state to maintain; keep method for interface completeness
        _ = (state, target, overlay_return, realized_vol_overlay)


class Exp3BanditTarget(OnlineVolTargetModel):
    """
    Treats discrete volatility targets as arms of an Exp3 bandit. Arms
    are rewarded using scaled overlay returns, enabling exploration of
    different risk regimes without assuming a fixed functional form.
    """

    def __init__(
        self,
        targets: Sequence[float],
        gamma: float = 0.07,
        seed: int = 42,
    ) -> None:
        if targets is None or len(targets) == 0:
            raise ValueError("Exp3BanditTarget requires at least one target level.")
        self.name = "Exp3 Bandit"
        self._targets = np.array(targets, dtype=float)
        self._gamma = gamma
        self._weights = np.ones_like(self._targets)
        self._rng = np.random.default_rng(seed)
        self._last_arm: Optional[int] = None
        self._last_probs: Optional[np.ndarray] = None

    def _probs(self) -> np.ndarray:
        weight_sum = self._weights.sum()
        base = (1.0 - self._gamma) * (self._weights / weight_sum)
        explore = self._gamma / len(self._weights)
        return base + explore

    def predict(self, state: MarketState) -> float:
        probs = self._probs()
        arm = self._rng.choice(len(self._targets), p=probs)
        self._last_arm = int(arm)
        self._last_probs = probs
        return float(self._targets[arm])

    def update(
        self,
        state: MarketState,
        target: float,
        overlay_return: float,
        realized_vol_overlay: float,
    ) -> None:
        if self._last_arm is None or self._last_probs is None:
            return
        reward = 0.5 + overlay_return / 0.1  # map returns to (roughly) [0,1]
        reward = float(np.clip(reward, 0.0, 1.0))
        arm = self._last_arm
        prob = self._last_probs[arm]
        est_reward = reward / prob
        self._weights[arm] *= math.exp(self._gamma * est_reward / len(self._targets))
        self._last_arm = None
        self._last_probs = None


###############################################################################
# Backtest runner
###############################################################################


def run_backtest(config: BacktestConfig) -> Dict[str, dict]:
    prices = download_prices(config.tickers, config.start_date, config.end_date)
    rp_returns, _ = compute_risk_parity_returns(prices, config.rp_lookback)
    uptrend_returns = compute_equal_weight_uptrend_returns(prices)

    rp_returns = rp_returns.dropna()
    rp_vol = rp_returns.rolling(window=config.vol_window).std() * math.sqrt(252)
    momentum = rp_returns.rolling(window=config.momentum_window).mean().fillna(0.0)

    models: List[OnlineVolTargetModel] = [
        EWMAVolatilityTarget(init_target=0.12, bounds=(config.min_vol_target, config.max_vol_target)),
        EWMAKellyVolTarget(
            init_target=0.12,
            rf_daily=config.rf_daily(),
            max_leverage=config.max_leverage,
            bounds=(config.min_vol_target, config.max_vol_target),
            alpha_mean=0.04,
            alpha_var=0.08,
            kelly_fraction=0.6,
            drawdown_scale=4.0,
            smoothing=0.25,
        ),
        SharpeThresholdVolTarget(
            target_high=config.max_vol_target,
            target_low=config.min_vol_target,
            rf_daily=config.rf_daily(),
            max_leverage=config.max_leverage,
            bounds=(config.min_vol_target, config.max_vol_target),
        ),
        SigmoidMomentumVolTarget(
            bounds=(config.min_vol_target, config.max_vol_target),
            max_leverage=config.max_leverage,
            momentum_scale=35.0,
            vol_scale=10.0,
            drawdown_scale=8.0,
            intercept=-0.3,
            smoothing=0.25,
        ),
        MomentumBreakoutVolTarget(
            high_target=0.14,
            threshold=5.9e-4,
            vol_cap=0.115,
            low_target=0.0,
        ),
        QuadraticFTRLVolTarget(
            init_target=0.12,
            rf_daily=config.rf_daily(),
            transaction_cost=config.transaction_cost(),
            max_leverage=config.max_leverage,
            bounds=(config.min_vol_target, config.max_vol_target),
            lambda_vol=1.6,
            lambda_drawdown=1.0,
            base_curvature=8.0,
        ),
        OnlineGradientDescentTarget(
            init_target=0.12,
            learning_rate=0.4,
            penalty=0.08,
            bounds=(config.min_vol_target, config.max_vol_target),
            max_leverage=config.max_leverage,
        ),
        RecursiveLeastSquaresTarget(
            init_target=0.12,
            lam=0.985,
            delta=20.0,
            bounds=(config.min_vol_target, config.max_vol_target),
        ),
        Exp3BanditTarget(
            targets=np.linspace(config.min_vol_target, config.max_vol_target, 7),
            gamma=0.08,
        ),
    ]

    tracks: Dict[str, ModelTrack] = {model.name: ModelTrack(model.name) for model in models}
    wealth: Dict[str, float] = {model.name: 1.0 for model in models}
    peak_wealth: Dict[str, float] = wealth.copy()
    prev_scaling: Dict[str, float] = {model.name: 1.0 for model in models}
    histories: Dict[str, List[float]] = {model.name: [] for model in models}

    rf_daily = config.rf_daily()
    tc = config.transaction_cost()
    min_date = max(rp_vol.dropna().index.min(), momentum.index.min())
    returns = rp_returns.loc[min_date:]

    for date, rp_ret in returns.items():
        sigma = float(rp_vol.loc[date]) if pd.notna(rp_vol.loc[date]) else float("nan")
        if not np.isfinite(sigma):
            continue
        mom = float(momentum.loc[date]) if pd.notna(momentum.loc[date]) else 0.0

        for model in models:
            name = model.name
            current_wealth = wealth[name]
            current_peak = peak_wealth[name]
            drawdown = (current_wealth / current_peak) - 1.0 if current_peak > 0 else 0.0
            state = MarketState(
                date=date,
                rp_return=rp_ret,
                rp_vol=sigma,
                momentum=mom,
                drawdown=drawdown,
                scaling=prev_scaling[name],
            )
            target = model.predict(state)
            leverage = np.clip(target / sigma, 0.0, config.max_leverage)
            cost = tc * abs(leverage - prev_scaling[name])
            overlay_return = leverage * rp_ret + (1.0 - leverage) * rf_daily - cost
            current_wealth *= 1.0 + overlay_return
            current_peak = max(current_peak, current_wealth)
            wealth[name] = current_wealth
            peak_wealth[name] = current_peak
            prev_scaling[name] = leverage

            history = histories[name]
            history.append(overlay_return)
            if len(history) > config.realized_window:
                history.pop(0)
            realized = realized_vol(history)

            track = tracks[name]
            track.returns.append(overlay_return)
            track.targets.append(target)
            track.scaling.append(leverage)
            track.costs.append(cost)
            track.realized_vol.append(realized)
            track.dates.append(date)

            model.update(state, target, overlay_return, realized)

    results: Dict[str, dict] = {}
    rp_wealth = (1.0 + rp_returns.loc[tracks[next(iter(tracks))].dates[0]:]).cumprod()

    for name, track in tracks.items():
        frame = track.as_frame()
        daily = frame["return"]
        wealth_path = (1.0 + daily).cumprod()
        metrics = compute_metrics(daily, config.rf_annual, wealth_path)
        realized_vs_target = (frame["realized_vol"] - frame["vol_target"]).abs()
        metrics["Realized MAE"] = float(realized_vs_target.mean())
        metrics["Target Std"] = float(frame["vol_target"].std())
        metrics["Terminal Wealth"] = wealth_path.iloc[-1]
        results[name] = {
            "metrics": metrics,
            "returns": daily,
            "wealth": wealth_path,
            "targets": frame["vol_target"],
            "scaling": frame["scaling"],
            "transaction_costs": frame["transaction_cost"],
        }

    results["Risk Parity (unscaled)"] = {
        "metrics": compute_metrics(rp_returns.loc[rp_wealth.index], config.rf_annual, rp_wealth),
        "returns": rp_returns.loc[rp_wealth.index],
        "wealth": rp_wealth,
        "targets": pd.Series(dtype=float),
        "scaling": pd.Series(dtype=float),
        "transaction_costs": pd.Series(dtype=float),
    }

    uptrend_returns = uptrend_returns.loc[uptrend_returns.index.intersection(rp_wealth.index)]
    if not uptrend_returns.empty:
        uptrend_wealth = (1.0 + uptrend_returns).cumprod()
        uptrend_metrics = compute_metrics(uptrend_returns, config.rf_annual, uptrend_wealth)
        uptrend_metrics["Terminal Wealth"] = uptrend_wealth.iloc[-1]
        results["Equal Weight Uptrend"] = {
            "metrics": uptrend_metrics,
            "returns": uptrend_returns,
            "wealth": uptrend_wealth,
            "targets": pd.Series(dtype=float),
            "scaling": pd.Series(dtype=float),
            "transaction_costs": pd.Series(dtype=float),
        }

    return results


###############################################################################
# Reporting utilities
###############################################################################


def build_summary_table(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, bundle in results.items():
        metrics = bundle["metrics"]
        rows.append(
            {
                "Strategy": name,
                "CAGR": metrics.get("CAGR", np.nan),
                "Volatility": metrics.get("Volatility", np.nan),
                "Sharpe": metrics.get("Sharpe", np.nan),
                "Max Drawdown": metrics.get("Max Drawdown", np.nan),
                "Realized MAE": metrics.get("Realized MAE", np.nan),
                "Target Std": metrics.get("Target Std", np.nan),
                "Terminal Wealth": metrics.get("Terminal Wealth", np.nan),
            }
        )
    summary = pd.DataFrame(rows).set_index("Strategy")
    return summary.sort_values("Sharpe", ascending=False)


def plot_results(results: Dict[str, dict], output_path: str | None = None) -> None:
    model_names = [name for name in results if "Risk Parity" not in name]
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Equity curves
    ax = axes[0]
    for name, bundle in results.items():
        wealth = bundle["wealth"]
        if wealth.empty:
            continue
        ax.plot(wealth.index, wealth.values, label=name)
    ax.set_ylabel("Wealth (log scale)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.set_title("Equity Curves")

    # Volatility targets
    ax = axes[1]
    for name in model_names:
        targets = results[name]["targets"]
        if targets.empty:
            continue
        ax.plot(targets.index, targets.values, label=name)
    ax.set_ylabel("Annual Vol Target")
    ax.axhline(0.12, color="gray", linestyle="--", linewidth=1.0, label="Static 12%")
    ax.legend(loc="best")
    ax.set_title("Volatility Targets Over Time")

    # Realized volatility vs targets
    ax = axes[2]
    for name in model_names:
        targets = results[name]["targets"]
        realized = results[name]["returns"].rolling(window=20).std() * math.sqrt(252)
        if realized.empty:
            continue
        ax.plot(realized.index, realized.values, label=f"{name} Realized")
    ax.set_ylabel("Annualized Volatility")
    ax.set_title("Realized Volatility (20-day)")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()


###############################################################################
# Entrypoint
###############################################################################


def main() -> None:
    config = BacktestConfig()
    results = run_backtest(config)
    summary = build_summary_table(results)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("=== Volatility Target Comparison ===")
    print(summary)
    try:
        plot_results(results)
    except Exception as exc:  # pragma: no cover - plotting may fail in headless envs
        print(f"Plotting failed: {exc}")


if __name__ == "__main__":
    main()
