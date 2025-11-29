from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat
import yfinance as yf


@dataclass(frozen=True)
class StatisticalFactorConfig:
    top_n: int = 50
    num_factors: int = 5
    lookback: int = 252


DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "NFLX",
    "AVGO",
    "ADBE",
    "CRM",
    "INTU",
    "AMD",
    "INTC",
    "CSCO",
    "QCOM",
    "TXN",
    "AMAT",
    "LRCX",
    "MU",
    "NOW",
    "SNOW",
    "SHOP",
    "UBER",
    "ABNB",
    "PDD",
    "BIDU",
    "BABA",
    "KO",
    "PEP",
    "MCD",
    "SBUX",
    "PG",
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "V",
    "MA",
    "AXP",
    "BAC",
    "JPM",
    "MS",
    "GS",
    "WMT",
    "HD",
    "DIS",
    "COST",
    "NKE",
    "CAT",
)
DEFAULT_START_DATE = "1990-01-01"


@dataclass(frozen=True)
class StrategyResult:
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    metrics: Dict[str, float]


def _to_datetime_index(tday: np.ndarray) -> pd.DatetimeIndex:
    values = np.asarray(tday).ravel()
    if values.size == 0:
        raise ValueError("Empty tday array in fundamental data file.")
    values = values.astype(int)
    return pd.to_datetime(values, format="%Y%m%d")


def _to_symbol_list(raw: np.ndarray) -> list[str]:
    if raw.dtype == object:
        symbols = []
        for item in raw.ravel():
            if isinstance(item, np.ndarray):
                symbols.append("".join(str(x) for x in item.flatten()).strip())
            else:
                symbols.append(str(item).strip())
        return symbols
    return ["".join(row).strip() for row in raw.astype(str)]


def load_fundamental_data(path: Path) -> Tuple[pd.DataFrame, pd.Index]:
    data = loadmat(path)
    try:
        tday = _to_datetime_index(data["tday"])
        syms = _to_symbol_list(data["syms"])
        mid = np.asarray(data["mid"], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing key {exc} in {path}") from exc
    frame = pd.DataFrame(mid, index=tday, columns=syms)
    return frame.sort_index(), frame.columns


def download_price_data(
    tickers: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker symbol is required.")
    clean = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not clean:
        raise ValueError("Tickers list contained only empty symbols.")
    data = yf.download(
        clean,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(
            f"No price data returned for tickers {', '.join(clean)} from {start}."
        )
    if isinstance(data.columns, pd.MultiIndex):
        for level in ("Adj Close", "Close"):
            if level in data.columns.levels[0]:
                data = data[level]
                break
        else:
            raise ValueError(
                "Unexpected data format from yfinance; Close prices unavailable."
            )
    else:
        for column in ("Adj Close", "Close"):
            if column in data.columns:
                data = data[column]
                break
        else:
            raise ValueError(
                "Unexpected data format from yfinance; Close prices unavailable."
            )
    frame = (
        data.rename(columns=str)
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
        .sort_index()
    )
    return frame


def calculate_returns(frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    return frame.pct_change(periods=lag)


def forward_shift(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    return frame.shift(-periods)


def _prepare_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    frame = price_frame.copy()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.dropna(axis=1, how="all")
    frame = frame.astype(float)
    return frame


def _run_statistical_factor_strategy(
    price_frame: pd.DataFrame, config: StatisticalFactorConfig
) -> StrategyResult:
    price_frame = _prepare_price_frame(price_frame)
    ret1 = calculate_returns(price_frame, 1)
    ret_fut1 = forward_shift(ret1, 1)

    dates = price_frame.index
    assets = price_frame.columns
    n_days, n_assets = ret1.shape
    positions = np.zeros((n_days, n_assets))
    ret_pred = np.full(n_assets, np.nan)

    ret1_values = ret1.to_numpy()
    ret_fut_values = ret_fut1.to_numpy()

    for t in range(config.lookback, n_days):
        start = t - config.lookback + 1
        end = t + 1
        if start < 0:
            continue
        window_ret = ret1_values[start:end]
        valid_mask = np.isfinite(window_ret).all(axis=0)
        valid_count = int(valid_mask.sum())
        if valid_count < 3:
            continue
        usable_top_n = min(config.top_n, max(1, (valid_count - 1) // 2))
        if usable_top_n <= 0:
            continue
        window_ret = window_ret[:, valid_mask]
        centered = window_ret - np.mean(window_ret, axis=0, keepdims=True)
        usable_factors = min(
            config.num_factors, centered.shape[0], centered.shape[1]
        )
        if usable_factors == 0:
            continue
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        factors = u[:, :usable_factors] * s[:usable_factors]
        if factors.shape[0] <= 1:
            continue
        X = factors[:-1]
        future_ret = ret_fut_values[start : end - 1, :][:, valid_mask]
        if not np.isfinite(future_ret).all():
            continue
        X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        beta, _, _, _ = np.linalg.lstsq(X_aug, future_ret, rcond=None)
        last_factor = np.concatenate(([1.0], factors[-1]))
        preds = last_factor @ beta

        ret_pred[:] = np.nan
        valid_indices = np.flatnonzero(valid_mask)
        ret_pred[valid_indices] = preds
        finite_idx = np.flatnonzero(np.isfinite(ret_pred))
        if finite_idx.size < 2 * usable_top_n:
            continue
        sorted_idx = finite_idx[np.argsort(ret_pred[finite_idx])]
        positions[t] = 0.0
        positions[t, sorted_idx[:usable_top_n]] = -1.0
        positions[t, sorted_idx[-usable_top_n:]] = 1.0

    positions_frame = pd.DataFrame(positions, index=dates, columns=assets)
    shifted_positions = positions_frame.shift(1).fillna(0.0)
    numerator = (shifted_positions * ret1).sum(axis=1)
    denominator = shifted_positions.abs().sum(axis=1)
    daily_returns = numerator.divide(denominator.where(denominator > 0), fill_value=0.0)
    daily_returns = daily_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    cumulative_returns = (1.0 + daily_returns).cumprod() - 1.0

    active = positions_frame.ne(0).any(axis=1).to_numpy()
    active_idx = np.flatnonzero(active)
    active_daily = (
        daily_returns.iloc[active_idx[0] :] if active_idx.size else daily_returns.iloc[0:0]
    )

    metrics = _compute_metrics(active_daily)
    return StrategyResult(
        daily_returns=daily_returns,
        cumulative_returns=cumulative_returns,
        positions=positions_frame,
        metrics=metrics,
    )


def _max_drawdown(series: pd.Series) -> Tuple[float, int]:
    wealth = 1.0 + series
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else float("nan")
    duration = 0
    worst_duration = 0
    for value in drawdown:
        if value < 0:
            duration += 1
            worst_duration = max(worst_duration, duration)
        else:
            duration = 0
    return max_dd, worst_duration


def _compute_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    if daily_returns.empty:
        return {
            "CAGR": float("nan"),
            "Sharpe": float("nan"),
            "Max Drawdown": float("nan"),
            "Max Drawdown Duration": float("nan"),
            "Calmar": float("nan"),
        }
    cagr = (1.0 + daily_returns).prod() ** (252 / len(daily_returns)) - 1.0
    volatility = daily_returns.std(ddof=0) * math.sqrt(252)
    sharpe = (
        math.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=0)
        if daily_returns.std(ddof=0) > 0
        else float("nan")
    )
    cumret = (1.0 + daily_returns).cumprod() - 1.0
    max_dd, max_ddd = _max_drawdown(cumret)
    calmar = -cagr / max_dd if max_dd < 0 else float("nan")
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Max Drawdown Duration": float(max_ddd),
        "Calmar": calmar,
    }


def run_statistical_factor_test(
    data_source: Union[Path, str, pd.DataFrame], config: StatisticalFactorConfig
) -> StrategyResult:
    if isinstance(data_source, pd.DataFrame):
        price_frame = data_source
    else:
        path = Path(data_source)
        price_frame, _ = load_fundamental_data(path)
    return _run_statistical_factor_strategy(price_frame, config)


def print_report(result: StrategyResult) -> None:
    print("Statistical factor prediction: Out-of-sample")
    for key, value in result.metrics.items():
        if math.isnan(value):
            display = "nan"
        elif key in {"CAGR", "Volatility", "Sharpe", "Calmar"}:
            display = f"{value:.6f}"
        elif key == "Max Drawdown":
            display = f"{value:.6f}"
        else:
            display = f"{value:.0f}"
        print(f"{key:>24}: {display}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the statistical factor strategy using yfinance prices or a MATLAB MAT file."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "Ticker symbols to download from yfinance. "
            "Use comma- or space-separated symbols. Defaults to a diversified US list."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date for yfinance downloads (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date for yfinance downloads (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--use-mat",
        action="store_true",
        help=(
            "Use the MATLAB fundamentalData.mat file instead of downloading prices."
        ),
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("fundamentalData.mat"),
        help="Path to the fundamentalData MAT file (used with --use-mat).",
    )
    parser.add_argument("--top-n", type=int, default=50, help="Number of long/short picks.")
    parser.add_argument(
        "--num-factors", type=int, default=5, help="Number of principal components to use."
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Minimum number of observations required for PCA.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = StatisticalFactorConfig(
        top_n=args.top_n, num_factors=args.num_factors, lookback=args.lookback
    )
    if args.use_mat:
        data_source: Union[pd.DataFrame, Path] = args.data_file
    else:
        raw_tickers = args.tickers if args.tickers is not None else list(DEFAULT_TICKERS)
        tickers = [
            ticker.strip()
            for item in raw_tickers
            for ticker in item.split(",")
            if ticker.strip()
        ]
        try:
            data_source = download_price_data(tickers, args.start_date, args.end_date)
        except Exception as exc:  # pragma: no cover - surface to CLI user
            parser.error(f"Failed to download price data: {exc}")
            return
    result = run_statistical_factor_test(data_source, config)
    print_report(result)


if __name__ == "__main__":
    main()
