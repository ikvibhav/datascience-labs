from pathlib import Path

import pandas as pd
from prefect import flow, task

from pipelines.data_ingestion import (
    EXPECTED_COLUMNS,
    build_filename,
    fetch_data,
    save_data,
    validate_data,
)
from utils.feature_engineering import (
    compute_atr,
    compute_bollinger_bands,
    compute_close_lag_features,
    compute_macd,
    compute_rsi,
)

RAW_PATH = Path("data/raw/")
RAW_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


def _build_filename(ticker: str, period: str, suffix: str = "") -> str:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem = f"{ticker}_{period}_{timestamp}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return f"{stem}.csv"


def _coerce_single_ticker_frame(
    data: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    """Flatten yfinance output to a single-ticker OHLCV frame."""
    if isinstance(data.columns, pd.MultiIndex):
        level_1 = set(data.columns.get_level_values(1))
        if ticker not in level_1:
            raise ValueError(
                "Ticker "
                f"{ticker} not found in downloaded data columns: "
                f"{sorted(level_1)}"
            )
        data = data.xs(ticker, axis=1, level=1)

    missing = [c for c in EXPECTED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing expected columns after flattening: "
            f"{missing}"
        )

    return data


@task
def build_feature_dataframe(
    data: pd.DataFrame,
    lag_days: tuple[int, ...] = (1, 5, 10, 20),
    drop_na: bool = False,
) -> pd.DataFrame:
    """Apply all FR-FE-001 and FR-FE-002 features in a fixed sequence."""
    featured = compute_close_lag_features(data, lags=lag_days, drop_na=False)
    featured = compute_rsi(featured, window=14)
    featured = compute_macd(featured, fast=12, slow=26, signal=9)
    featured = compute_bollinger_bands(featured, window=20, num_std=2.0)
    featured = compute_atr(featured, window=14)

    if drop_na:
        featured = featured.dropna()

    return featured


@task
def save_processed_data(data: pd.DataFrame, ticker: str, period: str) -> Path:
    filename = _build_filename(ticker=ticker, period=period, suffix="features")
    output = PROCESSED_PATH / filename
    data.to_csv(output, index=True)
    print(f"Processed feature data saved: {output}")
    return output


@flow
def feature_engineering_pipeline(
    ticker: str,
    period: str = "1y",
    save_raw: bool = True,
    save_processed: bool = True,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    End-to-end ingestion + feature generation pipeline.

    Steps:
    1) Download OHLCV data for one ticker.
    2) Flatten/validate columns.
    3) Compute lag + technical indicator features.
    4) Optionally save raw and processed CSV outputs.
    """
    # Reuse FR-DI tasks for download + base validation.
    downloaded = fetch_data(ticker=[ticker], period=period)
    if not validate_data(downloaded):
        raise ValueError("Data ingestion validation failed.")

    flat_data = _coerce_single_ticker_frame(downloaded, ticker=ticker)

    if save_raw:
        save_data(flat_data, build_filename([ticker], period))

    featured = build_feature_dataframe(flat_data, drop_na=drop_na)

    if save_processed:
        save_processed_data(featured, ticker=ticker, period=period)

    return featured


if __name__ == "__main__":
    feature_engineering_pipeline(ticker="AAPL", period="1y")
