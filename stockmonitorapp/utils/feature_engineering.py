from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_CLOSE_LAGS = (1, 5, 10, 20)


def compute_close_lag_features(
    data: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_CLOSE_LAGS,
    close_column: str = "Close",
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Add lag features for the close price column.

    This function is leakage-safe by construction because each lag uses
    historical values only via ``shift(lag)``.

    Args:
        data: Input market data with a DateTime-like index.
        lags: Sequence of lag periods to compute.
        close_column: Source close-price column name.
        drop_na: Whether to drop rows with warmup nulls introduced by lags.

    Returns:
        A copy of the dataframe with lag columns added.

    Raises:
        ValueError: If input is invalid or close column is missing/ambiguous.
    """

    # 1. If data is empty, raise an error.
    if data.empty:
        raise ValueError("Input data is empty.")

    # 2. If close column is missing, raise an error.
    if close_column not in data.columns:
        raise ValueError(f"Missing required column: {close_column}")

    # 3. Ensure lags exist and are valid positive integers.
    lag_list = _normalize_lags(lags)

    # 4. Copy data to avoid modifying original and extract close series.
    engineered = data.copy()
    close_series = engineered[close_column]

    # 5. If multiple close columns are present (e.g. multi-ticker), raise an error to avoid ambiguity.
    # In multi-ticker data, the "Close" column might be a DataFrame with multiple columns (one per ticker)
    if isinstance(close_series, pd.DataFrame):
        raise ValueError(
            "Close column is ambiguous (multiple close columns found). "
            "Provide a single-ticker dataframe before computing lag features."
        )

    # 6. Compute lag features using shift.
    # pd.series.shift() shifts the index by specified number of periods, introducing NaNs for the first 'lag' rows, which is expected and leakage-safe.
    # Eg, For sample_input = [100, 101, 102, 103], lag=1 → lag_1 = [NaN, 100, 101, 102]
    # Why is it called lag? - Because it represents the value of the close price "lagging" behind the current row by a certain number of periods.
    for lag in lag_list:
        engineered[f"{close_column}_lag_{lag}"] = close_series.shift(lag)

    # 7. If drop_na is True, drop rows with NaNs in any of the lag columns.
    if drop_na:
        lag_columns = [f"{close_column}_lag_{lag}" for lag in lag_list]
        engineered = engineered.dropna(subset=lag_columns)

    return engineered


def _normalize_lags(lags: Iterable[int]) -> list[int]:
    lag_list = sorted(set(lags))
    if not lag_list:
        raise ValueError("At least one lag must be provided.")

    if any(not isinstance(lag, int) or lag <= 0 for lag in lag_list):
        raise ValueError("All lag values must be positive integers.")

    return lag_list
