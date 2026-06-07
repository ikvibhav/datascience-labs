from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DEFAULT_CLOSE_LAGS = (1, 5, 10, 20)


def _normalize_lags(lags: Iterable[int]) -> list[int]:
    lag_list = sorted(set(lags))
    if not lag_list:
        raise ValueError("At least one lag must be provided.")

    if any(not isinstance(lag, int) or lag <= 0 for lag in lag_list):
        raise ValueError("All lag values must be positive integers.")

    return lag_list


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


# ---------------------------------------------------------------------------
# FR-FE-002: Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(
    data: pd.DataFrame,
    window: int = 14,
    close_column: str = "Close",
) -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI) for the close price.

    RSI measures momentum by comparing the average gain vs average loss over a
    rolling window. The formula is:
        RS  = avg_gain / avg_loss  (Wilder smoothing: ewm with alpha=1/window)
        RSI = 100 - 100 / (1 + RS)

    Values range from 0 to 100. Conventionally:
        >70 → overbought (price rose fast relative to recent history)
        <30 → oversold  (price fell fast relative to recent history)

    Warmup: the first `window` rows will be NaN.

    Args:
        data: Input market data with a DatetimeIndex.
        window: Rolling window size in trading days (default 14).
        close_column: Source close-price column name.

    Returns:
        A copy of the dataframe with an ``rsi_<window>`` column added.
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    if close_column not in data.columns:
        raise ValueError(f"Missing required column: {close_column}")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    engineered = data.copy()
    close = engineered[close_column]

    # Daily price change: positive = gain, negative = loss.
    delta = close.diff()

    # Separate gains (positive deltas) and losses (absolute negative deltas).
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing: exponential moving average with alpha = 1/window.
    # min_periods=window ensures the first value only appears after a full window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    engineered[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return engineered


def compute_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    close_column: str = "Close",
) -> pd.DataFrame:
    """
    Compute the MACD line, signal line, and histogram.

    MACD (Moving Average Convergence Divergence) measures trend direction and
    momentum by comparing two exponential moving averages of the close price.

        macd_line      = EMA(close, fast) - EMA(close, slow)
        signal_line    = EMA(macd_line, signal)
        macd_histogram = macd_line - signal_line

    Interpretation:
        macd_line > 0     → short-term trend is above long-term trend (bullish)
        histogram > 0     → momentum is accelerating upward
        signal crossover  → common buy/sell trigger

    Warmup: the first `slow + signal - 1` rows will be NaN.

    Args:
        data: Input market data with a DatetimeIndex.
        fast: Short EMA window (default 12).
        slow: Long EMA window (default 26).
        signal: Signal line EMA window (default 9).
        close_column: Source close-price column name.

    Returns:
        A copy of the dataframe with three new columns:
        ``macd_line``, ``macd_signal``, ``macd_histogram``.
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    if close_column not in data.columns:
        raise ValueError(f"Missing required column: {close_column}")
    if not (fast > 0 and slow > 0 and signal > 0):
        raise ValueError("fast, slow, and signal must all be positive integers.")
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window.")

    engineered = data.copy()
    close = engineered[close_column]

    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()

    engineered["macd_line"] = macd_line
    engineered["macd_signal"] = signal_line
    engineered["macd_histogram"] = macd_line - signal_line

    return engineered


def compute_bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    close_column: str = "Close",
) -> pd.DataFrame:
    """
    Compute Bollinger Bands for the close price.

    Bollinger Bands measure price volatility relative to a rolling mean:
        middle_band = rolling mean of close over `window` days
        upper_band  = middle_band + num_std * rolling std
        lower_band  = middle_band - num_std * rolling std
        bandwidth   = (upper_band - lower_band) / middle_band

    Interpretation:
        Price near upper band → relatively expensive vs recent history
        Price near lower band → relatively cheap vs recent history
        Narrow bandwidth      → low volatility (potential breakout ahead)
        Wide bandwidth        → high volatility

    Warmup: the first `window - 1` rows will be NaN.

    Args:
        data: Input market data with a DatetimeIndex.
        window: Rolling window size in trading days (default 20).
        num_std: Number of standard deviations for band width (default 2.0).
        close_column: Source close-price column name.

    Returns:
        A copy of the dataframe with four new columns:
        ``bb_middle``, ``bb_upper``, ``bb_lower``, ``bb_bandwidth``.
    """
    if data.empty:
        raise ValueError("Input data is empty.")
    if close_column not in data.columns:
        raise ValueError(f"Missing required column: {close_column}")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    engineered = data.copy()
    close = engineered[close_column]

    rolling = close.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    std = rolling.std(ddof=1)  # sample std, consistent with standard TA practice

    engineered["bb_middle"] = middle
    engineered["bb_upper"] = middle + num_std * std
    engineered["bb_lower"] = middle - num_std * std
    # Bandwidth: normalised spread, useful as a standalone volatility feature.
    engineered["bb_bandwidth"] = (engineered["bb_upper"] - engineered["bb_lower"]) / middle

    return engineered


def compute_atr(
    data: pd.DataFrame,
    window: int = 14,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
) -> pd.DataFrame:
    """
    Compute the Average True Range (ATR).

    ATR measures daily price volatility regardless of direction. The True Range
    for each day is the largest of:
        1. High - Low              (intraday range)
        2. |High - previous Close| (gap up then pull-back)
        3. |Low  - previous Close| (gap down then recovery)

    ATR = rolling mean of True Range over `window` days (Wilder smoothing).

    Interpretation:
        Higher ATR → more volatile days (larger price swings)
        Lower ATR  → quieter, tighter trading sessions

    Warmup: the first `window` rows will be NaN (first row also lacks
    previous Close, so True Range is NaN there too).

    Args:
        data: Input market data with a DatetimeIndex.
        window: Rolling window size in trading days (default 14).
        high_column: Source high-price column name.
        low_column: Source low-price column name.
        close_column: Source close-price column name.

    Returns:
        A copy of the dataframe with two new columns:
        ``true_range`` and ``atr_<window>``.
    """
    required = {high_column, low_column, close_column}
    if data.empty:
        raise ValueError("Input data is empty.")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    engineered = data.copy()
    high = engineered[high_column]
    low = engineered[low_column]
    prev_close = engineered[close_column].shift(1)

    # True range is the maximum of the three range measures.
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    engineered["true_range"] = tr
    # Wilder smoothing matches the original Wilder (1978) ATR definition.
    engineered[f"atr_{window}"] = tr.ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean()

    return engineered


