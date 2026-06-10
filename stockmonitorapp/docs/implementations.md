
### Feature Engineering Implementations

##### FR-FE-001 Implementation Section

**Implementation file:** `utils/feature_engineering.py`

**Primary function:** `compute_close_lag_features(data, lags=(1, 5, 10, 20), close_column="Close", drop_na=False)`

**Output columns:**
- `Close_lag_1`
- `Close_lag_5`
- `Close_lag_10`
- `Close_lag_20`

**Implementation notes:**
- Lag features are computed with `shift(lag)` to ensure historical-only inputs.
- The function returns a copy of the dataframe and does not mutate input data.
- Warmup rows with nulls are expected for the first 20 rows by default.
- If `drop_na=True`, rows missing lag values are removed.

**Acceptance criteria:**
1. All four lag columns are present after transformation.
2. For any timestamp `t`, `Close_lag_k[t] == Close[t-k]` for each configured lag `k`.
3. No future values are used when computing lag features.
4. Output dataframe retains original columns and appends lag columns.
5. Function raises a clear error when the Close column is missing.

##### FR-FE-002 Implementation Section

**Implementation file:** `utils/feature_engineering.py`

**Functions and output columns:**

| Function | Output columns | Inputs required | Warmup rows |
|---|---|---|---|
| `compute_rsi(data, window=14)` | `rsi_14` | Close | 14 |
| `compute_macd(data, fast=12, slow=26, signal=9)` | `macd_line`, `macd_signal`, `macd_histogram` | Close | 34 |
| `compute_bollinger_bands(data, window=20, num_std=2.0)` | `bb_middle`, `bb_upper`, `bb_lower`, `bb_bandwidth` | Close | 20 |
| `compute_atr(data, window=14)` | `true_range`, `atr_14` | High, Low, Close | 14 |

**What each indicator measures:**
- **RSI** — momentum: how fast price has risen or fallen relative to recent history (0–100 scale).
- **MACD** — trend and momentum: difference between fast and slow exponential moving averages of Close.
- **Bollinger Bands** — volatility: rolling mean ± 2 standard deviations of Close.
- **ATR** — volatility magnitude: rolling mean of the daily true range (High/Low/previous Close).

**Implementation notes:**
- All indicators use pure pandas; no external TA library is required.
- All functions return a copy of the input dataframe and do not mutate it.
- RSI and ATR use Wilder smoothing (`ewm` with `alpha=1/window`).
- Bollinger Bands use sample standard deviation (`ddof=1`).
- Each function validates required input columns and raises `ValueError` on bad input.

**Acceptance criteria:**
1. All output columns listed above are present after transformation.
2. RSI values are bounded in [0, 100] for all non-NaN rows.
3. `macd_histogram` equals `macd_line - macd_signal` for all non-NaN rows.
4. `bb_upper >= bb_middle >= bb_lower` for all non-NaN rows.
5. `atr_14 > 0` for all non-NaN rows (true range is always positive).
6. All functions raise `ValueError` when a required column is missing.
