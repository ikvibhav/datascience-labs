# [Part 7/8] MLOps Systems: Feature Engineering: Exponential Moving Averages: Wilder Smoothing vs Standard EMA in RSI, MACD, and ATR

In production momentum pipelines, not all exponential moving averages are the same — and confusing them silently corrupts your signals. Without understanding the distinction between Wilder's smoothing and the standard span-based EMA, it is easy to produce RSI or ATR values that look plausible but diverge significantly from industry benchmarks. In this post, I implement three indicators — RSI, MACD, and ATR — in Python and pandas, exposing the exact EMA variant each requires and why. By the end, you will be able to implement, validate, and configure both forms of EMA for any momentum or volatility use case. This article is part of the MLOps Systems: Feature Engineering track, which progresses from raw OHLCV data to production-grade feature pipelines.

![Image Placeholder: Side-by-side comparison of Wilder EMA vs standard span EMA on the same price series, showing divergence over a 14-day window]

---

## Two Flavours of EMA: What Changes and Why It Matters

An Exponential Moving Average assigns exponentially decaying weights to past observations. The formula is:

$$
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

The key is how $\alpha$ is derived. There are two conventions:

| Convention | Formula | pandas param | Used in |
|---|---|---|---|
| **Wilder smoothing** | $\alpha = \frac{1}{N}$ | `ewm(alpha=1/N)` | RSI, ATR |
| **Standard EMA** | $\alpha = \frac{2}{N+1}$ | `ewm(span=N)` | MACD |

For a 14-period window:
- Wilder: $\alpha = 0.0714$ → very slow decay, gives strong weight to older data
- Standard: $\alpha = 0.1333$ → twice as fast, more responsive to recent moves

Using the wrong variant produces signals that diverge by several percentage points — enough to flip RSI above or below the 70/30 threshold, or generate false MACD crossovers.

---

## RSI: Wilder Smoothing on Gains and Losses

The Relative Strength Index measures momentum by comparing average gains to average losses over a rolling window.

$$
RS = \frac{\overline{\text{Gain}}}{\overline{\text{Loss}}}, \quad RSI = 100 - \frac{100}{1 + RS}
$$

Wilder's original 1978 specification applies his slower EMA to both gain and loss series — this is the convention used by Bloomberg, TradingView, and most institutional platforms.

### Implementation

```python
def compute_rsi(
    data: pd.DataFrame,
    window: int = 14,
    close_column: str = "Close",
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Input data is empty.")
    if close_column not in data.columns:
        raise ValueError(f"Missing required column: {close_column}")
    if window <= 0:
        raise ValueError("window must be a positive integer.")

    engineered = data.copy()
    close = engineered[close_column]

    # Step 1: Daily price change
    delta = close.diff()

    # Step 2: Separate gains and losses
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Step 3: Wilder smoothing — alpha = 1/window, not span
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    # Step 4: RS and RSI (division by zero handled by pandas: inf → RSI=100)
    rs = avg_gain / avg_loss
    engineered[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return engineered
```

### Step-by-Step Walkthrough

**Daily delta and gain/loss separation:**

```python
delta = close.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
```

`clip(lower=0)` zeroes out days where the price moved in the opposite direction. For `delta = [NaN, +1, -2, +3]`:
- `gain = [NaN, 1, 0, 3]`
- `loss = [NaN, 0, 2, 0]`

**Wilder EMA — the critical line:**

```python
avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
```

- `alpha=1/window` — Wilder's decay factor. For `window=14`, $\alpha = 0.0714$.
- `adjust=False` — uses the recursive formula above rather than the weighted-window formulation. This matches Wilder's original definition exactly.
- `min_periods=window` — suppresses output until a full window of data exists.

**Division by zero:**

When `avg_loss` is zero (all gains, no losses over the window), `rs` becomes `inf`, and the RSI formula produces `100 - 0 = 100`. pandas handles this correctly via IEEE 754 arithmetic — no explicit guard is needed.

> **Pro Tip:** RSI values above 70 and below 30 are often described as overbought/oversold thresholds. In practice these are guidelines, not rules. During trending markets, RSI can stay above 70 for weeks. Use RSI divergence (price making new highs while RSI does not) as a stronger signal than the absolute threshold.

---

## MACD: Standard Span-Based EMA on Three Layers

MACD (Moving Average Convergence Divergence) computes the gap between a fast and slow standard EMA of the close price, then applies a third EMA to that gap to produce a signal line.

$$
\text{MACD Line} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})
$$

$$
\text{Signal Line} = \text{EMA}_{9}(\text{MACD Line})
$$

$$
\text{Histogram} = \text{MACD Line} - \text{Signal Line}
$$

### Implementation

```python
def compute_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    close_column: str = "Close",
) -> pd.DataFrame:
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
```

### Key Design Points

**`span=` not `alpha=`:**

```python
ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
```

MACD uses the standard span convention: $\alpha = 2/(N+1)$. For `span=12`, $\alpha = 0.154$. This is faster-decaying than Wilder's method and more responsive to recent closes — which is what you want for a trend-change indicator.

**Warmup arithmetic:**

The signal line is an EMA applied to the MACD line. The full warmup for `macd_histogram` is `slow + signal - 1 = 26 + 9 - 1 = 34` rows. Any training set shorter than 34 rows will produce entirely NaN histograms.

**Constraint enforcement:**

```python
if fast >= slow:
    raise ValueError("fast window must be smaller than slow window.")
```

If `fast >= slow`, `macd_line` would be flat or inverted. This is a configuration error, not a runtime edge case, and should be caught at pipeline initialisation.

> **Pro Tip:** The histogram (not the crossover) is often the most actionable MACD feature. A shrinking histogram with the same sign indicates slowing momentum — a leading warning before the crossover happens. When using MACD as a model feature, include `macd_histogram` as a standalone column rather than engineering a boolean crossover flag.

---

## ATR: Wilder Smoothing on True Range

Average True Range measures daily price volatility by accounting for overnight gaps — the kind of move that the High-Low range alone misses.

$$
\text{True Range}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)
$$

$$
\text{ATR}_t = \alpha \cdot \text{TR}_t + (1 - \alpha) \cdot \text{ATR}_{t-1}, \quad \alpha = \frac{1}{N}
$$

ATR uses the same Wilder smoothing as RSI — slow decay, emphasis on historical volatility.

### Implementation

```python
def compute_atr(
    data: pd.DataFrame,
    window: int = 14,
    high_column: str = "High",
    low_column: str = "Low",
    close_column: str = "Close",
) -> pd.DataFrame:
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

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    engineered["true_range"] = tr
    engineered[f"atr_{window}"] = tr.ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean()

    return engineered
```

### The True Range Construction

```python
tr = pd.concat(
    [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
    axis=1,
).max(axis=1)
```

This builds a three-column DataFrame and takes the row-wise maximum. The three components are:
1. **High − Low** — the intraday range (standard candle body)
2. **|High − Previous Close|** — a gap up followed by an intraday pullback
3. **|Low − Previous Close|** — a gap down followed by an intraday recovery

The `prev_close = close.shift(1)` introduces a NaN on the first row. Combined with `min_periods=window`, this means `atr_14` will be NaN for the first 15 rows (rows 0 and 1–14).

> **Pro Tip:** ATR is most useful as a position-sizing input: `position_size = risk_per_trade / (k * ATR)`. Unlike percentage-based stops, ATR-based stops automatically widen during volatile regimes and tighten during quiet markets. Normalise ATR by the close price (`ATR / Close`) to get a dimensionless volatility ratio comparable across tickers.

---

## Comparing the Two EMA Conventions Side by Side

```python
import pandas as pd

# Simulate 30 days of gains
gains = pd.Series([1.0] * 30)

wilder_ema = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
standard_ema = gains.ewm(span=14, min_periods=14, adjust=False).mean()

print("Wilder (last 5):", wilder_ema.tail(5).values)
print("Standard (last 5):", standard_ema.tail(5).values)
```

On a flat series both converge to 1.0, but they arrive at different rates. On a changing series the divergence is material — Wilder is more conservative and slower, standard EMA is more sensitive to recent observations.

---

## Warmup Summary

| Indicator | Warmup rows (NaN) | Reason |
|---|---|---|
| RSI (window=14) | 14 | Wilder EMA on gain/loss needs full window |
| MACD (12,26,9) | 34 | `slow + signal - 1 = 26 + 9 - 1` |
| ATR (window=14) | 15 | `prev_close` shift + Wilder EMA window |

Always account for these warmup periods when trimming your training dataset.

---

## Key Takeaways

- **RSI and ATR use Wilder's EMA** (`alpha=1/N`): slow, conservative decay that matches the original indicator specifications.
- **MACD uses the standard EMA** (`span=N`): faster decay, better for trend-change detection.
- Using the wrong EMA variant will produce plausible-looking but incorrect indicator values — always verify against a reference dataset.
- `adjust=False` is mandatory in both cases: it enforces the recursive EMA formula rather than the weighted-window approximation.
- `min_periods=window` is non-negotiable for production pipelines: it prevents warmup leakage into model training data.

---

---

*Series: MLOps Systems — Feature Engineering*

| | |
|---|---|
| **← Previous** | [Part 6/8 — Average True Range](../average-true-range-blog.md) |
| **This post** | Part 7/8 — Wilder Smoothing vs Standard EMA |
| **Next →** | [Part 8/8 — Types of Technical Indicators: A Taxonomy](../types-of-technical-indicators-blog.md) |
