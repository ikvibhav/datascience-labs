# [Part 6/8] MLOps Systems: Feature Engineering: Average True Range: Gap-Aware Volatility That the High-Low Range Misses

In production risk pipelines, measuring daily volatility correctly is the foundation of position sizing, stop placement, and regime detection. Without accounting for overnight gaps, your volatility measure systematically understates risk on the days that matter most — the ones where price opens far from the previous close. In this post, I implement Average True Range (ATR) in Python and pandas: a volatility indicator that captures intraday range, gap-up risk, and gap-down risk in a single number using Wilder's exponential smoothing. By the end, you will be able to compute ATR, interpret it for position sizing, and understand exactly why each of the three True Range components exists. This article is part of the MLOps Systems: Feature Engineering track, which progresses from raw OHLCV data to production-grade feature pipelines.

![Image Placeholder: Candlestick chart with a True Range annotation showing a gap-up open and the three TR components labeled]

---

## The Problem: Intraday Range Is Not Enough

The naive measure of daily price range is `High - Low`. For a normal trading day, this is accurate. But markets often open above or below the previous close — an earnings release overnight, a macro announcement before the open, or a gap from another market.

Consider two days:

| Day | High | Low | Close | Prev Close | High−Low | True Range |
|---|---|---|---|---|---|---|
| Normal | 105 | 100 | 103 | 102 | 5 | 5 |
| Gap up | 120 | 117 | 118 | 103 | 3 | **17** |

On the gap-up day, the High−Low range is only 3, but the actual volatility experienced — from the previous close at 103 to the intraday high at 120 — was 17. A volatility measure that ignores this gap dramatically understates risk.

True Range corrects this by taking the maximum of three measures that together cover all possible day types.

---

## The True Range Formula

$$
\text{TR}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)
$$

Each component handles a different day type:

| Component | Formula | Captures |
|---|---|---|
| Intraday range | $H_t - L_t$ | Normal trading day |
| Gap up + pullback | $\|H_t - C_{t-1}\|$ | Opens above prev close, trades lower |
| Gap down + recovery | $\|L_t - C_{t-1}\|$ | Opens below prev close, trades higher |

ATR is then a Wilder-smoothed rolling mean of True Range:

$$
\text{ATR}_t = \alpha \cdot \text{TR}_t + (1 - \alpha) \cdot \text{ATR}_{t-1}, \quad \alpha = \frac{1}{N}
$$

Wilder's slow decay ($\alpha = 1/14 \approx 0.071$ for the default 14-day window) means ATR responds conservatively to single-day spikes. A single extreme day raises ATR gradually, which is the correct behaviour for a risk metric — you do not want position sizes to collapse after one anomalous session.

---

## Implementation

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
```

### Step-by-Step Breakdown

**Multi-column validation**

```python
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
```

ATR requires three columns — not just Close. The set-difference check catches any missing column in one pass, and `sorted()` ensures the error message is deterministic and readable.

**Previous close**

```python
prev_close = engineered[close_column].shift(1)
```

`shift(1)` aligns the previous day's close with the current row. This introduces a `NaN` on row 0, which correctly propagates into the True Range on that row — there is no previous close for the first observation.

**True Range construction**

```python
tr = pd.concat(
    [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
    axis=1,
).max(axis=1)
```

This builds a three-column DataFrame with each TR component as a column, then takes the row-wise maximum. It is both readable and efficient — no Python loop, no conditional logic. The `.abs()` calls are essential: gaps can be in either direction and we want the magnitude, not the signed difference.

**Wilder EMA**

```python
engineered[f"atr_{window}"] = tr.ewm(
    alpha=1 / window, min_periods=window, adjust=False
).mean()
```

Three parameters matter here:

- `alpha=1/window` — Wilder's decay factor. For `window=14`, $\alpha \approx 0.071$.
- `adjust=False` — enforces the recursive EMA formula, matching Wilder's original definition exactly.
- `min_periods=window` — suppresses output until a full window is available, preventing warmup leakage.

---

## Why Wilder Smoothing, Not a Simple Rolling Mean?

The naive alternative is `tr.rolling(window=window).mean()`. This is simpler but produces a different result — and one that mismatches the industry standard.

The key difference: a rolling mean gives equal weight to every observation in the window and drops the oldest value abruptly when it exits the window. Wilder's EMA gives decaying weight to older values and never fully forgets them. On a day where a large spike exits a simple rolling mean window, ATR drops discontinuously. With Wilder smoothing, the spike fades gradually, which is more representative of persistent volatility regimes.

> **Pro Tip:** If your ATR values match closely but not exactly with TradingView or Bloomberg, you are likely using a simple rolling mean instead of `ewm(alpha=1/window)`. The Wilder variant is the canonical specification — always use it.

---

## Warmup Period

With `window=14`:
- Row 0: `true_range = NaN` (no previous close)
- Rows 1–14: `true_range` is valid, but `atr_14 = NaN` (warmup period)
- Row 15 onward: both columns are fully valid

```
Total NaN rows for atr_14: 15 (rows 0 through 14)
```

> **Pro Tip:** Never drop only the `NaN` ATR rows using `dropna()` on the ATR column alone — you would silently discard valid rows if other columns also have NaNs from different warmup periods. Handle warmup explicitly per feature column.

---

## Practical Applications

**Position sizing (the primary use case):**

```python
risk_per_trade = 0.01 * portfolio_value    # 1% risk per trade
stop_distance  = 2 * result["atr_14"]      # 2 ATR stop

position_size = risk_per_trade / stop_distance
```

ATR-based stops automatically widen during volatile periods and tighten during quiet ones — no manual adjustment needed as market conditions change.

**Normalised volatility for cross-ticker comparison:**

```python
result["atr_pct"] = result["atr_14"] / result["Close"]
```

Dividing ATR by Close produces a dimensionless volatility ratio, making it comparable across stocks at very different price levels.

**Regime detection:**

```python
result["atr_z"] = (result["atr_14"] - result["atr_14"].rolling(63).mean()) \
                  / result["atr_14"].rolling(63).std()
```

A Z-score of ATR against its own 63-day (quarterly) history flags when volatility is unusually high or low relative to recent norms — useful for switching between risk-on and risk-off position sizing.

---

## Common Pitfalls

| Pitfall | Impact | Fix |
|---|---|---|
| Using `rolling().mean()` instead of `ewm()` | Produces non-standard ATR; mismatches industry tools | Use `ewm(alpha=1/window, adjust=False)` |
| Forgetting `.abs()` on gap components | Gap-down days produce negative TR components | Always use `.abs()` on the gap measures |
| Comparing ATR across price levels | A $5 ATR on a $10 stock vs a $1,000 stock is incomparable | Normalise: `ATR / Close` |
| Dropping only ATR NaN rows | May silently discard valid rows from other features | Handle warmup rows explicitly per feature |
| Using ATR as a directional signal | ATR measures magnitude, not direction | Combine with trend indicators for directional context |

---

## Key Takeaways

- True Range extends the High−Low range to account for overnight gaps — the three components together cover every possible day type.
- ATR applies Wilder's slow EMA ($\alpha = 1/N$) to True Range, producing a conservatively smoothed volatility measure that degrades gracefully after a spike.
- `adjust=False` and `min_periods=window` are mandatory for correctness and leakage safety.
- ATR is most valuable as a position-sizing denominator and regime-detection input — not as a directional signal.
- Normalise ATR by Close price to compare volatility across tickers at different price levels.

---

---

*Series: MLOps Systems — Feature Engineering*

| | |
|---|---|
| **← Previous** | [Part 5/8 — Bollinger Bands](bollinger-bands-blog.md) |
| **This post** | Part 6/8 — Average True Range |
| **Next →** | [Part 7/8 — Wilder Smoothing vs Standard EMA in RSI, MACD, and ATR](to-review/exponential-moving-average-blog.md) |
