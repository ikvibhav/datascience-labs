# [Part 5/8] MLOps Systems: Feature Engineering: Bollinger Bands: Dynamic Volatility Bands From a Rolling Mean and Standard Deviation

In production stock analytics, price context matters as much as price itself — a close at $150 is meaningless without knowing whether that is stretched, compressed, or neutral relative to recent history. Without a volatility-aware reference frame, alert thresholds are arbitrary, model features lack distributional meaning, and dashboards report numbers in a vacuum. In this post, I implement Bollinger Bands in Python and pandas: a technique that wraps a Simple Moving Average with rolling standard-deviation envelopes and a normalised bandwidth signal. By the end, you will be able to compute upper and lower price boundaries and a squeeze-detection feature, all leakage-safe and pipeline-ready. This article is part of the MLOps Systems: Feature Engineering track, which progresses from raw OHLCV data to production-grade feature pipelines.

![Image Placeholder: Bollinger Bands chart on a stock price series showing upper, middle, and lower bands with bandwidth narrowing before a breakout]

---

## The Problem: Price Without a Baseline

Consider two stocks both trading at $150. One has been oscillating between $148 and $152 for a month; the other just recovered from $120. The first is near its recent mean; the second is far above it. Raw price tells you nothing about which is stretched and which is normal.

A Simple Moving Average gives you the baseline — the expected price over the last N days. But the SMA alone cannot quantify how far the current price has drifted from that baseline. Two stocks at their SMA may have completely different distributions of prices around it.

Bollinger Bands solve this by enveloping the SMA with a range proportional to the actual standard deviation of recent prices. When price is near the upper band, it is statistically extended upward. When it is near the lower band, it has dropped far below recent norms. The SMA becomes an active reference frame, not a passive trend line.

---

## How Bollinger Bands Are Defined

Given a rolling window of $N$ trading days:

$$
\text{Middle Band} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Close}_{t-i}
$$

$$
\text{Upper Band} = \text{Middle} + k \cdot \sigma_{\text{rolling}}
$$

$$
\text{Lower Band} = \text{Middle} - k \cdot \sigma_{\text{rolling}}
$$

Where $k = 2.0$ by default (covering roughly 95% of normally distributed values) and $\sigma$ is the sample standard deviation (`ddof=1`) over the same window.

A fourth derived feature — **bandwidth** — normalises the spread:

$$
\text{Bandwidth} = \frac{\text{Upper} - \text{Lower}}{\text{Middle}}
$$

Narrow bandwidth signals low volatility; it is the primary input to squeeze strategies. Wide bandwidth signals that the market is already in a volatile regime.

---

## Implementation

```python
def compute_bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
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

    rolling = close.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    std = rolling.std(ddof=1)  # sample std, consistent with standard TA practice

    engineered["bb_middle"] = middle
    engineered["bb_upper"] = middle + num_std * std
    engineered["bb_lower"] = middle - num_std * std
    engineered["bb_bandwidth"] = (engineered["bb_upper"] - engineered["bb_lower"]) / middle

    return engineered
```

### Step-by-Step Breakdown

**Input validation**

The function raises early on empty data, missing columns, and non-positive window sizes. These are configuration errors — catching them at the function boundary prevents silent NaN propagation downstream.

**Rolling mean — the SMA middle band**

```python
rolling = close.rolling(window=window, min_periods=window)
middle = rolling.mean()
```

`min_periods=window` is the critical safety constraint. It forces the first computed value to appear only once a full window of observations is available. Without it, pandas computes partial means from row 1 onward, producing unreliable warmup values that can contaminate model training data.

**Rolling standard deviation**

```python
std = rolling.std(ddof=1)
```

`ddof=1` applies Bessel's correction — it divides by $N-1$ rather than $N$. This is the sample standard deviation, which is the standard TA convention. Using `ddof=0` (the population std) systematically underestimates the spread, particularly on smaller windows.

**Band construction**

```python
engineered["bb_upper"] = middle + num_std * std
engineered["bb_lower"] = middle - num_std * std
```

The `num_std` parameter is exposed for configuration. Some strategies use 1.5 for tighter bands on low-volatility assets, or 2.5 during high-volatility regimes.

**Bandwidth**

```python
engineered["bb_bandwidth"] = (engineered["bb_upper"] - engineered["bb_lower"]) / middle
```

The division by `middle` normalises the spread into a dimensionless ratio. A $10 stock and a $1,000 stock are then directly comparable on bandwidth — a 5% bandwidth means the same thing regardless of absolute price level.

---

## Practical Usage

```python
import pandas as pd
from feature_engineering import compute_bollinger_bands

df = pd.read_csv("AAPL_1y.csv", index_col="Date", parse_dates=True)

result = compute_bollinger_bands(df, window=20, num_std=2.0)

print(result[["Close", "bb_lower", "bb_middle", "bb_upper", "bb_bandwidth"]].tail())
```

Example output:

```
            Close   bb_lower  bb_middle   bb_upper  bb_bandwidth
Date
2026-06-24  189.3   182.14    186.70      191.26     0.049
2026-06-25  190.1   182.50    186.90      191.30     0.047
2026-06-26  191.4   182.80    187.10      191.40     0.046
```

When `Close` approaches `bb_upper`, the stock is at the top of its recent volatility range. When bandwidth drops to a multi-month low, a **Bollinger Squeeze** is forming — a period of compression that often precedes a significant directional move.

---

## Warmup Period and Leakage Safety

With `window=20`, the first 19 rows of all four Bollinger columns will be `NaN`. This is correct and intentional. Never backfill or forward-fill these values — doing so introduces lookahead bias into your feature matrix.

> **Pro Tip:** When constructing a train/test split, calculate your feature warmup and trim accordingly. With `window=20` and 252 trading days of history, only rows 19–251 have valid features. Drop warmup rows explicitly before fitting any model; do not assume they are safely filtered downstream.

---

## Interpreting Signals

| Condition | Interpretation |
|---|---|
| `Close > bb_upper` | Price statistically extended above recent mean — possible reversion candidate |
| `Close < bb_lower` | Price statistically compressed below recent mean — possible reversion candidate |
| `bb_bandwidth` at rolling low | Squeeze: volatility compressed, breakout likely |
| `bb_bandwidth` expanding | Volatility expanding, trend may be accelerating |
| Price walking the upper band | Strong uptrend — band touches are not reversals |

The last row is critical. During strong trends, price can "walk" along the upper band for extended periods. Band touches alone are not reversal signals — they must be combined with momentum indicators (such as RSI or MACD) to reduce false signals.

---

## Common Pitfalls

| Pitfall | Impact | Fix |
|---|---|---|
| `min_periods` omitted | Partial-window means corrupt early rows | Always set `min_periods=window` |
| `ddof=0` for std | Underestimates spread on small windows | Use `ddof=1` |
| Forward-filling NaNs in warmup | Introduces lookahead bias | Accept NaNs; filter before modelling |
| Treating band touches as signals | High false-positive rate in trends | Combine with momentum confirmation |
| Applying to multi-ticker DataFrames | `Close` column is ambiguous | Slice to a single ticker first |

---

## Tradeoffs and Design Choices

**SMA vs EMA for the middle band:** Classic Bollinger Bands use a simple rolling mean. This gives equal weight to all observations in the window, producing a stable but slower-responding baseline. An EMA-centred variant exists (sometimes called "Keltner Channels" in a related form), but it diverges from the standard Bollinger specification and should be named and documented separately to avoid confusion.

**Window size:** The default of 20 trading days approximates one calendar month. Shorter windows (10 days) produce tighter, more reactive bands suited to swing trading. Longer windows (50 days) widen the bands and are appropriate for monthly position-sizing signals.

**`num_std`:** At 2.0, roughly 95% of observations fall within the bands under a normal distribution. Stock returns are fat-tailed, so in practice a wider spread (2.5) reduces false breakout signals during volatile regimes.

---

## Key Takeaways

- Bollinger Bands turn the Simple Moving Average into a dynamic reference frame that encodes both trend and volatility in a single, self-scaling feature set.
- `min_periods=window` and `ddof=1` are correctness requirements, not optional parameters — omitting either produces wrong values.
- The bandwidth feature is the most underrated output: it normalises volatility across tickers and detects squeeze setups before price moves.
- All four output columns are leakage-safe — no row contains a value computed from fewer than `window` observations.

---

---

*Series: MLOps Systems — Feature Engineering*

| | |
|---|---|
| **← Previous** | [Part 4/8 — MACD](macd-blog.md) |
| **This post** | Part 5/8 — Bollinger Bands |
| **Next →** | [Part 6/8 — Average True Range](average-true-range-blog.md) |
