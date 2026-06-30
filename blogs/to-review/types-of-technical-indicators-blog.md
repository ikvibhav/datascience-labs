# [Part 8/8] MLOps Systems: Feature Engineering: Types of Technical Indicators in Finance: A Taxonomy for Production Feature Pipelines

In production ML pipelines for financial data, not all features are created equal — and using the wrong type of indicator as a model input is one of the most common sources of silent model degradation. Without a clear taxonomy, engineers conflate trend indicators with momentum indicators, use volatility measures as directional signals, and miss entire classes of useful features. In this post, I map the full landscape of technical indicators used in finance, anchored to concrete Python implementations of RSI, MACD, Bollinger Bands, ATR, and lag features. By the end, you will be able to select the right indicator class for a given modelling objective and understand the design constraints each class imposes on your pipeline. This article is the conceptual foundation for the MLOps Systems: Feature Engineering track.

![Image Placeholder: Taxonomy diagram showing five indicator classes — trend, momentum, volatility, volume, and calendar — with example indicators in each category]

---

## Why a Taxonomy Matters in Production

A technical indicator is a transformation of raw OHLCV (Open, High, Low, Close, Volume) data into a derived signal. Each class of indicator answers a different question:

| Class | Question answered | Example |
|---|---|---|
| **Trend** | Which direction is price moving? | SMA, EMA, MACD line |
| **Momentum** | How fast is price moving? | RSI, MACD histogram, Rate of Change |
| **Volatility** | How large are recent price swings? | Bollinger Bands, ATR, Standard Deviation |
| **Volume** | Is price movement confirmed by trading activity? | OBV, Volume-weighted MA |
| **Calendar** | Are there systematic time-of-year patterns? | Day of week, month, quarter start/end |

Conflating these classes leads to two failure modes in ML:
1. **Redundancy**: Adding three trend indicators and treating them as independent features — they are highly correlated and add no orthogonal information.
2. **Misuse**: Using a volatility indicator (like ATR) as a directional feature — it is symmetric by design and carries no trend information.

---

## Class 1: Trend Indicators

Trend indicators smooth price over time to reveal the underlying direction, filtering out short-term noise.

**Simple Moving Average (SMA)** — equal-weight rolling mean:

$$
\text{SMA}_t = \frac{1}{N} \sum_{i=0}^{N-1} \text{Close}_{t-i}
$$

**Exponential Moving Average (EMA)** — decaying-weight mean, more responsive to recent prices:

$$
\text{EMA}_t = \alpha \cdot \text{Close}_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

Both are implemented in the feature pipeline via `pd.Series.rolling().mean()` (SMA) and `pd.Series.ewm().mean()` (EMA).

**MACD line** — the difference between fast and slow EMAs — is a trend indicator at its core: it is positive when the short-term trend is above the long-term trend, and negative when reversed:

```python
ema_fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
ema_slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
```

**Pipeline constraint:** Trend indicators have a warmup period equal to their window size. Longer windows mean more NaN rows at the start of your training set.

---

## Class 2: Momentum Indicators

Momentum indicators measure the rate and strength of price change, independent of absolute level. They are bounded or oscillating, which makes them useful for detecting overbought/oversold conditions and divergence from price.

**RSI (Relative Strength Index)** — compares average gains to average losses over a window, producing a 0–100 oscillator:

$$
RS = \frac{\overline{\text{Gain}_{14}}}{\overline{\text{Loss}_{14}}}, \quad RSI = 100 - \frac{100}{1 + RS}
$$

```python
delta = close.diff()
gain  = delta.clip(lower=0)
loss  = (-delta).clip(lower=0)

avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

rsi = 100 - (100 / (1 + avg_gain / avg_loss))
```

**MACD histogram** — the difference between MACD line and signal line — measures momentum acceleration:

$$
\text{Histogram} = \text{MACD Line} - \text{Signal Line}
$$

A positive and growing histogram means upward momentum is accelerating. A positive but shrinking histogram is the first warning of momentum decay — often a leading indicator before the crossover event itself.

**Pipeline constraint:** Momentum indicators are bounded (RSI: 0–100; MACD histogram: unbounded but mean-reverting). They are less susceptible to covariate shift than raw price features, which makes them more stable as model inputs across different market regimes.

> **Pro Tip:** RSI divergence — where price makes a new high but RSI does not — is a stronger signal than the absolute 70/30 threshold. For model features, consider engineering `rsi_divergence = close_new_high & ~rsi_new_high` as a boolean column rather than relying on the raw RSI value alone.

---

## Class 3: Volatility Indicators

Volatility indicators measure the magnitude of price swings without expressing a direction. They are symmetric — high volatility is equally consistent with a sharp rally or a sharp sell-off.

**Bollinger Bands** — wrap a rolling SMA with ±k standard deviation envelopes:

$$
\text{Upper} = \text{SMA}_{20} + 2\sigma_{20}, \quad \text{Lower} = \text{SMA}_{20} - 2\sigma_{20}
$$

```python
rolling = close.rolling(window=20, min_periods=20)
middle  = rolling.mean()
std     = rolling.std(ddof=1)

bb_upper = middle + 2.0 * std
bb_lower = middle - 2.0 * std
bb_bandwidth = (bb_upper - bb_lower) / middle
```

The **bandwidth** feature is the most pipeline-useful output: it is a normalised, dimensionless measure of current volatility relative to the middle band.

**ATR (Average True Range)** — gap-aware daily volatility, computed from High, Low, and previous Close:

$$
\text{TR}_t = \max\left(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\right)
$$

$$
\text{ATR} = \text{Wilder EMA of TR}
$$

```python
tr = pd.concat(
    [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
    axis=1,
).max(axis=1)

atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
```

**Pipeline constraint:** Volatility features are symmetric and should never be used as directional signals. To use ATR directionally, you must combine it with a trend indicator (e.g. `atr * sign(macd_line)` as a signed volatility feature).

---

## Class 4: Volume Indicators

Volume indicators confirm whether a price move is supported by trading activity. A large price move on high volume is more credible than the same move on thin volume.

The current pipeline does not implement volume indicators (the implemented features focus on OHLCV price data), but for completeness:

| Indicator | What it measures |
|---|---|
| On-Balance Volume (OBV) | Cumulative buy/sell pressure |
| Volume-Weighted Average Price (VWAP) | Average price weighted by volume — used for execution benchmarking |
| Accumulation/Distribution | Whether volume is flowing into or out of a stock |

**Pipeline constraint:** Volume data is noisier and more exchange-specific than price data. Volume indicators are most valuable as confirmation signals, not primary features.

---

## Class 5: Calendar Features

Calendar features encode systematic time-based patterns in market behaviour — day-of-week effects, month-end rebalancing, earnings season seasonality, and quarter boundaries.

```python
engineered["day_of_week"]      = dt_index.dayofweek      # 0=Monday
engineered["month"]            = dt_index.month
engineered["quarter"]          = dt_index.quarter
engineered["is_month_end"]     = dt_index.is_month_end
engineered["is_quarter_start"] = dt_index.is_quarter_start
```

These are derived purely from the DatetimeIndex — they carry no price information and introduce no warmup period. They are also perfectly reproducible and leakage-safe by construction.

**Pipeline constraint:** Calendar features are categorical. Tree-based models handle them natively; linear models require one-hot encoding. Boolean calendar features (`is_month_end`, `is_quarter_start`) should remain as booleans for tree models — converting them to integers is fine but introduces no additional information.

---

## Class 6: Lag Features

Lag features bring historical price values forward as explicit inputs to the model, allowing it to learn time-series dependencies without requiring sequential model architectures.

```python
for lag in [1, 5, 10, 20]:
    engineered[f"Close_lag_{lag}"] = close.shift(lag)
```

`shift(lag)` is leakage-safe by construction — `Close_lag_1` on today's row contains yesterday's close, which is historical information.

**Pipeline constraint:** Lag features have variable warmup periods. `Close_lag_20` requires 20 valid rows before producing its first non-NaN output. When combining lag features of different lengths, the effective warmup is the maximum lag.

---

## Combining Indicator Classes: Practical Guidelines

| Objective | Recommended classes |
|---|---|
| Trend-following model | Trend + Momentum (confirm trend strength) |
| Mean-reversion model | Volatility + Momentum (detect stretched conditions) |
| Regime detection | Volatility (bandwidth, ATR) + Calendar |
| Position sizing | Volatility only (ATR) |
| Breakout detection | Bollinger Bandwidth squeeze + Momentum crossover |

**Avoid:**
- Multiple indicators from the same class without orthogonality checks (high mutual information)
- Treating bounded oscillators (RSI) and unbounded trend features (EMA) with the same scaling strategy
- Using raw price columns (`Close`) alongside lag features — the lag features already encode the price history

---

## Warmup Summary Across All Implemented Indicators

| Feature | Warmup rows (NaN) | Window parameter |
|---|---|---|
| `Close_lag_N` | N | lags |
| `rsi_14` | 14 | window=14 |
| `macd_histogram` | 34 | fast=12, slow=26, signal=9 |
| `bb_*` | 19 | window=20 |
| `atr_14` | 15 | window=14 |
| Calendar features | 0 | — |

When stacking features from multiple classes, the effective warmup for the combined feature matrix is the **maximum individual warmup** — 34 rows in the case of the full indicator set above.

---

## Key Takeaways

- Technical indicators fall into five classes: trend, momentum, volatility, volume, and calendar — each answering a distinct question about price behaviour.
- Conflating classes produces redundant features and silent model degradation; treat class membership as a first-class design constraint.
- Volatility indicators are symmetric — combine them with trend indicators before using them as directional features.
- The effective warmup for a combined feature matrix is the maximum individual warmup across all features; account for this before train/test splitting.
- Lag features and calendar features are the cleanest inputs for ML models — no smoothing assumptions, no EMA convention choices, no warmup ambiguity (for calendar).

---

*Deeper dives: [Part 5/8 — Bollinger Bands](bollinger-bands-blog.md) covers the rolling mean and standard deviation pipeline in detail. [Part 6/8 — Average True Range](average-true-range-blog.md) covers True Range construction and Wilder smoothing. [Part 2/8 — EMA Primer](exponential-moving-average-primer-blog.md) covers the smoothing mechanics underlying RSI, MACD, and ATR.*

---

*Series: MLOps Systems — Feature Engineering*

| | |
|---|---|
| **← Previous** | [Part 7/8 — Wilder Smoothing vs Standard EMA](to-review/exponential-moving-average-blog.md) |
| **This post** | Part 8/8 — Types of Technical Indicators: A Taxonomy |
