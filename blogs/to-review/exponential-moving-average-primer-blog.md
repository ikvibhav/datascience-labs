# [Part 2/8] MLOps Systems: Feature Engineering: Exponential Moving Average Primer: From a Loop to pandas in Plain Python

In time-series analysis, raw data almost always contains noise — and filtering that noise is one of the first steps toward building reliable signals. Without a smoothing step, every spike, gap, and one-off reading feeds directly into your models and dashboards, producing unstable outputs. In this post, I build an Exponential Moving Average (EMA) filter from scratch in plain Python, then show how the same logic maps directly onto a single pandas line. You will gain an intuitive grasp of why EMA weights recent data more heavily than old data, and exactly what the smoothing factor controls. This post is the primer for the MLOps Systems: Feature Engineering track — read this before tackling the RSI, MACD, and ATR articles that build on top of it.

![Image Placeholder: Animated diagram showing how EMA weights decay exponentially from the most recent observation backward in time]

---

## Start Here: What Is a Moving Average?

A **moving average** reduces noise in a sequence of numbers by replacing each value with a summary of the values around it.

The simplest version is the **Simple Moving Average (SMA)**: take the last N values and compute their mean. Every observation in the window gets the same weight — the value from 20 days ago counts exactly as much as yesterday's value.

```python
prices = [100, 102, 101, 105, 107, 110, 108]

def sma(values, window):
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)   # not enough history yet
        else:
            window_slice = values[i - window + 1 : i + 1]
            result.append(sum(window_slice) / window)
    return result

print(sma(prices, window=3))
# [None, None, 101.0, 102.67, 104.33, 107.33, 108.33]
```

This works, but equal weighting has a problem: old data and new data carry identical influence. If the price jumps significantly yesterday, the SMA takes N days to fully reflect that change.

---

## The Exponential Moving Average: Recent Data Matters More

An **Exponential Moving Average (EMA)** solves this by assigning weights that decay exponentially as you look further back in time.

Instead of a sliding window, EMA is a **recursive filter** — each new output is a blend of the current input and the previous output:

$$
\text{EMA}_t = \alpha \cdot x_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

Where:
- $x_t$ is the new value (e.g. today's price)
- $\text{EMA}_{t-1}$ is the previous EMA value
- $\alpha$ is the **smoothing factor**, a number between 0 and 1

The parameter $\alpha$ controls the trade-off:

| $\alpha$ value | Behaviour |
|---|---|
| Close to 1.0 | Weights almost entirely the current value — very reactive, almost no smoothing |
| Close to 0.0 | Weights almost entirely the previous EMA — very smooth, slow to respond |
| 0.1 – 0.3 | Typical range for financial smoothing |

---

## Building EMA From Scratch in Plain Python

```python
def ema(values, alpha):
    """
    Compute the Exponential Moving Average of a list of numbers.

    Args:
        values: A list of numeric values (e.g. daily close prices).
        alpha:  Smoothing factor. Must be between 0 (exclusive) and 1 (inclusive).

    Returns:
        A list of EMA values, same length as the input.
        The first output is seeded with the first input value.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between 0 (exclusive) and 1 (inclusive).")
    if not values:
        return []

    result = [values[0]]           # seed: first EMA equals the first observation

    for x in values[1:]:
        prev_ema = result[-1]
        new_ema = alpha * x + (1 - alpha) * prev_ema
        result.append(new_ema)

    return result
```

Let us trace through a small example manually to make the formula concrete:

```python
prices = [100, 102, 101, 105, 107]
alpha  = 0.3

output = ema(prices, alpha=0.3)
print([round(v, 2) for v in output])
```

Step-by-step:

| Step | $x_t$ | Calculation | $\text{EMA}_t$ |
|---|---|---|---|
| 0 | 100 | seed | 100.00 |
| 1 | 102 | 0.3 × 102 + 0.7 × 100 | 100.60 |
| 2 | 101 | 0.3 × 101 + 0.7 × 100.60 | 100.72 |
| 3 | 105 | 0.3 × 105 + 0.7 × 100.72 | 101.00 (approx) |
| 4 | 107 | 0.3 × 107 + 0.7 × 101.00 | 103.30 (approx) |

Notice how the EMA tracks the upward move in prices 3–4, but stays below the raw values. That lag is the smoothing effect — the EMA is slower to react because it carries memory of earlier, lower values.

---

## Visualising the Effect of Alpha

```python
prices = [100, 102, 98, 110, 105, 112, 108, 115, 111, 120]

slow_ema = ema(prices, alpha=0.1)    # heavy smoothing
fast_ema = ema(prices, alpha=0.5)    # light smoothing

for i, (p, s, f) in enumerate(zip(prices, slow_ema, fast_ema)):
    print(f"Day {i:2d}  price={p:5.1f}  slow(α=0.1)={s:6.2f}  fast(α=0.5)={f:6.2f}")
```

```
Day  0  price=100.0  slow(α=0.1)=100.00  fast(α=0.5)=100.00
Day  1  price=102.0  slow(α=0.1)=100.20  fast(α=0.5)=101.00
Day  2  price= 98.0  slow(α=0.1)=100.18  fast(α=0.5)= 99.50
Day  3  price=110.0  slow(α=0.1)=101.16  fast(α=0.5)=104.75
Day  4  price=105.0  slow(α=0.1)=101.55  fast(α=0.5)=104.88
Day  5  price=112.0  slow(α=0.1)=102.59  fast(α=0.5)=108.44
...
```

The slow EMA barely moves; the fast EMA follows the price closely. Neither is "better" in isolation — the right choice depends on how much lag you can tolerate versus how much noise you want to suppress.

---

## The Same Thing in pandas

pandas' `ewm()` method implements exactly the same recursive formula. The `alpha` parameter maps directly:

```python
import pandas as pd

prices = pd.Series([100, 102, 98, 110, 105, 112, 108, 115, 111, 120])

slow_ema = prices.ewm(alpha=0.1, adjust=False).mean()
fast_ema = prices.ewm(alpha=0.5, adjust=False).mean()

print(slow_ema.round(2))
print(fast_ema.round(2))
```

Two details matter here:

**`adjust=False`** tells pandas to use the recursive formula:

$$
\text{EMA}_t = \alpha \cdot x_t + (1-\alpha) \cdot \text{EMA}_{t-1}
$$

Without it, pandas defaults to `adjust=True`, which uses a weighted-window formulation that produces slightly different results — especially at the start of the series. Always use `adjust=False` when you want the standard recursive EMA definition.

**Seeding:** pandas seeds the EMA with the first observation, identical to the plain Python implementation above.

---

## Connecting Alpha to a Window Size

In financial contexts, it is often more intuitive to think in terms of a window (e.g. "14-day EMA") rather than a raw `alpha` value. Two common conversions exist:

**Standard EMA (span convention):**

$$
\alpha = \frac{2}{N + 1}
$$

A 12-day EMA uses $\alpha = 2/13 \approx 0.154$.

```python
# Equivalent: ewm(alpha=2/13) and ewm(span=12)
ema_span = prices.ewm(span=12, adjust=False).mean()
```

**Wilder smoothing:**

$$
\alpha = \frac{1}{N}
$$

A 14-day Wilder EMA uses $\alpha = 1/14 \approx 0.071$ — a much slower filter. This is the convention used in RSI and ATR.

```python
# Wilder's method (used in RSI, ATR)
wilder_ema = prices.ewm(alpha=1/14, adjust=False).mean()
```

> **Pro Tip:** If you compare your RSI output against TradingView or Bloomberg and the values are close but not identical, you are almost certainly using `span=14` (standard EMA) instead of `alpha=1/14` (Wilder's EMA). The two produce different smoothing strengths, and only the Wilder version matches the original 1978 specification.

---

## Why EMA Over SMA?

| Property | SMA | EMA |
|---|---|---|
| Memory | Fixed window only | Entire history (decaying) |
| Recency bias | None — equal weights | Built-in — recent = heavier |
| Computation | Requires storing window | Single previous value only |
| Responsiveness | Lags by N/2 periods | Controlled by α |
| Drop-off effect | Sharp — old values are forgotten abruptly | Smooth — old values fade gradually |

The **drop-off effect** is the most underappreciated difference. An SMA "forgets" the oldest value abruptly when it drops off the window edge. If that value was an outlier, the SMA jumps discontinuously. EMA has no such cliff — old values simply fade toward zero weight.

---

## Complete Working Example

```python
import pandas as pd

# Load a price series (or substitute your own DataFrame)
prices = pd.Series(
    [100, 102, 98, 110, 105, 112, 108, 115, 111, 120,
     118, 125, 122, 130, 128, 135, 132, 140, 137, 145],
    name="Close"
)

# Compute three smoothing variants
sma_5    = prices.rolling(window=5, min_periods=5).mean()
ema_std  = prices.ewm(span=5, adjust=False).mean()          # standard EMA
ema_wld  = prices.ewm(alpha=1/5, adjust=False).mean()       # Wilder EMA

result = pd.DataFrame({
    "Close":      prices,
    "SMA_5":      sma_5.round(2),
    "EMA_std_5":  ema_std.round(2),
    "EMA_wld_5":  ema_wld.round(2),
})

print(result)
```

This lets you compare the three side-by-side and see the smoothing differences directly on real or simulated data.

---

## Key Takeaways

- An EMA is a recursive filter: each output is a weighted blend of the current input and the previous output.
- The smoothing factor $\alpha$ controls the speed: higher $\alpha$ = more reactive, lower $\alpha$ = smoother but slower.
- `adjust=False` in pandas enforces the standard recursive definition — always set it explicitly.
- Two conventions map window size to $\alpha$: span ($\alpha = 2/(N+1)$) and Wilder ($\alpha = 1/N$). They produce meaningfully different results.
- EMA has no sharp drop-off effect and requires only one previous value in memory — both practical advantages over SMA.

---

---

*Series: MLOps Systems — Feature Engineering*

| | |
|---|---|
| **← Previous** | [Part 1/8 — Moving Averages](moving-average-blog.md) |
| **This post** | Part 2/8 — EMA Primer |
| **Next →** | [Part 3/8 — RSI: Relative Strength Index](rsi-blog.md) |
