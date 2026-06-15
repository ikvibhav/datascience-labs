# Building a Leakage-Safe RSI Feature in Pandas for Stock Pipelines

The Relative Strength Index (RSI) is one of the most widely used momentum indicators in technical analysis, but its real engineering value shows up when it is implemented as a reusable, pipeline-safe feature. In this implementation, RSI is treated as a deterministic transformation over market data, which makes it suitable for analytics pipelines and downstream modeling.

[Image Placeholder]

## Why RSI Matters in Feature Engineering

Raw price levels rarely say enough on their own. What often matters more is whether recent gains are consistently stronger than recent losses. RSI captures exactly that balance by turning directional price movement into a bounded momentum score between 0 and 100.

That bounded scale is useful because it is easier to compare across dates and instruments than raw returns or price deltas.

## How This Implementation Works

The relevant function validates input, computes daily price changes, separates gains from losses, and then applies Wilder-style smoothing before deriving the final RSI column.

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
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    engineered[f"rsi_{window}"] = 100 - (100 / (1 + rs))

    return engineered
```

### Step 1: Validate the input

The function rejects empty data, missing close-price columns, and non-positive windows. That matters because technical indicators fail in subtle ways when fed malformed data, and bad validation usually surfaces later as misleading features rather than immediate exceptions.

### Step 2: Convert prices into directional changes

`diff()` transforms the close series into day-over-day movement. That creates the raw material for momentum analysis.

### Step 3: Split movement into gains and losses

Positive deltas are retained in the gain series; negative deltas are inverted into the loss series. This is the core abstraction behind RSI: not how much price moved in total, but how much of that movement was upward versus downward.

### Step 4: Apply Wilder smoothing

The implementation uses exponential weighting with `alpha = 1 / window`, which is a standard way to model Wilder's smoothing in pandas. This makes the indicator responsive without being overly noisy.

### Step 5: Normalize to a bounded oscillator

The final score is derived from the relative strength ratio:

- `RS = average gain / average loss`
- `RSI = 100 - (100 / (1 + RS))`

This yields a bounded feature that is easy to interpret and safe to join into a broader feature table.

## Design Decisions and Tradeoffs

This implementation makes several good engineering decisions:

- It copies the dataframe instead of mutating the caller's object.
- It parameterizes the close column and window size.
- It names the output column dynamically as `rsi_<window>`, which keeps multiple RSI windows composable.

The tradeoff is that warmup rows remain `NaN`, and this function delegates missing-value policy to the caller. That is usually the right choice in pipelines, but it means downstream steps must decide whether to drop or preserve early rows.

## Pitfalls and Edge Cases

- A `window` of zero or less is invalid and correctly rejected.
- Missing close-price columns raise immediately, preventing silent data corruption.
- The first `window` rows will be `NaN` because smoothing requires historical context.
- If average loss becomes zero for a sustained uptrend, the RSI approaches 100, which is mathematically expected but should still be interpreted carefully in models.

## Pro Tips

> **Pro Tips**
> - Generate multiple RSI windows, such as 7, 14, and 21, when you want short- and medium-term momentum views in the same dataset.
> - Keep RSI as one feature among many; on its own, it is descriptive, not predictive.
> - Document your warmup-row policy explicitly so training and scoring pipelines treat `NaN` periods consistently.

## Conclusion

This RSI implementation is more than a technical-analysis helper. It is a solid example of how to turn market behavior into a reusable, pipeline-friendly feature with clear validation and predictable output semantics. In a production stock workflow, those design decisions matter as much as the formula itself.

## Further Reading

- Wilder-style smoothing versus simple rolling averages
- Momentum oscillators in time-series feature sets
- Handling warmup nulls in financial feature engineering
