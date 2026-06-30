# Generated Blog Index

Source path analyzed: `stockmonitorapp/utils/feature_engineering.py`

## Topics

- Moving Average (Simple Moving Average / Bollinger Bands)
- Exponential Moving Average (Wilder Smoothing in RSI, MACD, and ATR)
- Exponential Moving Average Primer (From a loop to pandas)
- Bollinger Bands
- Average True Range
- Types of Technical Indicators in Finance

## Generated Files

- `blogs/movingaverage-blog.md`
- `blogs/exponential-moving-average-blog.md`
- `blogs/exponential-moving-average-primer-blog.md`
- `blogs/bollinger-bands-blog.md`
- `blogs/average-true-range-blog.md`
- `blogs/types-of-technical-indicators-blog.md`

## Topic Summaries

- **Moving Average**: Explains how the Simple Moving Average underpins Bollinger Bands — combining rolling mean and rolling standard deviation into dynamic volatility bands with a normalised bandwidth feature, including warmup safety and production pitfalls.
- **Exponential Moving Average**: Explains the two distinct EMA conventions (Wilder's `alpha=1/N` vs standard `span=N`) used across RSI, MACD, and ATR, covering why mixing them produces incorrect signals and how `adjust=False` enforces the recursive EMA formula.
- **Exponential Moving Average Primer**: A beginner-friendly introduction to smoothing filters — builds EMA from a plain Python loop, maps it to `ewm(adjust=False)` in pandas, and introduces the span vs Wilder alpha conventions as a bridge to the advanced indicators post.
- **Bollinger Bands**: Covers how `compute_bollinger_bands` uses a rolling SMA and `ddof=1` standard deviation to build self-scaling volatility bands, with a bandwidth feature for squeeze detection and detailed warmup and leakage guidance.
- **Average True Range**: Explains True Range construction across all three day types (intraday, gap-up, gap-down), Wilder EMA smoothing, and practical ATR applications for position sizing, normalised volatility, and regime detection.
- **Types of Technical Indicators in Finance**: A conceptual taxonomy of indicator classes — trend, momentum, volatility, volume, and calendar — anchored to the full pipeline implementation, with per-class pipeline constraints and a combined warmup summary.
