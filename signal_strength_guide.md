# Signal Strength Calculation Guide

This document explains how the integrated strategy calculates signal strength to make trading decisions.

## Overview

The integrated strategy ("him all along") uses a signal strength calculation mechanism to determine the conviction level for trade signals. Each indicator contributes to the overall signal strength, and the final decision is based on the weighted average of these contributions.

## Signal Strength Scale

Signal strength is measured on a scale from 0.0 to 1.0:

- **0.0**: No confidence
- **0.5**: Moderate confidence
- **1.0**: Maximum confidence

A minimum signal strength threshold (default: 0.65) must be exceeded for a trade to be executed.

## Signal Strength Components

The following components contribute to the signal strength calculation:

### 1. EMA Component

```
EMA9 vs EMA21 relationship
```

- **Bullish**: EMA9 > EMA21 (contributes to BUY signal)
- **Bearish**: EMA9 < EMA21 (contributes to SELL signal)
- **Strength modifier**: Distance between EMAs relative to average range

### 2. RSI Component

```
RSI extreme values
```

- **Bullish**: RSI < 30 (oversold condition, contributes to BUY signal)
- **Bearish**: RSI > 70 (overbought condition, contributes to SELL signal)
- **Strength modifier**: Distance from neutral (50) relative to extremes

### 3. MACD Component

```
MACD vs Signal line relationship
```

- **Bullish**: MACD > Signal line (contributes to BUY signal)
- **Bearish**: MACD < Signal line (contributes to SELL signal)
- **Strength modifier**: Histogram height relative to average

### 4. ADX Component

```
ADX trend strength
```

- **Both directions**: Higher ADX indicates stronger trend
- **Strength modifier**: ADX value relative to threshold (25)

### 5. ARIMA Forecast Component

```
Linear regression forecast direction and magnitude
```

- **Bullish**: Target price > Current price (contributes to BUY signal)
- **Bearish**: Target price < Current price (contributes to SELL signal)
- **Strength modifier**: Magnitude of predicted move relative to ATR

## Weighted Combination

Each component's signal strength is weighted based on its historical reliability:

```
Final Strength = (w1*EMA + w2*RSI + w3*MACD + w4*ADX + w5*ARIMA) / (w1 + w2 + w3 + w4 + w5)
```

Where:
- `w1`, `w2`, etc. are the weights for each component
- Default weights give slightly higher priority to RSI and ARIMA components

## Signal Competition

For each potential trade direction (BUY/SELL):

1. Calculate the weighted signal strength
2. Compare the strength to the minimum threshold
3. Compare BUY signal strength vs SELL signal strength
4. The higher strength wins (if above threshold)
5. If both below threshold, no trade is executed

## Example Log Output

```
【INTEGRATED】 Signal Strength: EMA=0.78, RSI=0.85, MACD=0.62, ADX=0.71, ARIMA=0.84
【INTEGRATED】 Final Signal Strength: 0.76 (BUY 🟢)
```

In this example:
- Each component's strength is shown (all on 0.0-1.0 scale)
- The final weighted strength is 0.76
- The signal direction is BUY
- The strength exceeds the threshold, so a trade would be executed

## Market Condition Modifiers

Additional factors that may adjust the final signal strength:

1. **Volatility filter**: Reduces strength in high volatility conditions
2. **Market trend**: Reduces strength when trading against the overall trend
3. **Price barriers**: Reduces strength when price is near key support/resistance

## Using Signal Strength for Position Sizing

In the integrated strategy, signal strength can also influence position sizing:

- Higher signal strength = Larger position size (up to maximum)
- Lower signal strength = Smaller position size

This allows the strategy to commit more capital to high-conviction trades and reduce risk on less certain opportunities.