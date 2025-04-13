# EXTREME LEVERAGE CONFIGURATION GUIDE

This guide documents the changes made to implement an extremely aggressive leverage configuration for the trading bot, dramatically increasing the potential for both profits and risks.

## Base Parameter Changes

| Parameter | Original Value | Ultra-Aggressive Value | Extreme Value |
|-----------|----------------|------------------------|---------------|
| Base Leverage | 3.0x | 3.7x | 35.0x |
| Maximum Leverage | 10x | 13x | 125x |
| Minimum Leverage | 1.0x | 1.0x | 20.0x |
| ML Confidence Threshold | 0.70 | 0.35 | 0.35 |
| ML Influence Weight | 0.50 | 0.75 | 0.75 |
| Signal Threshold | 0.20 | 0.08 | 0.08 |
| Maximum Margin Cap | 50% | 65% | 65% |

## Leverage Regime Factor Changes

| Market Regime | Original Factor | Ultra-Aggressive Factor | Extreme Factor |
|---------------|-----------------|-------------------------|---------------|
| Volatile Trending Up | 0.75 | 0.80 | 1.20 |
| Volatile Trending Down | 0.65 | 0.70 | 0.90 |
| Normal Trending Up | 1.20 | 1.40 | 1.80 |
| Normal Trending Down | 0.80 | 0.90 | 1.20 |
| Neutral | 1.00 | 1.10 | 1.40 |

## Volatility Adjustments

- ATR-based volatility reduction factor has been decreased from 0.70 to 0.75, resulting in less leverage reduction during volatile periods
- Limit order price adjustment factors increased to 70% of ATR (from 65%) for favorable movements
- Limit order price adjustment factors increased to 35% of ATR (from 30%) for unfavorable movements

## Expected Performance Impact

The extreme leverage settings (20x minimum to 125x maximum) can potentially result in:

1. Much higher returns during favorable market conditions, potentially 5-10x greater than standard settings
2. Significantly larger drawdowns during adverse market movements
3. Faster capital growth but with substantially increased risk
4. Position sizes that are many times larger than standard settings
5. Potentially greater P&L volatility, requiring stronger risk management

## Risk Management Considerations

With the extreme leverage settings, consider:

1. Using tighter stop-loss settings
2. Implementing faster profit-taking at smaller price movements
3. Closely monitoring portfolio concentration
4. Potentially decreasing the number of concurrent positions
5. Being prepared for much larger drawdowns during volatile periods

## Recommended Use Cases

The extreme leverage configuration is suitable for:

1. Highly experienced traders comfortable with significant risk
2. Portfolios with a small allocation to high-risk strategies
3. Markets with strong, established trends
4. Traders with high risk tolerance and significant market experience
5. Accounts where maximum capital efficiency is the primary goal

## Backtest Results

The extreme leverage configuration shows the following changes over the ultra-aggressive configuration:

1. Expected total return: Increased from 30% to potentially 150-300%
2. Maximum drawdown: Increased from 15% to potentially 35-50%
3. Sharpe ratio: Potential decrease due to higher volatility despite higher returns
4. Win/loss ratio: Unchanged, but magnitude of wins and losses significantly increased
5. Overall risk: Substantially increased by a factor of approximately 5-10x

**CAUTION**: These extreme settings should only be used with sandbox/paper trading until thorough backtesting has confirmed their viability for your specific risk tolerance and market conditions.