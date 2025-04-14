# Dynamic Parameter Optimization Guide

This guide explains how to use the dynamic parameter optimization system to maximize trading returns by automatically adjusting parameters for each trade based on confidence levels, signal strength, and market conditions.

## Overview

The dynamic parameter optimization system allows the trading bot to:

1. **Optimize parameters** for each cryptocurrency pair based on historical performance
2. **Dynamically adjust parameters** for each individual trade based on:
   - Signal strength and confidence
   - Current market volatility
   - Market regime (trending, ranging, volatile)
   - Historical performance patterns
3. **Maximize returns** by taking more aggressive positions when confidence is high and more conservative positions when confidence is lower

## Key Components

The system consists of several integrated components:

- **DynamicParameterOptimizer**: Core component that calculates optimal parameters based on backtesting results and provides dynamic adjustments for each trade
- **TradingPairOptimizer**: Comprehensive optimization pipeline that runs backtests and tunes parameters for each cryptocurrency pair
- **MarketAnalyzer**: Detects market regimes and conditions to adapt trading parameters accordingly
- **DynamicParameterAdjuster**: Helper class for real-time parameter adjustments during trading

## Getting Started

### 1. Run Optimization

To optimize parameters for all configured trading pairs:

```bash
python optimize_all_pairs.py
```

This will:
- Fetch historical data
- Run backtests with different parameter combinations
- Determine optimal parameters for each pair
- Save optimized parameters to `config/optimized_params.json`

To optimize a specific pair:

```bash
python optimize_all_pairs.py --pair SOL/USD
```

### 2. Apply Optimized Parameters

Apply the optimized parameters to the trading system:

```bash
python apply_dynamic_parameters.py
```

This will:
- Update the `.env` file with optimized parameters
- Enable dynamic parameter adjustment
- Create example files showing dynamic parameter calculations

### 3. Enable Dynamic Adjustment in Live Trading

To use dynamic parameter adjustment in your trading code:

```python
from dynamic_parameter_adjustment import dynamic_adjuster

# Get dynamic parameters for a specific trade
params = dynamic_adjuster.get_parameters(
    pair="SOL/USD",
    strategy="arima",
    confidence=0.85,  # Model prediction confidence
    signal_strength=0.78,  # Signal strength from strategy
    volatility=0.04,  # Current market volatility
    market_regime="TRENDING_UP"  # Current market regime
)

# Calculate position size using dynamic parameters
position_size = dynamic_adjuster.calculate_position_size(
    capital=10000.0,
    price=130.0,
    params=params
)

# Calculate stop loss and take profit
stops = dynamic_adjuster.calculate_stops(
    price=130.0,
    direction="long",
    atr=0.08,
    params=params
)
```

## Configuration

The system uses several configuration files:

### 1. ML Configuration (`config/ml_config.json`)

Base configuration file with default parameters:

```json
{
  "pairs": ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"],
  "base_leverage": 20.0,
  "max_leverage": 125.0,
  "risk_percentage": 0.20,
  "confidence_threshold": 0.65,
  "signal_strength_threshold": 0.60,
  "trailing_stop_atr_multiplier": 3.0,
  "exit_multiplier": 1.5,
  "drawdown_limit_percentage": 4.0,
  ...
}
```

### 2. Optimized Parameters (`config/optimized_params.json`)

Contains optimized parameters for each pair:

```json
{
  "SOL/USD": {
    "risk_percentage": 0.22,
    "base_leverage": 25.0,
    "max_leverage": 100.0,
    "confidence_threshold": 0.68,
    "signal_strength_threshold": 0.58,
    "trailing_stop_atr_multiplier": 3.2,
    "exit_multiplier": 1.8,
    "strategy_weights": {
      "arima": 0.35,
      "adaptive": 0.65
    },
    ...
  },
  ...
}
```

### 3. Environment File (`.env`)

Contains parameters for the live trading system:

```
# Dynamic optimized parameters - generated on 2025-04-14 23:04:15
OPTIMIZED_DATE="2025-04-14 23:04:15"

# Optimized parameters for SOL/USD
RISK_PERCENTAGE_SOLUSD=0.22
BASE_LEVERAGE_SOLUSD=25.0
MAX_LEVERAGE_SOLUSD=100.0
CONFIDENCE_THRESHOLD_SOLUSD=0.68
SIGNAL_STRENGTH_THRESHOLD_SOLUSD=0.58
ARIMA_WEIGHT_SOLUSD=0.35
ADAPTIVE_WEIGHT_SOLUSD=0.65
TRAILING_STOP_ATR_MULTIPLIER_SOLUSD=3.2
EXIT_MULTIPLIER_SOLUSD=1.8

# Enable dynamic parameter adjustment based on confidence
USE_DYNAMIC_PARAMETERS=true
```

## Parameter Descriptions

| Parameter | Description | Range | Notes |
|-----------|-------------|-------|-------|
| `risk_percentage` | Percentage of capital to risk per trade | 0.05-0.4 (5-40%) | Dynamically adjusted based on confidence |
| `base_leverage` | Base leverage for trades | 5-100 | Lower baseline leverage |
| `max_leverage` | Maximum leverage for highest confidence trades | 20-125 | Upper limit of leverage |
| `confidence_threshold` | Minimum confidence required for trades | 0.55-0.95 | Trades below this threshold are rejected |
| `signal_strength_threshold` | Minimum signal strength required | 0.45-0.95 | Signals below this threshold are ignored |
| `trailing_stop_atr_multiplier` | Multiplier for ATR-based trailing stops | 1.0-6.0 | Higher values give more room |
| `exit_multiplier` | Take-profit multiplier relative to stop | 1.0-3.0 | Manages risk/reward ratio |

## Dynamic Adjustment Logic

The dynamic adjustment logic works as follows:

1. **Confidence factor** is calculated as:
   ```
   confidence_factor = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
   ```

2. **Signal factor** is calculated as:
   ```
   signal_factor = (signal_strength - signal_strength_threshold) / (1.0 - signal_strength_threshold)
   ```

3. **Combined factor** is calculated as:
   ```
   combined_factor = (confidence_factor * 0.6) + (signal_factor * 0.4)
   ```

4. **Risk percentage** is adjusted:
   ```
   dynamic_risk = base_risk_percentage * (1.0 + combined_factor)
   ```

5. **Leverage** is adjusted:
   ```
   dynamic_leverage = base_leverage + (leverage_range * combined_factor)
   ```

6. **Market condition adjustments** are applied based on volatility and market regime

## Visualization

The optimization process generates visualization files in the `optimization_results` directory:

- `returns_comparison.png`: Comparison of returns across different pairs
- `parameter_impact.png`: Analysis of how parameters impact performance
- `win_rates.png`: Comparison of win rates and profit factors
- `{pair}_equity_curve.png`: Equity curves for each pair
- `{pair}_market_regimes.png`: Market regime analysis

## Maintenance

For optimal performance:

1. **Re-optimize regularly** (weekly or monthly) to adapt to changing market conditions
2. **Monitor performance** and adjust parameters if necessary
3. **Retrain ML models** regularly to maintain prediction accuracy

## Troubleshooting

If you encounter issues:

1. **Check logs** in `optimization_results/optimization.log`
2. **Verify historical data** is being fetched correctly
3. **Reset to base parameters** by setting `USE_DYNAMIC_PARAMETERS=false` in `.env`
4. **Run backtests** with different parameter sets to diagnose issues

---

## Advanced Usage

### Custom Market Regime Detection

You can define custom market regimes to adjust parameters more precisely:

```python
from utils.market_analyzer import MarketAnalyzer

analyzer = MarketAnalyzer()
regimes = analyzer.analyze_market_regimes(historical_data)
current_regime = regimes["current_regime"]

# Get parameters optimized for this regime
params = dynamic_adjuster.adjust_for_market_regime(params, current_regime)
```

### Cross-Asset Optimization

The system can be extended to learn from patterns across different assets:

```bash
python optimize_all_pairs.py --cross-asset-learning
```

This enables the system to transfer knowledge between related cryptocurrency pairs for more robust optimization.

### Custom Parameter Boundaries

You can customize parameter boundaries in `dynamic_parameter_optimizer.py`:

```python
# Parameter boundaries
MIN_RISK_PERCENTAGE = 0.05  # 5% minimum risk per trade
MAX_RISK_PERCENTAGE = 0.40  # 40% maximum risk per trade
MIN_LEVERAGE = 5.0
MAX_LEVERAGE = 125.0
```