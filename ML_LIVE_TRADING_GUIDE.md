# ML Enhanced Trading Bot Guide

This guide explains how to use the machine learning-enhanced trading system with the Kraken trading bot.

## Overview

The ML-enhanced trading system adds advanced machine learning capabilities to the existing Kraken trading bot, providing:

1. ML-based trade signal generation with ensemble models
2. Dynamic position sizing with confidence-based leverage
3. Market regime detection and adaptation
4. Performance tracking and strategy optimization
5. Multi-asset ML model training and deployment

The ML system includes three major model types working together:
- **Transformer models**: Excellent at capturing long-range dependencies and patterns
- **TCN (Temporal Convolutional Network) models**: Effective for time series with hierarchical patterns
- **LSTM models**: Strong at learning sequential patterns with memory

## Installation

All required dependencies are already installed. The system uses TensorFlow, numpy, pandas, and other Python libraries.

## Directory Structure

- `models/`: Main directory for all ML models
  - `transformer/`: Transformer-based models
  - `tcn/`: Temporal Convolutional Network models
  - `lstm/`: LSTM models
  - `ensemble/`: Ensemble configurations and weights

- `training_data/`: Contains prepared data for model training
- `backtest_results/`: Stores backtest results for performance analysis

## Quick Start

To start trading with ML enhancements in sandbox mode:

```bash
python run_optimized_ml_trading.py --reset --optimize
```

This will:
1. Reset the portfolio to initial conditions
2. Run ML model optimization for default assets (SOL/USD, ETH/USD, BTC/USD)
3. Start the trading bot in sandbox mode with ML integration

## Command Options

The ML trading system can be customized with various command-line options:

### Basic Options

```bash
python run_optimized_ml_trading.py --assets "SOL/USD" "ETH/USD" --capital 25000 --interval 30
```

This starts trading with:
- SOL/USD and ETH/USD assets
- $25,000 initial capital
- 30-second trading interval

### Reset Portfolio

```bash
python run_optimized_ml_trading.py --reset
```

Resets the portfolio to the initial capital amount before starting trading.

### Run Optimization

```bash
python run_optimized_ml_trading.py --optimize
```

Runs ML model optimization before starting trading, including:
- Data preparation and feature engineering
- Training transformer, TCN, and LSTM models
- Creating ensemble configurations
- Building position sizing models
- Backtesting ensemble performance

### Live Trading

```bash
python run_optimized_ml_trading.py --live
```

**CAUTION**: This enables live trading with real money. Use with extreme caution.

### Limiting Iterations

```bash
python run_optimized_ml_trading.py --iterations 100
```

Runs the trading bot for exactly 100 iterations then stops.

## Advanced Usage

### Custom Model Training

For advanced users who want to customize model training:

```bash
python ml_live_training_optimizer.py --assets "SOL/USD" "ETH/USD" "BTC/USD"
```

This only runs the optimization process without starting trading.

### Adjusting Leverage

The system automatically uses asset-specific leverage settings:

- SOL/USD: 20x-125x leverage (default: 35x)
- ETH/USD: 15x-100x leverage (default: 30x)
- BTC/USD: 12x-85x leverage (default: 25x)

Leverage is automatically scaled based on model confidence.

## Understanding ML Signal Strength

ML signal strength is determined by ensemble model confidence:

- **High confidence (>0.90)**: Full position size with maximum allowed leverage
- **Medium-high confidence (0.80-0.90)**: 80% position size with scaled leverage
- **Medium confidence (0.70-0.80)**: 50% position size with scaled leverage
- **Low confidence (0.65-0.70)**: 30% position size with minimum leverage
- **Very low confidence (<0.65)**: No trade

## Market Regime Specialization

The system automatically detects market regimes:

- **Trending markets**: Emphasizes transformer models (45% weight)
- **Ranging markets**: Balanced approach with more weight on adaptive strategy (25% weight)
- **Volatile markets**: Heavily weighted towards ML models (90% combined weight)

## Performance Tracking

The system automatically tracks model performance and adapts strategy weights based on results. Performance metrics are displayed during trading.

## Troubleshooting

### Missing Models

If you encounter errors about missing models, run:

```bash
python run_optimized_ml_trading.py --optimize
```

### Low Model Performance

If models are underperforming, try retraining with more recent data:

```bash
python ml_live_training_optimizer.py --assets "SOL/USD"
```

## Warning About Extreme Leverage

The ML system is configured to use extreme leverage settings that can result in significant gains but also catastrophic losses. Always use sandbox mode for testing, and proceed with extreme caution when using live trading mode.

## Best Practices

1. Always start in sandbox mode
2. Monitor initial trades carefully
3. Periodically retrain models (every 1-2 weeks)
4. Start with lower capital when testing new strategies
5. Review performance metrics regularly

## Advanced Configuration

Advanced users can modify asset-specific configurations in:
- `ml_live_training_optimizer.py`: For training configurations
- `model_collaboration_integrator.py`: For strategy weighting
- `ml_live_trading_integration.py`: For leverage and position sizing