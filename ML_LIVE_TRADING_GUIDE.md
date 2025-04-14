# ML Live Trading Integration Guide

This guide provides instructions on implementing and training the machine learning aspects of the trading bot for live trading.

## Overview

The ML Live Trading Integration system connects our sophisticated machine learning models with our trading infrastructure for real-time market predictions. The system includes:

1. **Advanced ML Model Architecture**: Combining TCN (Temporal Convolutional Networks), CNN (Convolutional Neural Networks), LSTM/GRU (Long Short-Term Memory/Gated Recurrent Units), and Transformer encoders for state-of-the-art price prediction.

2. **Strategy Ensemble Training**: Optimizing strategy weights for different market regimes to ensure optimal strategy collaboration.

3. **Model Collaboration Integrator**: Facilitating real-time communication between strategies and ML models.

4. **Dynamic Position Sizing with ML**: Using machine learning to adjust leverage and position sizes based on market conditions and prediction confidence.

## Getting Started

### Prerequisites

- Python 3.8+ with TensorFlow, scikit-learn, pandas and numpy
- Historical data available in `historical_data/` directory
- Kraken API credentials configured in the `.env` file

### Quick Start

To train and deploy the entire ML trading system:

```bash
./run_ml_training_and_live_integration.py --sandbox
```

This will:
1. Train ML models for all configured assets
2. Train the strategy ensemble
3. Start the ML-enhanced trading bot in sandbox mode

For production trading:

```bash
./run_ml_training_and_live_integration.py --live
```

**CAUTION**: Using `--live` mode will place real trades using your Kraken account.

## Training ML Models

### Basic Training

To train the ML models for all assets:

```bash
./train_ml_live_integration.py
```

This trains models using default settings:
- 90 days of historical data
- Regular leverage settings
- No hyperparameter optimization

### Advanced Training

For hyper-optimized training:

```bash
./train_ml_live_integration.py --optimize --days 180 --extreme-leverage
```

Options:
- `--optimize`: Enable hyperparameter optimization
- `--days`: Number of days of historical data to use (more is better but slower)
- `--extreme-leverage`: Train models for extreme leverage settings (20-125x)
- `--force-retrain`: Force retrain even if models exist
- `--visualize`: Generate performance visualizations

## Strategy Ensemble Training

The strategy ensemble trainer optimizes how different strategies collaborate:

```bash
./strategy_ensemble_trainer.py
```

The trainer:
1. Detects market regimes in historical data
2. Evaluates strategy performance in each regime
3. Optimizes strategy weights for each regime
4. Generates collaboration configuration

## Model Collaboration

The model collaboration integrator connects ML predictions with strategy decisions:

```bash
# Integration happens automatically when running the bot
./bot_manager_integration.py
```

Features:
- Regime-specific strategy weighting
- Adaptive weight adjustment based on performance
- Signal arbitration with confidence scoring
- ML-enhanced position sizing

## ML Position Sizing

ML-enhanced position sizing adjusts leverage and position size based on:
- Prediction confidence
- Signal strength
- Market volatility
- Historical performance

Enable with:

```bash
./run_ml_training_and_live_integration.py --ml-position-sizing
```

## Extreme Leverage Settings

For aggressive trading with extreme leverage:

```bash
./run_ml_training_and_live_integration.py --extreme-leverage
```

Asset-specific leverage ranges:
- SOL/USD: 20x to 125x
- ETH/USD: 15x to 100x
- BTC/USD: 12x to 85x

**CAUTION**: Extreme leverage significantly increases risk of liquidation.

## Multi-Asset Trading

By default, the system trades multiple assets with this capital allocation:
- SOL/USD: 40%
- ETH/USD: 35%
- BTC/USD: 25%

To specify custom assets:

```bash
./run_ml_training_and_live_integration.py --assets SOL/USD ETH/USD
```

## Monitoring and Logging

All ML predictions and trading actions are logged to:
- `ml_training.log`: Training progress and metrics
- `model_collaboration.log`: Strategy collaboration details
- `bot_manager_integration.log`: Trading decisions and execution
- `ml_deployment.log`: Overall deployment information

## Troubleshooting

Common issues:

1. **Missing historical data**: Ensure data files exist in `historical_data/` directory
2. **API connection errors**: Verify Kraken API credentials in `.env`
3. **Model training failures**: Check for sufficient GPU memory and data quality
4. **Strategy conflicts**: Review strategy weights in ensemble configuration

## Next Steps

1. **Customizing ML architecture**: Modify `enhanced_transformer_models.py`
2. **Adding new strategies**: Update `strategy_ensemble_trainer.py`
3. **Tuning hyperparameters**: Explore `hyper_optimized_ml_training.py`
4. **Backtesting with ML**: Use `comprehensive_backtest.py`