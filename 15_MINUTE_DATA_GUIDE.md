# 15-Minute Data Training Guide

This guide explains how to use 15-minute historical data from Kraken to train more granular trading models.

## Overview

The 15-minute timeframe offers several advantages over longer timeframes:
- **More trading opportunities**: More frequent signals allow for more trading opportunities
- **Faster response to market changes**: Quicker adaptation to changing market conditions
- **Better profit-taking**: More precise entries and exits
- **Improved backtesting**: More data points for statistical significance

## Prerequisites

Before training with 15-minute data, ensure you have:
- Python 3.11 or later
- TensorFlow 2.x
- Internet connection to access Kraken API
- At least 16GB of RAM for larger datasets

## Fetching 15-Minute Data

Use the `fetch_kraken_15m_data.py` script to download 15-minute OHLCV data from Kraken:

```bash
# Fetch 15-minute data for BTC/USD for the last 7 days
python fetch_kraken_15m_data.py --pair BTC/USD --timeframe 15m --days 7

# Fetch data for multiple pairs
python fetch_kraken_15m_data.py --pairs ETH/USD,SOL/USD --timeframe 15m --days 7

# Fetch data for all supported pairs
python fetch_kraken_15m_data.py --all --timeframe 15m --days 30
```

The data will be saved to the `historical_data` directory with filenames like `BTC_USD_15m.csv`.

## Training Models with 15-Minute Data

After fetching the data, use the `train_with_15m_data.py` script to train models:

```bash
# Train a model for BTC/USD with default parameters
python train_with_15m_data.py --pair BTC/USD 

# Train with custom parameters
python train_with_15m_data.py --pair BTC/USD --epochs 50 --sequence_length 96 --predict_horizon 4
```

Parameters explained:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--sequence_length`: Number of intervals in each sequence (default: 96 = 24 hours of 15-min data)
- `--predict_horizon`: Number of intervals to predict ahead (default: 4 = 1 hour)

## Training All Pairs

To train models for multiple pairs at once:

```bash
# Train models for the default pairs
python train_all_15m_data.py

# Train models for all supported pairs
python train_all_15m_data.py --pairs extended

# Train with data fetching for missing pairs
python train_all_15m_data.py --fetch_data --days 7
```

## Creating Timeframe Ensembles

Once you have trained models for multiple timeframes (15m, 1h, 4h, 1d), you can create ensemble models that combine signals from all timeframes:

```bash
# Create ensemble for BTC/USD
python create_timeframe_ensemble.py --pair BTC/USD

# Create ensemble with specific timeframes
python create_timeframe_ensemble.py --pair BTC/USD --timeframes 15m,1h,1d
```

The ensemble model will be saved to the `ensemble_models` directory and automatically added to the ML configuration.

## Performance Metrics

Training with 15-minute data generates comprehensive performance reports including:
- Model accuracy (overall and by class)
- Direction accuracy
- Win rate
- Total return
- Sharpe ratio
- Maximum drawdown
- Profit factor

These reports are saved to the `training_results` directory.

## Configuration

The trained models are automatically added to the ML configuration file at `config/ml_config.json`, and they can be activated for trading using the existing activation scripts.

## Best Practices

1. **Start with recent data**: Begin with 7-30 days of 15-minute data to ensure relevance
2. **Balance sequence length**: Use 96 intervals (24 hours) for a good balance of pattern recognition and memory requirements
3. **Predict multiple steps ahead**: Set `predict_horizon` to 4 (1 hour) for better directional predictions
4. **Use dynamic leverage**: 15-minute models work best with more conservative leverage settings
5. **Consider ensemble approaches**: Combine 15-minute signals with longer timeframes for more robust trading decisions