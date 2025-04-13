# Machine Learning Enhancement for Kraken Trading Bot

This document outlines the machine learning enhancement for the Kraken Trading Bot, focusing on how to leverage historical data fetching for improved model training.

## Overview

The machine learning enhancement consists of three main components:

1. **Historical Data Fetcher** (`historical_data_fetcher.py`)
   - Fetches extended historical price data from Kraken API
   - Processes data and adds technical indicators
   - Creates datasets suitable for machine learning

2. **ML Data Integrator** (`ml_data_integrator.py`)
   - Manages the training pipeline for machine learning models
   - Prepares data in correct format for each model architecture
   - Handles multi-timeframe integration for enhanced prediction power

3. **ML Model Integrator** (`ml_model_integrator.py`)
   - Integrates trained models with the trading bot system
   - Generates signals based on ensemble model predictions
   - Dynamically adjusts model weights based on performance

## Getting Started

### Step 1: Fetch Historical Data

Run the following script to fetch historical price data from Kraken:

```bash
./fetch_historical_data.sh
```

This will:
- Create necessary directories for data storage
- Fetch historical OHLC data for SOL/USD
- Process data and add technical indicators
- Save data in the proper format for model training

The script fetches data for multiple timeframes (15m, 30m, 1h, 4h) to provide more comprehensive market context.

### Step 2: Train ML Models

After fetching historical data, you can train the machine learning models:

```bash
python ml_data_integrator.py
```

This will:
- Load the prepared datasets
- Train three different model architectures:
  - TCN (Temporal Convolutional Network)
  - CNN (Convolutional Neural Network)
  - LSTM (Long Short-Term Memory)
- Save the trained models and training history

The training process can take 1-3 hours depending on the amount of data and hardware capabilities.

### Step 3: Integrate ML Models with Trading Bot

After training, the models can be integrated with the trading system:

```bash
# No specific command needed - models will be automatically loaded
# when the trading bot starts, if they are available
```

## ML Model Details

### TCN (Temporal Convolutional Network)
- Best suited for capturing long-range dependencies in time series data
- Uses dilated causal convolutions to maintain temporal order
- Avoids looking ahead bias by restricting information flow

### CNN (Convolutional Neural Network)
- Extracts local patterns and features from price data
- Effective at identifying chart patterns and price formations
- Faster training compared to recurrent models

### LSTM (Long Short-Term Memory)
- Specialized recurrent neural network for sequence data
- Captures both short-term and long-term dependencies
- Maintains memory of important price movements

## Signal Generation and Integration

The machine learning models generate signals in the following way:

1. Recent market data is processed and fed into each model
2. Each model produces a prediction (direction and confidence)
3. Predictions are weighted based on each model's recent performance
4. A final ensemble signal is generated combining all model outputs
5. This signal is integrated with traditional strategy signals

## Performance Monitoring

The system continuously monitors the performance of each model:

1. Trade outcomes are used to update model performance metrics
2. Models with better performance receive higher weights
3. Poorly performing models have their influence reduced
4. The system automatically adapts to changing market conditions

## Examples of ML Signal Logs

Look for the following entries in the logs to see ML model signals:

```
[INFO] TCN model prediction: 0.7632
[INFO] CNN model prediction: 0.6945
[INFO] LSTM model prediction: 0.5534
[INFO] ML Ensemble Signal: BUY with strength 0.6837
```

## Updating Model Parameters

To adjust model parameters, edit the following files:

- `historical_data_fetcher.py` - For data fetching parameters
- `ml_data_integrator.py` - For training parameters
- `ml_model_integrator.py` - For signal generation parameters

## Conclusion

The machine learning enhancement provides the trading bot with advanced predictive capabilities by leveraging multiple model architectures and extended historical data. By combining traditional technical indicators with machine learning predictions, the bot can make more informed trading decisions.