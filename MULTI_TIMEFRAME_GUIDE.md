# Multi-Timeframe Trading System Guide

This guide explains the implementation of the multi-timeframe trading system that combines data from different timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d) for more robust trading decisions.

## Overview

The multi-timeframe trading system follows a three-phase approach:

1. **Individual Timeframe Models**: Train separate models for each timeframe to capture different market behaviors
2. **Unified Multi-Timeframe Model**: Build a model that combines inputs from all timeframes simultaneously
3. **Ensemble Meta-Model**: Combine predictions from individual models using a weighted approach

## Timeframe Specialization

Each timeframe captures different market behavior:

- **1m-15m**: Noise, scalping, short-term momentum
- **1h-4h**: Trend shifts, breakout patterns
- **1D+**: Macro sentiment, accumulation zones

## Implementation Details

### 1. Individual Timeframe Models

The system trains specialized models for each timeframe with architectures optimized for that timeframe's characteristics:

- **Lower Timeframes (1m, 5m, 15m)**: CNN-based architecture for pattern recognition and noise filtering
- **Medium Timeframes (30m, 1h)**: Bidirectional LSTM for sequence patterns
- **Higher Timeframes (4h, 1d)**: TCN-like (Temporal Convolutional Network) with dilated convolutions for long-term dependencies

Each model is evaluated independently using accuracy, direction accuracy, precision, recall, and F1 score metrics.

### 2. Multi-Timeframe Model

The unified MTF model:

- Takes input sequences from multiple timeframes simultaneously
- Uses specialized branches for each timeframe
- Includes attention mechanisms to focus on the most relevant timeframes
- Combines features through feature fusion layers
- Outputs a single prediction considering all timeframe inputs

### 3. Ensemble Model

The ensemble approach:

- Combines predictions from individual timeframe models
- Uses weighted voting based on each model's historical performance
- Can adapt weights dynamically based on recent performance
- Provides more robust predictions by considering multiple perspectives

## Technical Indicators by Timeframe

The system calculates timeframe-specific indicators:

### Lower Timeframes (1m, 5m, 15m)
- Micro-trend indicators
- Volume spikes
- Price acceleration
- Fast stochastic oscillator

### Medium Timeframes (30m, 1h)
- EMA crossovers
- ADX for trend strength
- Ichimoku Cloud components

### Higher Timeframes (4h, 1d)
- VWAP (Volume-Weighted Average Price)
- OBV (On-Balance Volume)
- Mass Index
- Chaikin Money Flow

## In-Trade Management

Once a trade is open, the system uses higher-resolution data (1m, 5m) for:

- Dynamic trailing stops
- Partial take-profit decisions
- Early exit signals based on microstructure changes
- Position scaling
- Leverage adaptation

## Dynamic Risk Management

The system implements advanced risk management:

- Base leverage: 5x
- Maximum leverage: 75x
- Dynamic adjustment based on model confidence
- Position sizing relative to portfolio equity
- Adaptive stop-loss placement based on volatility

## Usage Guide

### Training the System

```bash
# Train individual timeframe models
python multi_timeframe_trainer.py --pair BTC/USD --phase individual

# Train unified multi-timeframe model
python multi_timeframe_trainer.py --pair BTC/USD --phase mtf

# Build ensemble model
python multi_timeframe_trainer.py --pair BTC/USD --phase ensemble

# Run all phases at once
python multi_timeframe_trainer.py --pair BTC/USD --phase all
```

### Customization Options

- `--timeframes`: Specify which timeframes to use (default: 15m,1h,4h,1d)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--sequence_length`: Length of input sequences
- `--predict_horizon`: Number of intervals to predict ahead

### Deployment

The trained models are automatically added to the ML configuration file at `config/ml_config.json`, and they can be activated for trading using:

```bash
python activate_improved_model_trading.py --sandbox
```

## Results and Reports

After training, the system generates:

1. Individual model reports for each timeframe
2. Multi-timeframe model report
3. Ensemble configuration
4. Summary report with recommendations

Reports include metrics such as accuracy, direction accuracy, precision, recall, F1 score, and class-wise performance.

## Advanced Features

### Real-Time Portfolio Risk Optimization

The system dynamically optimizes portfolio risk exposure and leverage in real time:

1. Uses in-trade model predictions to calculate expected value (EV) of trade continuation
2. Based on EV, dynamically sets:
   - Leverage (L) = f(EV, volatility)
   - Position size (PS) = f(Portfolio, confidence, drawdown)
3. Adjusts margin allocation with a risk model

### Modular Architecture

The system is designed with a modular architecture:

| Module | Timeframe | Role |
|--------|-----------|------|
| Entry Signal Model | 15m-4h | Determines direction and initial entry |
| In-Trade Manager | 1m-5m | Adjusts stops, size, and leverage |
| Portfolio Risk Bot | Live | Adjusts margin and total exposure |