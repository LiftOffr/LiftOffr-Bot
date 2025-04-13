# Advanced Machine Learning Models Guide

This document provides comprehensive information about the advanced machine learning models implemented in the Kraken Trading Bot. The system uses a sophisticated ensemble approach that combines multiple model architectures with adaptive weighting mechanisms to handle the complex and volatile nature of cryptocurrency markets.

## Table of Contents

1. [Overview](#overview)
2. [Model Architectures](#model-architectures)
3. [Dynamic Weighted Ensemble](#dynamic-weighted-ensemble)
4. [Market Regime Detection](#market-regime-detection)
5. [Performance-Based Weight Adjustment](#performance-based-weight-adjustment)
6. [Training Process](#training-process)
7. [Prediction Pipeline](#prediction-pipeline)
8. [Integration with Trading Strategies](#integration-with-trading-strategies)

## Overview

The advanced machine learning system is designed to predict price movements and generate trading signals by combining multiple specialized model architectures. Each model type has strengths in capturing different aspects of market behavior:

- **TCN (Temporal Convolutional Network)**: Excels at capturing long-range dependencies while maintaining computational efficiency.
- **CNN (Convolutional Neural Network)**: Identifies local patterns and features in the price data.
- **LSTM/GRU (Long Short-Term Memory/Gated Recurrent Unit)**: Processes sequential data with memory, ideal for time series.
- **Attention Mechanism**: Focuses on relevant parts of the input sequence.
- **Transformer Architecture**: Captures complex relationships between different time points.
- **Hybrid Models**: Combines multiple architectures to leverage their complementary strengths.

The system dynamically adjusts the influence of each model based on:
1. Current market regime (trending, ranging, volatile, etc.)
2. Recent performance of each model
3. Confidence of predictions

## Model Architectures

### TCN (Temporal Convolutional Network)

TCNs use dilated causal convolutions to achieve a large receptive field while maintaining computational efficiency. Features:

- Causal convolutions prevent information leakage from future to past
- Dilated convolutions expand receptive field exponentially with depth
- Residual connections help with gradient flow and deep network training

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### CNN (Convolutional Neural Network)

The CNN architecture identifies local patterns in price movements and technical indicators:

- Multiple convolutional layers with increasing filter sizes
- Max pooling layers to reduce dimensionality
- Batch normalization for training stability
- Dense layers for final prediction

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### LSTM (Long Short-Term Memory)

LSTM networks are specialized recurrent neural networks capable of learning long-term dependencies:

- Memory cell to retain information over long sequences
- Input, forget, and output gates to control information flow
- Bidirectional variants capture dependencies in both directions

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### GRU (Gated Recurrent Unit)

GRU is a simplified variation of LSTM with fewer parameters:

- Reset and update gates (simpler than LSTM)
- Computationally more efficient than LSTM
- Often performs similarly to LSTM despite simplicity

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### Attention Mechanism

Attention mechanisms help the model focus on relevant parts of the input sequence:

- Self-attention scores relationships between all time steps
- Multi-head attention captures different types of relationships
- Enables the model to "pay attention" to important price movements

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### Transformer

Transformer architecture uses self-attention as its primary building block:

- Entirely attention-based, no recurrence
- Positional encoding to maintain temporal information
- Layer normalization and residual connections for stability

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

### Hybrid Model

The hybrid model combines CNN, LSTM, GRU, attention, and transformer components:

- CNN branch for local pattern recognition
- LSTM/GRU branches for sequential processing
- Attention and transformer components for focus on important signals
- Combines outputs from different branches for final prediction

```
Input Shape: (sequence_length, features)
Output: Predicted price movement (-1.0 to 1.0)
```

## Dynamic Weighted Ensemble

The heart of the ML system is the dynamic weighted ensemble that adaptively combines predictions from all model types:

```python
# Simplified example of ensemble prediction
def predict(market_data):
    # Get predictions from all models
    predictions = {}
    for model_type, model in models.items():
        predictions[model_type] = model.predict(market_data)
    
    # Adjust weights based on market regime
    adjusted_weights = adjust_weights_for_regime(weights, current_regime)
    
    # Further adjust weights based on recent performance
    final_weights = adjust_weights_by_performance(adjusted_weights)
    
    # Calculate weighted prediction
    ensemble_prediction = sum(pred * final_weights[model_type] 
                             for model_type, pred in predictions.items())
    
    return ensemble_prediction
```

Benefits:
- Reduces prediction variance compared to any single model
- Adapts to changing market conditions automatically
- Self-improves based on performance feedback

## Market Regime Detection

The system automatically detects the current market regime using volatility and trend measurements:

1. **Normal Ranging**: Low volatility, no clear trend
2. **Normal Trending**: Low volatility, clear direction
3. **Volatile Ranging**: High volatility, no clear trend
4. **Volatile Trending**: High volatility, clear direction

Different model architectures excel in different regimes:
- TCN and Transformer: Perform well in volatile markets
- CNN and LSTM: More effective in trending markets
- Ensemble approach: Maintains robustness across all regimes

## Performance-Based Weight Adjustment

The system continuously tracks the performance of each model and adjusts weights accordingly:

1. After each prediction, the actual market movement is compared to the prediction
2. Models with correct predictions receive increased weights
3. Models with incorrect predictions have weights reduced
4. Recent performance is weighted more heavily than older results

This creates a self-improving system that favors consistently accurate models while maintaining diversity in the ensemble.

## Training Process

The training process involves:

1. **Data Collection**: Fetching historical OHLCV data for multiple timeframes
2. **Feature Engineering**: Calculating technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. **Data Integration**: Combining data from multiple timeframes
4. **Sequence Creation**: Generating input sequences with target values
5. **Model Training**: Training all model architectures with early stopping
6. **Ensemble Calibration**: Initializing ensemble weights based on validation performance

Training is performed with a 70/15/15 split (train/validation/test) and uses early stopping to prevent overfitting.

## Prediction Pipeline

The prediction pipeline involves:

1. **Data Preparation**: Processing current market data into model input format
2. **Individual Predictions**: Generating predictions from each model
3. **Regime Detection**: Determining the current market regime
4. **Weight Adjustment**: Adjusting model weights based on regime and performance
5. **Ensemble Prediction**: Calculating the weighted ensemble prediction
6. **Confidence Estimation**: Estimating confidence in the prediction
7. **Signal Generation**: Converting prediction and confidence to a trading signal

## Integration with Trading Strategies

The ML system integrates with existing trading strategies through the `MLModelIntegrator` class:

1. The integrator maintains the ensemble model and provides high-level interfaces
2. Trading strategies can request predictions and trading signals
3. Strategies can incorporate ML signals with configurable strength
4. The system provides both directional predictions and confidence levels
5. Performance feedback is collected to improve future predictions

The ML models can operate in three modes:
1. **Advisory**: Providing signals for human traders to consider
2. **Hybrid**: Working alongside traditional strategies with a weighted influence
3. **Autonomous**: Directly generating trading decisions with risk management

By default, the system operates in hybrid mode, where ML signals are combined with traditional strategy signals based on their relative strength and confidence.