# ML-Enhanced Trading System

This document explains the Machine Learning (ML) enhancements added to the Kraken Trading Bot, including how to train the models, integrate them with existing strategies, and configure the ML parameters for optimal trading performance.

## Overview

The ML-enhanced trading system combines traditional technical analysis with advanced machine learning models to improve trading decisions. The system uses an ensemble of multiple model architectures to make predictions about market direction and volatility, which are then integrated with existing trading strategies.

## Key Components

1. **Advanced Ensemble Model** (`advanced_ensemble_model.py`)
   - Combines predictions from multiple model architectures
   - Dynamically adjusts model weights based on market conditions and past performance
   - Detects market regimes (volatile, trending, ranging, etc.) to optimize predictions

2. **ML Strategy Integrator** (`ml_strategy_integrator.py`)
   - Bridges ML predictions with traditional trading strategies
   - Adjusts position sizing and risk management based on ML confidence
   - Recommends optimal stop-loss levels based on volatility assessment

3. **ML-Enhanced Strategy** (`ml_enhanced_strategy.py`)
   - Wraps existing strategies with ML capabilities
   - Enhances entry/exit signals using ML predictions
   - Provides ML-influenced trailing stops and position sizing

## Model Architectures

The ensemble combines multiple advanced neural network architectures:

1. **TCN (Temporal Convolutional Network)**
   - Excellent at capturing long-range patterns in time series data
   - Handles variable-length inputs with dilated convolutions
   - Especially effective in volatile market conditions

2. **CNN (Convolutional Neural Network)**
   - Identifies local patterns and features in price data
   - Good at recognizing chart patterns and technical formations
   - Effective at finding short-term trading opportunities

3. **LSTM (Long Short-Term Memory)**
   - Captures long-term dependencies in sequential data
   - Maintains memory of past market states
   - Good for trend recognition and momentum strategies

4. **GRU (Gated Recurrent Unit)**
   - Simplified version of LSTM with comparable performance
   - Faster training and execution than LSTM
   - Balances short and medium-term pattern recognition

5. **BiLSTM (Bidirectional LSTM)**
   - Processes data in both forward and backward directions
   - Enhances context understanding in time series
   - Better at identifying reversals and pattern completions

6. **Attention Model**
   - Focuses on the most relevant parts of input data
   - Weighs importance of different time periods
   - Especially good at identifying key market turning points

7. **Transformer Model**
   - State-of-the-art architecture for sequential data
   - Captures complex relationships between different time points
   - Excels at multi-timeframe analysis

8. **Hybrid Model**
   - Combines CNN, LSTM, and Attention mechanisms
   - Leverages strengths of multiple architectures
   - Best overall performance across different market regimes

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Keras TCN
- NumPy, Pandas, Scikit-learn
- Historical data for your trading pairs

### Installation

1. Ensure all required packages are installed:
```bash
pip install tensorflow keras-tcn numpy pandas scikit-learn matplotlib
```

2. All ML-related scripts are included in the project directory, no additional installation steps required.

### Training the Models

1. Run the training script to train all model architectures:
```bash
bash start_high_accuracy_training.sh
```

This script will:
- Create necessary model directories
- Fetch historical data if not already present
- Automatically detect available timeframes (1h, 4h, 1d)
- Train each model architecture sequentially
- Save trained models in their respective directories

The training occurs in phases with increasing complexity:
- Phase 1: Training core models with single timeframe data
- Phase 2: Multi-timeframe integration with medium complexity
- Phase 3: Full ensemble with all model architectures

Training typically takes 2-4 hours depending on your system, with each model requiring 15-30 minutes.

> **Note about timeframes:** The training system is designed to work with available timeframes from Kraken. Currently, it automatically detects and uses available 1h and 4h timeframe data. The 1d timeframe is not supported by Kraken's API for some trading pairs (including SOL/USD).

### Running ML-Enhanced Trading

1. Start the ML-enhanced trading bot:
```bash
bash start_ml_enhanced_trading.sh
```

By default, the bot runs in sandbox mode. To run in live mode (use with caution!):
```bash
bash start_ml_enhanced_trading.sh --live
```

Additional parameters can be configured:
```bash
bash start_ml_enhanced_trading.sh --capital=50000 --leverage=10 --ml-influence=0.7 --confidence=0.8
```

## Configuration Parameters

### ML Influence

The `ml_influence` parameter (0.0-1.0) controls how much weight the ML predictions have in the final trading decision:

- 0.0: ML predictions are completely ignored
- 0.5: Equal weight between strategy signals and ML predictions (default)
- 1.0: Trading decisions are based entirely on ML predictions

Example: To set ML influence to 70%:
```bash
python start_ml_enhanced_trading.py --ml-influence=0.7
```

### Confidence Threshold

The `confidence_threshold` parameter (0.0-1.0) sets the minimum confidence level required for ML predictions to influence trading decisions:

- Lower values (e.g., 0.3) allow more ML predictions to influence decisions, potentially increasing trade frequency
- Higher values (e.g., 0.8) only allow high-confidence predictions to influence decisions, potentially reducing trade frequency but increasing quality

Example: To set confidence threshold to 80%:
```bash
python start_ml_enhanced_trading.py --confidence=0.8
```

## Performance Analysis

The ML-enhanced strategies track their performance, including:

- ML influence rate (percentage of decisions influenced by ML)
- Prediction accuracy (overall and recent)
- Model confidence levels
- Market regime detection

This information is available in the strategy status output and logs.

## Advanced Customization

### Modifying Ensemble Weights

You can adjust the base weights of different models in the ensemble by modifying the following in `advanced_ensemble_model.py`:

```python
# Default weights - equal weighting to start
default_weights = {
    "tcn": 1.0 / len(MODEL_TYPES),
    "cnn": 1.0 / len(MODEL_TYPES),
    # ... other models
}
```

### Adding New Features

To add new features for the ML models to consider:

1. Modify the `load_and_prepare_data` function in `train_ensemble_models.py`
2. Add your new features to the `feature_columns` list
3. Retrain the models using the training script

### Custom Market Regime Detection

You can customize how market regimes are detected by modifying the `detect_market_regime` function in `advanced_ensemble_model.py`.

## Troubleshooting

### Models Not Loading

If models fail to load, check:
- That all model files exist in their respective directories
- That the model architecture matches what's expected
- That you've installed all required packages, particularly keras-tcn

### Training Errors

If you encounter errors during training:
- Check that your historical data is correctly formatted
- Ensure you have enough RAM for model training
- Try reducing batch size or model complexity

### Missing Timeframe Data

If you encounter errors regarding missing timeframe data:
- The system is designed to automatically detect available timeframes
- It will adapt to use only the timeframes available (typically 1h and 4h)
- You can check available timeframes in the `historical_data` directory
- If you need to add more timeframes, modify the data fetcher scripts

Note: Kraken API may not support certain timeframes (like 1d) for all trading pairs. The system is designed to gracefully handle these limitations.

### Integration Issues

If ML predictions aren't properly influencing trading decisions:
- Check that the ML influence parameter is set > 0
- Verify that model confidence is exceeding the threshold
- Look for any errors in the logs related to prediction generation

## Contributing

Feel free to contribute to the ML-enhanced trading system by:
- Adding new model architectures
- Implementing additional features and indicators
- Optimizing existing models for better performance
- Improving market regime detection

## Future Enhancements

Planned enhancements for the ML system include:
- Reinforcement learning for dynamic strategy optimization
- Sentiment analysis integration from news and social media
- Multi-timeframe model coordination
- Automatic hyperparameter optimization