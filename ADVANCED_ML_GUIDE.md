# Advanced Machine Learning Trading System Guide

This guide explains the advanced machine learning components implemented in the Kraken trading bot to achieve 90% prediction accuracy and 1000%+ returns.

## Overview

The advanced ML trading system consists of five key components:

1. **Temporal Fusion Transformer (TFT)** - A sophisticated deep learning architecture for time series forecasting
2. **Multi-Asset Feature Fusion** - Analyzes correlations and relationships between multiple assets to improve predictions
3. **Adaptive Hyperparameter Tuning** - Automatically adjusts model parameters based on performance and market conditions
4. **Explainable AI** - Provides clear explanations for trading decisions with feature importance analysis
5. **Sentiment Analysis** - Incorporates news and social media sentiment for improved prediction during high-impact events

## Installation Requirements

To use the advanced ML components, you need the following dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy optuna shap matplotlib tensorflow-addons
pip install trafilatura requests beautifulsoup4 nltk
```

For sentiment analysis with transformer models (optional):
```bash
pip install transformers torch
```

## Using the Advanced ML System

### Training Models

To train the advanced ML models with historical data:

```bash
python train_advanced_ml_models.py --trading-pairs SOL/USD ETH/USD BTC/USD --force-retrain --use-sentiment
```

Optional arguments:
- `--trading-pairs`: List of trading pairs to train models for
- `--force-retrain`: Force retraining models even if they exist
- `--max-lookback-days`: Maximum days of historical data to use
- `--timeframes`: Timeframes to include (e.g., 1d, 4h, 1h)
- `--use-cross-asset`: Enable cross-asset feature fusion
- `--use-sentiment`: Include sentiment analysis

### Running Backtests

To evaluate model performance using historical data:

```bash
python run_advanced_ml_trading.py --mode backtest --trading-pairs SOL/USD ETH/USD --start-date 2024-01-01
```

Optional arguments:
- `--mode`: Trading mode (backtest, live, sandbox)
- `--trading-pairs`: List of trading pairs to trade
- `--start-date`: Start date for backtest (YYYY-MM-DD)
- `--end-date`: End date for backtest (YYYY-MM-DD)
- `--capital`: Initial capital for trading
- `--max-leverage`: Maximum leverage to use
- `--risk-per-trade`: Percentage of capital to risk per trade
- `--use-sentiment`: Include sentiment analysis
- `--explain-trades`: Generate trade explanations

### Live Trading

To run live trading with the advanced ML system:

```bash
python run_advanced_ml_trading.py --mode live --trading-pairs SOL/USD ETH/USD --use-sentiment --explain-trades
```

For sandbox trading (simulated):
```bash
python run_advanced_ml_trading.py --mode sandbox --trading-pairs SOL/USD ETH/USD --use-sentiment --explain-trades
```

## Component Details

### Temporal Fusion Transformer (TFT)

The TFT model combines recurrent networks with self-attention mechanisms to create a powerful architecture that excels at multi-horizon forecasting:

- **Variable selection networks** identify the most relevant input variables
- **Gated residual networks** control the flow of information through the model
- **Multi-head attention** focuses on the most important time steps
- **Interpretable components** allow for understanding prediction drivers

The TFT model is configured with:
- Sequence length: 60-120 days (asset-dependent)
- Hidden units: 64-256 (optimized per asset)
- Heads: 4-8 (optimized per asset)
- Asymmetric loss function to prioritize profitable trades

### Multi-Asset Feature Fusion

This component analyzes relationships between different assets to create more powerful predictive features:

- **Lead-lag relationships** identify assets that predict others' movements
- **Correlation analysis** finds assets that move together or oppositely
- **Cointegration testing** identifies pairs with long-term equilibrium relationships
- **Cross-asset features** such as price ratios, relative volatility, and inter-market spreads

Features are combined using:
- Principal component analysis (PCA) for dimensionality reduction
- Dedicated feature importance analysis to identify the most predictive relationships

### Adaptive Hyperparameter Tuning

This system automatically optimizes model parameters based on market conditions:

- **Market regime detection** to identify trending, ranging, or volatile markets
- **Parameter space exploration** using Bayesian optimization
- **Performance tracking** to correlate parameters with outcomes
- **Automatic adjustment** based on recent model performance

The tuning system targets:
- Aggressive parameters in trending markets
- Defensive parameters in volatile markets
- Balanced parameters in ranging markets

### Explainable AI

Makes model predictions transparent and interpretable:

- **SHAP values** show the contribution of each feature to predictions
- **Feature importance analysis** identifies the most significant factors
- **Decision boundary visualization** shows how predictions change with input values
- **Human-readable explanations** for every trading signal

Explanations cover:
- Key market factors influencing each decision
- Confidence levels and uncertainty estimates
- Comparisons with historical patterns and baseline expectations

### Sentiment Analysis

Incorporates market sentiment to improve predictions during major events:

- **News sentiment analysis** from financial and crypto news sources
- **Lexicon-based analysis** with domain-specific terms
- **Temporal tracking** of sentiment changes and trends
- **Adaptive weighting** of sentiment vs. technical indicators

## Integrating with Existing Strategies

The advanced ML models integrate with existing trading strategies (ARIMA, Adaptive) via the bot manager:

1. Both systems generate signals independently
2. The bot manager arbitrates between them based on signal strength/confidence
3. The strongest signal determines the final trading action
4. During conflicts, ML signals with high confidence can override traditional strategies

## ML Configuration

The ML system's behavior is controlled through the `ml_config.json` file, which includes:

- Asset-specific settings for each trading pair
- Model weights for different market regimes
- Leverage settings based on prediction confidence
- Position sizing parameters
- Risk management settings
- Training parameters

Example configuration snippet:
```json
{
  "asset_configs": {
    "SOL/USD": {
      "leverage_settings": {
        "min": 20.0,
        "default": 35.0,
        "max": 125.0,
        "confidence_threshold": 0.65
      },
      "model_weights": {
        "transformer": 0.40,
        "tcn": 0.30,
        "lstm": 0.10,
        "attention_gru": 0.20
      }
    }
  }
}
```

## Performance Optimization

To achieve 90% prediction accuracy and 1000%+ returns:

1. **Continuous training** on new market data
2. **Automatic pruning** of underperforming models
3. **Dynamic model selection** based on recent performance
4. **Asymmetric loss functions** to prioritize profitable trades
5. **Adaptive leverage** based on prediction confidence
6. **Custom position sizing** to maximize returns while managing risk

## Logs and Monitoring

The system generates detailed logs and visualizations:

- Training results in `training_results/`
- Backtest results in `backtest_results/`
- Prediction logs in `prediction_logs/`
- Sentiment analysis in `sentiment_data/`
- Explainable AI visualizations in `trading_explanations/`

## Troubleshooting

If you encounter issues:

1. Check the log files for error messages
2. Ensure all required dependencies are installed
3. Verify that historical data is available and properly formatted
4. Make sure API keys are valid for live trading
5. Check disk space for model storage

## Next Steps

Continuing to improve the system:

1. Regular retraining with new market data
2. Experimenting with new model architectures
3. Fine-tuning hyperparameters for specific market conditions
4. Adding more sentiment data sources
5. Expanding to additional trading pairs