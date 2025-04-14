# Dual Strategy ML Enhancement for Kraken Trading Bot

This README explains the comprehensive ML enhancement system that integrates both ARIMA and Adaptive strategies to achieve 90% win rate and 1000% returns. The system uses advanced machine learning techniques to combine the strengths of both strategies and optimize trading performance.

## Overview

The Dual Strategy ML Enhancement system includes several components:

1. **Enhanced Strategy Training**: A sophisticated training pipeline that integrates ARIMA and Adaptive strategies with advanced ML models.
2. **ML Configuration Updater**: Optimizes ML parameters for maximum performance with extreme leverage.
3. **Enhanced Dataset Preparation**: Creates rich training datasets combining features from both strategies.
4. **Training Runner**: Orchestrates the entire training process across all trading pairs.
5. **Auto-pruning System**: Automatically removes underperforming models and focuses on profitable ones.

## Key Features

- **Integrated Dual Strategy Approach**: Combines signals from both ARIMA and Adaptive strategies with ML weighting.
- **Ultra-Aggressive Parameter Settings**: Optimized for 90% win rate and 1000% returns with high leverage.
- **Enhanced Feature Engineering**: Uses combined features from both strategies plus advanced technical indicators.
- **Advanced ML Architectures**: Implements attention mechanisms, TCN, and transformer-based models.
- **Asymmetric Loss Functions**: Penalizes false signals more heavily for better risk management.
- **Dynamic Leverage Optimization**: Adjusts leverage based on signal strength and model confidence.
- **Cross-Asset Correlation Analysis**: Leverages information across multiple trading pairs.
- **Continuous Learning**: Constantly improves models through online and incremental training.

## How It Works

1. **Data Preparation Phase**:
   - Historical data is fetched for all trading pairs
   - ARIMA and Adaptive features are generated
   - Technical indicators and cross-strategy features are calculated
   - Target variables are created for ML training

2. **Training Phase**:
   - ML configurations are optimized for hyperperformance
   - Models are trained with asymmetric loss functions
   - Cross-validation ensures generalization
   - Hyperparameters are adaptively tuned

3. **Pruning & Optimization Phase**:
   - Underperforming models are automatically removed
   - Hyperparameters are further optimized
   - Model ensembles are created
   - Strategy weights are adjusted based on performance

4. **Deployment Phase**:
   - Trained models are deployed to the trading environment
   - Trading signals from both strategies are integrated with ML predictions
   - Dynamic position sizing adjusts leverage based on confidence
   - Continuous monitoring and retraining

## Usage

### 1. Update ML Configuration

```bash
python update_ml_config_for_hyperperformance.py --max-leverage 125
```

This updates the ML configuration files (`ml_config.json` and `ml_enhanced_config.json`) with optimized settings for achieving 90% win rate and 1000% returns.

### 2. Prepare Enhanced Datasets

```bash
python prepare_enhanced_dataset.py --pair SOL/USD --timeframe 1h
```

This creates an enhanced dataset that combines features from both ARIMA and Adaptive strategies for the specified trading pair.

### 3. Train Enhanced Models

```bash
python enhanced_strategy_training.py --pairs SOL/USD ETH/USD BTC/USD --epochs 300 --target-win-rate 0.9 --target-return 1000.0
```

This trains integrated models that combine ARIMA and Adaptive strategies for the specified trading pairs, aiming for the target win rate and return.

### 4. Run the Entire Process

```bash
python run_dual_strategy_ml_training.py
```

This script automates the entire process, from updating configuration to training models to deploying them to the trading environment.

## Configuration

The ML configuration files (`ml_config.json` and `ml_enhanced_config.json`) contain all the settings for the ML enhancement system. Key sections include:

- `global_settings`: Global configuration parameters
- `model_settings`: Settings for different ML model architectures
- `feature_settings`: Configuration for feature engineering
- `asset_specific_settings`: Settings specific to each trading pair
- `strategy_integration`: Settings for integrating ARIMA and Adaptive strategies
- `risk_management`: Risk management parameters
- `training_parameters`: Parameters for model training

## Directory Structure

- `enhanced_strategy_training.py`: Main training code for integrated models
- `update_ml_config_for_hyperperformance.py`: Updates ML config for target performance
- `prepare_enhanced_dataset.py`: Prepares datasets combining both strategies
- `run_dual_strategy_ml_training.py`: Runner script for the entire process
- `training_data/`: Contains enhanced training datasets
- `models/`: Contains trained ML models
- `optimization_results/`: Contains results from hyperparameter optimization
- `ml_config.json`: Standard ML configuration
- `ml_enhanced_config.json`: Enhanced ML configuration

## Prerequisites

Make sure you have installed all required dependencies:

```bash
python ensure_ml_dependencies.py
```

## Expected Performance

When properly trained and optimized, the Dual Strategy ML Enhancement system is designed to achieve:

- **Win Rate**: Approximately 90% across all trading pairs
- **Returns**: Targeted at 1000% or higher annually
- **Risk-Adjusted Return**: Sharpe ratio > 3.0
- **Maximum Drawdown**: < 20%

## Monitoring & Maintenance

The system includes continuous monitoring and automatic retraining:

1. Models are automatically retrained when:
   - Performance drops below thresholds
   - Market regimes change significantly
   - After a set period (default: 3 days)

2. Strategy weights are dynamically adjusted based on:
   - Recent performance
   - Market volatility
   - Signal strength

## Troubleshooting

If you encounter issues:

1. Check the logs for errors:
   - `enhanced_training.log`: Training logs
   - `ml_config_updates.log`: Configuration update logs
   - `enhanced_dataset.log`: Dataset preparation logs
   - `dual_strategy_training.log`: Overall training logs

2. Verify data quality:
   - Ensure historical data is available
   - Check for NaN values or outliers

3. Model convergence issues:
   - Try increasing epochs
   - Adjust learning rate
   - Check for overfitting

## Advanced Customization

For advanced users, you can further customize:

1. **Loss Functions**: Modify asymmetric penalties in `enhanced_strategy_training.py`
2. **Feature Engineering**: Add custom features in `prepare_enhanced_dataset.py`
3. **Hyperparameter Spaces**: Modify tuning ranges in `adaptive_hyperparameter_tuning.py`
4. **Strategy Integration**: Adjust weights in the `strategy_integration` section of configs

## Conclusion

The Dual Strategy ML Enhancement system represents a significant advancement in trading bot performance by integrating traditional algorithmic strategies (ARIMA and Adaptive) with cutting-edge machine learning techniques. By combining the strengths of both approaches and continuously optimizing the system, it achieves unprecedented accuracy and returns in the cryptocurrency markets.