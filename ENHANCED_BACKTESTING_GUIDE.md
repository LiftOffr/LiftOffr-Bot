# Enhanced Backtesting System

This guide explains the enhanced backtesting system for the Kraken Trading Bot, which has been designed to provide superior accuracy in backtesting and strategy optimization. The system incorporates several advanced techniques to better simulate real market conditions and achieve more realistic trade execution.

## Key Features

### 1. Enhanced Backtesting Engine

The `enhanced_backtesting.py` module implements a sophisticated backtesting framework with the following key features:

- **Realistic Order Execution**: Simulates order slippage, partial fills, and execution delays
- **Accurate Transaction Costs**: Dynamic fee modeling based on order types and market conditions
- **Market Regime Detection**: Identifies different market conditions for regime-specific analysis
- **Multi-Timeframe Support**: Integrates data from multiple timeframes for better decision-making
- **Walk-Forward Optimization**: Uses time-series cross-validation to prevent overfitting
- **Comprehensive Performance Metrics**: Detailed analytics including regime-specific breakdowns

### 2. Enhanced TCN Model

The `enhanced_tcn_model.py` module provides an advanced implementation of Temporal Convolutional Networks optimized for financial time series prediction:

- **Multi-Branch Architecture**: Combines TCN, CNN, LSTM, Attention, and Transformer components
- **Advanced Regularization**: Employs spatial dropout, batch normalization, layer normalization, and other techniques
- **Channel-Wise Attention**: Focus on the most informative features
- **Parameter Optimization**: Fine-tuned hyperparameters for financial time series
- **Market Regime Awareness**: Adapts to different market conditions

### 3. Automatic Model Pruning

The `auto_prune_ml_models.py` module implements an intelligent auto-pruning system for ML models:

- **Component-Level Analysis**: Identifies underperforming components within models
- **Performance-Based Pruning**: Removes elements that don't contribute to accuracy
- **Retraining Capabilities**: Automatically retrains pruned models
- **Ensemble Weight Optimization**: Adjusts weights based on model performance
- **Regime-Specific Evaluation**: Tests models under different market conditions

### 4. Strategy Parameter Optimization

The `optimize_strategy_parameters.py` module provides advanced optimization techniques:

- **Bayesian Optimization**: Efficient parameter search using Gaussian processes
- **Market Regime Optimization**: Finds optimal parameters for different market conditions
- **Multi-Objective Optimization**: Balances risk and reward metrics
- **Parameter Importance Analysis**: Identifies which parameters matter most
- **Cross-Validation**: Prevents overfitting with walk-forward testing

## Usage Guide

### Running Enhanced Backtests

To run an enhanced backtest for a specific trading strategy:

```bash
python run_enhanced_backtesting.py --strategy arima --symbol SOLUSD --timeframe 1h --plot
```

Additional options:
- `--optimize`: Optimize strategy parameters
- `--multi-strategy`: Run backtest with multiple strategies
- `--multi-asset`: Run backtest with multiple assets
- `--cross-timeframe`: Run backtest on multiple timeframes
- `--walk-forward`: Use walk-forward optimization
- `--use-best-params`: Use previously optimized parameters

### Training Enhanced TCN Models

To train an enhanced TCN model:

```bash
python enhanced_tcn_model.py --symbol SOLUSD --timeframe 1h
```

Additional options:
- `--sequence-length`: Sequence length for time series
- `--filters`: Number of filters in convolutional layers
- `--no-attention`: Disable attention mechanism
- `--no-transformer`: Disable transformer components
- `--multi-timeframe`: Use multi-timeframe data

### Auto-Pruning ML Models

To automatically prune underperforming ML models:

```bash
python auto_prune_ml_models.py --symbol SOLUSD --timeframe 1h --auto-prune-all
```

Additional options:
- `--model-path`: Prune a specific model
- `--ensemble-dir`: Prune a specific ensemble
- `--performance-threshold`: Minimum accuracy threshold
- `--symbols`: List of symbols to process
- `--timeframes`: List of timeframes to process

### Optimizing Strategy Parameters

To optimize the parameters of a trading strategy:

```bash
python optimize_strategy_parameters.py --strategy arima --symbol SOLUSD --timeframe 1h
```

Additional options:
- `--bayesian`: Use Bayesian optimization
- `--market-regimes`: Optimize for different market regimes
- `--scoring`: Metric to optimize (sharpe_ratio, total_return_pct, etc.)
- `--multi-asset`: Optimize for multiple assets
- `--multi-timeframe`: Optimize for multiple timeframes

## Integration with Trading Bot

The enhanced backtesting system integrates with the existing trading bot in the following ways:

1. **Strategy Optimization**: After finding optimal parameters with `optimize_strategy_parameters.py`, these can be used for live trading by saving them to a configuration file.

2. **ML Model Deployment**: Enhanced TCN models trained with `enhanced_tcn_model.py` can be used for prediction in the ML-enhanced trading strategy.

3. **Model Auto-Pruning**: The auto-pruning system helps maintain only the best-performing models in the ensemble, improving overall prediction accuracy.

4. **Market Regime Detection**: The market regime detection from `enhanced_backtesting.py` can be used in live trading to adapt to changing market conditions.

## Understanding Performance Metrics

The enhanced backtesting system provides comprehensive performance metrics:

- **Total Return**: Overall return of the strategy
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Maximum percentage loss from peak
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Regime Performance**: Performance breakdown by market regime

## Market Regime Classification

The system classifies market regimes as follows:

- **Normal**: Low volatility, neutral trend
- **Trending**: Low volatility, strong directional movement
- **Volatile**: High volatility, weak directional movement
- **Volatile Trending**: High volatility, strong directional movement

Each regime may require different strategy parameters for optimal performance, which is why the market regime optimization capability is so valuable.

## Conclusion

The enhanced backtesting system dramatically improves the accuracy and realism of strategy testing. By using this system, you can:

1. Get more realistic performance estimates before live trading
2. Identify optimal parameters for different market conditions
3. Improve model prediction accuracy through auto-pruning
4. Understand which components of your strategies are most important

For advanced users, the system also provides the ability to create custom strategies that adapt to different market regimes, applying different parameters based on detected market conditions.