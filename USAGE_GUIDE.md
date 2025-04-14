# Enhanced Trading Bot Usage Guide

This guide covers how to use the new scripts to optimize your trading bot performance, expand trading to additional pairs, and improve your existing strategies.

## 1. Optimizing ML Models to Reach 90% Accuracy

The `optimize_ml_hyperparameters.py` script uses Bayesian optimization to find the best hyperparameters for ML models, targeting 90%+ accuracy. It includes:

- Advanced feature engineering
- Cross-validation
- Multiple model architectures (LSTM, CNN, Transformer, etc.)
- Ensemble techniques
- Data augmentation

### Basic Usage:

```bash
python optimize_ml_hyperparameters.py --pairs SOL/USD,BTC/USD,ETH/USD
```

### Advanced Usage:

```bash
python optimize_ml_hyperparameters.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD --epochs 200 --trials 20 --cv-folds 5 --verbose
```

### Parameters:

- `--pairs`: Trading pairs to optimize (comma-separated)
- `--epochs`: Maximum number of training epochs (default: 200)
- `--batch-size`: Batch size for training (default: 32)
- `--patience`: Early stopping patience (default: 20)
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--trials`: Number of optimization trials (default: 20)
- `--output-dir`: Directory to save models (default: models)
- `--verbose`: Print detailed information

## 2. Adding More Trading Pairs

The `add_additional_trading_pairs.py` script streamlines the process of adding new trading pairs to your system, such as ADA/USD and LINK/USD.

### Basic Usage:

```bash
python add_additional_trading_pairs.py --pairs ADA/USD,LINK/USD --sandbox
```

### Advanced Usage:

```bash
python add_additional_trading_pairs.py --pairs ADA/USD,LINK/USD --days 365 --base-leverage 20.0 --max-leverage 125.0 --confidence-threshold 0.65 --risk-percentage 0.20 --sandbox
```

### Parameters:

- `--pairs`: New trading pairs to add (comma-separated)
- `--capital`: Starting capital (default: 20000.0)
- `--timeframe`: Timeframe for historical data (default: 1h)
- `--days`: Days of historical data to fetch (default: 365)
- `--base-leverage`: Base leverage for trading (default: 20.0)
- `--max-leverage`: Maximum leverage for high-confidence trades (default: 125.0)
- `--confidence-threshold`: ML confidence threshold (default: 0.65)
- `--risk-percentage`: Risk percentage per trade (default: 0.20)
- `--sandbox`: Use sandbox mode (default)
- `--live`: Use live trading mode (CAUTION!)
- `--skip-fetch`: Skip fetching historical data
- `--skip-train`: Skip training models

## 3. Improving Existing Strategies

The `improve_strategies.py` script analyzes backtest results and optimizes parameters for ARIMA and Adaptive strategies to enhance their performance.

### Basic Usage:

```bash
python improve_strategies.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD
```

### Advanced Usage:

```bash
python improve_strategies.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD --days 180 --trials 50 --only-analyze --verbose
```

### Parameters:

- `--pairs`: Trading pairs to optimize (comma-separated)
- `--days`: Number of days for backtesting (default: 180)
- `--timeframes`: Timeframes to test (comma-separated, default: 1h)
- `--strategies`: Strategies to optimize (comma-separated, default: arima,adaptive)
- `--only-analyze`: Only analyze, don't modify any files
- `--output-dir`: Directory to save results (default: strategy_optimization)
- `--trials`: Number of optimization trials (default: 50)
- `--verbose`: Print detailed information

## 4. Recommended Workflow

1. **Optimize Existing Strategies**:
   ```bash
   python improve_strategies.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD --only-analyze
   ```
   Review the results in `strategy_optimization/optimization_report.html`, then run without `--only-analyze` to apply the changes.

2. **Optimize ML Models**:
   ```bash
   python optimize_ml_hyperparameters.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD
   ```
   This will create improved ML models targeting 90%+ accuracy.

3. **Add More Trading Pairs**:
   ```bash
   python add_additional_trading_pairs.py --pairs ADA/USD,LINK/USD --sandbox
   ```
   This will add the new pairs with historical data and optimized settings.

4. **Activate Trading**:
   ```bash
   python quick_activate_ml.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD,ADA/USD,LINK/USD --sandbox
   ```
   This starts trading with the optimized models and strategies.

## 5. Tips for Best Results

1. **Start with Strategy Optimization**: Improve the existing strategies first, as they form the foundation of your trading system.

2. **Gradually Add Trading Pairs**: Add one pair at a time, verify it works well, then add more.

3. **Monitor Performance Regularly**: Check portfolio status and strategy performance daily.

4. **Balance Risk Across Pairs**: Adjust risk parameters based on each pair's volatility.

5. **Periodically Re-optimize**: Markets change over time; re-run the optimization scripts monthly.

6. **Always Use Sandbox First**: Test all changes in sandbox mode before considering live trading.

7. **Analyze Backtest Results**: Pay attention to max drawdown and win rate, not just total return.

## 6. Troubleshooting

If you encounter issues:

1. **Check Logs**: Look for error messages in the logs directory.

2. **Verify Data**: Ensure historical data is available for all pairs.

3. **Memory Issues**: For larger models, reduce batch size or simplify model architecture.

4. **Strategy Conflicts**: If strategies conflict, adjust signal strength thresholds.

5. **Reset Portfolio**: Use `close_all_positions.py` to reset the sandbox if needed.

By following this guide, you'll have a powerful trading system with optimized ML models (targeting 90%+ accuracy), multiple trading pairs, and improved trading strategies.