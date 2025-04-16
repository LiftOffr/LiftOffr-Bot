# Enhanced Cryptocurrency Trading System

## System Improvements

We've made significant enhancements to the trading system to improve performance, accuracy, and profitability:

### 1. Enhanced Model Architecture

The improved hybrid model architecture now incorporates:

- **Multi-Branch Neural Network**:
  - CNN branch for capturing local price patterns
  - LSTM branch with bidirectional layers for better sequence memory
  - Advanced attention mechanisms for focusing on important patterns

- **5-Class Prediction System**:
  - Strong bearish (-1.0)
  - Moderate bearish (-0.5)
  - Neutral (0.0)
  - Moderate bullish (0.5)
  - Strong bullish (1.0)

- **Comprehensive Technical Indicators**:
  - Over 40 technical indicators for more robust feature engineering
  - Multiple timeframe analysis within each model
  - Volatility and market regime detection features

### 2. Improved Risk Management

- **Dynamic Leverage Adjustment**:
  - Base leverage: 5x
  - Maximum leverage: 75x
  - Leverage scales with prediction confidence
  - More confident predictions receive higher leverage

- **Maximum Portfolio Risk Cap**:
  - 25% maximum portfolio risk
  - Protects against excessive drawdowns
  - Ensures capital preservation during adverse market conditions

- **Dynamic Position Sizing**:
  - Position size adjusted based on available risk
  - More aggressive during high-confidence signals
  - More conservative during uncertain market conditions

### 3. Multi-Timeframe Training System

We now train and evaluate models across multiple timeframes:

- **1h**: Hourly data for medium-term trading
- **4h**: 4-hour data for swing trading
- **1d**: Daily data for longer-term position trading

The system evaluates which timeframe performs best for each pair and emphasizes that timeframe for trading decisions.

### 4. Comprehensive Performance Metrics

Each model is now evaluated using a broader range of metrics:

- **Classification Accuracy**: How well the model predicts the 5 price direction classes
- **Direction Accuracy**: How well the model predicts just the price direction (up/down)
- **Win Rate**: Percentage of profitable trades in backtest
- **Total Return**: Overall return on investment
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough drop
- **Profit Factor**: Ratio of gross profits to gross losses

### 5. Data Integration Enhancements

- **Flexible Data Handling**:
  - Works with existing historical data
  - Support for additional data sources like Amberdata API
  - Ability to seamlessly integrate new data as it becomes available

- **Standardized Data Processing Pipeline**:
  - Consistent feature engineering across all pairs and timeframes
  - Robust data normalization and scaling
  - Missing data handling and outlier management

## Training Scripts

We've created several specialized training scripts:

1. **train_with_existing_data.py**: Trains a single model using existing historical data
2. **train_all_pairs_existing_data.py**: Trains all pairs and timeframes with existing data
3. **train_with_amberdata.py**: Trains models using Amberdata API (when API key is available)
4. **train_all_timeframes.py**: Orchestrates training across multiple timeframes

## Performance Reports

The system now generates detailed performance reports:

1. **Individual Model Reports**: Comprehensive metrics for each pair/timeframe combination
2. **Summary Reports**: Overall performance across all models
3. **Best Model Selection**: Automatically selects the best-performing model for each pair

## How to Use

1. **Train a Single Model**:
   ```
   python train_with_existing_data.py --pair BTC/USD --timeframe 1h --epochs 50
   ```

2. **Train All Available Pairs**:
   ```
   python train_all_pairs_existing_data.py --epochs 50
   ```

3. **Train with External Data** (requires API key):
   ```
   python train_with_amberdata.py --pair BTC/USD --timeframe 1h --epochs 50
   ```

4. **Activate the Models for Trading**:
   ```
   python activate_improved_model_trading.py --reset_portfolio
   ```

## Future Enhancements

1. **Ensemble Methods**: Combining predictions from multiple timeframes
2. **Reinforcement Learning**: Further optimization of entry/exit timing
3. **Adaptive Hyperparameter Tuning**: Automatically adjusting model parameters
4. **Real-time Model Updates**: Continuous learning from market data
5. **Cross-asset Correlation Analysis**: Leveraging relationships between different assets