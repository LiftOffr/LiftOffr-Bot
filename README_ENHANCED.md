# Enhanced Kraken Trading Bot

This document outlines the enhancements made to the Kraken Trading Bot, including machine learning models, real-time visualization, log management, and market context awareness.

## New Features

### 1. Machine Learning Models
Three advanced neural network architectures have been implemented to improve prediction accuracy:

- **TCN (Temporal Convolutional Network)**  
  Uses causal convolutions to capture temporal dependencies in price data while avoiding future information leakage.

- **CNN (Convolutional Neural Network)**  
  Extracts local patterns and features from market data.

- **LSTM (Long Short-Term Memory)**  
  Captures sequential dependencies and long-term patterns in market data.

### 2. Enhanced Logging System
The integrated strategy now provides detailed insights through categorized log markers:

- **Analysis markers** showing forecast direction and price targets
- **Indicator markers** displaying technical indicator values
- **Volatility markers** showing current market volatility
- **Bands markers** showing price relationship with Bollinger Bands and Keltner Channels
- **Signal markers** showing final trading decisions
- **Action markers** showing actual trade executions

The system also includes log rotation and archiving through the `log_manager.py` module.

### 3. Real-time Visualization
A web-based dashboard (`dashboard.py`) provides real-time visualization of:

- Portfolio performance metrics
- Price charts with technical indicators
- Signal strength distribution
- Recent trading signals and executed trades

### 4. Market Context Awareness
The `market_context.py` module provides broader market awareness to inform trading decisions:

- General crypto market data (BTC dominance, total market cap)
- Asset-specific metrics (volatility, correlations)
- Market sentiment analysis
- Risk scoring and position sizing recommendations

### 5. Performance Metrics
Advanced performance metrics have been added (`performance_metrics.py`):

- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Equity curve visualization
- Detailed performance reports

### 6. Signal Strength Optimization
The `signal_strength_optimizer.py` tool optimizes signal strength parameters based on historical performance:

- Evaluates different parameter combinations
- Optimizes weights for different indicators
- Provides parameter importance analysis
- Generates detailed optimization reports

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Required packages: tensorflow, scikit-learn, matplotlib, pandas, numpy, flask, etc.

### Running the Enhanced Trading System
To start the enhanced trading system:
```bash
./start_enhanced_trading.sh
```

This will:
1. Set up log rotation
2. Start the real-time dashboard on port 8050
3. Start the integrated trading strategy in sandbox mode

### Accessing the Dashboard
Open your browser and navigate to:
```
http://localhost:8050
```

### Stopping the System
To stop all components:
```bash
./stop_enhanced_trading.sh
```

## Key Components

### Machine Learning Models (`ml_models.py`)
Contains implementation of TCN, CNN, and LSTM models for market prediction.

### Model Training (`train_ml_models.py`)
Script for training machine learning models using historical data.

### Log Management (`log_manager.py`)
Handles log rotation, archiving, and analysis.

### Performance Metrics (`performance_metrics.py`)
Calculates and visualizes advanced performance metrics.

### Signal Strength Optimization (`signal_strength_optimizer.py`)
Optimizes signal strength parameters based on historical performance.

### Dashboard (`dashboard.py`)
Provides real-time visualization of trading data.

### Market Context (`market_context.py`)
Analyzes broader market conditions to inform trading decisions.

## Strategy Markers

The integrated strategy uses the following markers in logs:

- `【ANALYSIS】` - Overall market forecast
- `【INDICATORS】` - Technical indicator values
- `【VOLATILITY】` - Current market volatility
- `【BANDS】` - Price relationship with Bollinger/Keltner bands
- `【SIGNAL】` - Final trading signal determination
- `【ACTION】` - Trade execution details
- `【INTEGRATED】` - Detailed signal strength information