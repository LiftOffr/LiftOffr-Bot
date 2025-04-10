# Kraken Trading Bot

A sophisticated Python trading bot designed for the Kraken US exchange, leveraging advanced API and websocket technologies for real-time cryptocurrency trading and intelligent market analysis.

## Features

- Full Kraken US API integration for trading operations
- Real-time market data via Kraken US WebSockets
- Multiple advanced trading strategies:
  - Simple Moving Average (SMA) crossover
  - Relative Strength Index (RSI)
  - Adaptive strategy with multiple indicators (EMA, MACD, ATR, Bollinger Bands)
  - ARIMA time series forecasting
  - Linear regression forecasting
- Trailing stop implementation with ATR-based sizing
- Intelligent position management
- Sandbox/test mode for risk-free testing
- Customizable trading pairs and position sizing
- Robust error handling and automatic reconnection
- Clean, informative logging with categorized output

## Requirements

- Python 3.7+
- Required packages:
  - websocket-client
  - requests
  - numpy
  - pandas

## Configuration

The bot can be configured using environment variables or command line arguments:

### Environment Variables

- `KRAKEN_API_KEY`: Your Kraken API key
- `KRAKEN_API_SECRET`: Your Kraken API secret
- `TRADING_PAIR`: Trading pair to use (default: 'XBTUSD')
- `TRADE_QUANTITY`: Quantity to trade (default: 0.001)
- `STRATEGY_TYPE`: Trading strategy to use (default: 'simple_moving_average')
- `USE_SANDBOX`: Set to 'True' to run in sandbox/test mode (default: 'True')
- `SMA_SHORT_PERIOD`: Short period for SMA strategy (default: 9)
- `SMA_LONG_PERIOD`: Long period for SMA strategy (default: 21)
- `RSI_PERIOD`: Period for RSI calculation (default: 14)
- `RSI_OVERBOUGHT`: Overbought threshold for RSI (default: 70)
- `RSI_OVERSOLD`: Oversold threshold for RSI (default: 30)
- `LOOP_INTERVAL`: Sleep time between iterations in seconds (default: 60)

### Command Line Arguments

- `--pair`: Trading pair to use
- `--quantity`: Quantity to trade
- `--strategy`: Trading strategy to use
- `--sandbox`: Run in sandbox/test mode

## Trading Strategies

The bot supports several different trading strategies:

### Simple Moving Average (SMA)
- Uses crossovers between short and long moving averages to generate signals
- Buy when short MA crosses above long MA, sell when it crosses below
- Configurable periods for both moving averages

### Relative Strength Index (RSI)
- Uses RSI to identify overbought and oversold conditions
- Buy when RSI crosses above oversold threshold, sell when it crosses below overbought threshold
- Configurable period and thresholds

### Adaptive Strategy
- Combines multiple technical indicators (EMA, RSI, MACD, ADX, ATR)
- Uses Bollinger Bands and Keltner Channels for volatility assessment
- Applies dynamic risk management based on market conditions

### ARIMA Strategy
- Uses AutoRegressive Integrated Moving Average (ARIMA) forecasting
- Predicts future price movements based on time series analysis
- Generates buy signals when forecast is bullish, sell signals when forecast is bearish
- Applies ATR-based trailing stops for risk management
- Includes risk buffer multiplier to prevent liquidation

## Usage

1. Set your Kraken API key and secret as environment variables:

```bash
export KRAKEN_API_KEY='your_api_key'
export KRAKEN_API_SECRET='your_api_secret'
```

2. Run the trading bot with your desired configuration:

```bash
python main.py --strategy adaptive --pair SOLUSD --sandbox
```

3. For production use, remove the `--sandbox` flag (ensure you have your API keys set):

```bash
python main.py --strategy adaptive --pair SOLUSD
```

4. Alternatively, use the provided startup scripts:

```bash
# To run with the ARIMA strategy in sandbox mode:
./start_arima_trading.sh

# To run with the ARIMA strategy in live trading mode:
./start_arima_trading.sh --live

# To run with the default adaptive strategy:
./start_live_trading.sh
```

5. To customize further, you can edit the environment variables in the `.env` file:

```bash
# Example .env file
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
TRADING_PAIR=SOLUSD
STRATEGY_TYPE=arima
SENDGRID_API_KEY=your_sendgrid_key  # For email notifications
```
