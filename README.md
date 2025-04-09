# Kraken Trading Bot

A Python trading bot that connects to Kraken US exchange using their API and websockets. This bot implements various trading strategies and can be run in sandbox/test mode to validate functionality before trading with real funds.

## Features

- Connect to Kraken US API for trading
- Real-time market data via Kraken US websockets
- Multiple trading strategies:
  - Simple Moving Average (SMA) crossover
  - Relative Strength Index (RSI)
  - Combined strategy (SMA + RSI)
- Sandbox/test mode for safe testing
- Customizable trading pairs and quantities
- Error handling and reconnection logic
- Detailed logging

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

## Usage

1. Set your Kraken API key and secret as environment variables:

```bash
export KRAKEN_API_KEY='your_api_key'
export KRAKEN_API_SECRET='your_api_secret'
