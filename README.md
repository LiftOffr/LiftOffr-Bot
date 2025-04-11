# Kraken Trading Bot

A sophisticated Python trading bot designed for the Kraken US exchange, leveraging advanced API and websocket technologies for real-time cryptocurrency trading and intelligent market analysis.

## Features

- Kraken API integration for real-time market data
- Websocket real-time data streaming
- Multiple trading strategies:
  - ARIMA-based forecasting
  - Technical indicators (RSI, MACD, EMA)
  - Volatility analysis with ATR
- Comprehensive logging and error handling
- Sandbox mode for safe testing
- Margin trading support with configurable leverage
- Real-time portfolio tracking
- Email notifications for trades

## Setup

### Requirements

- Python 3.10+
- A Kraken US API key and secret
- (Optional) SendGrid API key for email notifications

### Environment Variables

Copy the `.env.example` file to `.env` and add your API keys:

```bash
KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_api_secret
SENDGRID_API_KEY=your_sendgrid_api_key  # Optional
```

### Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Running in Sandbox Mode

For testing without real funds:

```bash
python main.py --sandbox
```

### Running Live Trading

**Warning**: This will use real funds if API keys are provided:

```bash
python main.py --live
```

### Using the ARIMA Strategy Only

For ARIMA-based trading strategy:

```bash
python main.py --strategy arima --sandbox
```

### Checking Current Status

Check the current status of the trading bot:

```bash
./current
```

This will display:
- Current portfolio value
- Profit/loss metrics
- Active positions
- Recent trading signals
- Recent actions

### Start Scripts

#### Testing with Sandbox

```bash
./start_arima_trading.sh
```

#### Live Trading

```bash
./start_live_trading.sh
```

## Configuration

You can modify the trading parameters in `config.py`:

- `TRADING_PAIR`: The trading pair (default: "SOLUSD")
- `TRADE_QUANTITY`: The amount to trade in USD
- `INITIAL_PORTFOLIO_VALUE`: Starting portfolio value for metrics
- `USE_SANDBOX`: Whether to use sandbox mode
- `LEVERAGE`: Trading leverage (default: 5)

## Advanced Strategy Parameters

### ARIMA Strategy

In `arima_strategy.py`:

- `lookback_period`: Number of candles to analyze
- `atr_trailing_multiplier`: ATR multiplier for trailing stops
- `entry_atr_multiplier`: ATR multiplier for entry points
- `leverage`: Trading leverage
- `risk_buffer_multiplier`: Risk management buffer
- `arima_order`: ARIMA model parameters (p,d,q)

## Disclaimer

This trading bot is provided for educational and informational purposes only. Trading cryptocurrency involves substantial risk of loss and is not suitable for all investors. Do not trade with money you cannot afford to lose.