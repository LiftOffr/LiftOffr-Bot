import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Kraken API configuration
API_KEY = os.getenv('KRAKEN_API_KEY', '')
API_SECRET = os.getenv('KRAKEN_API_SECRET', '')

# Kraken API URLs
API_URL = "https://api.kraken.com"
API_VERSION = "0"  # Base API version

# Websocket URLs
WEBSOCKET_PUBLIC_URL = "wss://ws.kraken.com"
WEBSOCKET_PRIVATE_URL = "wss://ws-auth.kraken.com"

# Trading configuration
TRADING_PAIR = os.getenv('TRADING_PAIR', 'XBTUSD')  # Default to BTC/USD
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', '0.001'))  # Default to 0.001 BTC
USE_SANDBOX = os.getenv('USE_SANDBOX', 'True').lower() == 'true'  # Default to True for testing

# Strategy configuration
STRATEGY_TYPE = os.getenv('STRATEGY_TYPE', 'simple_moving_average')
SMA_SHORT_PERIOD = int(os.getenv('SMA_SHORT_PERIOD', '9'))
SMA_LONG_PERIOD = int(os.getenv('SMA_LONG_PERIOD', '21'))
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))

# Sleep time between iterations (in seconds)
LOOP_INTERVAL = int(os.getenv('LOOP_INTERVAL', '60'))
