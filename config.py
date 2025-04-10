import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Notification settings
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', 'cchapman.liftoffr@gmail.com')  # Default to email address
NOTIFICATION_ENABLED = SENDGRID_API_KEY != ''

# Log whether API keys are available
if API_KEY and API_SECRET:
    logging.info("Kraken API credentials loaded successfully")
else:
    logging.warning("Kraken API credentials not found - running in sandbox mode only")

# Log whether notification is enabled
if NOTIFICATION_ENABLED:
    logging.info("Email notifications enabled")
else:
    logging.warning("SendGrid API key not found - email notifications disabled")

# Kraken API URLs
API_URL = "https://api.kraken.com"
API_VERSION = "0"  # Base API version

# Websocket URLs
WEBSOCKET_PUBLIC_URL = "wss://ws.kraken.com"
WEBSOCKET_PRIVATE_URL = "wss://ws-auth.kraken.com"

# Trading configuration
TRADING_PAIR = os.getenv('TRADING_PAIR', 'SOLUSD')  # Default to SOL/USD
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', '0.001'))  # Default to 0.001 BTC
USE_SANDBOX = os.getenv('USE_SANDBOX', 'True').lower() in ['true', 't', 'yes', '1']  # Default to True for testing

# Position management
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '20000.0'))  # Starting portfolio value in USD
LEVERAGE = int(os.getenv('LEVERAGE', '25'))  # Leverage for trading (25x by default)
MARGIN_PERCENT = float(os.getenv('MARGIN_PERCENT', '0.25'))  # Percentage of portfolio used as margin

# Strategy configuration
STRATEGY_TYPE = os.getenv('STRATEGY_TYPE', 'adaptive')  # Default to adaptive strategy from original code
SMA_SHORT_PERIOD = int(os.getenv('SMA_SHORT_PERIOD', '9'))
SMA_LONG_PERIOD = int(os.getenv('SMA_LONG_PERIOD', '21'))
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))

# Advanced strategy parameters (from original code)
VOL_THRESHOLD = float(os.getenv('VOL_THRESHOLD', '0.006'))  # Normalized ATR threshold
ENTRY_ATR_MULTIPLIER = float(os.getenv('ENTRY_ATR_MULTIPLIER', '0.01'))  # Entry offset multiplier for limit orders
BREAKEVEN_PROFIT_TARGET = float(os.getenv('BREAKEVEN_PROFIT_TARGET', '1.0'))  # Profit target in ATR multiples
ARIMA_LOOKBACK = int(os.getenv('ARIMA_LOOKBACK', '10'))  # Lookback period for linear regression forecast

# Sleep time between iterations (in seconds)
LOOP_INTERVAL = int(os.getenv('LOOP_INTERVAL', '5'))  # Sleep 5 seconds between iterations
SIGNAL_INTERVAL = int(os.getenv('SIGNAL_INTERVAL', '30'))  # Update signals every 30 seconds for testing
LOOKBACK_HOURS = int(os.getenv('LOOKBACK_HOURS', '12'))  # Hours of historical data to fetch
STATUS_UPDATE_INTERVAL = int(os.getenv('STATUS_UPDATE_INTERVAL', '10'))  # Status update every 10 seconds

# Order execution settings
ORDER_TIMEOUT_SECONDS = 2 * LOOP_INTERVAL  # Time before reconsidering a pending order
