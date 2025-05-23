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

# Status update interval in seconds
STATUS_UPDATE_INTERVAL = 60  # Default to once per minute

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
MARGIN_PERCENT = float(os.getenv('MARGIN_PERCENT', '0.20'))  # Percentage of portfolio used as margin (20% per trade)

# Strategy configuration
STRATEGY_TYPE = os.getenv('STRATEGY_TYPE', 'adaptive')  # Options: 'adaptive', 'simple_moving_average', 'rsi', 'arima'
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

# Portfolio monitor update interval
MONITOR_UPDATE_INTERVAL = int(os.getenv('MONITOR_UPDATE_INTERVAL', '60'))  # Display portfolio status every 60 seconds (1 minute)

# Order execution settings
ORDER_TIMEOUT_SECONDS = 2 * LOOP_INTERVAL  # Time before reconsidering a pending order

# Cross-strategy exit configuration
ENABLE_CROSS_STRATEGY_EXITS = os.getenv('ENABLE_CROSS_STRATEGY_EXITS', 'True').lower() in ['true', 't', 'yes', '1']  # Default to enabled
CROSS_STRATEGY_EXIT_THRESHOLD = float(os.getenv('CROSS_STRATEGY_EXIT_THRESHOLD', '0.65'))  # Signal strength threshold (0.0 to 1.0) - lowered from 0.75 to 0.65
CROSS_STRATEGY_EXIT_CONFIRMATION_COUNT = int(os.getenv('CROSS_STRATEGY_EXIT_CONFIRMATION_COUNT', '2'))  # Number of confirmations needed - reduced from 3 to 2

# Signal strength arbitration configuration
STRONGER_SIGNAL_DOMINANCE = os.getenv('STRONGER_SIGNAL_DOMINANCE', 'True').lower() in ['true', 't', 'yes', '1']  # Default to enabled
SIGNAL_STRENGTH_ADVANTAGE = float(os.getenv('SIGNAL_STRENGTH_ADVANTAGE', '0.25'))  # Minimum difference to override (default 0.25)
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', '0.65'))  # Minimum strength to be considered significant

# Dynamic position sizing based on signal strength
ENABLE_DYNAMIC_POSITION_SIZING = os.getenv('ENABLE_DYNAMIC_POSITION_SIZING', 'True').lower() in ['true', 't', 'yes', '1']  # Default to enabled
BASE_MARGIN_PERCENT = float(os.getenv('BASE_MARGIN_PERCENT', str(MARGIN_PERCENT)))  # Base margin percentage (default to MARGIN_PERCENT)
MAX_MARGIN_PERCENT = float(os.getenv('MAX_MARGIN_PERCENT', '0.40'))  # Maximum margin percentage for very strong signals (40%)
STRONG_SIGNAL_THRESHOLD = float(os.getenv('STRONG_SIGNAL_THRESHOLD', '0.80'))  # Signals above this are considered strong
VERY_STRONG_SIGNAL_THRESHOLD = float(os.getenv('VERY_STRONG_SIGNAL_THRESHOLD', '0.90'))  # Signals above this are considered very strong

# Dual limit order configuration for signal reversals (exits)
DUAL_LIMIT_ORDER_PRICE_OFFSET = float(os.getenv('DUAL_LIMIT_ORDER_PRICE_OFFSET', '0.05'))  # Price offset for dual limit orders (default $0.05)
DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT = int(os.getenv('DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT', '300'))  # Timeout in seconds before executing market order failsafe (default 5 minutes)
ENABLE_DUAL_LIMIT_ORDERS = os.getenv('ENABLE_DUAL_LIMIT_ORDERS', 'True').lower() in ['true', 't', 'yes', '1']  # Whether to use dual limit orders for signal reversals

# Dual limit order configuration for entries
ENABLE_DUAL_LIMIT_ENTRIES = os.getenv('ENABLE_DUAL_LIMIT_ENTRIES', 'True').lower() in ['true', 't', 'yes', '1']  # Whether to use dual limit orders for entries
DUAL_LIMIT_ENTRY_PRICE_OFFSET = float(os.getenv('DUAL_LIMIT_ENTRY_PRICE_OFFSET', '0.05'))  # Price offset for dual limit entry orders (default $0.05)
DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT = int(os.getenv('DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT', '300'))  # Timeout in seconds before executing market order failsafe (default 5 minutes)
