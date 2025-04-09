import time
import hmac
import base64
import hashlib
import urllib.parse
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def get_kraken_signature(urlpath, data, secret):
    """
    Generate Kraken API signature
    
    Args:
        urlpath (str): API endpoint path
        data (dict): API request parameters
        secret (str): API secret key
    
    Returns:
        str: API request signature
    """
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    
    signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(signature.digest())
    
    return sigdigest.decode()

def get_nonce():
    """
    Get unique nonce for API requests
    
    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * 1000)

def calculate_sma(data, period):
    """
    Calculate Simple Moving Average
    
    Args:
        data (list): List of price data
        period (int): Period for SMA calculation
    
    Returns:
        float: SMA value
    """
    return np.mean(data[-period:]) if len(data) >= period else None

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data (list): List of price data
        period (int): Period for RSI calculation
    
    Returns:
        float: RSI value
    """
    if len(data) < period + 1:
        return None
    
    # Convert to numpy array for calculations
    price_array = np.array(data)
    # Calculate price differences
    deltas = np.diff(price_array)
    
    # Calculate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Calculate subsequent values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def format_kraken_symbol(symbol):
    """
    Format trading pair symbol to Kraken format
    
    Args:
        symbol (str): Trading pair symbol (e.g. 'BTCUSD')
    
    Returns:
        str: Kraken formatted symbol (e.g. 'XBT/USD')
    """
    # Kraken uses XBT instead of BTC
    symbol = symbol.replace('BTC', 'XBT')
    
    # Handle special cases for Kraken
    if '/' not in symbol:
        # If no separator, insert one between base and quote currency
        if len(symbol) >= 6:  # Most pairs are 6+ characters (XBTUSD, ETHUSD, etc.)
            base = symbol[:-3]
            quote = symbol[-3:]
            return f"{base}/{quote}"
        else:
            return symbol
    
    return symbol

def parse_kraken_ohlc(ohlc_data):
    """
    Parse OHLC data from Kraken API into pandas DataFrame
    
    Args:
        ohlc_data (list): List of OHLC data from Kraken API
    
    Returns:
        pd.DataFrame: DataFrame with OHLC data
    """
    df = pd.DataFrame(ohlc_data, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    
    # Convert types
    df['time'] = pd.to_datetime(df['time'], unit='s')
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
        df[col] = df[col].astype(float)
    
    df['count'] = df['count'].astype(int)
    
    return df
