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

def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average
    
    Args:
        data (list or numpy array): List of price data
        period (int): Period for EMA calculation
    
    Returns:
        float: EMA value
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) < period:
        return None
    
    # Convert to pandas Series for easy EMA calculation
    series = pd.Series(data)
    return series.ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range
    
    Args:
        high (list or numpy array): List of high prices
        low (list or numpy array): List of low prices
        close (list or numpy array): List of close prices
        period (int): Period for ATR calculation
    
    Returns:
        float: ATR value
    """
    if isinstance(high, list):
        high = np.array(high)
    if isinstance(low, list):
        low = np.array(low)
    if isinstance(close, list):
        close = np.array(close)
    
    if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
        return None
    
    # Calculate true range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR
    atr = np.mean(tr[-period:])
    
    return atr

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

def calculate_bollinger_bands(data, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data (list or numpy array): List of price data
        period (int): Period for SMA calculation
        num_std (int): Number of standard deviations for bands
    
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) < period:
        return None, None, None
    
    # Middle band is SMA
    middle_band = np.mean(data[-period:])
    
    # Standard deviation for period
    std_dev = np.std(data[-period:])
    
    # Upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_keltner_channels(close, high, low, ema_period=20, atr_period=14, atr_multiplier=2):
    """
    Calculate Keltner Channels
    
    Args:
        close (list or numpy array): List of close prices
        high (list or numpy array): List of high prices
        low (list or numpy array): List of low prices
        ema_period (int): Period for EMA calculation
        atr_period (int): Period for ATR calculation
        atr_multiplier (int): Multiplier for ATR
    
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    # Middle band is EMA of close
    middle_band = calculate_ema(close, ema_period)
    
    # ATR for channel width
    atr = calculate_atr(high, low, close, atr_period)
    
    if middle_band is None or atr is None:
        return None, None, None
    
    # Upper and lower bands
    upper_band = middle_band + (atr * atr_multiplier)
    lower_band = middle_band - (atr * atr_multiplier)
    
    return upper_band, middle_band, lower_band

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

def record_trade(order_type, symbol, quantity, price, profit=None, profit_percent=None, position=None):
    """
    Record trade to log and CSV
    
    Args:
        order_type (str): Type of order (buy/sell)
        symbol (str): Trading symbol
        quantity (float): Order quantity
        price (float): Order execution price
        profit (float, optional): Trade profit/loss
        profit_percent (float, optional): Trade profit/loss as percentage
        position (str, optional): New position after trade (long/short/none)
    """
    import os
    import csv
    from datetime import datetime
    
    # Log the trade
    if profit is not None:
        logger.info(f"{order_type.upper()} {symbol}: {quantity:.6f} @ {price:.2f}, Profit: ${profit:.2f} ({profit_percent:.2f}%)")
    else:
        logger.info(f"{order_type.upper()} {symbol}: {quantity:.6f} @ {price:.2f}")
    
    # Record to CSV
    filename = "trades.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp", "symbol", "order_type", "quantity", "price", 
            "profit", "profit_percent", "position"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "order_type": order_type.upper(),
            "quantity": f"{quantity:.6f}",
            "price": f"{price:.2f}",
            "profit": f"{profit:.2f}" if profit is not None else "",
            "profit_percent": f"{profit_percent:.2f}" if profit_percent is not None else "",
            "position": position if position is not None else ""
        })
