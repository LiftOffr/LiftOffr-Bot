#!/usr/bin/env python3
"""
Kraken API Client

This module handles interactions with the Kraken API,
providing functions to fetch real-time price data.
"""
import os
import json
import time
import hmac
import base64
import urllib.parse
import hashlib
import logging
import requests
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger("kraken_api")

# Kraken API endpoints
BASE_URL = "https://api.kraken.com"
PUBLIC_ENDPOINT = "/0/public/"
PRIVATE_ENDPOINT = "/0/private/"

# Cache for API responses to avoid rate limits
api_cache = {}
CACHE_EXPIRY = 5  # 5 seconds cache


def get_kraken_signature(urlpath: str, data: Dict, secret: str) -> str:
    """
    Generates Kraken API signature for private endpoints.
    
    Args:
        urlpath: API endpoint path
        data: Request data containing API nonce
        secret: API secret key
        
    Returns:
        API request signature
    """
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()


def make_public_request(method: str, params: Optional[Dict] = None) -> Dict:
    """
    Makes a public API request to Kraken.
    
    Args:
        method: API method name
        params: Request parameters
        
    Returns:
        API response as dictionary
    """
    # Check cache first
    cache_key = f"{method}:{json.dumps(params) if params else 'None'}"
    current_time = time.time()
    
    if cache_key in api_cache:
        cached_time, cached_response = api_cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            return cached_response
    
    url = f"{BASE_URL}{PUBLIC_ENDPOINT}{method}"
    logger.debug(f"Making public request to {method}")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        
        if 'error' in result and result['error']:
            logger.error(f"Kraken API error: {result['error']}")
            return {"error": result['error']}
        
        # Cache the response
        api_cache[cache_key] = (current_time, result)
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return {"error": str(e)}


def make_private_request(method: str, params: Optional[Dict] = None) -> Dict:
    """
    Makes a private API request to Kraken.
    
    Args:
        method: API method name
        params: Request parameters
        
    Returns:
        API response as dictionary
    """
    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("API key and secret required for private requests")
        return {"error": "API key and secret required"}
    
    # Set up request data
    url = f"{BASE_URL}{PRIVATE_ENDPOINT}{method}"
    data = params or {}
    data['nonce'] = int(time.time() * 1000)
    
    # Generate API signature
    signature = get_kraken_signature(f"{PRIVATE_ENDPOINT}{method}", data, api_secret)
    headers = {
        'API-Key': api_key,
        'API-Sign': signature
    }
    
    logger.debug(f"Making private request to {method}")
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        result = response.json()
        
        if 'error' in result and result['error']:
            logger.error(f"Kraken API error: {result['error']}")
            return {"error": result['error']}
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return {"error": str(e)}


def get_ticker_info(pairs: Union[str, List[str]]) -> Dict:
    """
    Gets current ticker information for the specified pairs.
    
    Args:
        pairs: Single trading pair or list of pairs (e.g., "XBTUSD" or ["XBTUSD", "ETHUSD"])
        
    Returns:
        Dictionary of ticker information
    """
    # Convert list to comma-separated string
    if isinstance(pairs, list):
        pairs = ",".join(pairs)
    
    return make_public_request("Ticker", {"pair": pairs})


def get_ohlc_data(pair: str, interval: int = 1, since: Optional[int] = None) -> Dict:
    """
    Gets OHLC (candlestick) data for a pair.
    
    Args:
        pair: Trading pair (e.g., "XBTUSD")
        interval: Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since: Return data since given ID
        
    Returns:
        Dictionary of OHLC data
    """
    params = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    
    return make_public_request("OHLC", params)


def get_kraken_pair_name(standard_pair: str) -> str:
    """
    Converts standard pair names to Kraken API format.
    
    Args:
        standard_pair: Pair in standard format (e.g., "BTC/USD")
        
    Returns:
        Pair in Kraken API format (e.g., "XXBTZUSD")
    """
    # Common mappings for cryptocurrency pairs
    mappings = {
        "BTC/USD": "XXBTZUSD",
        "ETH/USD": "XETHZUSD",
        "SOL/USD": "SOLUSD",
        "ADA/USD": "ADAUSD",
        "DOT/USD": "DOTUSD",
        "LINK/USD": "LINKUSD",
        "AVAX/USD": "AVAXUSD",
        "MATIC/USD": "MATICUSD",
        "UNI/USD": "UNIUSD",
        "ATOM/USD": "ATOMUSD",
        
        # We can easily add more mappings here as needed
    }
    
    return mappings.get(standard_pair, standard_pair.replace("/", ""))


def get_current_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Gets current prices for multiple trading pairs.
    
    Args:
        pairs: List of trading pairs in standard format (e.g., ["BTC/USD", "ETH/USD"])
        
    Returns:
        Dictionary mapping pairs to current prices
    """
    # Convert standard pairs to Kraken format
    kraken_pairs = [get_kraken_pair_name(pair) for pair in pairs]
    
    # Make API request
    response = get_ticker_info(kraken_pairs)
    
    # Parse response
    prices = {}
    if 'result' in response:
        for i, pair in enumerate(pairs):
            kraken_pair = kraken_pairs[i]
            if kraken_pair in response['result']:
                # Use the last trade price (c[0])
                try:
                    prices[pair] = float(response['result'][kraken_pair]['c'][0])
                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error parsing price for {pair}: {e}")
                    prices[pair] = None
    
    # Fill in any missing prices with None
    for pair in pairs:
        if pair not in prices:
            prices[pair] = None
    
    return prices


if __name__ == "__main__":
    # Simple test to fetch current prices
    test_pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    print(f"Fetching prices for {test_pairs}")
    prices = get_current_prices(test_pairs)
    for pair, price in prices.items():
        print(f"{pair}: ${price:.2f}" if price else f"{pair}: N/A")