#!/usr/bin/env python3
"""
Kraken API Client

This module provides functionality to interact with the Kraken cryptocurrency
exchange REST API for fetching market data and trading information.
"""
import os
import time
import json
import logging
import hmac
import base64
import hashlib
import urllib.parse
from typing import Dict, List, Optional, Union, Any
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger("kraken_api")

# API endpoints
API_URL = "https://api.kraken.com"
API_VERSION = "0"
PUBLIC_ENDPOINT = f"{API_URL}/{API_VERSION}/public"
PRIVATE_ENDPOINT = f"{API_URL}/{API_VERSION}/private"

# Rate limiting
REQUEST_MIN_INTERVAL = 0.5  # Minimum interval between requests in seconds
last_request_time = 0

# Default request timeout
DEFAULT_TIMEOUT = 10  # seconds

# Kraken API key and secret from environment variables
API_KEY = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")


def get_nonce() -> int:
    """Get a unique nonce for API requests"""
    return int(time.time() * 1000)


def get_kraken_signature(urlpath: str, data: Dict, secret: str) -> str:
    """
    Create Kraken API signature.
    
    Args:
        urlpath: API endpoint path
        data: Request data including nonce
        secret: API secret key
        
    Returns:
        API request signature
    """
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    
    signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(signature.digest())
    
    return sigdigest.decode()


def make_request(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict] = None,
    private: bool = False,
    retry_count: int = 3
) -> Dict:
    """
    Make a request to the Kraken API.
    
    Args:
        endpoint: API endpoint
        method: HTTP method (GET or POST)
        params: Request parameters
        private: Whether this is a private API request
        retry_count: Number of retries for failed requests
        
    Returns:
        Response data as dictionary
    """
    global last_request_time
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < REQUEST_MIN_INTERVAL:
        time.sleep(REQUEST_MIN_INTERVAL - time_since_last)
    
    url = PRIVATE_ENDPOINT if private else PUBLIC_ENDPOINT
    url = f"{url}/{endpoint}"
    
    headers = {}
    data = {}
    
    if private:
        if not API_KEY or not API_SECRET:
            logger.error("API key and secret required for private API requests")
            return {"error": ["API credentials not configured"]}
        
        if not params:
            params = {}
        
        params["nonce"] = get_nonce()
        headers = {
            "API-Key": API_KEY,
            "API-Sign": get_kraken_signature(
                f"/{API_VERSION}/private/{endpoint}",
                params,
                API_SECRET
            )
        }
        data = params
    
    for attempt in range(retry_count):
        try:
            if method == "GET":
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT
                )
            else:  # POST
                response = requests.post(
                    url,
                    data=data,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT
                )
            
            last_request_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("error") and result["error"]:
                    logger.error(f"API error: {result['error']}")
                
                return result
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                
                # If we get a 5xx error, retry
                if response.status_code >= 500:
                    logger.info(f"Retrying in {2**attempt} seconds...")
                    time.sleep(2**attempt)
                    continue
                
                return {"error": [f"HTTP error {response.status_code}: {response.text}"]}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {2**attempt} seconds...")
                time.sleep(2**attempt)
            else:
                return {"error": [f"Request failed after {retry_count} attempts: {e}"]}
    
    return {"error": ["Maximum retry attempts reached"]}


def get_server_time() -> Dict:
    """
    Get Kraken server time.
    
    Returns:
        Server time information
    """
    return make_request("Time")


def get_asset_info(assets: Optional[List[str]] = None) -> Dict:
    """
    Get information about assets.
    
    Args:
        assets: List of assets to get info for (or None for all)
        
    Returns:
        Asset information
    """
    params = {}
    if assets:
        params["asset"] = ",".join(assets)
    
    return make_request("Assets", params=params)


def get_asset_pairs(pairs: Optional[List[str]] = None) -> Dict:
    """
    Get information about asset pairs.
    
    Args:
        pairs: List of pairs to get info for (or None for all)
        
    Returns:
        Asset pair information
    """
    params = {}
    if pairs:
        params["pair"] = ",".join(pairs)
    
    return make_request("AssetPairs", params=params)


def get_ticker(pairs: List[str]) -> Dict:
    """
    Get ticker information.
    
    Args:
        pairs: List of pairs to get ticker for
        
    Returns:
        Ticker information
    """
    params = {"pair": ",".join(pairs)}
    return make_request("Ticker", params=params)


def get_current_prices(pairs: List[str]) -> Dict[str, Optional[float]]:
    """
    Get current prices for the specified pairs.
    
    Args:
        pairs: List of trading pairs (e.g., ["BTC/USD", "ETH/USD"])
        
    Returns:
        Dictionary mapping pairs to prices
    """
    # Convert standard pairs to Kraken format
    kraken_pairs = [pair.replace("/", "") for pair in pairs]
    
    # Get ticker data
    result = get_ticker(kraken_pairs)
    
    # Parse the result
    prices = {}
    if "error" in result and result["error"]:
        logger.error(f"Error getting prices: {result['error']}")
        for pair in pairs:
            prices[pair] = None
        return prices
    
    # Extract price data
    if "result" in result:
        data = result["result"]
        
        # Map pairs back to standard format
        pair_mapping = {pair.replace("/", ""): pair for pair in pairs}
        
        for kraken_pair, pair_data in data.items():
            # Get the standard pair format
            # Sometimes Kraken adds X/Z prefixes, so we need to handle that
            base_pair = kraken_pair
            for k, v in pair_mapping.items():
                if k in kraken_pair:
                    base_pair = k
                    break
            
            std_pair = pair_mapping.get(base_pair)
            if std_pair:
                try:
                    # 'c' is the array with [price, lot volume]
                    prices[std_pair] = float(pair_data["c"][0])
                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error parsing price for {std_pair}: {e}")
                    prices[std_pair] = None
        
        # Make sure all requested pairs are in the result
        for pair in pairs:
            if pair not in prices:
                prices[pair] = None
    
    return prices


def get_ohlc(
    pair: str, 
    interval: int = 1,
    since: Optional[int] = None
) -> Dict:
    """
    Get OHLC (candle) data.
    
    Args:
        pair: Asset pair
        interval: Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since: Return data since given ID (optional)
        
    Returns:
        OHLC data
    """
    params = {
        "pair": pair.replace("/", ""),
        "interval": str(interval)
    }
    if since:
        params["since"] = str(since)
    
    return make_request("OHLC", params=params)


def get_order_book(pair: str, count: int = 100) -> Dict:
    """
    Get order book.
    
    Args:
        pair: Asset pair
        count: Maximum number of asks/bids
        
    Returns:
        Order book data
    """
    params = {
        "pair": pair.replace("/", ""),
        "count": str(count)
    }
    
    return make_request("Depth", params=params)


def get_recent_trades(
    pair: str,
    since: Optional[int] = None
) -> Dict:
    """
    Get recent trades.
    
    Args:
        pair: Asset pair
        since: Return data since given ID (optional)
        
    Returns:
        Recent trades data
    """
    params = {"pair": pair.replace("/", "")}
    if since:
        params["since"] = str(since)
    
    return make_request("Trades", params=params)


def get_account_balance() -> Dict:
    """
    Get account balance (private).
    
    Returns:
        Account balance
    """
    return make_request("Balance", method="POST", private=True)


def get_trade_balance(asset: str = "ZUSD") -> Dict:
    """
    Get trade balance (private).
    
    Args:
        asset: Base asset to show results in
        
    Returns:
        Trade balance
    """
    params = {"asset": asset}
    return make_request("TradeBalance", method="POST", params=params, private=True)


def get_open_orders() -> Dict:
    """
    Get open orders (private).
    
    Returns:
        Open orders
    """
    return make_request("OpenOrders", method="POST", private=True)


def get_closed_orders() -> Dict:
    """
    Get closed orders (private).
    
    Returns:
        Closed orders
    """
    return make_request("ClosedOrders", method="POST", private=True)


def query_orders(txid: Union[str, List[str]]) -> Dict:
    """
    Query orders (private).
    
    Args:
        txid: Transaction ID(s)
        
    Returns:
        Order information
    """
    if isinstance(txid, list):
        txid = ",".join(txid)
    
    params = {"txid": txid}
    return make_request("QueryOrders", method="POST", params=params, private=True)


def get_trades_history() -> Dict:
    """
    Get trades history (private).
    
    Returns:
        Trades history
    """
    return make_request("TradesHistory", method="POST", private=True)


def query_trades(txid: Union[str, List[str]]) -> Dict:
    """
    Query trades (private).
    
    Args:
        txid: Transaction ID(s)
        
    Returns:
        Trade information
    """
    if isinstance(txid, list):
        txid = ",".join(txid)
    
    params = {"txid": txid}
    return make_request("QueryTrades", method="POST", params=params, private=True)


def add_order(
    pair: str,
    type: str,  # "buy" or "sell"
    ordertype: str,  # "market", "limit", "stop-loss", etc.
    volume: float,
    price: Optional[float] = None,
    price2: Optional[float] = None,
    leverage: Optional[float] = None,
    oflags: Optional[List[str]] = None,
    starttm: Optional[int] = None,
    expiretm: Optional[int] = None,
    userref: Optional[int] = None,
    validate: bool = False,
    close_order: Optional[Dict] = None
) -> Dict:
    """
    Add a new order (private).
    
    Args:
        pair: Asset pair
        type: Type of order (buy/sell)
        ordertype: Order type
        volume: Order volume
        price: Price (optional depending on ordertype)
        price2: Secondary price (optional)
        leverage: Leverage (optional)
        oflags: Order flags (optional)
        starttm: Scheduled start time (optional)
        expiretm: Expiration time (optional)
        userref: User reference ID (optional)
        validate: Validate only (no actual order) (optional)
        close_order: Conditional close order (optional)
        
    Returns:
        Order information
    """
    params = {
        "pair": pair.replace("/", ""),
        "type": type,
        "ordertype": ordertype,
        "volume": str(volume)
    }
    
    if price:
        params["price"] = str(price)
    
    if price2:
        params["price2"] = str(price2)
    
    if leverage:
        params["leverage"] = str(leverage)
    
    if oflags:
        params["oflags"] = ",".join(oflags)
    
    if starttm:
        params["starttm"] = str(starttm)
    
    if expiretm:
        params["expiretm"] = str(expiretm)
    
    if userref:
        params["userref"] = str(userref)
    
    if validate:
        params["validate"] = "true"
    
    if close_order:
        for k, v in close_order.items():
            params[f"close[{k}]"] = v
    
    return make_request("AddOrder", method="POST", params=params, private=True)


def cancel_order(txid: str) -> Dict:
    """
    Cancel an open order (private).
    
    Args:
        txid: Transaction ID
        
    Returns:
        Cancellation status
    """
    params = {"txid": txid}
    return make_request("CancelOrder", method="POST", params=params, private=True)


def check_api_key_validity() -> bool:
    """
    Check if the API key is valid by making a simple private request.
    
    Returns:
        True if valid, False otherwise
    """
    if not API_KEY or not API_SECRET:
        logger.warning("API key and secret not configured")
        return False
    
    result = make_request("TradeVolume", method="POST", params={"pair": "XBTUSD"}, private=True)
    
    # Check if there's an "Invalid key" error
    if "error" in result:
        for error in result["error"]:
            if "Invalid key" in error:
                logger.error("Invalid API key")
                return False
    
    # If there's an error but not "Invalid key", the key might be valid
    # but some other error occurred
    if "error" in result and result["error"]:
        logger.warning(f"API error but key may be valid: {result['error']}")
        return True
    
    # No error means the key is valid
    logger.info("API key is valid")
    return True


if __name__ == "__main__":
    # Test the API client
    print("\nTesting Kraken API client:")
    
    # Public endpoints
    print("\n1. Server time:")
    result = get_server_time()
    print(json.dumps(result, indent=2))
    
    print("\n2. Current prices:")
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    prices = get_current_prices(pairs)
    for pair, price in prices.items():
        print(f"{pair}: ${price}")
    
    # Check if we have API credentials
    if API_KEY and API_SECRET:
        print("\n3. Checking API key validity:")
        is_valid = check_api_key_validity()
        print(f"API key valid: {is_valid}")
        
        if is_valid:
            print("\n4. Account balance:")
            balance = get_account_balance()
            print(json.dumps(balance, indent=2))
    else:
        print("\nSkipping private API tests - no API credentials configured")