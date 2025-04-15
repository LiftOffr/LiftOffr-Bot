#!/usr/bin/env python3
"""
Kraken API Client

This module provides functions to interact with the Kraken API for
real-time price data and trading using both REST and WebSocket APIs.
"""
import os
import json
import time
import base64
import hashlib
import hmac
import urllib.parse
import logging
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Try to import websockets, but handle the case when it's not available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("Warning: websockets module not available, using sandbox mode only")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API Configuration
KRAKEN_API_URL = "https://api.kraken.com"
KRAKEN_WEBSOCKET_URL = "wss://ws.kraken.com"
KRAKEN_API_VERSION = "0"
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

# Mapping between common pair names and Kraken-specific asset IDs
PAIR_MAPPING = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
    "ADA/USD": "ADAUSD",
    "DOT/USD": "DOTUSD",
    "LINK/USD": "LINKUSD",
    "AVAX/USD": "AVAXUSD",
    "MATIC/USD": "MATICUSD",
    "UNI/USD": "UNIUSD",
    "ATOM/USD": "ATOMUSD"
}

# Reverse mapping for websocket feed
REVERSE_PAIR_MAPPING = {v: k for k, v in PAIR_MAPPING.items()}

class KrakenAPIError(Exception):
    """Exception raised for Kraken API errors"""
    pass

def get_kraken_signature(urlpath: str, data: Dict[str, Any], secret: str) -> str:
    """
    Generate Kraken API signature
    
    Args:
        urlpath: API endpoint path
        data: Request data
        secret: API secret key
        
    Returns:
        API request signature
    """
    # Convert data dictionary to POST data string
    post_data = urllib.parse.urlencode(data)
    
    # Create signature
    encoded = (str(data.get('nonce', '')) + post_data).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    signature = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sig_digest = base64.b64encode(signature.digest()).decode()
    
    return sig_digest

def make_request(endpoint: str, data: Dict[str, Any] = None, 
                private: bool = False) -> Dict[str, Any]:
    """
    Make a request to the Kraken API
    
    Args:
        endpoint: API endpoint (e.g., '/0/public/Ticker')
        data: Request data
        private: Whether this is a private API endpoint requiring authentication
        
    Returns:
        API response data
    """
    if data is None:
        data = {}
    
    # Add nonce to private requests
    if private:
        data['nonce'] = int(time.time() * 1000)
    
    # Construct URL path
    urlpath = f'/{KRAKEN_API_VERSION}/{("private" if private else "public")}/{endpoint}'
    url = f'{KRAKEN_API_URL}{urlpath}'
    
    try:
        if private:
            if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
                logger.error("Kraken API credentials not found")
                raise KrakenAPIError("API credentials not configured")
            
            headers = {
                'API-Key': KRAKEN_API_KEY,
                'API-Sign': get_kraken_signature(urlpath, data, KRAKEN_API_SECRET)
            }
            response = requests.post(url, headers=headers, data=data)
        else:
            response = requests.post(url, data=data) if data else requests.get(url)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Check for API errors
        if result.get('error') and len(result['error']) > 0:
            error_msg = ", ".join(result['error'])
            logger.error(f"Kraken API error: {error_msg}")
            raise KrakenAPIError(error_msg)
        
        return result.get('result', {})
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise KrakenAPIError(f"Request failed: {str(e)}")
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise KrakenAPIError(f"Invalid response format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise KrakenAPIError(f"Unexpected error: {str(e)}")

def get_server_time() -> Dict[str, Any]:
    """Get Kraken server time"""
    return make_request('Time')

def get_asset_info(assets: List[str] = None) -> Dict[str, Any]:
    """Get asset information"""
    data = {}
    if assets:
        data['asset'] = ','.join(assets)
    return make_request('Assets', data)

def get_tradable_pairs(pairs: List[str] = None) -> Dict[str, Any]:
    """Get tradable asset pairs"""
    data = {}
    if pairs:
        kraken_pairs = [PAIR_MAPPING.get(pair, pair) for pair in pairs]
        data['pair'] = ','.join(kraken_pairs)
    return make_request('AssetPairs', data)

def get_ticker_info(pairs: List[str]) -> Dict[str, Any]:
    """
    Get ticker information for specified pairs
    
    Args:
        pairs: List of trading pairs (e.g., ["BTC/USD", "ETH/USD"])
        
    Returns:
        Dictionary of ticker data keyed by pair name
    """
    kraken_pairs = [PAIR_MAPPING.get(pair, pair) for pair in pairs]
    data = {'pair': ','.join(kraken_pairs)}
    result = make_request('Ticker', data)
    
    # Convert Kraken pair names back to standard format
    normalized_result = {}
    for k, v in result.items():
        # Find the original pair name
        original_pair = None
        for pair, kraken_pair in PAIR_MAPPING.items():
            if k == kraken_pair:
                original_pair = pair
                break
        
        # Use original name if found, otherwise keep Kraken name
        normalized_key = original_pair or k
        normalized_result[normalized_key] = v
    
    return normalized_result

def get_ohlc_data(pair: str, interval: int = 60, since: int = None) -> Dict[str, Any]:
    """
    Get OHLC (candlestick) data
    
    Args:
        pair: Trading pair
        interval: Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        since: Return data since given ID
        
    Returns:
        OHLC data
    """
    kraken_pair = PAIR_MAPPING.get(pair, pair)
    data = {
        'pair': kraken_pair,
        'interval': interval
    }
    if since:
        data['since'] = since
    
    result = make_request('OHLC', data)
    return result

def get_order_book(pair: str, count: int = 100) -> Dict[str, Any]:
    """
    Get order book data
    
    Args:
        pair: Trading pair
        count: Maximum number of asks/bids
        
    Returns:
        Order book data
    """
    kraken_pair = PAIR_MAPPING.get(pair, pair)
    data = {
        'pair': kraken_pair,
        'count': count
    }
    result = make_request('Depth', data)
    return result

def get_recent_trades(pair: str, since: int = None) -> Dict[str, Any]:
    """
    Get recent trades
    
    Args:
        pair: Trading pair
        since: Return trades since given ID
        
    Returns:
        Recent trades data
    """
    kraken_pair = PAIR_MAPPING.get(pair, pair)
    data = {'pair': kraken_pair}
    if since:
        data['since'] = since
    
    result = make_request('Trades', data)
    return result

def get_recent_spreads(pair: str, since: int = None) -> Dict[str, Any]:
    """
    Get recent spreads
    
    Args:
        pair: Trading pair
        since: Return spreads since given ID
        
    Returns:
        Recent spreads data
    """
    kraken_pair = PAIR_MAPPING.get(pair, pair)
    data = {'pair': kraken_pair}
    if since:
        data['since'] = since
    
    result = make_request('Spread', data)
    return result

def get_account_balance() -> Dict[str, Any]:
    """
    Get account balance (private API)
    
    Returns:
        Account balance data
    """
    return make_request('Balance', private=True)

def get_trade_balance(asset: str = 'ZUSD') -> Dict[str, Any]:
    """
    Get trade balance (private API)
    
    Args:
        asset: Base asset for balance
        
    Returns:
        Trade balance information
    """
    data = {'asset': asset}
    return make_request('TradeBalance', data, private=True)

def get_open_orders(trades: bool = False, userref: int = None) -> Dict[str, Any]:
    """
    Get open orders (private API)
    
    Args:
        trades: Include trades
        userref: Restrict to orders with specified user reference id
        
    Returns:
        Open orders information
    """
    data = {}
    if trades:
        data['trades'] = trades
    if userref:
        data['userref'] = userref
    
    return make_request('OpenOrders', data, private=True)

def get_closed_orders(trades: bool = False, userref: int = None, 
                     start: int = None, end: int = None, 
                     ofs: int = None, closetime: str = 'both') -> Dict[str, Any]:
    """
    Get closed orders (private API)
    
    Args:
        trades: Include trades
        userref: Restrict to orders with specified user reference id
        start: Starting unix timestamp or order tx ID
        end: Ending unix timestamp or order tx ID
        ofs: Result offset for pagination
        closetime: Which time to use ('open', 'close', or 'both')
        
    Returns:
        Closed orders information
    """
    data = {'closetime': closetime}
    if trades:
        data['trades'] = trades
    if userref:
        data['userref'] = userref
    if start:
        data['start'] = start
    if end:
        data['end'] = end
    if ofs:
        data['ofs'] = ofs
    
    return make_request('ClosedOrders', data, private=True)

def place_order(pair: str, type: str, ordertype: str, volume: str, 
               price: str = None, price2: str = None, leverage: str = None, 
               oflags: str = None, starttm: int = None, expiretm: int = None, 
               userref: int = None, validate: bool = False, 
               close_ordertype: str = None, close_price: str = None, 
               close_price2: str = None, trading_agreement: str = 'agree') -> Dict[str, Any]:
    """
    Place order (private API)
    
    Args:
        pair: Asset pair
        type: Type of order (buy/sell)
        ordertype: Order type (market/limit/stop-loss/etc.)
        volume: Order volume in lots
        price: Price (dependent on ordertype)
        price2: Secondary price (dependent on ordertype)
        leverage: Amount of leverage
        oflags: Order flags
        starttm: Scheduled start time
        expiretm: Expiration time
        userref: User reference id
        validate: Validate inputs only, don't place order
        close_ordertype: Close order type
        close_price: Close order price
        close_price2: Close order secondary price
        trading_agreement: Agreement confirmation
        
    Returns:
        Order placement result
    """
    kraken_pair = PAIR_MAPPING.get(pair, pair)
    
    data = {
        'pair': kraken_pair,
        'type': type,
        'ordertype': ordertype,
        'volume': volume,
        'trading_agreement': trading_agreement
    }
    
    if price:
        data['price'] = price
    if price2:
        data['price2'] = price2
    if leverage:
        data['leverage'] = leverage
    if oflags:
        data['oflags'] = oflags
    if starttm:
        data['starttm'] = starttm
    if expiretm:
        data['expiretm'] = expiretm
    if userref:
        data['userref'] = userref
    if validate:
        data['validate'] = validate
    
    # Close order details
    if close_ordertype:
        data['close[ordertype]'] = close_ordertype
    if close_price:
        data['close[price]'] = close_price
    if close_price2:
        data['close[price2]'] = close_price2
    
    return make_request('AddOrder', data, private=True)

def cancel_order(txid: str) -> Dict[str, Any]:
    """
    Cancel open order (private API)
    
    Args:
        txid: Transaction ID
        
    Returns:
        Cancellation result
    """
    data = {'txid': txid}
    return make_request('CancelOrder', data, private=True)

class KrakenWebSocketClient:
    """
    Kraken WebSocket Client for real-time data streaming
    
    This client connects to Kraken's WebSocket API to receive real-time
    market data and trading updates.
    """
    def __init__(self, sandbox: bool = True):
        """
        Initialize WebSocket client
        
        Args:
            sandbox: Use sandbox environment (default: True)
        """
        self.ws = None
        self.running = False
        self.callbacks = {}
        self.sandbox = sandbox
        
        # If in sandbox mode, use fake data to avoid hitting API limits
        self.sandbox_data = {
            "BTC/USD": {"price": 29735.50, "bid": 29734.25, "ask": 29736.75, "volume": 123.45},
            "ETH/USD": {"price": 1865.25, "bid": 1864.75, "ask": 1865.75, "volume": 567.89},
            "SOL/USD": {"price": 22.68, "bid": 22.67, "ask": 22.69, "volume": 12345.67},
            "ADA/USD": {"price": 0.381, "bid": 0.380, "ask": 0.382, "volume": 234567.89},
            "DOT/USD": {"price": 5.42, "bid": 5.41, "ask": 5.43, "volume": 34567.89},
            "LINK/USD": {"price": 6.51, "bid": 6.50, "ask": 6.52, "volume": 12345.67},
            "AVAX/USD": {"price": 14.28, "bid": 14.27, "ask": 14.29, "volume": 7890.12},
            "MATIC/USD": {"price": 0.665, "bid": 0.664, "ask": 0.666, "volume": 123456.78},
            "UNI/USD": {"price": 4.75, "bid": 4.74, "ask": 4.76, "volume": 23456.78},
            "ATOM/USD": {"price": 8.92, "bid": 8.91, "ask": 8.93, "volume": 9876.54}
        }
    
    async def _sandbox_ticker_updates(self):
        """
        Generate sandbox ticker updates for testing
        
        This method simulates real-time ticker updates when in sandbox mode
        to avoid hitting API rate limits during development and testing.
        """
        while self.running:
            try:
                for pair in self.callbacks.keys():
                    if pair in self.sandbox_data:
                        # Simulate small price movements (±0.5%)
                        data = self.sandbox_data[pair]
                        price_change_pct = (2 * (0.5 / 100) * (0.5 - asyncio.get_event_loop().time() % 1))
                        
                        # Update price with simulated movement
                        data["price"] *= (1 + price_change_pct)
                        data["bid"] = data["price"] * 0.9995
                        data["ask"] = data["price"] * 1.0005
                        
                        # Create ticker update message similar to Kraken format
                        ticker_update = {
                            "pair": pair,
                            "time": datetime.now().timestamp(),
                            "price": data["price"],
                            "bid": data["bid"],
                            "ask": data["ask"],
                            "volume": data["volume"] * (1 + 0.1 * (asyncio.get_event_loop().time() % 1 - 0.5))
                        }
                        
                        # Call registered callbacks with this data
                        for callback in self.callbacks.get(pair, []):
                            callback(ticker_update)
                
                # Simulate update every 1 second
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in sandbox ticker simulation: {e}")
                await asyncio.sleep(5)
    
    async def connect(self):
        """Connect to WebSocket API"""
        if self.sandbox or not WEBSOCKETS_AVAILABLE:
            logger.info("Starting simulated WebSocket connection in sandbox mode")
            self.running = True
            asyncio.create_task(self._sandbox_ticker_updates())
            return
        
        try:
            logger.info(f"Connecting to Kraken WebSocket API: {KRAKEN_WEBSOCKET_URL}")
            self.ws = await websockets.connect(KRAKEN_WEBSOCKET_URL)
            self.running = True
            logger.info("Connected to Kraken WebSocket API")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raiseet connection in sandbox mode")
            self.running = True
            asyncio.create_task(self._sandbox_ticker_updates())
            return
        
        try:
            logger.info(f"Connecting to Kraken WebSocket API: {KRAKEN_WEBSOCKET_URL}")
            self.ws = await websockets.connect(KRAKEN_WEBSOCKET_URL)
            self.running = True
            logger.info("Connected to Kraken WebSocket API")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket API"""
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
            logger.info("Disconnected from Kraken WebSocket API")
    
    async def send(self, message: Dict[str, Any]):
        """
        Send message to WebSocket
        
        Args:
            message: Message to send
        """
        if not self.ws and not self.sandbox:
            raise KrakenAPIError("WebSocket not connected")
        
        if self.sandbox:
            # Log but don't actually send in sandbox mode
            logger.debug(f"Sandbox mode: Would send message: {message}")
            return
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            raise KrakenAPIError(f"WebSocket send error: {e}")
    
    async def receive(self):
        """
        Receive message from WebSocket
        
        Returns:
            Parsed message
        """
        if not self.ws and not self.sandbox:
            raise KrakenAPIError("WebSocket not connected")
        
        if self.sandbox:
            # In sandbox mode, we don't receive actual messages
            await asyncio.sleep(1)
            return None
        
        try:
            message = await self.ws.recv()
            return json.loads(message)
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            raise KrakenAPIError(f"WebSocket receive error: {e}")
    
    def register_ticker_callback(self, pair: str, callback: Callable):
        """
        Register callback for ticker updates
        
        Args:
            pair: Trading pair
            callback: Callback function to be called with ticker updates
        """
        if pair not in self.callbacks:
            self.callbacks[pair] = []
        
        self.callbacks[pair].append(callback)
        logger.debug(f"Registered ticker callback for {pair}")
    
    async def subscribe_ticker(self, pairs: List[str]):
        """
        Subscribe to ticker updates
        
        Args:
            pairs: List of trading pairs
        """
        if self.sandbox:
            logger.info(f"Sandbox mode: Subscribed to ticker for {pairs}")
            return
        
        kraken_pairs = [PAIR_MAPPING.get(pair, pair) for pair in pairs]
        
        subscription = {
            "name": "subscribe",
            "reqid": int(time.time()),
            "pair": kraken_pairs,
            "subscription": {
                "name": "ticker"
            }
        }
        
        await self.send(subscription)
        logger.info(f"Subscribed to ticker for {pairs}")
    
    async def process_messages(self):
        """
        Process incoming WebSocket messages
        
        This method runs in a loop to process incoming messages and
        dispatch them to the appropriate callbacks.
        """
        if self.sandbox:
            # In sandbox mode, updates are generated by _sandbox_ticker_updates
            while self.running:
                await asyncio.sleep(1)
            return
        
        while self.running:
            try:
                message = await self.receive()
                
                # Process ticker updates
                if isinstance(message, list) and len(message) >= 2:
                    channel_data = message[1]
                    pair_name = message[-1]
                    
                    # Convert Kraken pair name to our standard format
                    standard_pair = REVERSE_PAIR_MAPPING.get(pair_name, pair_name)
                    
                    # Check if this is a ticker update
                    if message[2] == "ticker":
                        # Process ticker data
                        ticker_data = {
                            "pair": standard_pair,
                            "time": time.time(),
                            "bid": float(channel_data.get("b", [0])[0]),
                            "ask": float(channel_data.get("a", [0])[0]),
                            "last": float(channel_data.get("c", [0])[0]),
                            "volume": float(channel_data.get("v", [0])[1]),
                            "vwap": float(channel_data.get("p", [0])[1]),
                            "low": float(channel_data.get("l", [0])[1]),
                            "high": float(channel_data.get("h", [0])[1])
                        }
                        
                        # Call registered callbacks
                        for callback in self.callbacks.get(standard_pair, []):
                            callback(ticker_data)
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting to reconnect...")
                try:
                    await self.connect()
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")
                    await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await asyncio.sleep(1)

async def get_real_time_prices(pairs: List[str], duration: int = 60, 
                              callback: Callable = None) -> Dict[str, float]:
    """
    Get real-time prices via WebSocket
    
    Args:
        pairs: List of trading pairs
        duration: Duration in seconds to collect data
        callback: Optional callback function to process updates in real-time
        
    Returns:
        Dictionary of latest prices by pair
    """
    prices = {pair: None for pair in pairs}
    
    # Define update handler
    def update_handler(ticker_data):
        pair = ticker_data.get("pair")
        price = ticker_data.get("price")
        if pair and price:
            prices[pair] = price
            if callback:
                callback(pair, price, ticker_data)
    
    # Create and connect WebSocket client
    client = KrakenWebSocketClient(sandbox=not (KRAKEN_API_KEY and KRAKEN_API_SECRET))
    
    try:
        await client.connect()
        
        # Register callbacks
        for pair in pairs:
            client.register_ticker_callback(pair, update_handler)
        
        # Subscribe to ticker updates
        await client.subscribe_ticker(pairs)
        
        # Process messages for specified duration
        end_time = time.time() + duration
        while time.time() < end_time:
            await asyncio.sleep(1)
        
        await client.disconnect()
        return prices
    
    except Exception as e:
        logger.error(f"Error getting real-time prices: {e}")
        if client:
            await client.disconnect()
        return prices

def get_current_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Get current prices for specified pairs
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary of current prices by pair
    """
    try:
        # Check if API keys are available
        if KRAKEN_API_KEY and KRAKEN_API_SECRET:
            # Use REST API to get ticker info
            ticker_info = get_ticker_info(pairs)
            
            # Extract last trade price for each pair
            prices = {}
            for pair, data in ticker_info.items():
                if 'c' in data and len(data['c']) >= 1:
                    prices[pair] = float(data['c'][0])
                else:
                    logger.warning(f"No price data available for {pair}")
            
            return prices
        else:
            # Use sandbox prices
            logger.info("Using sandbox prices as API keys not available")
            sandbox_prices = {
                "BTC/USD": 29735.50,
                "ETH/USD": 1865.25,
                "SOL/USD": 22.68,
                "ADA/USD": 0.381,
                "DOT/USD": 5.42,
                "LINK/USD": 6.51,
                "AVAX/USD": 14.28,
                "MATIC/USD": 0.665,
                "UNI/USD": 4.75,
                "ATOM/USD": 8.92
            }
            
            # Return only requested pairs
            return {pair: sandbox_prices.get(pair, 100.0) for pair in pairs}
    
    except Exception as e:
        logger.error(f"Error fetching current prices: {e}")
        # Return None for each pair to indicate error
        return {pair: None for pair in pairs}

# Asynchronous function to continuously monitor prices
async def monitor_prices(pairs: List[str], callback: Callable, interval: int = 1):
    """
    Continuously monitor prices for specified pairs
    
    Args:
        pairs: List of trading pairs
        callback: Callback function to process price updates
        interval: Update interval in seconds
    """
    client = KrakenWebSocketClient(sandbox=not (KRAKEN_API_KEY and KRAKEN_API_SECRET))
    
    try:
        await client.connect()
        
        # Register callbacks
        for pair in pairs:
            client.register_ticker_callback(pair, callback)
        
        # Subscribe to ticker updates
        await client.subscribe_ticker(pairs)
        
        # Process messages until stopped
        await client.process_messages()
    
    except Exception as e:
        logger.error(f"Error monitoring prices: {e}")
    
    finally:
        await client.disconnect()

# Helper function to run the asyncio event loop for price monitoring
def start_price_monitoring(pairs: List[str], callback: Callable, interval: int = 1):
    """
    Start price monitoring in a separate thread
    
    Args:
        pairs: List of trading pairs
        callback: Callback function to process price updates
        interval: Update interval in seconds
    """
    async def run():
        await monitor_prices(pairs, callback, interval)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run())
    except Exception as e:
        logger.error(f"Error in price monitoring thread: {e}")

if __name__ == "__main__":
    # Simple test of the API functionality
    try:
        # Test REST API
        print("Testing REST API...")
        server_time = get_server_time()
        print(f"Server time: {server_time}")
        
        # Get current prices
        pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
        prices = get_current_prices(pairs)
        print(f"Current prices: {prices}")
        
        # Test WebSocket API
        print("\nTesting WebSocket API...")
        
        async def websocket_test():
            def print_update(ticker_data):
                pair = ticker_data.get("pair")
                price = ticker_data.get("price")
                print(f"Update for {pair}: price={price}")
            
            prices = await get_real_time_prices(pairs, duration=10, callback=print_update)
            print(f"Final prices: {prices}")
        
        asyncio.run(websocket_test())
        
        print("\nAPI tests completed successfully")
    
    except Exception as e:
        print(f"Error testing API: {e}")