#!/usr/bin/env python3
"""
Kraken Realtime Price Fetcher

This module connects to Kraken's API to fetch real-time price data
for multiple trading pairs. It maintains a websocket connection for
streaming price updates and provides an interface for other components
to access the latest price data.
"""

import os
import json
import time
import hmac
import base64
import hashlib
import logging
import threading
import requests
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
KRAKEN_API_URL = "https://api.kraken.com/0/public"
KRAKEN_WS_URL = "wss://ws.kraken.com"
KRAKEN_KEY = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

# Trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD",
    "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD",
    "UNI/USD", "ATOM/USD"
]

# Standardize pair format for Kraken API (e.g., "SOL/USD" -> "SOL/USD")
def standardize_pair(pair: str) -> str:
    """Convert pair to Kraken websocket format"""
    return pair.replace("/", "/")  # Already in correct format

# Convert pair to Kraken REST API format (e.g., "SOL/USD" -> "SOLUSD")
def rest_api_pair_format(pair: str) -> str:
    """Convert pair to Kraken REST API format"""
    return pair.replace("/", "")


class KrakenRealtimePriceFetcher:
    """Fetches and maintains real-time price data from Kraken"""
    
    def __init__(self, pairs: Optional[List[str]] = None):
        """
        Initialize the price fetcher
        
        Args:
            pairs: List of pairs to fetch prices for (e.g., ["SOL/USD", "BTC/USD"])
        """
        self.pairs = pairs or DEFAULT_PAIRS
        self.latest_prices = {}  # Store latest prices
        self.ws_connected = False
        self.running = False
        self.last_update_time = {}  # Track last update time for each pair
        self.update_callbacks = []  # Callbacks to notify on price updates
        
        # Initialize with REST API prices
        self._initialize_prices()
    
    def _initialize_prices(self):
        """Initialize prices using REST API"""
        for pair in self.pairs:
            try:
                price = self._fetch_ticker_price(pair)
                if price:
                    self.latest_prices[pair] = price
                    self.last_update_time[pair] = datetime.now(timezone.utc)
                    logger.info(f"Initialized {pair} price: {price}")
                else:
                    logger.warning(f"Failed to initialize price for {pair}")
            except Exception as e:
                logger.error(f"Error initializing {pair} price: {e}")
    
    def _fetch_ticker_price(self, pair: str) -> Optional[float]:
        """
        Fetch current ticker price from Kraken REST API
        
        Args:
            pair: Trading pair
            
        Returns:
            Current price or None if fetch failed
        """
        kraken_pair = rest_api_pair_format(pair)
        try:
            url = f"{KRAKEN_API_URL}/Ticker"
            response = requests.get(url, params={"pair": kraken_pair})
            data = response.json()
            
            if response.status_code != 200 or "error" in data and data["error"]:
                logger.error(f"Kraken API error: {data.get('error', response.status_code)}")
                return None
            
            # Extract price from the response (c[0] is the last trade price)
            if "result" in data and kraken_pair in data["result"]:
                price = float(data["result"][kraken_pair]["c"][0])
                logger.debug(f"Fetched {pair} price: {price}")
                return price
            
            logger.warning(f"No price data for {pair}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {pair} price: {e}")
            return None
    
    def start(self):
        """Start the price fetcher"""
        if self.running:
            logger.warning("Price fetcher is already running")
            return
        
        self.running = True
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self._run_websocket_loop)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Start fallback price update thread
        self.fallback_thread = threading.Thread(target=self._fallback_price_updater)
        self.fallback_thread.daemon = True
        self.fallback_thread.start()
        
        logger.info("Started Kraken price fetcher")
    
    def stop(self):
        """Stop the price fetcher"""
        self.running = False
        logger.info("Stopping Kraken price fetcher")
    
    def _fallback_price_updater(self):
        """
        Fallback price updater to ensure we always have recent prices
        even if the WebSocket connection fails
        """
        while self.running:
            if not self.ws_connected:
                logger.info("WebSocket not connected, using REST API for prices")
                for pair in self.pairs:
                    # Check if price is stale (no update in last 30 seconds)
                    last_update = self.last_update_time.get(pair)
                    now = datetime.now(timezone.utc)
                    
                    if not last_update or (now - last_update).total_seconds() > 30:
                        try:
                            price = self._fetch_ticker_price(pair)
                            if price:
                                self.latest_prices[pair] = price
                                self.last_update_time[pair] = now
                                self._notify_price_update(pair, price)
                                logger.info(f"Fallback update for {pair}: {price}")
                        except Exception as e:
                            logger.error(f"Error in fallback update for {pair}: {e}")
            
            # Sleep for 15 seconds before next check
            for _ in range(15):
                if not self.running:
                    break
                time.sleep(1)
    
    def _run_websocket_loop(self):
        """Run the WebSocket connection loop"""
        while self.running:
            try:
                # Create and run the asyncio event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._connect_websocket())
                loop.close()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            # Don't reconnect immediately on failure
            if self.running:
                logger.info("Reconnecting WebSocket in 5 seconds...")
                time.sleep(5)
    
    async def _connect_websocket(self):
        """Connect to Kraken WebSocket and subscribe to ticker channels"""
        try:
            logger.info("Connecting to Kraken WebSocket...")
            self.ws_connected = False
            
            # Format pairs for subscription
            ws_pairs = [standardize_pair(pair) for pair in self.pairs]
            
            # Connect to WebSocket
            async with websockets.connect(KRAKEN_WS_URL) as websocket:
                logger.info("Connected to Kraken WebSocket")
                self.ws_connected = True
                
                # Subscribe to ticker data
                subscribe_msg = {
                    "name": "subscribe",
                    "reqid": 1,
                    "pair": ws_pairs,
                    "subscription": {
                        "name": "ticker"
                    }
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to ticker updates for: {', '.join(ws_pairs)}")
                
                # Process messages
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        await self._process_websocket_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        try:
                            pong = await websocket.ping()
                            await asyncio.wait_for(pong, timeout=5)
                            logger.debug("WebSocket ping successful")
                        except Exception as e:
                            logger.warning(f"WebSocket ping failed: {e}")
                            break
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        if not self.running:
                            break
                
                # Unsubscribe before closing
                if self.running:
                    try:
                        unsubscribe_msg = {
                            "name": "unsubscribe",
                            "reqid": 2,
                            "pair": ws_pairs,
                            "subscription": {
                                "name": "ticker"
                            }
                        }
                        await websocket.send(json.dumps(unsubscribe_msg))
                        logger.info("Unsubscribed from ticker updates")
                    except Exception as e:
                        logger.error(f"Error unsubscribing: {e}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.ws_connected = False
    
    async def _process_websocket_message(self, message: str):
        """
        Process incoming WebSocket message
        
        Args:
            message: JSON message from WebSocket
        """
        try:
            data = json.loads(message)
            
            # Check if this is a ticker update (array with channelID, data, ticker, pair)
            if isinstance(data, list) and len(data) == 4 and data[2] == "ticker":
                pair_data = data[1]
                pair_name = data[3]
                
                # Convert Kraken WebSocket format to our standard format
                standard_pair = pair_name.replace("XBT", "BTC")  # Handle XBT/USD -> BTC/USD
                
                # Extract last trade price (c[0])
                if "c" in pair_data and len(pair_data["c"]) > 0:
                    try:
                        price = float(pair_data["c"][0])
                        self.latest_prices[standard_pair] = price
                        self.last_update_time[standard_pair] = datetime.now(timezone.utc)
                        self._notify_price_update(standard_pair, price)
                        logger.debug(f"WebSocket update for {standard_pair}: {price}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid price data for {standard_pair}: {e}")
            # Subscription status message
            elif isinstance(data, dict) and "event" in data:
                if data["event"] == "subscriptionStatus":
                    status = data.get("status")
                    if status == "subscribed":
                        logger.info(f"Successfully subscribed to {data.get('pair')}")
                    elif status == "error":
                        logger.error(f"Subscription error: {data.get('errorMessage')}")
        except json.JSONDecodeError:
            logger.warning(f"Received invalid JSON: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def get_latest_price(self, pair: str) -> Optional[float]:
        """
        Get the latest price for a pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Latest price or None if not available
        """
        if pair not in self.latest_prices:
            # Try to fetch price on demand if not available
            price = self._fetch_ticker_price(pair)
            if price:
                self.latest_prices[pair] = price
                self.last_update_time[pair] = datetime.now(timezone.utc)
                return price
            return None
        
        return self.latest_prices.get(pair)
    
    def get_all_latest_prices(self) -> Dict[str, float]:
        """
        Get latest prices for all pairs
        
        Returns:
            Dictionary of {pair: price}
        """
        return self.latest_prices.copy()
    
    def register_update_callback(self, callback: Callable[[str, float], None]):
        """
        Register a callback for price updates
        
        Args:
            callback: Function that takes (pair, price) arguments
        """
        if callback not in self.update_callbacks:
            self.update_callbacks.append(callback)
    
    def unregister_update_callback(self, callback: Callable[[str, float], None]):
        """
        Unregister a callback
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def _notify_price_update(self, pair: str, price: float):
        """
        Notify all registered callbacks of a price update
        
        Args:
            pair: Trading pair
            price: Updated price
        """
        for callback in self.update_callbacks:
            try:
                callback(pair, price)
            except Exception as e:
                logger.error(f"Error in price update callback: {e}")


# Usage example
if __name__ == "__main__":
    import signal
    
    # Set up signal handling
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        fetcher.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Define a simple price update callback
    def price_update_callback(pair, price):
        logger.info(f"PRICE UPDATE: {pair} = {price}")
    
    # Create and start the price fetcher
    fetcher = KrakenRealtimePriceFetcher()
    fetcher.register_update_callback(price_update_callback)
    fetcher.start()
    
    logger.info("Press Ctrl+C to exit")
    
    # Keep the main thread running
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    
    fetcher.stop()