#!/usr/bin/env python3
"""
Kraken WebSocket Client for Real-time Market Data

This module implements a WebSocket client for the Kraken exchange,
providing real-time market data updates for trading pairs.
"""
import json
import logging
import threading
import time
from datetime import datetime
import websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class KrakenWebSocketClient:
    """
    Kraken WebSocket Client for real-time market data.
    Connects to Kraken's public WebSocket API to receive ticker updates.
    """
    def __init__(self, pairs=None, callback=None):
        """
        Initialize the WebSocket client.
        
        Args:
            pairs: List of trading pairs to subscribe to (e.g., ["XBT/USD", "ETH/USD"])
            callback: Function to call with ticker updates
        """
        self.ws = None
        self.pairs = pairs or []
        self.callback = callback
        self.running = False
        self.connected = False
        self.last_prices = {}
        self.thread = None
        self.last_message_time = 0
        self.watchdog_thread = None
        
        # Convert pairs to Kraken format (remove / and lowercase)
        self.kraken_pairs = [p.replace("/", "").lower() for p in self.pairs]
        
        # Start the connection watchdog
        self._start_watchdog()
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle heartbeat message
            if isinstance(data, dict) and data.get("event") == "heartbeat":
                logger.debug("Received heartbeat")
                return
                
            # Handle system status
            if isinstance(data, dict) and data.get("event") == "systemStatus":
                status = data.get("status")
                logger.info(f"Kraken WebSocket status: {status}")
                return
                
            # Handle subscription status
            if isinstance(data, dict) and data.get("event") == "subscriptionStatus":
                status = data.get("status")
                pair = data.get("pair")
                subscription = data.get("subscription", {}).get("name")
                if status == "subscribed":
                    logger.info(f"Successfully subscribed to {subscription} for {pair}")
                else:
                    logger.warning(f"Subscription status for {pair}: {status}")
                return
            
            # Handle ticker data (main price updates)
            if isinstance(data, list) and len(data) >= 2:
                # Kraken format is [channelID, data, channelName, pair]
                if len(data) >= 4 and data[2] == "ticker":
                    pair = data[3]  # This is the pair name
                    ticker_data = data[1]  # This contains the ticker data
                    
                    # Extract close price (current price)
                    if 'c' in ticker_data and len(ticker_data['c']) > 0:
                        price = float(ticker_data['c'][0])
                        
                        # Standardize pair format (XBT/USD -> BTC/USD)
                        standard_pair = pair
                        if pair.startswith("XBT"):
                            standard_pair = "BTC" + pair[3:]
                        
                        # Update last price
                        self.last_prices[standard_pair] = price
                        logger.debug(f"Price update for {standard_pair}: ${price}")
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(standard_pair, price)
                            
                        # Log connection status periodically on price updates
                        if standard_pair.endswith("/USD") and int(time.time()) % 60 == 0:
                            logger.info(f"WebSocket connection active, prices updating for {standard_pair}: ${price}")
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}, raw message: {message[:100]}...")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Try to reconnect after a delay if still running
        if self.running:
            delay = 5
            logger.info(f"Attempting to reconnect in {delay} seconds...")
            
            # Don't block the callback thread with sleep
            reconnect_thread = threading.Thread(target=self._delayed_reconnect, args=(delay,))
            reconnect_thread.daemon = True
            reconnect_thread.start()
            
    def _delayed_reconnect(self, delay):
        """Reconnect after a delay"""
        try:
            time.sleep(delay)
            logger.info("Reconnecting WebSocket...")
            self.connect()
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            # Exponential backoff for the next retry
            next_delay = min(delay * 2, 60)  # Maximum 60 second delay
            logger.info(f"Will retry again in {next_delay} seconds...")
            self._delayed_reconnect(next_delay)
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.connected = True
        
        # Subscribe to ticker for all pairs
        self._subscribe()
    
    def _subscribe(self):
        """Subscribe to ticker data for configured pairs"""
        if not self.connected or not self.ws:
            logger.warning("Cannot subscribe - WebSocket not connected")
            return
        
        try:
            # Kraken expects the subscription format as:
            # {"name": "subscribe", "reqid": ID, "pair": ["XBT/USD",...], "subscription": {"name": "ticker"}}
            subscribe_msg = {
                "name": "subscribe",
                "reqid": int(time.time()),
                "pair": self.pairs,  # Use original pair format with slash
                "subscription": {
                    "name": "ticker"
                }
            }
            
            logger.info(f"Sending subscription request for {self.pairs}")
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscription request sent for {len(self.pairs)} pairs")
        except Exception as e:
            logger.error(f"Error subscribing to ticker feed: {e}")
    
    def connect(self):
        """Connect to Kraken WebSocket API"""
        if self.connected:
            return
        
        try:
            # Use Kraken's public WebSocket API endpoint
            self.ws = websocket.WebSocketApp(
                "wss://ws.kraken.com/",
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Set running flag
            self.running = True
            
            # Start WebSocket connection in a separate thread
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self.ws.run_forever)
                self.thread.daemon = True
                self.thread.start()
            
            logger.info("Started WebSocket client thread")
            
            # Wait for connection to establish
            timeout = 10
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                logger.warning("WebSocket connection timed out")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Kraken WebSocket API"""
        self.running = False
        if self.ws:
            self.ws.close()
        self.connected = False
        logger.info("WebSocket client disconnected")
    
    def get_current_price(self, pair):
        """
        Get the current price for a trading pair.
        
        Args:
            pair: Trading pair symbol (e.g., "BTC/USD")
            
        Returns:
            float: Current price if available, None otherwise
        """
        return self.last_prices.get(pair)
    
    def get_all_prices(self):
        """
        Get all current prices.
        
        Returns:
            dict: Dictionary of current prices by pair
        """
        return self.last_prices.copy()


def price_update_callback(pair, price):
    """Example callback for price updates"""
    logger.info(f"Price update: {pair} = ${price}")


if __name__ == "__main__":
    # Example usage
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "LINK/USD"]
    
    # Create WebSocket client
    client = KrakenWebSocketClient(pairs=pairs, callback=price_update_callback)
    
    try:
        # Connect to WebSocket
        if client.connect():
            logger.info("Connected to Kraken WebSocket")
            
            # Keep the script running
            while True:
                time.sleep(10)
                print("Current prices:", client.get_all_prices())
        else:
            logger.error("Failed to connect to Kraken WebSocket")
    
    except KeyboardInterrupt:
        logger.info("WebSocket client stopped by user")
    finally:
        client.disconnect()