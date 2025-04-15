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
        
        # Convert pairs to Kraken format (remove / and lowercase)
        self.kraken_pairs = [p.replace("/", "").lower() for p in self.pairs]
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle ticker data
            if isinstance(data, list) and len(data) >= 2 and data[2] == 'ticker':
                pair = data[3]
                ticker_data = data[1]
                
                # Extract close price (current price)
                if 'c' in ticker_data and len(ticker_data['c']) > 0:
                    price = float(ticker_data['c'][0])
                    
                    # Find the original pair format
                    original_pair = None
                    for p in self.pairs:
                        if p.replace("/", "").lower() == pair.lower():
                            original_pair = p
                            break
                    
                    if original_pair:
                        # Update last price
                        self.last_prices[original_pair] = price
                        logger.debug(f"WebSocket price update for {original_pair}: ${price}")
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(original_pair, price)
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Try to reconnect after a delay if still running
        if self.running:
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self.connect()
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        logger.info("WebSocket connection established")
        self.connected = True
        
        # Subscribe to ticker for all pairs
        self._subscribe()
    
    def _subscribe(self):
        """Subscribe to ticker data for configured pairs"""
        if not self.connected or not self.ws:
            return
        
        # Create subscription message
        subscription = {
            "name": "ticker"
        }
        
        # Subscribe to each pair
        for pair in self.kraken_pairs:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": {
                    "name": "ticker",
                    "token": "invalid"  # Public API doesn't require token
                },
                "req_id": int(time.time())
            }
            
            try:
                self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {pair} ticker")
            except Exception as e:
                logger.error(f"Error subscribing to {pair}: {e}")
    
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