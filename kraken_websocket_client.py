#!/usr/bin/env python3
"""
Kraken WebSocket Client

This module provides functionality to connect to Kraken's WebSocket API
and receive real-time updates on cryptocurrency prices.
"""
import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Callable, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger("kraken_websocket")

# Try to import websockets library, but provide fallback if not available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available - will use REST API fallback")

# WebSocket API endpoint
WEBSOCKET_URI = "wss://ws.kraken.com"

# Store latest ticker data
latest_ticker_data = {}
ticker_callbacks = []


class KrakenWebSocketClient:
    """Kraken WebSocket client for real-time market data"""
    
    def __init__(self):
        """Initialize the WebSocket client"""
        self.ws = None
        self.running = False
        self.subscriptions = []
        self.background_task = None
        
        # Create a shared asyncio event loop
        self.loop = asyncio.new_event_loop()
        
        # Store callbacks for each subscription
        self.callbacks = {}
    
    async def connect(self) -> bool:
        """
        Connect to Kraken WebSocket API.
        
        Returns:
            True if connected successfully, False otherwise
        """
        # Check if websockets is available
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSocket connection not possible - websockets library not available")
            return False
            
        try:
            self.ws = await websockets.connect(WEBSOCKET_URI)
            self.running = True
            logger.info("Connected to Kraken WebSocket API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Kraken WebSocket API"""
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.running = False
        logger.info("Disconnected from Kraken WebSocket API")
    
    async def subscribe(self, subscription: Dict) -> bool:
        """
        Subscribe to a WebSocket feed.
        
        Args:
            subscription: Subscription request
            
        Returns:
            True if subscription was successful, False otherwise
        """
        if not self.ws:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            await self.ws.send(json.dumps(subscription))
            self.subscriptions.append(subscription)
            logger.info(f"Subscribed to {subscription}")
            return True
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False
    
    async def unsubscribe(self, subscription: Dict) -> bool:
        """
        Unsubscribe from a WebSocket feed.
        
        Args:
            subscription: Subscription request to unsubscribe from
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        if not self.ws:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            # Create unsubscribe message
            unsub_msg = {
                "name": "unsubscribe",
                "reqid": int(time.time()),
                "pair": subscription.get("pair", []),
                "subscription": {"name": subscription.get("subscription", {}).get("name")}
            }
            
            await self.ws.send(json.dumps(unsub_msg))
            
            # Remove from subscriptions list
            if subscription in self.subscriptions:
                self.subscriptions.remove(subscription)
            
            logger.info(f"Unsubscribed from {subscription}")
            return True
        except Exception as e:
            logger.error(f"Unsubscription error: {e}")
            return False
    
    async def listen(self) -> None:
        """Listen for WebSocket messages"""
        if not self.ws:
            logger.error("Not connected to WebSocket")
            return
        
        try:
            while self.running:
                try:
                    message = await self.ws.recv()
                    await self.process_message(message)
                except Exception as e:
                    if WEBSOCKETS_AVAILABLE and "ConnectionClosed" in str(e.__class__):
                        logger.warning("WebSocket connection closed")
                        break
                    else:
                        logger.error(f"WebSocket receive error: {e}")
                        break
        except Exception as e:
            logger.error(f"WebSocket listen error: {e}")
        
        # Try to reconnect if still running
        if self.running:
            logger.info("Attempting to reconnect...")
            
            # Wait a bit before reconnecting
            await asyncio.sleep(5)
            
            # Try to reconnect
            connected = await self.connect()
            if connected:
                # Resubscribe to all previous subscriptions
                for subscription in self.subscriptions:
                    await self.subscribe(subscription)
                
                # Continue listening
                await self.listen()
    
    async def process_message(self, message: str) -> None:
        """
        Process a WebSocket message.
        
        Args:
            message: WebSocket message to process
        """
        try:
            data = json.loads(message)
            
            # Check if it's a ticker update
            if isinstance(data, list) and len(data) > 1 and isinstance(data[1], dict) and "c" in data[1]:
                # This is a ticker update
                channel_name = data[2]
                pair = data[3]
                
                if channel_name == "ticker":
                    # Parse the ticker data
                    ticker_data = data[1]
                    
                    # Store latest price (from the 'c' field)
                    try:
                        latest_price = float(ticker_data["c"][0])
                        standard_pair = self._convert_to_standard_pair(pair)
                        
                        # Update latest ticker data
                        latest_ticker_data[standard_pair] = latest_price
                        
                        # Call registered callbacks
                        for callback in ticker_callbacks:
                            callback(standard_pair, latest_price)
                        
                        # Log occasionally (not every tick to avoid flooding)
                        if int(time.time()) % 30 == 0:  # Every 30 seconds
                            logger.debug(f"Ticker update - {standard_pair}: ${latest_price:.2f}")
                    except (KeyError, IndexError, ValueError) as e:
                        logger.error(f"Error parsing ticker data: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _convert_to_standard_pair(self, kraken_pair: str) -> str:
        """
        Convert Kraken pair format to standard format.
        
        Args:
            kraken_pair: Pair in Kraken format (e.g., "XBT/USD")
            
        Returns:
            Pair in standard format (e.g., "BTC/USD")
        """
        # Common mappings for cryptocurrency pairs
        reverse_mappings = {
            "XBT": "BTC",
            "XXBT": "BTC",
            "XETH": "ETH",
        }
        
        # Split the pair
        if "/" in kraken_pair:
            base, quote = kraken_pair.split("/")
        else:
            # Try to extract base and quote without separator
            if len(kraken_pair) <= 6:
                # Simple pairs like BTCUSD
                base = kraken_pair[:-3]
                quote = kraken_pair[-3:]
            else:
                # Complex pairs with X/Z prefixes
                if kraken_pair.startswith("X") and "Z" in kraken_pair:
                    z_pos = kraken_pair.find("Z")
                    base = kraken_pair[1:z_pos]
                    quote = kraken_pair[z_pos+1:]
                else:
                    # Default fallback
                    base = kraken_pair[:-3]
                    quote = kraken_pair[-3:]
        
        # Convert base currency if needed
        if base in reverse_mappings:
            base = reverse_mappings[base]
        
        # Convert quote currency if needed
        if quote in reverse_mappings:
            quote = reverse_mappings[quote]
        
        return f"{base}/{quote}"
    
    def start_background_task(self) -> None:
        """Start listening for WebSocket updates in background thread"""
        if self.background_task and self.background_task.is_alive():
            logger.warning("Background task already running")
            return
        
        def run_event_loop():
            """Run the asyncio event loop in the background thread"""
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._background_task())
        
        self.background_task = threading.Thread(target=run_event_loop, daemon=True)
        self.background_task.start()
        logger.info("Started WebSocket background task")
    
    async def _background_task(self) -> None:
        """Background task to handle WebSocket connection and subscriptions"""
        try:
            # Connect to WebSocket
            connected = await self.connect()
            if not connected:
                logger.error("Failed to connect in background task")
                return
            
            # Listen for updates
            await self.listen()
        except Exception as e:
            logger.error(f"Error in background task: {e}")
        finally:
            await self.disconnect()
    
    def subscribe_to_ticker(self, pairs: List[str]) -> bool:
        """
        Subscribe to ticker updates for the specified pairs.
        
        Args:
            pairs: List of trading pairs (e.g., ["BTC/USD", "ETH/USD"])
            
        Returns:
            True if subscription was initiated, False otherwise
        """
        # Convert standard pairs to Kraken format
        kraken_pairs = [pair.replace("/", "") for pair in pairs]
        
        # Create subscription message
        subscription = {
            "name": "subscribe",
            "reqid": int(time.time()),
            "pair": kraken_pairs,
            "subscription": {"name": "ticker"}
        }
        
        # Schedule the subscription in the event loop
        future = asyncio.run_coroutine_threadsafe(self.subscribe(subscription), self.loop)
        try:
            return future.result(5)  # Wait up to 5 seconds
        except Exception as e:
            logger.error(f"Failed to subscribe to ticker: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the WebSocket client"""
        self.running = False
        
        # Schedule disconnection in the event loop
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.disconnect(), self.loop)
        
        # Wait for background task to complete
        if self.background_task and self.background_task.is_alive():
            self.background_task.join(1)  # Wait up to 1 second
        
        logger.info("Stopped WebSocket client")


def get_latest_price(pair: str) -> Optional[float]:
    """
    Get the latest price for a trading pair.
    
    Args:
        pair: Trading pair in standard format (e.g., "BTC/USD")
        
    Returns:
        Latest price or None if not available
    """
    return latest_ticker_data.get(pair)


def register_ticker_callback(callback: Callable[[str, float], None]) -> None:
    """
    Register a callback for ticker updates.
    
    Args:
        callback: Function to call with (pair, price) when a new price is received
    """
    ticker_callbacks.append(callback)


def unregister_ticker_callback(callback: Callable[[str, float], None]) -> None:
    """
    Unregister a ticker callback.
    
    Args:
        callback: Callback function to unregister
    """
    if callback in ticker_callbacks:
        ticker_callbacks.remove(callback)


if __name__ == "__main__":
    # Simple test to demonstrate WebSocket usage
    def price_update_callback(pair, price):
        print(f"UPDATE: {pair} - ${price:.2f}")
    
    client = KrakenWebSocketClient()
    
    # Register our callback
    register_ticker_callback(price_update_callback)
    
    # Start background task
    client.start_background_task()
    
    # Subscribe to some pairs
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    client.subscribe_to_ticker(pairs)
    
    try:
        print("Listening for updates (press Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        client.stop()