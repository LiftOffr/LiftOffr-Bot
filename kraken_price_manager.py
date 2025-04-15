#!/usr/bin/env python3
"""
Kraken Price Manager

This module handles real-time price data from Kraken with fallbacks.
It combines WebSocket for real-time updates with REST API as a fallback.
"""
import time
import logging
import threading
from typing import Dict, List, Optional, Callable

# Import our Kraken API clients
from kraken_api_client import get_current_prices
from kraken_websocket_client import (
    KrakenWebSocketClient, 
    get_latest_price, 
    register_ticker_callback,
    unregister_ticker_callback
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger("kraken_price_manager")

# Store cached prices with fallback
cached_prices = {}
price_callbacks = []

# WebSocket client instance
ws_client = None

# Time when prices were last updated (for each pair)
last_update_times = {}

# Fallback prices in case API is unreachable
fallback_prices = {
    "BTC/USD": 63800.0,
    "ETH/USD": 3040.0,
    "SOL/USD": 148.5,
    "ADA/USD": 0.465,
    "DOT/USD": 6.85,
    "LINK/USD": 16.32,
    "AVAX/USD": 34.25,
    "MATIC/USD": 0.73,
    "UNI/USD": 9.86,
    "ATOM/USD": 8.15
}


def init(pairs: List[str], use_websocket: bool = True) -> bool:
    """
    Initialize the price manager.
    
    Args:
        pairs: List of trading pairs to track
        use_websocket: Whether to use WebSocket for real-time updates
        
    Returns:
        True if initialization was successful, False otherwise
    """
    global ws_client
    
    # Initialize cache with fallback prices
    for pair in pairs:
        cached_prices[pair] = fallback_prices.get(pair, 1000.0)
        last_update_times[pair] = 0
    
    # Try to get initial prices from REST API
    try:
        prices = get_current_prices(pairs)
        for pair, price in prices.items():
            if price is not None:
                cached_prices[pair] = price
                last_update_times[pair] = time.time()
        logger.info(f"Got initial prices for {len(prices)} pairs")
    except Exception as e:
        logger.error(f"Failed to get initial prices: {e}")
        logger.info("Using fallback prices")
    
    # Start WebSocket client if requested
    if use_websocket:
        try:
            # Initialize WebSocket client
            ws_client = KrakenWebSocketClient()
            
            # Register callback for WebSocket updates
            register_ticker_callback(handle_price_update)
            
            # Start WebSocket client in background
            ws_client.start_background_task()
            
            # Subscribe to ticker updates
            ws_client.subscribe_to_ticker(pairs)
            
            logger.info("Started WebSocket client for real-time price updates")
        except Exception as e:
            logger.error(f"Failed to start WebSocket client: {e}")
            logger.info("Will use REST API for price updates")
            return False
    
    # Start background refresh timer if not using WebSocket
    if not use_websocket:
        start_background_refresh(pairs)
    
    return True


def handle_price_update(pair: str, price: float) -> None:
    """
    Handle a price update from the WebSocket.
    
    Args:
        pair: Trading pair
        price: Current price
    """
    if pair in cached_prices:
        cached_prices[pair] = price
        last_update_times[pair] = time.time()
        
        # Call registered callbacks
        for callback in price_callbacks:
            callback(pair, price)


def register_price_callback(callback: Callable[[str, float], None]) -> None:
    """
    Register a callback for price updates.
    
    Args:
        callback: Function to call with (pair, price) when price is updated
    """
    price_callbacks.append(callback)


def unregister_price_callback(callback: Callable[[str, float], None]) -> None:
    """
    Unregister a price callback.
    
    Args:
        callback: Callback function to unregister
    """
    if callback in price_callbacks:
        price_callbacks.remove(callback)


def get_price(pair: str) -> float:
    """
    Get the current price for a trading pair.
    
    Args:
        pair: Trading pair in standard format (e.g., "BTC/USD")
        
    Returns:
        Current price (or cached/fallback price if not available)
    """
    # First check WebSocket price
    if ws_client:
        ws_price = get_latest_price(pair)
        if ws_price is not None:
            return ws_price
    
    # If we haven't updated in a while, try to refresh
    current_time = time.time()
    last_update = last_update_times.get(pair, 0)
    if current_time - last_update > 60:  # More than 60 seconds old
        try:
            prices = get_current_prices([pair])
            if pair in prices and prices[pair] is not None:
                cached_prices[pair] = prices[pair]
                last_update_times[pair] = current_time
        except Exception as e:
            logger.error(f"Failed to refresh price for {pair}: {e}")
    
    # Return cached price or fallback
    return cached_prices.get(pair, fallback_prices.get(pair, 1000.0))


def get_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Get current prices for multiple trading pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary mapping pairs to prices
    """
    result = {}
    for pair in pairs:
        result[pair] = get_price(pair)
    return result


def start_background_refresh(pairs: List[str], interval: int = 30) -> None:
    """
    Start a background thread to periodically refresh prices.
    
    Args:
        pairs: List of trading pairs to refresh
        interval: Refresh interval in seconds
    """
    def refresh_loop():
        while True:
            try:
                prices = get_current_prices(pairs)
                for pair, price in prices.items():
                    if price is not None:
                        old_price = cached_prices.get(pair)
                        cached_prices[pair] = price
                        last_update_times[pair] = time.time()
                        
                        # Only log significant changes to avoid spam
                        if old_price and abs(price - old_price) / old_price > 0.002:  # 0.2% change
                            logger.info(f"Price update: {pair} - ${price:.2f}")
                        
                        # Call registered callbacks
                        for callback in price_callbacks:
                            callback(pair, price)
            except Exception as e:
                logger.error(f"Error in price refresh: {e}")
            
            # Sleep until next refresh
            time.sleep(interval)
    
    refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
    refresh_thread.start()
    logger.info(f"Started background price refresh (every {interval}s)")


def cleanup() -> None:
    """Clean up resources"""
    global ws_client
    
    if ws_client:
        ws_client.stop()
        ws_client = None
    
    logger.info("Cleaned up Kraken price manager")


if __name__ == "__main__":
    # Simple test to demonstrate price manager usage
    def price_callback(pair, price):
        print(f"Price update: {pair} - ${price:.2f}")
    
    # Initialize with some pairs
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    # Register a callback
    register_price_callback(price_callback)
    
    # Initialize the price manager
    init(pairs)
    
    try:
        print("Monitoring prices (press Ctrl+C to stop)...")
        while True:
            # Print current prices every 5 seconds
            time.sleep(5)
            prices = get_prices(pairs)
            for pair, price in prices.items():
                print(f"Current {pair}: ${price:.2f}")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cleanup()