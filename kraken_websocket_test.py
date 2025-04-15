#!/usr/bin/env python3
"""
Test Kraken WebSocket Connection

This script tests the connection to Kraken's WebSocket API
"""
import sys
import time
import logging
import asyncio
import websocket
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Kraken WebSocket URL
KRAKEN_WS_URL = "wss://ws.kraken.com"

async def test_websocket_connection(timeout=10):
    """
    Test connection to Kraken WebSocket API
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Create connection
        ws = websocket.create_connection(KRAKEN_WS_URL, timeout=timeout)
        
        # Subscribe to heartbeat to test the connection
        subscribe_msg = {
            "name": "subscribe",
            "reqid": 1,
            "pair": ["XBT/USD"],
            "subscription": {
                "name": "ticker"
            }
        }
        
        ws.send(json.dumps(subscribe_msg))
        
        # Wait for response
        result = ws.recv()
        logger.info(f"Received response: {result[:100]}...")
        
        # Close connection
        ws.close()
        
        return True
    
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        return False

def test_connection():
    """Run the async test in a synchronous context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_websocket_connection())
    finally:
        loop.close()

if __name__ == "__main__":
    logger.info("Testing connection to Kraken WebSocket API...")
    
    if test_connection():
        logger.info("✓ Connection successful")
        sys.exit(0)
    else:
        logger.error("× Connection failed")
        sys.exit(1)