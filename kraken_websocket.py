import json
import time
import logging
import threading
from websocket import WebSocketApp
import hmac
import base64
import hashlib
import urllib.parse
from typing import Dict, List, Callable, Any, Optional, Union
from config import (
    API_KEY, API_SECRET, WEBSOCKET_PUBLIC_URL, 
    WEBSOCKET_PRIVATE_URL, TRADING_PAIR
)
from utils import get_nonce

logger = logging.getLogger(__name__)

class KrakenWebsocket:
    """
    Class to handle Kraken WebSocket connections
    """
    def __init__(self, api_key: str = API_KEY, api_secret: str = API_SECRET):
        """
        Initialize the KrakenWebsocket class
        
        Args:
            api_key (str): Kraken API key
            api_secret (str): Kraken API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.public_ws_url = WEBSOCKET_PUBLIC_URL
        self.private_ws_url = WEBSOCKET_PRIVATE_URL
        
        # WebSocket connections
        self.public_ws = None
        self.private_ws = None
        
        # Connection status
        self.public_connected = False
        self.private_connected = False
        
        # Callbacks
        self.ticker_callbacks = []
        self.ohlc_callbacks = []
        self.trade_callbacks = []
        self.book_callbacks = []
        self.own_trades_callbacks = []
        self.open_orders_callbacks = []
        
        # Message buffer for reconnection
        self.public_buffer = []
        self.private_buffer = []
        
        # Threading
        self.public_thread = None
        self.private_thread = None
        self.reconnect_thread = None
        self.public_ping_thread = None
        self.private_ping_thread = None
        self.reconnect_interval = 5  # seconds
        self.ping_interval = 30  # seconds - send a ping every 30 seconds to keep connection alive
        self.max_reconnect_attempts = 10  # maximum number of consecutive reconnection attempts
        self.reconnect_attempts = 0  # current number of reconnection attempts
        self.reconnect_backoff = 2  # exponential backoff factor
        self.keep_running = True
        
        # Subscription tracking
        self.public_subscriptions = []  # Track subscriptions for automatic resubscription after reconnect
        self.private_subscriptions = []
    
    def _get_auth_token(self) -> Dict:
        """
        Generate authentication token for private WebSocket API
        
        Returns:
            dict: Authentication token
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret required for private WebSocket connection")
        
        nonce = get_nonce()
        token_data = {
            'token': self.api_key,
            'nonce': nonce
        }
        
        # Create signature
        urlpath = "/ws/auth"
        postdata = urllib.parse.urlencode(token_data)
        encoded = (str(nonce) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest())
        
        token_data['signature'] = sigdigest.decode()
        
        return token_data
    
    def _on_public_message(self, ws, message):
        """
        Handle public WebSocket messages
        
        Args:
            ws: WebSocket connection
            message: Message received
        """
        try:
            msg = json.loads(message)
            
            # Handle heartbeat
            if isinstance(msg, dict) and 'event' in msg and msg['event'] == 'heartbeat':
                return
            
            # Handle subscription status
            if isinstance(msg, dict) and 'event' in msg and msg['event'] == 'subscriptionStatus':
                status = msg.get('status', 'unknown')
                name = msg.get('subscription', {}).get('name', 'unknown')
                logger.info(f"Subscription status: {status} for {name}")
                
                # Log more details if there's an error
                if status == 'error':
                    error_msg = msg.get('errorMessage', 'Unknown error')
                    logger.error(f"WebSocket subscription error: {error_msg} for {name}")
                    logger.debug(f"Full error response: {msg}")
                return
            
            # Handle actual data messages (in array format)
            if isinstance(msg, list):
                try:
                    # Log the raw message for debugging
                    logger.debug(f"Received WebSocket data: {str(msg)[:200]}...")
                    
                    # Determine message type from the subscription details
                    if len(msg) >= 4 and isinstance(msg[2], str):
                        channel_name = msg[2]
                        pair = msg[3]
                        data = msg[1]
                        
                        logger.debug(f"Processing {channel_name} data for {pair}")
                        
                        if channel_name == 'ticker':
                            logger.debug(f"Ticker data: {str(data)[:100]}...")
                            for callback in self.ticker_callbacks:
                                try:
                                    callback(pair, data)
                                except Exception as e:
                                    logger.error(f"Error in ticker callback: {e}")
                        
                        elif channel_name == 'ohlc':
                            for callback in self.ohlc_callbacks:
                                callback(pair, data)
                        
                        elif channel_name == 'trade':
                            for callback in self.trade_callbacks:
                                callback(pair, data)
                        
                        elif channel_name == 'book':
                            for callback in self.book_callbacks:
                                callback(pair, data)
                    else:
                        # Alternative format for some Kraken WebSocket messages
                        # For example, some callbacks might be in format [channelID, data, channelName, pair]
                        if len(msg) >= 3 and isinstance(msg[0], int) and isinstance(msg[2], str):
                            channel_id = msg[0]
                            data = msg[1]
                            channel_name = msg[2]
                            pair = msg[3] if len(msg) > 3 else None
                            
                            logger.debug(f"Alternative format: {channel_name} for {pair}")
                            
                            if channel_name == 'ticker':
                                logger.debug(f"Alt format ticker data: {str(data)[:100]}...")
                                for callback in self.ticker_callbacks:
                                    try:
                                        callback(pair, data)
                                    except Exception as e:
                                        logger.error(f"Error in alt ticker callback: {e}")
                            
                            elif 'ohlc' in channel_name:
                                for callback in self.ohlc_callbacks:
                                    callback(pair, data)
                            
                            elif channel_name == 'trade':
                                for callback in self.trade_callbacks:
                                    callback(pair, data)
                            
                            elif 'book' in channel_name:
                                for callback in self.book_callbacks:
                                    callback(pair, data)
                except IndexError:
                    logger.error(f"Unexpected message format: {msg}")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}, Message: {message}")
    
    def _on_private_message(self, ws, message):
        """
        Handle private WebSocket messages
        
        Args:
            ws: WebSocket connection
            message: Message received
        """
        try:
            msg = json.loads(message)
            
            # Handle heartbeat
            if isinstance(msg, dict) and 'event' in msg and msg['event'] == 'heartbeat':
                return
            
            # Handle subscription status
            if isinstance(msg, dict) and 'event' in msg and msg['event'] == 'subscriptionStatus':
                status = msg.get('status', 'unknown')
                name = msg.get('subscription', {}).get('name', 'unknown')
                logger.info(f"Private subscription status: {status} for {name}")
                
                # Log more details if there's an error
                if status == 'error':
                    error_msg = msg.get('errorMessage', 'Unknown error')
                    logger.error(f"Private WebSocket subscription error: {error_msg} for {name}")
                    logger.debug(f"Full error response: {msg}")
                return
            
            # Handle authentication status
            if isinstance(msg, dict) and 'event' in msg and msg['event'] == 'authenticationStatus':
                logger.info(f"Authentication status: {msg['status']}")
                
                # Once authenticated, send the subscription messages from buffer
                if msg['status'] == 'success':
                    for buffered_msg in self.private_buffer:
                        self.private_ws.send(json.dumps(buffered_msg))
                    self.private_buffer = []
                return
            
            # Handle actual data messages (in array format)
            if isinstance(msg, list):
                channel_name = msg[1]
                data = msg[0]
                
                if channel_name == 'ownTrades':
                    for callback in self.own_trades_callbacks:
                        callback(data)
                
                elif channel_name == 'openOrders':
                    for callback in self.open_orders_callbacks:
                        callback(data)
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse private message: {message}")
        except Exception as e:
            logger.error(f"Error handling private message: {e}")
    
    def _on_public_error(self, ws, error):
        """
        Handle public WebSocket errors
        
        Args:
            ws: WebSocket connection
            error: Error
        """
        logger.error(f"Public WebSocket error: {error}")
    
    def _on_private_error(self, ws, error):
        """
        Handle private WebSocket errors
        
        Args:
            ws: WebSocket connection
            error: Error
        """
        logger.error(f"Private WebSocket error: {error}")
    
    def _on_public_close(self, ws, close_status_code, close_msg):
        """
        Handle public WebSocket close
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        logger.info(f"Public WebSocket closed: {close_msg} (Code: {close_status_code})")
        self.public_connected = False
        
        # Start reconnection if still running
        if self.keep_running and not self.reconnect_thread:
            self.reconnect_thread = threading.Thread(target=self._reconnect)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()
    
    def _on_private_close(self, ws, close_status_code, close_msg):
        """
        Handle private WebSocket close
        
        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        logger.info(f"Private WebSocket closed: {close_msg} (Code: {close_status_code})")
        self.private_connected = False
        
        # Start reconnection if still running
        if self.keep_running and not self.reconnect_thread:
            self.reconnect_thread = threading.Thread(target=self._reconnect)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()
    
    def _on_public_open(self, ws):
        """
        Handle public WebSocket open
        
        Args:
            ws: WebSocket connection
        """
        logger.info("Public WebSocket connected")
        self.public_connected = True
        
        # Send subscription messages from buffer
        for msg in self.public_buffer:
            ws.send(json.dumps(msg))
        self.public_buffer = []
    
    def _on_private_open(self, ws):
        """
        Handle private WebSocket open
        
        Args:
            ws: WebSocket connection
        """
        logger.info("Private WebSocket connected")
        self.private_connected = True
        
        # Authenticate
        auth_token = self._get_auth_token()
        ws.send(json.dumps({
            "event": "authenticate",
            "token": auth_token['token'],
            "nonce": auth_token['nonce'],
            "signature": auth_token['signature']
        }))
    
    def _send_ping(self, ws, ws_type):
        """
        Send periodic ping to keep WebSocket connection alive
        
        Args:
            ws: WebSocket connection
            ws_type: Type of WebSocket (public/private)
        """
        while self.keep_running:
            try:
                if ws and ((ws_type == 'public' and self.public_connected) or 
                           (ws_type == 'private' and self.private_connected)):
                    # Send ping message to prevent timeouts
                    ws.send(json.dumps({"event": "ping"}))
                    logger.debug(f"Sent ping to {ws_type} WebSocket")
                else:
                    logger.debug(f"{ws_type} WebSocket not connected, skipping ping")
                    break
            except Exception as e:
                logger.error(f"Error sending ping to {ws_type} WebSocket: {e}")
                break
                
            # Sleep for ping interval
            time.sleep(self.ping_interval)
    
    def _reconnect(self):
        """
        Handle reconnection for WebSockets with exponential backoff
        """
        # Increase reconnect attempts
        self.reconnect_attempts += 1
        
        # Calculate backoff time (with maximum cap at 5 minutes)
        backoff_time = min(300, self.reconnect_interval * (self.reconnect_backoff ** (self.reconnect_attempts - 1)))
        
        logger.info(f"Attempting to reconnect in {backoff_time:.1f} seconds... (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        time.sleep(backoff_time)
        
        try:
            if not self.public_connected and self.public_ws:
                self.connect_public()
                # Reset reconnect attempts on successful connection
                self.reconnect_attempts = 0
            
            if not self.private_connected and self.private_ws:
                self.connect_private()
                # Reset reconnect attempts on successful connection
                self.reconnect_attempts = 0
            
            self.reconnect_thread = None
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            
            # If we haven't reached max attempts, retry
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_thread = threading.Thread(target=self._reconnect)
                self.reconnect_thread.daemon = True
                self.reconnect_thread.start()
            else:
                logger.error(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                self.reconnect_thread = None
                
                # Reset counter for future attempts
                self.reconnect_attempts = 0
    
    def connect_public(self):
        """
        Connect to public WebSocket
        """
        self.public_ws = WebSocketApp(
            self.public_ws_url,
            on_open=self._on_public_open,
            on_message=self._on_public_message,
            on_error=self._on_public_error,
            on_close=self._on_public_close
        )
        
        self.public_thread = threading.Thread(target=self.public_ws.run_forever)
        self.public_thread.daemon = True
        self.public_thread.start()
    
    def connect_private(self):
        """
        Connect to private WebSocket
        """
        if not self.api_key or not self.api_secret:
            logger.warning("API key and secret not provided. Private WebSocket not connected.")
            return
        
        self.private_ws = WebSocketApp(
            self.private_ws_url,
            on_open=self._on_private_open,
            on_message=self._on_private_message,
            on_error=self._on_private_error,
            on_close=self._on_private_close
        )
        
        self.private_thread = threading.Thread(target=self.private_ws.run_forever)
        self.private_thread.daemon = True
        self.private_thread.start()
    
    def subscribe_ticker(self, pairs: List[str], callback: Callable):
        """
        Subscribe to ticker information
        
        Args:
            pairs (list): List of asset pairs
            callback (callable): Callback function for ticker updates
        """
        self.ticker_callbacks.append(callback)
        
        # For Kraken, we need to subscribe to each pair individually
        for pair in pairs:
            # Format pair according to Kraken's ISO 4217-A3 requirements
            # Kraken requires ISO 4217-A3 format such as "SOL/USD" for websockets
            if '/' not in pair:
                # Add the slash if it's not there (SOLUSD -> SOL/USD)
                if len(pair) == 6:  # For 6 character pairs like SOLUSD
                    formatted_pair = pair[:3] + '/' + pair[3:]
                else:
                    # For pairs with different length, try to split at standard locations
                    formatted_pair = pair
            else:
                formatted_pair = pair
                
            # Special case for BTC (Kraken uses XBT)
            if 'BTC' in formatted_pair:
                formatted_pair = formatted_pair.replace('BTC', 'XBT')
                
            logger.info(f"Formatted pair for WebSocket: {pair} → {formatted_pair}")
            
            subscription = {
                "name": "ticker"
            }
            
            message = {
                "event": "subscribe",
                "pair": [formatted_pair],
                "subscription": subscription
            }
            
            if self.public_connected and self.public_ws:
                try:
                    self.public_ws.send(json.dumps(message))
                    logger.info(f"Subscribed to ticker for {formatted_pair}")
                except Exception as e:
                    logger.error(f"Error subscribing to ticker: {e}")
                    self.public_buffer.append(message)
            else:
                self.public_buffer.append(message)
    
    def subscribe_ohlc(self, pairs: List[str], callback: Callable, interval: int = 1):
        """
        Subscribe to OHLC data
        
        Args:
            pairs (list): List of asset pairs
            interval (int, optional): Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            callback (callable): Callback function for OHLC updates
        """
        self.ohlc_callbacks.append(callback)
        
        # For Kraken, we need to subscribe to each pair individually
        for pair in pairs:
            # Format pair according to Kraken's ISO 4217-A3 requirements
            # Kraken requires ISO 4217-A3 format such as "SOL/USD" for websockets
            if '/' not in pair:
                # Add the slash if it's not there (SOLUSD -> SOL/USD)
                if len(pair) == 6:  # For 6 character pairs like SOLUSD
                    formatted_pair = pair[:3] + '/' + pair[3:]
                else:
                    # For pairs with different length, try to split at standard locations
                    formatted_pair = pair
            else:
                formatted_pair = pair
                
            # Special case for BTC (Kraken uses XBT)
            if 'BTC' in formatted_pair:
                formatted_pair = formatted_pair.replace('BTC', 'XBT')
                
            logger.info(f"Formatted pair for OHLC WebSocket: {pair} → {formatted_pair}")
            
            subscription = {
                "name": "ohlc",
                "interval": interval
            }
            
            message = {
                "event": "subscribe",
                "pair": [formatted_pair],
                "subscription": subscription
            }
            
            if self.public_connected and self.public_ws:
                try:
                    self.public_ws.send(json.dumps(message))
                    logger.info(f"Subscribed to OHLC for {formatted_pair}")
                except Exception as e:
                    logger.error(f"Error subscribing to OHLC: {e}")
                    self.public_buffer.append(message)
            else:
                self.public_buffer.append(message)
    
    def subscribe_trades(self, pairs: List[str], callback: Callable):
        """
        Subscribe to trades
        
        Args:
            pairs (list): List of asset pairs
            callback (callable): Callback function for trade updates
        """
        self.trade_callbacks.append(callback)
        
        # For Kraken, we need to subscribe to each pair individually
        for pair in pairs:
            # Format pair according to Kraken's ISO 4217-A3 requirements
            # Kraken requires ISO 4217-A3 format such as "SOL/USD" for websockets
            if '/' not in pair:
                # Add the slash if it's not there (SOLUSD -> SOL/USD)
                if len(pair) == 6:  # For 6 character pairs like SOLUSD
                    formatted_pair = pair[:3] + '/' + pair[3:]
                else:
                    # For pairs with different length, try to split at standard locations
                    formatted_pair = pair
            else:
                formatted_pair = pair
                
            # Special case for BTC (Kraken uses XBT)
            if 'BTC' in formatted_pair:
                formatted_pair = formatted_pair.replace('BTC', 'XBT')
                
            logger.info(f"Formatted pair for trades WebSocket: {pair} → {formatted_pair}")
            
            subscription = {
                "name": "trade"
            }
            
            message = {
                "event": "subscribe",
                "pair": [formatted_pair],
                "subscription": subscription
            }
            
            if self.public_connected and self.public_ws:
                try:
                    self.public_ws.send(json.dumps(message))
                    logger.info(f"Subscribed to trades for {formatted_pair}")
                except Exception as e:
                    logger.error(f"Error subscribing to trades: {e}")
                    self.public_buffer.append(message)
            else:
                self.public_buffer.append(message)
    
    def subscribe_book(self, pairs: List[str], callback: Callable, depth: int = 10):
        """
        Subscribe to order book
        
        Args:
            pairs (list): List of asset pairs
            depth (int, optional): Order book depth (10, 25, 100, 500, 1000)
            callback (callable): Callback function for order book updates
        """
        self.book_callbacks.append(callback)
        
        # For Kraken, we need to subscribe to each pair individually
        for pair in pairs:
            # Format pair according to Kraken's ISO 4217-A3 requirements
            # Kraken requires ISO 4217-A3 format such as "SOL/USD" for websockets
            if '/' not in pair:
                # Add the slash if it's not there (SOLUSD -> SOL/USD)
                if len(pair) == 6:  # For 6 character pairs like SOLUSD
                    formatted_pair = pair[:3] + '/' + pair[3:]
                else:
                    # For pairs with different length, try to split at standard locations
                    formatted_pair = pair
            else:
                formatted_pair = pair
                
            # Special case for BTC (Kraken uses XBT)
            if 'BTC' in formatted_pair:
                formatted_pair = formatted_pair.replace('BTC', 'XBT')
                
            logger.info(f"Formatted pair for book WebSocket: {pair} → {formatted_pair}")
            
            subscription = {
                "name": "book",
                "depth": depth
            }
            
            message = {
                "event": "subscribe",
                "pair": [formatted_pair],
                "subscription": subscription
            }
            
            if self.public_connected and self.public_ws:
                try:
                    self.public_ws.send(json.dumps(message))
                    logger.info(f"Subscribed to book for {formatted_pair}")
                except Exception as e:
                    logger.error(f"Error subscribing to book: {e}")
                    self.public_buffer.append(message)
            else:
                self.public_buffer.append(message)
    
    def subscribe_own_trades(self, callback: Callable, snapshot: bool = True):
        """
        Subscribe to own trades
        
        Args:
            callback (callable): Callback function for own trades updates
            snapshot (bool, optional): Include initial snapshot
        """
        self.own_trades_callbacks.append(callback)
        
        subscription = {
            "name": "ownTrades",
            "snapshot": snapshot
        }
        
        message = {
            "event": "subscribe",
            "subscription": subscription
        }
        
        logger.info(f"Preparing to subscribe to own trades (private WebSocket)")
        
        if self.private_connected and self.private_ws:
            try:
                self.private_ws.send(json.dumps(message))
                logger.info("Subscribed to own trades")
            except Exception as e:
                logger.error(f"Error subscribing to own trades: {e}")
                self.private_buffer.append(message)
        else:
            logger.warning("Private WebSocket not connected, buffering own trades subscription")
            self.private_buffer.append(message)
    
    def subscribe_open_orders(self, callback: Callable, snapshot: bool = True):
        """
        Subscribe to open orders
        
        Args:
            callback (callable): Callback function for open orders updates
            snapshot (bool, optional): Include initial snapshot
        """
        self.open_orders_callbacks.append(callback)
        
        subscription = {
            "name": "openOrders",
            "snapshot": snapshot
        }
        
        message = {
            "event": "subscribe",
            "subscription": subscription
        }
        
        logger.info(f"Preparing to subscribe to open orders (private WebSocket)")
        
        if self.private_connected and self.private_ws:
            try:
                self.private_ws.send(json.dumps(message))
                logger.info("Subscribed to open orders")
            except Exception as e:
                logger.error(f"Error subscribing to open orders: {e}")
                self.private_buffer.append(message)
        else:
            logger.warning("Private WebSocket not connected, buffering open orders subscription")
            self.private_buffer.append(message)
    
    def disconnect(self):
        """
        Disconnect from WebSockets
        """
        self.keep_running = False
        
        if self.public_ws:
            self.public_ws.close()
        
        if self.private_ws:
            self.private_ws.close()
        
        logger.info("WebSocket connections closed")
