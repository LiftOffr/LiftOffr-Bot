import logging
import time
from typing import Dict, List, Optional, Union, Any
from kraken_api import KrakenAPI
from kraken_websocket import KrakenWebsocket
from trading_strategy import get_strategy, TradingStrategy
from config import (
    TRADING_PAIR, TRADE_QUANTITY, LOOP_INTERVAL, STRATEGY_TYPE,
    USE_SANDBOX
)

logger = logging.getLogger(__name__)

class KrakenTradingBot:
    """
    Trading bot for Kraken exchange
    """
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 trading_pair: str = TRADING_PAIR, 
                 trade_quantity: float = TRADE_QUANTITY,
                 strategy_type: str = STRATEGY_TYPE):
        """
        Initialize the trading bot
        
        Args:
            api_key (str, optional): Kraken API key
            api_secret (str, optional): Kraken API secret
            trading_pair (str, optional): Trading pair to trade
            trade_quantity (float, optional): Quantity to trade
            strategy_type (str, optional): Type of trading strategy
        """
        self.trading_pair = trading_pair
        self.trade_quantity = trade_quantity
        
        # Initialize API clients
        self.api = KrakenAPI(api_key, api_secret)
        self.ws = KrakenWebsocket(api_key, api_secret)
        
        # Initialize trading strategy
        self.strategy = get_strategy(strategy_type, trading_pair)
        
        # Trading state
        self.current_price = None
        self.in_position = False
        self.position_price = None
        
        # WebSocket data
        self.ohlc_data = []
        self.ticker_data = {}
        self.order_book = {}
        self.open_orders = {}
        
        # Bot control
        self.running = False
        self.sandbox_mode = USE_SANDBOX
        
        if self.sandbox_mode:
            logger.warning("Running in sandbox/test mode. No real trades will be executed.")
    
    def _handle_ticker_update(self, pair: str, data: Dict):
        """
        Handle ticker updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (dict): Ticker data
        """
        # Convert ticker data to easier format
        ticker = {
            'ask': float(data['a'][0]),
            'bid': float(data['b'][0]),
            'close': float(data['c'][0]),
            'volume': float(data['v'][1]),
            'vwap': float(data['p'][1]),
            'low': float(data['l'][1]),
            'high': float(data['h'][1]),
            'open': float(data['o'])
        }
        
        self.ticker_data[pair] = ticker
        
        # Update current price and strategy
        if pair == self.trading_pair:
            self.current_price = ticker['close']
            self.strategy.update_price(self.current_price)
            
            # Check for buy/sell signals
            self._check_signals()
    
    def _handle_ohlc_update(self, pair: str, data: List):
        """
        Handle OHLC updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (list): OHLC data
        """
        # data format: [channelID, [time, open, high, low, close, vwap, volume, count], channelName, pair]
        candle = {
            'time': float(data[0]),
            'open': float(data[1]),
            'high': float(data[2]),
            'low': float(data[3]),
            'close': float(data[4]),
            'vwap': float(data[5]),
            'volume': float(data[6]),
            'count': int(data[7])
        }
        
        # Add to OHLC data
        self.ohlc_data.append(candle)
        
        # Keep only the last 1000 candles
        if len(self.ohlc_data) > 1000:
            self.ohlc_data = self.ohlc_data[-1000:]
        
        # Update current price and strategy
        if pair == self.trading_pair:
            self.current_price = candle['close']
            self.strategy.update_price(self.current_price)
            
            # Check for buy/sell signals
            self._check_signals()
    
    def _handle_trade_update(self, pair: str, data: List):
        """
        Handle trade updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (list): Trade data
        """
        # Only process the most recent trade
        if len(data) > 0:
            latest_trade = data[-1]
            price = float(latest_trade[0])
            
            # Update current price and strategy
            if pair == self.trading_pair:
                self.current_price = price
                self.strategy.update_price(price)
                
                # Check for buy/sell signals
                self._check_signals()
    
    def _handle_book_update(self, pair: str, data: Dict):
        """
        Handle order book updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (dict): Order book data
        """
        # Initialize order book if not exists
        if pair not in self.order_book:
            self.order_book[pair] = {'asks': {}, 'bids': {}}
        
        # Update asks
        if 'a' in data:
            for ask in data['a']:
                price = float(ask[0])
                volume = float(ask[1])
                
                if volume == 0:
                    if price in self.order_book[pair]['asks']:
                        del self.order_book[pair]['asks'][price]
                else:
                    self.order_book[pair]['asks'][price] = volume
        
        # Update bids
        if 'b' in data:
            for bid in data['b']:
                price = float(bid[0])
                volume = float(bid[1])
                
                if volume == 0:
                    if price in self.order_book[pair]['bids']:
                        del self.order_book[pair]['bids'][price]
                else:
                    self.order_book[pair]['bids'][price] = volume
    
    def _handle_own_trades(self, data: Dict):
        """
        Handle own trades updates from WebSocket
        
        Args:
            data (dict): Own trades data
        """
        # Process trades to update position
        for trade_id, trade in data.items():
            symbol = trade['pair']
            
            if symbol == self.trading_pair:
                if trade['type'] == 'buy':
                    self.in_position = True
                    self.position_price = float(trade['price'])
                    logger.info(f"Position entered at {self.position_price}")
                    
                elif trade['type'] == 'sell':
                    self.in_position = False
                    self.position_price = None
                    logger.info("Position exited")
                
                # Update strategy with position information
                self.strategy.update_position(self.in_position, self.position_price)
    
    def _handle_open_orders(self, data: Dict):
        """
        Handle open orders updates from WebSocket
        
        Args:
            data (dict): Open orders data
        """
        self.open_orders = data
    
    def _check_signals(self):
        """
        Check for trading signals and execute trades
        """
        if not self.running:
            return
        
        try:
            if self.strategy.should_buy():
                self._execute_buy()
            
            elif self.strategy.should_sell():
                self._execute_sell()
        
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
    
    def _execute_buy(self):
        """
        Execute buy order
        """
        if not self.current_price:
            logger.warning("Cannot execute buy: current price unknown")
            return
        
        logger.info(f"BUY SIGNAL at price {self.current_price}")
        
        # Execute order only if not in sandbox mode
        if not self.sandbox_mode:
            try:
                order_result = self.api.place_order(
                    pair=self.trading_pair,
                    type_="buy",
                    ordertype="market",
                    volume=str(self.trade_quantity)
                )
                
                logger.info(f"Buy order executed: {order_result}")
                
                # Update position status
                self.in_position = True
                self.position_price = self.current_price
                self.strategy.update_position(True, self.current_price)
            
            except Exception as e:
                logger.error(f"Error executing buy order: {e}")
        else:
            logger.info("SANDBOX MODE: Buy order would be executed here")
            
            # Simulate position status update in sandbox mode
            self.in_position = True
            self.position_price = self.current_price
            self.strategy.update_position(True, self.current_price)
    
    def _execute_sell(self):
        """
        Execute sell order
        """
        if not self.current_price:
            logger.warning("Cannot execute sell: current price unknown")
            return
        
        logger.info(f"SELL SIGNAL at price {self.current_price}")
        
        # Execute order only if not in sandbox mode
        if not self.sandbox_mode:
            try:
                order_result = self.api.place_order(
                    pair=self.trading_pair,
                    type_="sell",
                    ordertype="market",
                    volume=str(self.trade_quantity)
                )
                
                logger.info(f"Sell order executed: {order_result}")
                
                # Update position status
                self.in_position = False
                self.position_price = None
                self.strategy.update_position(False, None)
            
            except Exception as e:
                logger.error(f"Error executing sell order: {e}")
        else:
            logger.info("SANDBOX MODE: Sell order would be executed here")
            
            # Simulate position status update in sandbox mode
            self.in_position = False
            self.position_price = None
            self.strategy.update_position(False, None)
    
    def check_position(self):
        """
        Check current position status from closed orders
        """
        try:
            # Get closed orders to determine current position
            closed_orders = self.api.get_closed_orders()
            
            # Sort orders by closing time
            orders_list = []
            for order_id, order in closed_orders.get('closed', {}).items():
                if order['pair'] == self.trading_pair and order['status'] == 'closed':
                    orders_list.append({
                        'order_id': order_id,
                        'close_time': order['closetm'],
                        'type': order['descr']['type'],
                        'price': float(order['price']),
                        'vol_exec': float(order['vol_exec'])
                    })
            
            # Sort by close time, most recent first
            orders_list.sort(key=lambda x: x['close_time'], reverse=True)
            
            # Check most recent order to determine position
            if orders_list:
                last_order = orders_list[0]
                
                if last_order['type'] == 'buy':
                    self.in_position = True
                    self.position_price = last_order['price']
                else:
                    self.in_position = False
                    self.position_price = None
                
                logger.info(f"Position status from order history: in_position={self.in_position}, price={self.position_price}")
                self.strategy.update_position(self.in_position, self.position_price)
        
        except Exception as e:
            logger.error(f"Error checking position: {e}")
    
    def start(self):
        """
        Start the trading bot
        """
        logger.info("Starting Kraken Trading Bot")
        
        # Connect to WebSockets
        self.ws.connect_public()
        self.ws.connect_private()
        
        # Subscribe to data streams
        self.ws.subscribe_ticker([self.trading_pair], self._handle_ticker_update)
        self.ws.subscribe_ohlc([self.trading_pair], self._handle_ohlc_update, 1)
        self.ws.subscribe_trades([self.trading_pair], self._handle_trade_update)
        self.ws.subscribe_book([self.trading_pair], self._handle_book_update, 10)
        
        # Subscribe to private data streams
        self.ws.subscribe_own_trades(self._handle_own_trades)
        self.ws.subscribe_open_orders(self._handle_open_orders)
        
        # Check current position
        self.check_position()
        
        # Fetch initial price data
        try:
            ohlc_data = self.api.get_ohlc(self.trading_pair, 1)
            pair_key = list(ohlc_data.keys())[0]  # Get the pair key
            
            # Load historical price data into strategy
            for candle in ohlc_data[pair_key]:
                self.strategy.update_price(float(candle[4]))  # 4 is the close price index
            
            logger.info(f"Loaded {len(ohlc_data[pair_key])} historical candles")
        
        except Exception as e:
            logger.error(f"Error fetching initial price data: {e}")
        
        # Start trading loop
        self.running = True
        
        logger.info(f"Trading bot started for pair {self.trading_pair}")
        logger.info(f"Strategy: {self.strategy.__class__.__name__}")
        logger.info(f"Trade quantity: {self.trade_quantity}")
        logger.info(f"Sandbox mode: {self.sandbox_mode}")
    
    def stop(self):
        """
        Stop the trading bot
        """
        logger.info("Stopping Kraken Trading Bot")
        self.running = False
        self.ws.disconnect()
        logger.info("Trading bot stopped")
    
    def run(self):
        """
        Run the trading bot in a loop
        """
        try:
            self.start()
            
            while self.running:
                # Main loop sleeps to avoid excessive CPU usage
                # Trading signals are processed in WebSocket callbacks
                time.sleep(LOOP_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        
        except Exception as e:
            logger.error(f"Bot error: {e}")
        
        finally:
            self.stop()
