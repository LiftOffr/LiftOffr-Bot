import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from kraken_api import KrakenAPI
from kraken_websocket import KrakenWebsocket
from trading_strategy import get_strategy, TradingStrategy
from utils import record_trade, parse_kraken_ohlc
from config import (
    TRADING_PAIR, TRADE_QUANTITY, LOOP_INTERVAL, STRATEGY_TYPE,
    USE_SANDBOX, INITIAL_CAPITAL, LEVERAGE, MARGIN_PERCENT,
    SIGNAL_INTERVAL, VOL_THRESHOLD, ENTRY_ATR_MULTIPLIER,
    LOOKBACK_HOURS, ORDER_TIMEOUT_SECONDS
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
        
        # Portfolio and position tracking (from original code)
        self.portfolio_value = INITIAL_CAPITAL
        self.total_profit = 0.0
        self.total_profit_percent = 0.0
        self.trade_count = 0
        
        # Trading state
        self.current_price = None
        self.position = None    # None, "long", or "short"
        self.entry_price = None
        
        # For trailing stops and exits
        self.trailing_max_price = None  # For long positions
        self.trailing_min_price = None  # For short positions
        
        # Order management (from original code)
        self.pending_order = None
        self.trailing_stop_order = None
        self.breakeven_order = None
        self.liquidity_exit_order = None
        
        # Signal timing
        self.last_signal_update = 0
        self.last_candle_time = None
        
        # WebSocket data
        self.ohlc_data = []
        self.ticker_data = {}
        self.order_book = {}
        self.open_orders = {}
        
        # Cached data for signal calculations
        self.cached_df = None
        self.cached_indicators = None
        self.current_atr = None
        
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
        try:
            # Convert ticker data to easier format (handle different data formats gracefully)
            ticker = {}
            if isinstance(data, dict) and 'a' in data and 'b' in data and 'c' in data:
                ticker = {
                    'ask': float(data['a'][0]) if isinstance(data['a'], list) and len(data['a']) > 0 else 0.0,
                    'bid': float(data['b'][0]) if isinstance(data['b'], list) and len(data['b']) > 0 else 0.0,
                    'close': float(data['c'][0]) if isinstance(data['c'], list) and len(data['c']) > 0 else 0.0,
                    'volume': float(data['v'][1]) if isinstance(data['v'], list) and len(data['v']) > 1 else 0.0,
                    'vwap': float(data['p'][1]) if isinstance(data['p'], list) and len(data['p']) > 1 else 0.0,
                    'low': float(data['l'][1]) if isinstance(data['l'], list) and len(data['l']) > 1 else 0.0,
                    'high': float(data['h'][1]) if isinstance(data['h'], list) and len(data['h']) > 1 else 0.0,
                    'open': float(data['o']) if 'o' in data else 0.0
                }
            else:
                logger.warning(f"Unexpected ticker data format: {data}")
                return
            
            self.ticker_data[pair] = ticker
            
            # Update current price and trailing stops
            if pair == self.trading_pair:
                self.current_price = ticker['close']
                
                # Update trailing stop prices
                if self.position == "long" and self.trailing_max_price is not None:
                    self.trailing_max_price = max(self.trailing_max_price, self.current_price)
                elif self.position == "short" and self.trailing_min_price is not None:
                    self.trailing_min_price = min(self.trailing_min_price, self.current_price)
                
                # Check for trailing stop triggers and pending orders
                self._check_orders()
        except Exception as e:
            logger.error(f"Error processing ticker update: {e}, Data: {data}")
    
    def _handle_ohlc_update(self, pair: str, data: List):
        """
        Handle OHLC updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (list): OHLC data
        """
        try:
            # Check if we have valid OHLC data
            if not isinstance(data, list) or len(data) < 8:
                logger.warning(f"Invalid OHLC data format: {data}")
                return
            
            # data format: [time, open, high, low, close, vwap, volume, count]
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
                
                # Update strategy with OHLC data
                self.strategy.update_ohlc(candle['open'], candle['high'], candle['low'], candle['close'])
                
                # Check if we need to update signals
                if time.time() - self.last_signal_update >= SIGNAL_INTERVAL or self.last_signal_update == 0:
                    self._update_signals()
        except Exception as e:
            logger.error(f"Error processing OHLC update: {e}, Data: {data}")
    
    def _handle_trade_update(self, pair: str, data: List):
        """
        Handle trade updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (list): Trade data
        """
        try:
            # Validate the data structure
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"Invalid trade data format: {data}")
                return
                
            # Only process the most recent trade
            latest_trade = data[-1]
            
            # Make sure we have a valid trade data element
            if not isinstance(latest_trade, list) or len(latest_trade) < 1:
                logger.warning(f"Invalid trade element format: {latest_trade}")
                return
                
            price = float(latest_trade[0])
            
            # Update current price
            if pair == self.trading_pair:
                self.current_price = price
                
                # Update trailing stop prices
                if self.position == "long" and self.trailing_max_price is not None:
                    self.trailing_max_price = max(self.trailing_max_price, self.current_price)
                elif self.position == "short" and self.trailing_min_price is not None:
                    self.trailing_min_price = min(self.trailing_min_price, self.current_price)
                
                # Check for trailing stop triggers and pending orders
                self._check_orders()
        except Exception as e:
            logger.error(f"Error processing trade update: {e}, Data: {data}")
    
    def _handle_book_update(self, pair: str, data: Dict):
        """
        Handle order book updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (dict): Order book data
        """
        try:
            # Validate data
            if not isinstance(data, dict):
                logger.warning(f"Invalid order book data format: {data}")
                return
                
            # Initialize order book if not exists
            if pair not in self.order_book:
                self.order_book[pair] = {'asks': {}, 'bids': {}}
            
            # Update asks
            if 'a' in data and isinstance(data['a'], list):
                for ask in data['a']:
                    if not isinstance(ask, list) or len(ask) < 2:
                        logger.warning(f"Invalid ask format: {ask}")
                        continue
                        
                    try:
                        price = float(ask[0])
                        volume = float(ask[1])
                        
                        if volume == 0:
                            if price in self.order_book[pair]['asks']:
                                del self.order_book[pair]['asks'][price]
                        else:
                            self.order_book[pair]['asks'][price] = volume
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing ask: {e}, Data: {ask}")
            
            # Update bids
            if 'b' in data and isinstance(data['b'], list):
                for bid in data['b']:
                    if not isinstance(bid, list) or len(bid) < 2:
                        logger.warning(f"Invalid bid format: {bid}")
                        continue
                        
                    try:
                        price = float(bid[0])
                        volume = float(bid[1])
                        
                        if volume == 0:
                            if price in self.order_book[pair]['bids']:
                                del self.order_book[pair]['bids'][price]
                        else:
                            self.order_book[pair]['bids'][price] = volume
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing bid: {e}, Data: {bid}")
        except Exception as e:
            logger.error(f"Error processing book update: {e}, Data: {data}")
    
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
                    self.position = "long"
                    self.entry_price = float(trade['price'])
                    self.trailing_max_price = self.entry_price
                    self.trailing_min_price = None
                    logger.info(f"LONG position entered at {self.entry_price}")
                    
                elif trade['type'] == 'sell':
                    self.position = "short" if self.position is None else None
                    
                    if self.position == "short":
                        self.entry_price = float(trade['price'])
                        self.trailing_min_price = self.entry_price
                        self.trailing_max_price = None
                        logger.info(f"SHORT position entered at {self.entry_price}")
                    else:
                        # This was a close of a long position
                        trade_profit = (float(trade['price']) - self.entry_price) * float(trade['vol'])
                        profit_percent = (trade_profit / (self.portfolio_value * MARGIN_PERCENT)) * 100.0
                        
                        # Update portfolio
                        self.portfolio_value += trade_profit
                        self.total_profit += trade_profit
                        self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
                        self.trade_count += 1
                        
                        logger.info(f"LONG position exited at {float(trade['price'])}, Profit=${trade_profit:.2f} ({profit_percent:.2f}%)")
                        
                        self.entry_price = None
                        self.trailing_max_price = None
                        self.trailing_min_price = None
                
                # Update strategy with position information
                self.strategy.update_position(self.position, self.entry_price)
    
    def _handle_open_orders(self, data: Dict):
        """
        Handle open orders updates from WebSocket
        
        Args:
            data (dict): Open orders data
        """
        self.open_orders = data
    
    def _update_signals(self):
        """
        Update trading signals based on current market data
        """
        try:
            if len(self.ohlc_data) < 30:
                logger.warning("Not enough OHLC data to update signals")
                return
            
            # Convert OHLC data to DataFrame
            df = pd.DataFrame(self.ohlc_data)
            
            # Get the latest candle time
            current_last_candle = df['time'].max()
            
            # Only update indicators if we have a new candle
            if self.last_candle_time is None or current_last_candle > self.last_candle_time:
                self.last_candle_time = current_last_candle
                
                # Calculate signals
                buy_signal, sell_signal, atr_value = self.strategy.calculate_signals(df)
                
                # Store ATR value
                self.current_atr = atr_value
                
                # Store the DataFrame with indicators
                self.cached_df = df
                
                # Check for entry signals
                if self.position is None and buy_signal:
                    self._place_buy_order()
                elif self.position == "long" and sell_signal:
                    self._place_sell_order(exit_only=True)
                # We could add short selling here, but keeping it simple for now
                
                # Update signal timestamp
                self.last_signal_update = time.time()
                logger.info(f"Signals updated: Buy={buy_signal}, Sell={sell_signal}, ATR={atr_value:.4f}")
                
        except Exception as e:
            logger.error(f"Error updating signals: {e}")
    
    def _check_orders(self):
        """
        Check and manage pending orders and trailing stops
        """
        if not self.running or self.current_price is None:
            return
        
        try:
            # Check if pending order is filled based on live price
            if self.pending_order is not None:
                # Check for timeout
                if time.time() - self.pending_order["time"] > ORDER_TIMEOUT_SECONDS:
                    logger.info(f"Pending {self.pending_order['type']} order timed out")
                    self.pending_order = None
                    return
                
                # Check if price conditions are met
                if self.pending_order["type"] == "buy" and self.current_price <= self.pending_order["price"]:
                    logger.info(f"Buy limit order filled at {self.pending_order['price']:.4f}")
                    self._execute_buy()
                    self.pending_order = None
                
                elif self.pending_order["type"] == "sell" and self.current_price >= self.pending_order["price"]:
                    logger.info(f"Sell limit order filled at {self.pending_order['price']:.4f}")
                    self._execute_sell()
                    self.pending_order = None
            
            # Check trailing stop for long position
            if self.position == "long" and self.trailing_stop_order is not None:
                if self.current_price <= self.trailing_stop_order["price"]:
                    logger.info(f"Trailing stop triggered at {self.trailing_stop_order['price']:.4f}")
                    self._place_sell_order(exit_only=True)
                    self.trailing_stop_order = None
            
            # Check breakeven exit for long position
            if self.position == "long" and self.breakeven_order is not None:
                if self.current_price >= self.breakeven_order["price"]:
                    logger.info(f"Breakeven exit triggered at {self.breakeven_order['price']:.4f}")
                    self._place_sell_order(exit_only=True)
                    self.breakeven_order = None
            
            # Update trailing stop based on current price for long position
            if self.position == "long" and self.trailing_max_price is not None and self.current_atr is not None:
                new_stop_price = self.trailing_max_price - (2.0 * self.current_atr)
                
                if self.trailing_stop_order is None:
                    self.trailing_stop_order = {
                        "type": "sell", 
                        "price": new_stop_price, 
                        "time": time.time()
                    }
                    logger.info(f"Setting trailing stop at {new_stop_price:.4f}")
                
                # Only move stop up, never down
                elif new_stop_price > self.trailing_stop_order["price"]:
                    self.trailing_stop_order["price"] = new_stop_price
                    logger.info(f"Updated trailing stop to {new_stop_price:.4f}")
        
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
    
    def _place_buy_order(self):
        """
        Place a buy limit order based on ATR
        """
        if self.current_price is None or self.current_atr is None:
            logger.warning("Cannot place buy order: price or ATR unknown")
            return
        
        if self.position is not None:
            logger.info(f"Already in {self.position} position; skipping buy order")
            return
        
        # Calculate limit price based on ATR (from original code)
        limit_price = self.current_price - (self.current_atr * ENTRY_ATR_MULTIPLIER)
        
        # Create pending order
        self.pending_order = {
            "type": "buy",
            "price": limit_price,
            "time": time.time()
        }
        
        logger.info(f"Placed buy limit order at {limit_price:.4f}")
    
    def _place_sell_order(self, exit_only=False):
        """
        Place a sell limit order
        
        Args:
            exit_only (bool): Whether this is just to exit a position
        """
        if self.current_price is None:
            logger.warning("Cannot place sell order: current price unknown")
            return
        
        if exit_only and self.position != "long":
            logger.info("No long position to exit; skipping sell order")
            return
        
        if not exit_only and self.position is not None:
            logger.info(f"Already in {self.position} position; skipping sell order")
            return
        
        # For exit orders, use market price
        price = self.current_price
        
        # Create pending order for immediate execution
        self.pending_order = {
            "type": "sell",
            "price": price,
            "time": time.time(),
            "exit_only": exit_only
        }
        
        logger.info(f"Placed sell {'exit' if exit_only else 'limit'} order at {price:.4f}")
        
        # Execute immediately
        if exit_only:
            self._execute_sell(exit_only=True)
            self.pending_order = None
    
    def _execute_buy(self):
        """
        Execute buy order
        """
        if not self.current_price:
            logger.warning("Cannot execute buy: current price unknown")
            return
        
        logger.info(f"BUY SIGNAL at price {self.current_price}")
        
        # Calculate position size based on margin
        margin_amount = self.portfolio_value * MARGIN_PERCENT
        notional = margin_amount * LEVERAGE
        quantity = notional / self.current_price
        
        # Execute order only if not in sandbox mode
        if not self.sandbox_mode:
            try:
                order_result = self.api.place_order(
                    pair=self.trading_pair,
                    type_="buy",
                    ordertype="market",
                    volume=str(quantity)
                )
                
                logger.info(f"Buy order executed: {order_result}")
                
                # Update position status
                self.position = "long"
                self.entry_price = self.current_price
                self.trailing_max_price = self.current_price
                self.trailing_min_price = None
                self.strategy.update_position("long", self.current_price)
                
                # Record the trade
                record_trade("buy", self.trading_pair, quantity, self.current_price, 
                            profit=None, profit_percent=None, position="long")
            
            except Exception as e:
                logger.error(f"Error executing buy order: {e}")
        else:
            logger.info(f"SANDBOX MODE: Buy order for {quantity:.6f} units would be executed at {self.current_price:.4f}")
            
            # Simulate position status update in sandbox mode
            self.position = "long"
            self.entry_price = self.current_price
            self.trailing_max_price = self.current_price
            self.trailing_min_price = None
            self.strategy.update_position("long", self.current_price)
            
            # Record the trade
            record_trade("buy", self.trading_pair, quantity, self.current_price, 
                        profit=None, profit_percent=None, position="long")
    
    def _execute_sell(self, exit_only=False):
        """
        Execute sell order
        
        Args:
            exit_only (bool): Whether this is just to exit a position
        """
        if not self.current_price:
            logger.warning("Cannot execute sell: current price unknown")
            return
        
        logger.info(f"SELL SIGNAL at price {self.current_price}")
        
        # If exiting a long position
        if exit_only and self.position == "long":
            # Calculate actual trade quantity based on entry
            margin_amount = self.portfolio_value * MARGIN_PERCENT
            notional = margin_amount * LEVERAGE
            quantity = notional / self.entry_price
            
            # Calculate profit
            trade_profit = (self.current_price - self.entry_price) * quantity
            profit_percent = (trade_profit / margin_amount) * 100.0
            
            # Execute order only if not in sandbox mode
            if not self.sandbox_mode:
                try:
                    order_result = self.api.place_order(
                        pair=self.trading_pair,
                        type_="sell",
                        ordertype="market",
                        volume=str(quantity)
                    )
                    
                    logger.info(f"Sell order executed: {order_result}")
                    
                    # Update portfolio
                    self.portfolio_value += trade_profit
                    self.total_profit += trade_profit
                    self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
                    self.trade_count += 1
                    
                    # Update position status
                    self.position = None
                    self.entry_price = None
                    self.trailing_max_price = None
                    self.trailing_min_price = None
                    self.strategy.update_position(None, None)
                    
                    # Record the trade
                    record_trade("sell", self.trading_pair, quantity, self.current_price, 
                               profit=trade_profit, profit_percent=profit_percent, position="none")
                
                except Exception as e:
                    logger.error(f"Error executing sell order: {e}")
            else:
                logger.info(f"SANDBOX MODE: Sell order for {quantity:.6f} units would be executed at {self.current_price:.4f}")
                logger.info(f"SANDBOX MODE: Profit would be ${trade_profit:.2f} ({profit_percent:.2f}%)")
                
                # Update portfolio in sandbox mode
                self.portfolio_value += trade_profit
                self.total_profit += trade_profit
                self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
                self.trade_count += 1
                
                # Update position status
                self.position = None
                self.entry_price = None
                self.trailing_max_price = None
                self.trailing_min_price = None
                self.strategy.update_position(None, None)
                
                # Record the trade
                record_trade("sell", self.trading_pair, quantity, self.current_price, 
                           profit=trade_profit, profit_percent=profit_percent, position="none")
        else:
            logger.warning("Sell order only supported for exit_only=True at this time")
    
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
                    self.position = "long"
                    self.entry_price = last_order['price']
                    self.trailing_max_price = self.entry_price
                else:
                    self.position = None
                    self.entry_price = None
                    self.trailing_max_price = None
                
                logger.info(f"Position status from order history: position={self.position}, price={self.entry_price}")
                self.strategy.update_position(self.position, self.entry_price)
        
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
            
            # Convert to DataFrame for easier processing
            df = parse_kraken_ohlc(ohlc_data[pair_key])
            
            # Load historical OHLC data into strategy
            for _, row in df.iterrows():
                self.strategy.update_ohlc(row['open'], row['high'], row['low'], row['close'])
            
            logger.info(f"Loaded {len(df)} historical candles")
            
            # Calculate initial signals
            self._update_signals()
        
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
                # Check orders and trailing stops periodically
                self._check_orders()
                
                # Check if we need to update signals
                if time.time() - self.last_signal_update >= SIGNAL_INTERVAL:
                    self._update_signals()
                
                # Log portfolio status
                logger.info(f"Portfolio: ${self.portfolio_value:.2f} | "
                          f"Total Profit: ${self.total_profit:.2f} "
                          f"({self.total_profit_percent:.2f}%)")
                
                time.sleep(LOOP_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        
        except Exception as e:
            logger.error(f"Bot error: {e}")
        
        finally:
            self.stop()
