import logging
import time
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from kraken_api import KrakenAPI
from kraken_websocket import KrakenWebsocket
from trading_strategy import get_strategy, TradingStrategy
from utils import record_trade, parse_kraken_ohlc
from notifications import send_trade_entry_notification, send_trade_exit_notification
from config import (
    TRADING_PAIR, TRADE_QUANTITY, LOOP_INTERVAL, STRATEGY_TYPE,
    USE_SANDBOX, INITIAL_CAPITAL, LEVERAGE, MARGIN_PERCENT,
    SIGNAL_INTERVAL, VOL_THRESHOLD, ENTRY_ATR_MULTIPLIER,
    LOOKBACK_HOURS, ORDER_TIMEOUT_SECONDS, STATUS_UPDATE_INTERVAL,
    ENABLE_DYNAMIC_POSITION_SIZING, BASE_MARGIN_PERCENT, MAX_MARGIN_PERCENT,
    STRONG_SIGNAL_THRESHOLD, VERY_STRONG_SIGNAL_THRESHOLD
)

logger = logging.getLogger(__name__)

class KrakenTradingBot:
    """
    Trading bot for Kraken exchange
    """
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 trading_pair: str = TRADING_PAIR, 
                 trade_quantity: float = TRADE_QUANTITY,
                 strategy_type: str = STRATEGY_TYPE,
                 margin_percent: float = MARGIN_PERCENT,
                 leverage: int = LEVERAGE):
        """
        Initialize the trading bot
        
        Args:
            api_key (str, optional): Kraken API key
            api_secret (str, optional): Kraken API secret
            trading_pair (str, optional): Trading pair to trade
            trade_quantity (float, optional): Quantity to trade
            strategy_type (str, optional): Type of trading strategy
            margin_percent (float, optional): Percentage of portfolio used as margin
            leverage (int, optional): Leverage for trading
        """
        # Store strategy-specific settings
        self.margin_percent = margin_percent
        self.leverage = leverage
        
        # For portfolio management with multiple strategies
        self.available_funds = INITIAL_CAPITAL
        self.bot_manager = None  # Reference to BotManager, set when added to manager
        # Format the trading pair correctly for Kraken
        # Kraken WebSocket requires ISO 4217-A3 format with / (e.g. "SOL/USD")
        self.original_pair = trading_pair  # Keep original for REST API calls
        
        # Format pair for WebSockets (with slash)
        if '/' not in trading_pair:
            if len(trading_pair) == 6:  # For 6 character pairs like SOLUSD
                self.trading_pair = trading_pair[:3] + '/' + trading_pair[3:]
            else:
                # For other length pairs, just use as is for now
                self.trading_pair = trading_pair
        else:
            self.trading_pair = trading_pair
            
        # Special case for BTC (Kraken uses XBT)
        if 'BTC' in self.trading_pair:
            self.trading_pair = self.trading_pair.replace('BTC', 'XBT')
            
        logger.info(f"Using trading pair: {self.trading_pair} (original: {trading_pair})")
        self.trade_quantity = trade_quantity
        
        # Use API keys from params or environment variables
        from config import API_KEY, API_SECRET
        api_key = api_key or API_KEY
        api_secret = api_secret or API_SECRET
        
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
        self.trailing_stop_order = None  # In-memory representation
        self.trailing_stop_limit_order_id = None  # Actual order ID in the exchange
        self.breakeven_order = None
        self.liquidity_exit_order = None
        
        # Dual limit order tracking for signal reversals
        self.signal_reversal_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": None,
            "lower_price": None,
            "created_time": None,
            "last_check_time": None,
            "is_forecast_reversal": False,
            "failsafe_timeout": 300
        }
        
        # Dual limit order tracking for entries
        self.entry_limit_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": None,
            "lower_price": None,
            "created_time": None,
            "last_check_time": None,
            "is_forecast_reversal": False,  # Flag to indicate if these are forecast reversal orders
            "failsafe_timeout": 300  # 5 minutes timeout for failsafe market order
        }
        
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
        
        # Get sandbox mode from environment variable (may have been changed by command line)
        self.sandbox_mode = os.environ.get('USE_SANDBOX', 'True').lower() in ['true', 't', 'yes', '1']
        
        if self.sandbox_mode:
            logger.warning("Running in sandbox/test mode. No real trades will be executed.")
        else:
            logger.info("Running in LIVE mode with REAL trading enabled.")
    
    def _handle_ticker_update(self, pair: str, data: Dict):
        """
        Handle ticker updates from WebSocket
        
        Args:
            pair (str): Trading pair
            data (dict): Ticker data
        """
        try:
            # Log raw data for debugging
            logger.debug(f"Received ticker data for {pair}: {str(data)[:200]}...")
            
            # Convert ticker data to easier format (handle different data formats gracefully)
            ticker = {}
            # Kraken ticker format: https://docs.kraken.com/websockets/#message-ticker
            if isinstance(data, dict) and 'a' in data and 'b' in data and 'c' in data:
                # Kraken ticker format is different from Binance
                # Example: {'a': ['119.07000', 125, '125.27527800'], 'b': ['119.06000', 0, '0.16798252'], ...}
                try:
                    ticker = {
                        'ask': float(data['a'][0]) if isinstance(data['a'], list) and len(data['a']) > 0 else 0.0,
                        'bid': float(data['b'][0]) if isinstance(data['b'], list) and len(data['b']) > 0 else 0.0,
                        'close': float(data['c'][0]) if isinstance(data['c'], list) and len(data['c']) > 0 else 0.0,
                        'volume': float(data['v'][1]) if isinstance(data['v'], list) and len(data['v']) > 1 else 0.0,
                        'vwap': float(data['p'][1]) if isinstance(data['p'], list) and len(data['p']) > 1 else 0.0,
                        'low': float(data['l'][1]) if isinstance(data['l'], list) and len(data['l']) > 1 else 0.0,
                        'high': float(data['h'][1]) if isinstance(data['h'], list) and len(data['h']) > 1 else 0.0,
                        'open': float(data['o'][1]) if isinstance(data['o'], list) and len(data['o']) > 1 else 0.0
                    }
                    # Log at INFO level now that we've fixed the visibility issues
                    logger.info(f"【TICKER】 {pair} = ${ticker['close']:.2f} | Bid: ${ticker['bid']:.2f} | Ask: ${ticker['ask']:.2f}")
                except Exception as e:
                    logger.error(f"Error parsing ticker fields: {e}")
                    logger.debug(f"Raw ticker data: {data}")
                    # Create a ticker with safer parsing
                    try:
                        ticker = {
                            'ask': float(str(data['a'][0]).strip()) if isinstance(data['a'], list) and len(data['a']) > 0 else 0.0,
                            'bid': float(str(data['b'][0]).strip()) if isinstance(data['b'], list) and len(data['b']) > 0 else 0.0,
                            'close': float(str(data['c'][0]).strip()) if isinstance(data['c'], list) and len(data['c']) > 0 else 0.0,
                            'volume': 0.0,  # Default
                            'vwap': 0.0,    # Default
                            'low': 0.0,     # Default
                            'high': 0.0,    # Default
                            'open': 0.0     # Default
                        }
                        logger.info(f"Used fallback parsing for {pair}: close=${ticker['close']:.2f}")
                    except Exception:
                        logger.error(f"Failed to parse minimum ticker data, skipping update")
                        return
            else:
                logger.warning(f"Unexpected ticker data format for {pair}: {data}")
                return
            
            self.ticker_data[pair] = ticker
            
            # Update current price and trailing stops
            if pair == self.trading_pair:
                # Update current price
                self.current_price = ticker['close']
                
                # Update trailing stop prices
                if self.position == "long" and self.trailing_max_price is not None:
                    self.trailing_max_price = max(self.trailing_max_price, self.current_price)
                    logger.debug(f"Updated trailing max price: {self.trailing_max_price}")
                elif self.position == "short" and self.trailing_min_price is not None:
                    self.trailing_min_price = min(self.trailing_min_price, self.current_price)
                    logger.debug(f"Updated trailing min price: {self.trailing_min_price}")
                
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
                    
                    # Set up breakeven order for long position if ATR is available
                    if self.current_atr is not None:
                        breakeven_price = self.entry_price + (self.current_atr * 1.0)  # Set breakeven 1 ATR above entry
                        self.breakeven_order = {
                            "type": "sell",
                            "price": breakeven_price,
                            "time": time.time()
                        }
                        logger.info(f"Setting breakeven exit for long position at {breakeven_price:.4f}")
                    
                    # Send trade entry notification
                    if self.current_atr:
                        stop_price = self.entry_price - (self.current_atr * 2.0)
                        send_trade_entry_notification(
                            self.trading_pair, 
                            self.entry_price, 
                            self.trade_quantity, 
                            self.current_atr, 
                            stop_price
                        )
                    
                elif trade['type'] == 'sell':
                    self.position = "short" if self.position is None else None
                    
                    if self.position == "short":
                        self.entry_price = float(trade['price'])
                        self.trailing_min_price = self.entry_price
                        self.trailing_max_price = None
                        logger.info(f"SHORT position entered at {self.entry_price}")
                        
                        # Set up breakeven order for short position if ATR is available
                        if self.current_atr is not None:
                            breakeven_price = self.entry_price - (self.current_atr * 1.0)  # Set breakeven 1 ATR below entry
                            self.breakeven_order = {
                                "type": "buy",
                                "price": breakeven_price,
                                "time": time.time()
                            }
                            logger.info(f"Setting breakeven exit for short position at {breakeven_price:.4f}")
                    else:
                        # This was a close of a long position
                        trade_profit = (float(trade['price']) - self.entry_price) * float(trade['vol'])
                        profit_percent = (trade_profit / (self.portfolio_value * MARGIN_PERCENT)) * 100.0
                        
                        # Update portfolio
                        self.portfolio_value += trade_profit
                        self.total_profit += trade_profit
                        self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
                        self.trade_count += 1
                        
                        exit_price = float(trade['price'])
                        logger.info(f"LONG position exited at {exit_price}, Profit=${trade_profit:.2f} ({profit_percent:.2f}%)")
                        
                        # Send trade exit notification
                        send_trade_exit_notification(
                            self.trading_pair, 
                            exit_price, 
                            self.entry_price, 
                            float(trade['vol']), 
                            trade_profit,
                            profit_percent,
                            self.portfolio_value,
                            self.total_profit,
                            self.total_profit_percent,
                            self.trade_count
                        )
                        
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
            
            # Get strategy name for logging purposes
            strategy_name = self.strategy.__class__.__name__
            
            # Calculate signals (always calculate to get updated analysis)
            buy_signal, sell_signal, atr_value = self.strategy.calculate_signals(df)
            
            # Store ATR value
            self.current_atr = atr_value
            
            # Store the DataFrame with indicators
            self.cached_df = df
            
            # Only execute trades if we have a new candle
            if self.last_candle_time is None or current_last_candle > self.last_candle_time:
                self.last_candle_time = current_last_candle
                
                # Check for entry signals and exit signals based on strategy position
                # Get the current strategy's position from its own state
                strategy_position = self.strategy.get_status().get('position', None)
                
                # Each strategy instance can have its own position
                if strategy_position is None:
                    if buy_signal:
                        # Check if we have enough funds before entering
                        if self.available_funds >= (self.portfolio_value * self.margin_percent):
                            logger.info(f"Strategy {strategy_name} is signaling to enter LONG position")
                            # Use dual limit orders for entries if enabled
                            from config import ENABLE_DUAL_LIMIT_ENTRIES
                            if ENABLE_DUAL_LIMIT_ENTRIES:
                                logger.info(f"Using dual limit orders for LONG entry")
                                self._place_dual_limit_orders_for_entry(entry_type="long")
                            else:
                                self._place_buy_order()  # Enter long position with standard method
                        else:
                            logger.info(f"Skipping {strategy_name} long signal - insufficient funds ({self.available_funds:.2f} < {self.portfolio_value * self.margin_percent:.2f})")
                    elif sell_signal:
                        # Check if we have enough funds before entering
                        if self.available_funds >= (self.portfolio_value * self.margin_percent):
                            logger.info(f"Strategy {strategy_name} is signaling to enter SHORT position")
                            # Use dual limit orders for entries if enabled
                            from config import ENABLE_DUAL_LIMIT_ENTRIES
                            if ENABLE_DUAL_LIMIT_ENTRIES:
                                logger.info(f"Using dual limit orders for SHORT entry")
                                self._place_dual_limit_orders_for_entry(entry_type="short")
                            else:
                                self._place_short_order()  # Enter short position with standard method
                        else:
                            logger.info(f"Skipping {strategy_name} short signal - insufficient funds ({self.available_funds:.2f} < {self.portfolio_value * self.margin_percent:.2f})")
                elif strategy_position == "long" and sell_signal:
                    logger.info(f"Strategy {strategy_name} is signaling to exit LONG position")
                    # Use dual limit orders for signal reversal exits
                    is_forecast_reversal = False
                    if self.strategy.__class__.__name__ == 'ARIMAStrategy':
                        # Only ARIMA strategy has forecast_direction property
                        is_forecast_reversal = hasattr(self.strategy, 'forecast_direction') and self.strategy.forecast_direction == 'bearish'
                    if is_forecast_reversal:
                        logger.info(f"Using dual limit orders for forecast reversal exit from LONG position")
                        self._place_dual_limit_orders_for_reversal()
                    else:
                        # Use regular exit for other cases
                        self._place_sell_order(exit_only=True)
                elif strategy_position == "short" and buy_signal:
                    logger.info(f"Strategy {strategy_name} is signaling to exit SHORT position")
                    # Use dual limit orders for signal reversal exits
                    is_forecast_reversal = False
                    if self.strategy.__class__.__name__ == 'ARIMAStrategy':
                        # Only ARIMA strategy has forecast_direction property
                        is_forecast_reversal = hasattr(self.strategy, 'forecast_direction') and self.strategy.forecast_direction == 'bullish'
                    if is_forecast_reversal:
                        logger.info(f"Using dual limit orders for forecast reversal exit from SHORT position")
                        self._place_dual_limit_orders_for_reversal()
                    else:
                        # Use regular exit for other cases
                        self._place_buy_order(exit_only=True)
            
            # Update signal timestamp (even for test updates)
            self.last_signal_update = time.time()
            
            # Format the signals log in a clean, easy-to-read format
            signal_emoji = "🟢" if buy_signal else "🔴" if sell_signal else "⚪"
            direction = "BUY" if buy_signal else "SELL" if sell_signal else "HOLD"
            
            # Calculate volatility stop (ATR-based) using current price from dataframe
            stop_price = 0
            current_df_price = df.iloc[-1]['close']  # Get most recent price from dataframe
            
            # Always use the most recent price we have for stop calculation
            used_price = self.current_price if self.current_price else current_df_price
            
            # Calculate volatility-based stop level (2 x ATR below price)
            if used_price and atr_value:
                stop_price = used_price - (atr_value * 2.0)
            
            # Display the ACTION line for decisions with strategy name
            logger.info(f"【ACTION】 {signal_emoji} {direction} | ATR: ${atr_value:.4f} | Volatility Stop: ${stop_price:.2f}")
            
            # Set signal type and strength in strategy object
            if buy_signal:
                self.strategy.signal_type = "buy"
                self.strategy.buy_strength = 0.8  # Strong buy signal
                self.strategy.sell_strength = 0.0
            elif sell_signal:
                self.strategy.signal_type = "sell"
                self.strategy.sell_strength = 0.8  # Strong sell signal
                self.strategy.buy_strength = 0.0
            else:
                self.strategy.signal_type = "neutral"
                self.strategy.buy_strength = 0.0
                self.strategy.sell_strength = 0.0
                
            # Register signals with BotManager for cross-strategy coordination
            if hasattr(self, 'bot_manager') and self.bot_manager:
                bot_id = f"{strategy_name}-{self.trading_pair}"
                signal_type = self.strategy.signal_type
                
                if signal_type == "buy":
                    self.bot_manager.register_strategy_signal(bot_id, signal_type, self.strategy.buy_strength, self.trading_pair)
                elif signal_type == "sell":
                    self.bot_manager.register_strategy_signal(bot_id, signal_type, self.strategy.sell_strength, self.trading_pair)
                elif signal_type == "neutral":
                    self.bot_manager.register_strategy_signal(bot_id, signal_type, 0.3, self.trading_pair)
                
                # Check for cross-strategy exit signals if we have a position
                strategy_position = self.strategy.get_status().get('position', None)
                if strategy_position is not None:
                    cross_strategy_exit = self.bot_manager.check_cross_strategy_exit(
                        bot_id,
                        strategy_position
                    )
                    
                    if cross_strategy_exit:
                        logger.info(f"Cross-strategy exit triggered for {strategy_name} by other strategies' opposing signals")
                        
                        # Exit long position
                        if strategy_position == "long":
                            logger.info(f"Executing cross-strategy exit for {strategy_name} LONG position")
                            # Use dual limit orders for signal reversal exits
                            is_forecast_reversal = False
                            if self.strategy.__class__.__name__ == 'ARIMAStrategy':
                                # Only ARIMA strategy has forecast_direction property
                                is_forecast_reversal = hasattr(self.strategy, 'forecast_direction') and self.strategy.forecast_direction == 'bearish'
                            if is_forecast_reversal:
                                logger.info(f"Using dual limit orders for cross-strategy forecast reversal exit from LONG position")
                                self._place_dual_limit_orders_for_reversal()
                            else:
                                # Use regular exit for other cases
                                self._place_sell_order(exit_only=True, cross_strategy_exit=True)
                        # Exit short position
                        elif strategy_position == "short":
                            logger.info(f"Executing cross-strategy exit for {strategy_name} SHORT position")
                            # Use dual limit orders for signal reversal exits
                            is_forecast_reversal = False
                            if self.strategy.__class__.__name__ == 'ARIMAStrategy':
                                # Only ARIMA strategy has forecast_direction property
                                is_forecast_reversal = hasattr(self.strategy, 'forecast_direction') and self.strategy.forecast_direction == 'bullish'
                            if is_forecast_reversal:
                                logger.info(f"Using dual limit orders for cross-strategy forecast reversal exit from SHORT position")
                                self._place_dual_limit_orders_for_reversal()
                            else:
                                # Use regular exit for other cases
                                self._place_buy_order(exit_only=True, cross_strategy_exit=True)
            
        except Exception as e:
            logger.error(f"Error updating signals: {e}")
    
    def _check_portfolio_request(self):
        """
        Check if there's a request for portfolio value and respond accordingly
        """
        request_file = "portfolio_request.txt"
        response_file = "portfolio_value.txt"
        
        if os.path.exists(request_file):
            try:
                # Get the request timestamp
                with open(request_file, "r") as f:
                    request_time = float(f.read().strip())
                
                # Only respond to requests that are less than 30 seconds old
                if time.time() - request_time < 30:
                    # Prepare portfolio data
                    portfolio_data = {
                        'value': self.portfolio_value,
                        'profit': self.total_profit,
                        'profit_percent': self.total_profit_percent,
                        'trade_count': self.trade_count,
                        'position': self.position,
                        'entry_price': self.entry_price if self.entry_price else 0,
                        'current_price': self.current_price if self.current_price else 0,
                        'timestamp': time.time()
                    }
                    
                    # Write response
                    with open(response_file, "w") as f:
                        f.write(json.dumps(portfolio_data))
                    
                    logger.info("Portfolio value request detected - data written to file")
                
                # Remove the request file
                os.remove(request_file)
                
            except Exception as e:
                logger.error(f"Error processing portfolio request: {e}")
                # Clean up the request file even if there was an error
                if os.path.exists(request_file):
                    os.remove(request_file)
    
    def _place_dual_limit_orders_for_reversal(self):
        """
        Place dual limit orders for signal reversal exits - one order slightly above
        and one slightly below the current price
        """
        from config import DUAL_LIMIT_ORDER_PRICE_OFFSET, DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT, ENABLE_DUAL_LIMIT_ORDERS
        
        # Skip if dual limit orders are disabled
        if not ENABLE_DUAL_LIMIT_ORDERS:
            logger.info("Dual limit orders are disabled in config; using regular exit instead")
            if self.position == "long":
                self._place_sell_order(exit_only=True)
            elif self.position == "short":
                self._place_buy_order(exit_only=True)
            return False
            
        if self.current_price is None:
            logger.warning("Cannot place dual limit orders: current price unknown")
            return False
        
        if self.position is None:
            logger.info("No active position to exit; skipping dual limit orders")
            return False
        
        # Set price offsets for the dual limit orders based on config
        # These are small offsets to ensure at least one gets filled in volatile conditions
        upper_price = self.current_price + DUAL_LIMIT_ORDER_PRICE_OFFSET
        lower_price = self.current_price - DUAL_LIMIT_ORDER_PRICE_OFFSET
        
        # Create tracking dictionary for the orders
        self.signal_reversal_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": upper_price,
            "lower_price": lower_price,
            "created_time": time.time(),
            "last_check_time": time.time(),
            "is_forecast_reversal": True,
            "failsafe_timeout": DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT
        }
        
        logger.info(f"【FORECAST REVERSAL】 Placing dual limit orders at ${upper_price:.4f} (above) and ${lower_price:.4f} (below)")
        
        # In sandbox mode, we simulate the orders
        if self.sandbox_mode:
            self.signal_reversal_orders["upper_order_id"] = "sandbox-upper-" + str(int(time.time()))
            self.signal_reversal_orders["lower_order_id"] = "sandbox-lower-" + str(int(time.time()))
            return True
        # For real trading, we would place the actual orders here with the exchange
        else:
            # Place upper price order
            if self.position == "long":
                # For long positions, sell at upper and lower prices
                upper_result = self._place_limit_sell_order(upper_price)
                lower_result = self._place_limit_sell_order(lower_price)
            else:  # short position
                # For short positions, buy at upper and lower prices
                upper_result = self._place_limit_buy_order(upper_price)
                lower_result = self._place_limit_buy_order(lower_price)
            
            if upper_result and lower_result:
                self.signal_reversal_orders["upper_order_id"] = upper_result
                self.signal_reversal_orders["lower_order_id"] = lower_result
                return True
            else:
                logger.error("Failed to place one or both dual limit orders for reversal")
                return False
    
    def _check_dual_reversal_orders(self):
        """
        Check and manage the dual limit orders placed for forecast reversals
        """
        if not self.signal_reversal_orders or not self.signal_reversal_orders.get("created_time"):
            return
        
        # Update the last check time
        self.signal_reversal_orders["last_check_time"] = time.time()
        
        # First, check if we're in sandbox mode
        if self.sandbox_mode:
            # In sandbox mode, simulate one of the orders being filled based on price
            if self.position == "long":
                # For long positions, check if price crosses upper or lower price
                if self.current_price >= self.signal_reversal_orders["upper_price"]:
                    logger.info(f"SANDBOX: Upper sell order filled at ${self.signal_reversal_orders['upper_price']:.4f}")
                    self._execute_sell(exit_only=True)
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling lower sell order at ${self.signal_reversal_orders['lower_price']:.4f}")
                    self._clear_signal_reversal_orders()
                    return
                elif self.current_price <= self.signal_reversal_orders["lower_price"]:
                    logger.info(f"SANDBOX: Lower sell order filled at ${self.signal_reversal_orders['lower_price']:.4f}")
                    self._execute_sell(exit_only=True)
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling upper sell order at ${self.signal_reversal_orders['upper_price']:.4f}")
                    self._clear_signal_reversal_orders()
                    return
            else:  # short position
                # For short positions, check if price crosses upper or lower price
                if self.current_price >= self.signal_reversal_orders["upper_price"]:
                    logger.info(f"SANDBOX: Upper buy order filled at ${self.signal_reversal_orders['upper_price']:.4f}")
                    self._execute_buy(exit_only=True)
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling lower buy order at ${self.signal_reversal_orders['lower_price']:.4f}")
                    self._clear_signal_reversal_orders()
                    return
                elif self.current_price <= self.signal_reversal_orders["lower_price"]:
                    logger.info(f"SANDBOX: Lower buy order filled at ${self.signal_reversal_orders['lower_price']:.4f}")
                    self._execute_buy(exit_only=True)
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling upper buy order at ${self.signal_reversal_orders['upper_price']:.4f}")
                    self._clear_signal_reversal_orders()
                    return
        else:
            # In real trading, check the order status from the exchange
            # This would involve API calls to check if either order is filled
            # If one is filled, cancel the other one
            pass
        
        # Check for failsafe timeout - if orders are not filled within the timeout,
        # cancel both and execute a market order
        elapsed_time = time.time() - self.signal_reversal_orders["created_time"]
        if elapsed_time > self.signal_reversal_orders["failsafe_timeout"]:
            logger.warning(f"Reversal orders failsafe timeout after {elapsed_time:.0f} seconds - executing market order")
            
            # Cancel both orders
            if not self.sandbox_mode:
                # Cancel the orders via exchange API
                pass
            
            # Execute market order
            if self.position == "long":
                logger.info("Executing market sell order as failsafe")
                self._execute_sell(exit_only=True)
            else:  # short position
                logger.info("Executing market buy order as failsafe")
                self._execute_buy(exit_only=True)
            
            self._clear_signal_reversal_orders()
    
    def _place_dual_limit_orders_for_entry(self, entry_type="long"):
        """
        Place dual limit orders for entry - one order slightly above
        and one slightly below the current price
        
        Args:
            entry_type (str): Type of entry, "long" or "short"
        """
        from config import DUAL_LIMIT_ENTRY_PRICE_OFFSET, DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT

        if self.current_price is None:
            logger.warning("Cannot place dual limit entry orders: current price unknown")
            return False
        
        if self.position is not None:
            logger.info(f"Already in {self.position} position; skipping dual limit entry orders")
            return False
        
        # Set price offsets for the dual limit orders based on config
        # These are small offsets to ensure at least one gets filled in volatile conditions
        upper_price = self.current_price + DUAL_LIMIT_ENTRY_PRICE_OFFSET
        lower_price = self.current_price - DUAL_LIMIT_ENTRY_PRICE_OFFSET
        
        # Calculate position size and risk
        position_size_usd = self.trade_quantity * self.current_price
        account_risk_percent = (position_size_usd / self.portfolio_value) * 100
        
        # Create tracking dictionary for the orders
        self.entry_limit_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": upper_price,
            "lower_price": lower_price,
            "created_time": time.time(),
            "last_check_time": time.time(),
            "entry_type": entry_type,
            "failsafe_timeout": DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT
        }
        
        logger.info(f"【DUAL LIMIT ENTRY】 Placing dual limit {entry_type.upper()} orders at ${upper_price:.4f} (above) and ${lower_price:.4f} (below)")
        logger.info(f"【POSITION】 Size: {self.trade_quantity} ({position_size_usd:.2f} USD) | Leverage: {LEVERAGE}x | Risk: {account_risk_percent:.2f}%")
        
        # In sandbox mode, we simulate the orders
        if self.sandbox_mode:
            self.entry_limit_orders["upper_order_id"] = "sandbox-upper-entry-" + str(int(time.time()))
            self.entry_limit_orders["lower_order_id"] = "sandbox-lower-entry-" + str(int(time.time()))
            return True
        # For real trading, we would place the actual orders here with the exchange
        else:
            if entry_type == "long":
                # For long entries, place buy orders at upper and lower prices
                upper_result = self._place_limit_buy_order(upper_price)
                lower_result = self._place_limit_buy_order(lower_price)
            else:  # short entry
                # For short entries, place sell orders at upper and lower prices
                upper_result = self._place_limit_sell_order(upper_price)
                lower_result = self._place_limit_sell_order(lower_price)
            
            if upper_result and lower_result:
                self.entry_limit_orders["upper_order_id"] = upper_result
                self.entry_limit_orders["lower_order_id"] = lower_result
                return True
            else:
                logger.error("Failed to place one or both dual limit entry orders")
                return False
    
    def _check_dual_entry_orders(self):
        """
        Check and manage the dual limit orders placed for entries
        """
        if not hasattr(self, 'entry_limit_orders') or not self.entry_limit_orders or not self.entry_limit_orders.get("created_time"):
            return
        
        # Update the last check time
        self.entry_limit_orders["last_check_time"] = time.time()
        
        # First, check if we're in sandbox mode
        if self.sandbox_mode:
            # In sandbox mode, simulate one of the orders being filled based on price
            entry_type = self.entry_limit_orders.get("entry_type", "long")
            
            if entry_type == "long":
                # For long entries, check if price crosses upper or lower price
                if self.current_price >= self.entry_limit_orders["upper_price"]:
                    logger.info(f"SANDBOX: Upper buy order filled at ${self.entry_limit_orders['upper_price']:.4f}")
                    self._execute_buy()
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling lower buy order at ${self.entry_limit_orders['lower_price']:.4f}")
                    self._clear_entry_limit_orders()
                    return
                elif self.current_price <= self.entry_limit_orders["lower_price"]:
                    logger.info(f"SANDBOX: Lower buy order filled at ${self.entry_limit_orders['lower_price']:.4f}")
                    self._execute_buy()
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling upper buy order at ${self.entry_limit_orders['upper_price']:.4f}")
                    self._clear_entry_limit_orders()
                    return
            else:  # short entry
                # For short entries, check if price crosses upper or lower price
                if self.current_price >= self.entry_limit_orders["upper_price"]:
                    logger.info(f"SANDBOX: Upper sell order filled at ${self.entry_limit_orders['upper_price']:.4f}")
                    self._execute_sell()
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling lower sell order at ${self.entry_limit_orders['lower_price']:.4f}")
                    self._clear_entry_limit_orders()
                    return
                elif self.current_price <= self.entry_limit_orders["lower_price"]:
                    logger.info(f"SANDBOX: Lower sell order filled at ${self.entry_limit_orders['lower_price']:.4f}")
                    self._execute_sell()
                    # Cancel the other order
                    logger.info(f"SANDBOX: Cancelling upper sell order at ${self.entry_limit_orders['upper_price']:.4f}")
                    self._clear_entry_limit_orders()
                    return
        else:
            # In real trading, check the order status from the exchange
            # This would involve API calls to check if either order is filled
            # If one is filled, cancel the other one
            pass
        
        # Check for failsafe timeout - if orders are not filled within the timeout,
        # cancel both and execute a market order
        elapsed_time = time.time() - self.entry_limit_orders["created_time"]
        if elapsed_time > self.entry_limit_orders["failsafe_timeout"]:
            logger.warning(f"Entry orders failsafe timeout after {elapsed_time:.0f} seconds - executing market order")
            
            # Cancel both orders
            if not self.sandbox_mode:
                # Cancel the orders via exchange API
                pass
            
            # Execute market order
            entry_type = self.entry_limit_orders.get("entry_type", "long")
            if entry_type == "long":
                logger.info("Executing market buy order as failsafe for entry")
                self._execute_buy()
            else:  # short entry
                logger.info("Executing market sell order as failsafe for entry")
                self._execute_sell()
            
            self._clear_entry_limit_orders()
            
    def _clear_entry_limit_orders(self):
        """
        Clear the entry limit orders tracking
        """
        from config import DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT
        self.entry_limit_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": None,
            "lower_price": None,
            "created_time": None,
            "last_check_time": None,
            "entry_type": None,
            "failsafe_timeout": DUAL_LIMIT_ENTRY_FAILSAFE_TIMEOUT
        }

    def _clear_signal_reversal_orders(self):
        """
        Clear the signal reversal orders tracking
        """
        from config import DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT
        self.signal_reversal_orders = {
            "upper_order_id": None,
            "lower_order_id": None,
            "upper_price": None,
            "lower_price": None,
            "created_time": None,
            "last_check_time": None,
            "is_forecast_reversal": False,
            "failsafe_timeout": DUAL_LIMIT_ORDER_FAILSAFE_TIMEOUT
        }
    
    def _place_limit_sell_order(self, price):
        """
        Place a limit sell order with the exchange
        
        Args:
            price (float): Limit price for the order
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        # In a real implementation, this would call the exchange API
        # For now, return a dummy order ID
        return f"limit-sell-{int(time.time())}"
    
    def _place_limit_buy_order(self, price):
        """
        Place a limit buy order with the exchange
        
        Args:
            price (float): Limit price for the order
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        # In a real implementation, this would call the exchange API
        # For now, return a dummy order ID
        return f"limit-buy-{int(time.time())}"
    
    def _check_orders(self):
        """
        Check and manage pending orders and trailing stops
        """
        if not self.running or self.current_price is None:
            return
        
        try:
            # Check for cross-strategy exit signals if we have a position
            if self.position and hasattr(self, 'bot_manager') and self.bot_manager:
                bot_id = f"{self.strategy.__class__.__name__}-{self.trading_pair}"
                if self.bot_manager.check_cross_strategy_exit(bot_id, self.position):
                    logger.info(f"Cross-strategy exit signal detected for {self.position} position")
                    if self.position == 'long':
                        self._place_sell_order(exit_only=True, cross_strategy_exit=True)
                    elif self.position == 'short':
                        self._place_buy_order(exit_only=True, cross_strategy_exit=True)
            
            # Check dual reversal orders first
            if self.signal_reversal_orders.get("created_time") is not None:
                self._check_dual_reversal_orders()
                return  # Skip other checks if we have active reversal orders
            
            # Check dual entry orders next
            if hasattr(self, 'entry_limit_orders') and self.entry_limit_orders and self.entry_limit_orders.get("created_time") is not None:
                self._check_dual_entry_orders()
                return  # Skip other checks if we have active entry orders
            
            # Check if pending order is filled based on live price
            if self.pending_order is not None:
                # If ENTRY_ATR_MULTIPLIER is 0, execute immediately as a market order
                if ENTRY_ATR_MULTIPLIER == 0 and self.pending_order["type"] == "buy":
                    logger.info(f"Executing market buy at current price ${self.current_price:.4f}")
                    self._execute_buy()
                    self.pending_order = None
                    return
                
                # Check for timeout for limit orders
                if time.time() - self.pending_order["time"] > ORDER_TIMEOUT_SECONDS:
                    logger.info(f"Pending {self.pending_order['type']} order timed out")
                    self.pending_order = None
                    return
                
                # Check if price conditions are met for limit orders
                if self.pending_order["type"] == "buy" and self.current_price <= self.pending_order["price"]:
                    logger.info(f"Buy limit order filled at {self.pending_order['price']:.4f}")
                    self._execute_buy()
                    self.pending_order = None
                
                elif self.pending_order["type"] == "sell" and self.current_price >= self.pending_order["price"]:
                    logger.info(f"Sell limit order filled at {self.pending_order['price']:.4f}")
                    self._execute_sell()
                    self.pending_order = None
            
            # Check trailing stop for long position (in-memory tracking only)
            if self.position == "long" and self.trailing_stop_order is not None:
                if self.current_price <= self.trailing_stop_order["price"] and self.trailing_stop_limit_order_id is None:
                    logger.info(f"Trailing stop triggered at {self.trailing_stop_order['price']:.4f}")
                    self._place_sell_order(exit_only=True, cross_strategy_exit=False)
                    self.trailing_stop_order = None
            
            # Check breakeven exit for long position
            if self.position == "long" and self.breakeven_order is not None:
                if self.current_price >= self.breakeven_order["price"]:
                    logger.info(f"Breakeven exit triggered at {self.breakeven_order['price']:.4f}")
                    self._place_sell_order(exit_only=True, cross_strategy_exit=False)
                    self.breakeven_order = None
            
            # Check breakeven exit for short position
            if self.position == "short" and self.breakeven_order is not None:
                if self.current_price <= self.breakeven_order["price"]:
                    logger.info(f"Short position breakeven exit triggered at {self.breakeven_order['price']:.4f}")
                    self._place_buy_order(exit_only=True, cross_strategy_exit=False)
                    self.breakeven_order = None
            
            # Check trailing stop for short position (in-memory tracking only)
            if self.position == "short" and self.trailing_stop_order is not None:
                if self.current_price >= self.trailing_stop_order["price"] and self.trailing_stop_limit_order_id is None:
                    logger.info(f"Short position trailing stop triggered at {self.trailing_stop_order['price']:.4f}")
                    self._place_buy_order(exit_only=True, cross_strategy_exit=False)
                    self.trailing_stop_order = None
            
            # Update trailing stop based on current price for long position
            if self.position == "long" and self.trailing_max_price is not None and self.current_atr is not None:
                new_stop_price = self.trailing_max_price - (2.0 * self.current_atr)
                
                # Initialize trailing stop if none exists
                if self.trailing_stop_order is None:
                    self.trailing_stop_order = {
                        "type": "sell", 
                        "price": new_stop_price, 
                        "time": time.time()
                    }
                    logger.info(f"Setting trailing stop at {new_stop_price:.4f}")
                    
                    # Place actual trailing stop limit order if not in sandbox mode
                    if not self.sandbox_mode:
                        self._place_trailing_stop_limit_order(new_stop_price)
                
                # Only move stop up, never down
                elif new_stop_price > self.trailing_stop_order["price"]:
                    old_price = self.trailing_stop_order["price"]
                    self.trailing_stop_order["price"] = new_stop_price
                    logger.info(f"Updated trailing stop from {old_price:.4f} to {new_stop_price:.4f}")
                    
                    # Update actual limit order in the exchange
                    if not self.sandbox_mode and self.trailing_stop_limit_order_id is not None:
                        self._update_trailing_stop_limit_order(new_stop_price)
            
            # Update trailing stop based on current price for short position
            if self.position == "short" and self.trailing_min_price is not None and self.current_atr is not None:
                new_stop_price = self.trailing_min_price + (2.0 * self.current_atr)
                
                # Initialize trailing stop if none exists
                if self.trailing_stop_order is None:
                    self.trailing_stop_order = {
                        "type": "buy", 
                        "price": new_stop_price, 
                        "time": time.time()
                    }
                    logger.info(f"Setting short position trailing stop at {new_stop_price:.4f}")
                    
                    # Place actual trailing stop limit order if not in sandbox mode
                    if not self.sandbox_mode:
                        self._place_trailing_stop_limit_order(new_stop_price)
                
                # Only move stop down, never up
                elif new_stop_price < self.trailing_stop_order["price"]:
                    old_price = self.trailing_stop_order["price"]
                    self.trailing_stop_order["price"] = new_stop_price
                    logger.info(f"Updated short position trailing stop from {old_price:.4f} to {new_stop_price:.4f}")
                    
                    # Update actual limit order in the exchange
                    if not self.sandbox_mode and self.trailing_stop_limit_order_id is not None:
                        self._update_trailing_stop_limit_order(new_stop_price)
        
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
    
    def _place_buy_order(self, exit_only=False, cross_strategy_exit=False):
        """
        Place a buy limit order based on ATR
        
        Args:
            exit_only (bool): Whether this is just to exit a short position
            cross_strategy_exit (bool): Whether this is triggered by a cross-strategy exit signal
        """
        if self.current_price is None or self.current_atr is None:
            logger.warning("Cannot place buy order: price or ATR unknown")
            return
        
        if exit_only and self.position != "short":
            logger.info("No short position to exit; skipping buy order")
            return
            
        if not exit_only and self.position is not None:
            logger.info(f"Already in {self.position} position; skipping buy order")
            return
        
        # Calculate price for buy order
        if ENTRY_ATR_MULTIPLIER > 0 and not exit_only:
            # Use limit price based on ATR if multiplier is positive
            limit_price = self.current_price - (self.current_atr * ENTRY_ATR_MULTIPLIER)
            order_type = "limit"
        else:
            # Use current price if multiplier is 0 (no offset) or exiting position
            limit_price = self.current_price
            order_type = "market"
        
        # Calculate position size and risk
        position_size_usd = self.trade_quantity * self.current_price
        account_risk_percent = (position_size_usd / self.portfolio_value) * 100
        
        # Create pending order
        self.pending_order = {
            "type": "buy",
            "price": limit_price,
            "time": time.time(),
            "exit_only": exit_only
        }
        
        logger.info(f"【ORDER】 {'EXIT SHORT' if exit_only else 'LONG'} - Placed buy {order_type} order at ${limit_price:.4f}")
        logger.info(f"【POSITION】 Size: {self.trade_quantity} ({position_size_usd:.2f} USD) | Leverage: {LEVERAGE}x | Risk: {account_risk_percent:.2f}%")
        
        # Execute immediately for exit orders
        if exit_only:
            self._execute_buy(exit_only=True, cross_strategy_exit=cross_strategy_exit)
            self.pending_order = None
    
    def _place_short_order(self):
        """
        Place a short sell order based on ATR
        """
        if self.current_price is None or self.current_atr is None:
            logger.warning("Cannot place short order: price or ATR unknown")
            return
        
        if self.position is not None:
            logger.info(f"Already in {self.position} position; skipping short order")
            return
        
        # Calculate price for short order
        # If ENTRY_ATR_MULTIPLIER is positive, sell price is slightly above current price
        # If ENTRY_ATR_MULTIPLIER is 0, use market order at current price
        if ENTRY_ATR_MULTIPLIER > 0:
            # Use limit price based on ATR if multiplier is positive
            limit_price = self.current_price + (self.current_atr * ENTRY_ATR_MULTIPLIER)
            order_type = "limit"
        else:
            # Use current price if multiplier is 0 (no offset)
            limit_price = self.current_price
            order_type = "market"
        
        # Calculate position size and risk
        position_size_usd = self.trade_quantity * self.current_price
        account_risk_percent = (position_size_usd / self.portfolio_value) * 100
        
        # Create pending order
        self.pending_order = {
            "type": "sell",
            "price": limit_price,
            "time": time.time(),
            "exit_only": False
        }
        
        logger.info(f"【ORDER】 SHORT - Placed sell {order_type} order at ${limit_price:.4f}")
        logger.info(f"【POSITION】 Size: {self.trade_quantity} ({position_size_usd:.2f} USD) | Leverage: {LEVERAGE}x | Risk: {account_risk_percent:.2f}%")
        
        # Execute immediately
        self._execute_sell(exit_only=False)
        self.pending_order = None
    
    def _place_sell_order(self, exit_only=False, cross_strategy_exit=False):
        """
        Place a sell limit order
        
        Args:
            exit_only (bool): Whether this is just to exit a position
            cross_strategy_exit (bool): Whether this is triggered by a cross-strategy exit signal
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
        
        # Use ATR offset for limit orders if ATR is available and not an exit order
        if self.current_atr is not None and ENTRY_ATR_MULTIPLIER > 0 and not exit_only:
            # For short entries, use a limit price slightly above the current price
            limit_price = self.current_price + (self.current_atr * ENTRY_ATR_MULTIPLIER)
            order_type = "limit"
        else:
            # For exit orders or if no ATR available, use current price
            limit_price = self.current_price
            order_type = "market" if exit_only else "limit"
        
        # Calculate position size and risk if entering a new position
        if not exit_only:
            position_size_usd = self.trade_quantity * self.current_price
            account_risk_percent = (position_size_usd / self.portfolio_value) * 100
            logger.info(f"【POSITION】 Size: {self.trade_quantity} ({position_size_usd:.2f} USD) | Leverage: {self.leverage}x | Risk: {account_risk_percent:.2f}%")
        
        # Create pending order
        self.pending_order = {
            "type": "sell",
            "price": limit_price,
            "time": time.time(),
            "exit_only": exit_only
        }
        
        logger.info(f"【ORDER】 {'EXIT LONG' if exit_only else 'SHORT'} - Placed sell {order_type} order at ${limit_price:.4f}")
        
        # Execute immediately for exit orders
        if exit_only:
            self._execute_sell(exit_only=True, cross_strategy_exit=cross_strategy_exit)
            self.pending_order = None
    
    def _calculate_dynamic_margin_percent(self, signal_strength=0.0):
        """
        Calculate dynamic margin percentage based on signal strength
        
        Args:
            signal_strength (float): Strength of the signal (0.0 to 1.0)
            
        Returns:
            float: Calculated margin percentage
        """
        from dynamic_position_sizing import calculate_dynamic_margin_percent
        
        # Use the centralized function from dynamic_position_sizing module
        return calculate_dynamic_margin_percent(signal_strength, self.margin_percent)
            
    def _execute_buy(self, exit_only=False, cross_strategy_exit=False, signal_strength=0.0):
        """
        Execute buy order
        
        Args:
            exit_only (bool): Whether this is just to exit a short position
            cross_strategy_exit (bool): Whether this is triggered by a cross-strategy exit signal
            signal_strength (float): Strength of the signal (0.0 to 1.0)
        """
        if not self.current_price:
            logger.warning("Cannot execute buy: current price unknown")
            return
        
        logger.info(f"【EXECUTE】 {'EXIT SHORT' if exit_only else 'LONG'} - Buy order at ${self.current_price:.4f}")
        
        # Save previous position for tracking position changes
        previous_position = self.position
        
        # Calculate volatility stop price for notifications only
        stop_price = 0
        if self.current_atr:
            stop_price = self.current_price - (self.current_atr * 2.0)
        
        # Use fixed allocation percentages for position sizing (25% for Adaptive, 10% for ARIMA)
        # Always use portfolio value directly instead of available funds to ensure consistent allocation
        funds_to_use = self.portfolio_value
        logger.info(f"Using portfolio value: ${funds_to_use:.2f} with fixed margin percentage: {self.margin_percent*100}%")
            
        # Calculate position size using dynamic margin percent based on signal strength
        dynamic_margin_percent = self._calculate_dynamic_margin_percent(signal_strength)
        margin_amount = funds_to_use * dynamic_margin_percent
        notional = margin_amount * self.leverage
        logger.info(f"Position sizing (dynamic): Portfolio: ${funds_to_use:.2f}, Margin: ${margin_amount:.2f} ({dynamic_margin_percent*100:.1f}%), Leverage: {self.leverage}x")
        quantity = notional / self.current_price
        
        # Track strategy name in logs
        strategy_name = self.strategy.__class__.__name__
        logger.info(f"Strategy {strategy_name} executing buy order")
        
        # Execute order only if not in sandbox mode
        if not self.sandbox_mode:
            try:
                # Use the price from the pending order if it exists, which is a limit price
                # with the small ATR offset. Otherwise, use a market order at current price.
                if self.pending_order and 'price' in self.pending_order:
                    limit_price = self.pending_order['price']
                    order_result = self.api.place_order(
                        pair=self.trading_pair,
                        type_="buy",
                        ordertype="limit",
                        price=str(limit_price),
                        volume=str(quantity),
                        leverage=str(self.leverage)
                    )
                    logger.info(f"Buy limit order placed at ${limit_price:.4f} with {ENTRY_ATR_MULTIPLIER} ATR offset")
                else:
                    # Fallback to market order if no pending order price
                    order_result = self.api.place_order(
                        pair=self.trading_pair,
                        type_="buy",
                        ordertype="market",
                        volume=str(quantity),
                        leverage=str(self.leverage)
                    )
                
                logger.info(f"Buy order executed: {order_result}")
                
                # Get previous position before updating
                previous_position = self.position
                
                # Update position status
                # If we're using a limit order, the entry price will be the limit price, not current price
                entry_price = self.pending_order['price'] if self.pending_order and 'price' in self.pending_order else self.current_price
                self.position = "long"
                self.entry_price = entry_price
                self.trailing_max_price = entry_price
                self.trailing_min_price = None
                
                # Update strategy position
                self.strategy.update_position("long", entry_price)
                
                # Get strategy type for logs and records
                strategy_name = self.strategy.__class__.__name__
                
                # Notify BotManager about position change
                if self.bot_manager is not None:
                    # Get strategy type and pair for bot ID
                    bot_id = f"{strategy_name.lower()}-{self.original_pair}"
                    self.bot_manager.track_position_change(
                        bot_id=bot_id,
                        new_position=self.position, 
                        previous_position=previous_position,
                        margin_percent=self.margin_percent
                    )
                    logger.info(f"Notified BotManager of position change for {strategy_name}: {previous_position} -> long")
                
                # Record the trade with strategy information
                record_trade(
                    "buy", 
                    self.trading_pair, 
                    quantity, 
                    self.current_price, 
                    profit=None, 
                    profit_percent=None, 
                    position="long", 
                    strategy=strategy_name
                )
                
                # Send trade entry notification
                if self.current_atr:
                    stop_price = self.current_price - (self.current_atr * 2.0)
                    send_trade_entry_notification(
                        self.trading_pair, 
                        self.current_price, 
                        quantity, 
                        self.current_atr, 
                        stop_price
                    )
            
            except Exception as e:
                logger.error(f"Error executing buy order: {e}")
        else:
            logger.info(f"SANDBOX MODE: Buy order for {quantity:.6f} units would be executed at {self.current_price:.4f}")
            
            # Simulate position status update in sandbox mode
            previous_position = self.position
            self.position = "long"
            self.entry_price = self.current_price
            self.trailing_max_price = self.current_price
            self.trailing_min_price = None
            self.strategy.update_position("long", self.current_price)
            
            # Update bot manager about position change
            if self.bot_manager is not None:
                # Get strategy type and pair for bot ID
                bot_id = f"{self.strategy.__class__.__name__}-{self.original_pair}"
                self.bot_manager.track_position_change(
                    bot_id=bot_id,
                    new_position=self.position,
                    previous_position=previous_position,
                    margin_percent=self.margin_percent
                )
                # Get strategy name for logs and tracking
                strategy_name = self.strategy.__class__.__name__
                logger.info(f"SANDBOX: Notified BotManager of position change for {strategy_name}: {previous_position} -> long")
            
            # Record the trade with strategy information
            record_trade(
                "buy", 
                self.trading_pair, 
                quantity, 
                self.current_price, 
                profit=None, 
                profit_percent=None, 
                position="long",
                strategy=strategy_name
            )
    
    def _execute_sell(self, exit_only=False, cross_strategy_exit=False):
        """
        Execute sell order
        
        Args:
            exit_only (bool): Whether this is just to exit a position
            cross_strategy_exit (bool): Whether this is triggered by a cross-strategy exit signal
        """
        if not self.current_price:
            logger.warning("Cannot execute sell: current price unknown")
            return
        
        # Save previous position for tracking position changes
        previous_position = self.position
        
        # Track strategy name in logs
        strategy_name = self.strategy.__class__.__name__
        logger.info(f"【EXECUTE】 {'EXIT LONG' if exit_only else 'SHORT'} - Sell order at ${self.current_price:.4f}")
        logger.info(f"Strategy {strategy_name} executing sell order")
        
        # If exiting a long position
        if exit_only and self.position == "long":
            # Calculate actual trade quantity based on entry using strategy-specific settings
            margin_amount = self.portfolio_value * self.margin_percent
            notional = margin_amount * self.leverage
            quantity = notional / self.entry_price
            logger.info(f"Position sizing for exit: Portfolio: ${self.portfolio_value:.2f}, Margin: ${margin_amount:.2f} ({self.margin_percent*100}%), Leverage: {self.leverage}x")
            
            # Calculate profit
            trade_profit = (self.current_price - self.entry_price) * quantity
            profit_percent = (trade_profit / margin_amount) * 100.0
            
            # Execute order only if not in sandbox mode
            if not self.sandbox_mode:
                try:
                    # Use the price from the pending order if it exists, which is a limit price
                    # with the small ATR offset. Otherwise, use a market order at current price.
                    if self.pending_order and 'price' in self.pending_order:
                        limit_price = self.pending_order['price']
                        order_result = self.api.place_order(
                            pair=self.trading_pair,
                            type_="sell",
                            ordertype="limit",
                            price=str(limit_price),
                            volume=str(quantity),
                            leverage=str(self.leverage)
                        )
                        logger.info(f"Exit long limit order placed at ${limit_price:.4f} with {ENTRY_ATR_MULTIPLIER} ATR offset")
                    else:
                        # Fallback to market order if no pending order price
                        order_result = self.api.place_order(
                            pair=self.trading_pair,
                            type_="sell",
                            ordertype="market",
                            volume=str(quantity),
                            leverage=str(self.leverage)
                        )
                    
                    logger.info(f"Sell order executed: {order_result}")
                    
                    # Update portfolio
                    self.portfolio_value += trade_profit
                    self.total_profit += trade_profit
                    self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
                    self.trade_count += 1
                    
                    # Update shared portfolio via BotManager if available
                    if self.bot_manager is not None:
                        # Get strategy type and pair for bot ID
                        bot_id = f"{self.strategy.__class__.__name__}-{self.original_pair}"
                        self.bot_manager.update_portfolio(
                            bot_id=bot_id,
                            new_portfolio_value=self.portfolio_value,
                            trade_profit=trade_profit
                        )
                        logger.info(f"Updated shared portfolio via BotManager, new value: ${self.portfolio_value:.2f}")
                    
                    # Clear any trailing stop limit orders
                    if self.trailing_stop_limit_order_id is not None:
                        try:
                            cancel_result = self.api.cancel_order(self.trailing_stop_limit_order_id)
                            logger.info(f"Cancelled trailing stop order: {self.trailing_stop_limit_order_id}")
                        except Exception as e:
                            logger.error(f"Error cancelling trailing stop order: {e}")
                    
                    # Get previous position before updating
                    previous_position = self.position
                    
                    # Update position status
                    self.position = None
                    self.entry_price = None
                    self.trailing_max_price = None
                    self.trailing_min_price = None
                    self.trailing_stop_limit_order_id = None
                    self.strategy.update_position(None, None)
                    
                    # Get strategy name for logs and tracking
                    strategy_name = self.strategy.__class__.__name__
                    
                    # Notify BotManager about position change
                    if self.bot_manager is not None:
                        # Get strategy type and pair for bot ID
                        bot_id = f"{strategy_name.lower()}-{self.original_pair}"
                        self.bot_manager.track_position_change(
                            bot_id=bot_id,
                            new_position=self.position, 
                            previous_position=previous_position,
                            margin_percent=self.margin_percent
                        )
                        logger.info(f"Notified BotManager of position change for {strategy_name}: {previous_position} -> none")
                    
                    # Record the trade with strategy information
                    record_trade(
                        "sell", 
                        self.trading_pair, 
                        quantity, 
                        self.current_price, 
                        profit=trade_profit, 
                        profit_percent=profit_percent, 
                        position="none",
                        strategy=strategy_name
                    )
                
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
                
                # Update shared portfolio via BotManager if available (in sandbox mode too)
                if self.bot_manager is not None:
                    # Get strategy type and pair for bot ID
                    bot_id = f"{self.strategy.__class__.__name__}-{self.original_pair}"
                    self.bot_manager.update_portfolio(
                        bot_id=bot_id,
                        new_portfolio_value=self.portfolio_value,
                        trade_profit=trade_profit
                    )
                    logger.info(f"SANDBOX: Updated shared portfolio via BotManager, new value: ${self.portfolio_value:.2f}")
                
                # Get previous position before updating
                previous_position = self.position
                
                # Update position status
                self.position = None
                self.entry_price = None
                self.trailing_max_price = None
                self.trailing_min_price = None
                self.trailing_stop_limit_order_id = None
                self.strategy.update_position(None, None)
                
                # Get strategy name for logs and tracking
                strategy_name = self.strategy.__class__.__name__
                
                # Notify BotManager about position change in sandbox mode
                if self.bot_manager is not None:
                    # Get strategy type and pair for bot ID
                    bot_id = f"{strategy_name.lower()}-{self.original_pair}"
                    self.bot_manager.track_position_change(
                        bot_id=bot_id,
                        new_position=self.position, 
                        previous_position=previous_position,
                        margin_percent=self.margin_percent
                    )
                    logger.info(f"SANDBOX: Notified BotManager of position change for {strategy_name}: {previous_position} -> none")
                
                # Record the trade with strategy information
                record_trade(
                    "sell", 
                    self.trading_pair, 
                    quantity, 
                    self.current_price, 
                    profit=trade_profit, 
                    profit_percent=profit_percent, 
                    position="none",
                    strategy=strategy_name
                )
                
                # Send trade exit notification in sandbox mode too
                send_trade_exit_notification(
                    self.trading_pair, 
                    self.current_price, 
                    self.entry_price, 
                    quantity, 
                    trade_profit,
                    profit_percent,
                    self.portfolio_value,
                    self.total_profit,
                    self.total_profit_percent,
                    self.trade_count
                )
        # Entering a short position
        elif not exit_only:
            # Use fixed allocation percentages for position sizing (25% for Adaptive, 10% for ARIMA)
            # Always use portfolio value directly instead of available funds to ensure consistent allocation
            funds_to_use = self.portfolio_value
            logger.info(f"Using portfolio value: ${funds_to_use:.2f} with fixed margin percentage: {self.margin_percent*100}%")
                
            # Calculate position size using fixed percentage of portfolio
            margin_amount = funds_to_use * self.margin_percent
            notional = margin_amount * self.leverage
            logger.info(f"Position sizing (fixed): Portfolio: ${funds_to_use:.2f}, Margin: ${margin_amount:.2f} ({self.margin_percent*100}%), Leverage: {self.leverage}x")
            quantity = notional / self.current_price
            
            # Execute order only if not in sandbox mode
            if not self.sandbox_mode:
                try:
                    # Use the price from the pending order if it exists, which is a limit price
                    # with the small ATR offset. Otherwise, use a market order at current price.
                    if self.pending_order and 'price' in self.pending_order:
                        limit_price = self.pending_order['price']
                        order_result = self.api.place_order(
                            pair=self.trading_pair,
                            type_="sell",
                            ordertype="limit",
                            price=str(limit_price),
                            volume=str(quantity),
                            leverage=str(self.leverage)
                        )
                        logger.info(f"Short sell limit order placed at ${limit_price:.4f} with {ENTRY_ATR_MULTIPLIER} ATR offset")
                    else:
                        # Fallback to market order if no pending order price
                        order_result = self.api.place_order(
                            pair=self.trading_pair,
                            type_="sell",
                            ordertype="market",
                            volume=str(quantity),
                            leverage=str(self.leverage)
                        )
                    
                    logger.info(f"Short sell order executed: {order_result}")
                    
                    # Update position status
                    # If we're using a limit order, the entry price will be the limit price, not current price
                    entry_price = self.pending_order['price'] if self.pending_order and 'price' in self.pending_order else self.current_price
                    self.position = "short"
                    self.entry_price = entry_price
                    self.trailing_max_price = None
                    self.trailing_min_price = entry_price
                    self.strategy.update_position("short", entry_price)
                    
                    # Get strategy name for logs and tracking
                    strategy_name = self.strategy.__class__.__name__
                    
                    # Record the trade with strategy information
                    record_trade(
                        "sell", 
                        self.trading_pair, 
                        quantity, 
                        self.current_price, 
                        profit=None, 
                        profit_percent=None, 
                        position="short",
                        strategy=strategy_name
                    )
                    
                    # Send trade entry notification
                    if self.current_atr:
                        stop_price = self.current_price + (self.current_atr * 2.0)
                        send_trade_entry_notification(
                            self.trading_pair, 
                            self.current_price, 
                            quantity, 
                            self.current_atr, 
                            stop_price,
                            position_type="short"
                        )
                
                except Exception as e:
                    logger.error(f"Error executing short sell order: {e}")
            else:
                logger.info(f"SANDBOX MODE: Short sell order for {quantity:.6f} units would be executed at {self.current_price:.4f}")
                
                # Simulate position status update in sandbox mode
                previous_position = self.position
                self.position = "short"
                self.entry_price = self.current_price
                self.trailing_max_price = None
                self.trailing_min_price = self.current_price
                self.strategy.update_position("short", self.current_price)
                
                # Get strategy name for logs and tracking
                strategy_name = self.strategy.__class__.__name__
                
                # Update bot manager about position change
                if self.bot_manager is not None:
                    # Get strategy type and pair for bot ID
                    bot_id = f"{strategy_name.lower()}-{self.original_pair}"
                    self.bot_manager.track_position_change(
                        bot_id=bot_id,
                        new_position=self.position,
                        previous_position=previous_position,
                        margin_percent=self.margin_percent
                    )
                    logger.info(f"SANDBOX: Notified BotManager of position change for {strategy_name}: {previous_position} -> short")
                
                # Record the trade with strategy information
                record_trade(
                    "sell", 
                    self.trading_pair, 
                    quantity, 
                    self.current_price, 
                    profit=None, 
                    profit_percent=None, 
                    position="short",
                    strategy=strategy_name
                )
        else:
            logger.warning("Invalid sell order configuration")
    
    def _place_trailing_stop_limit_order(self, stop_price):
        """
        Place a trailing stop limit order at the specified price
        
        Args:
            stop_price (float): Price at which to place the stop limit order
        """
        try:
            # Only proceed if we have a valid position
            if self.position != "long" or self.entry_price is None:
                logger.warning("Cannot place trailing stop: no long position exists")
                return
            
            # Use fixed allocation percentages for position sizing (25% for Adaptive, 10% for ARIMA)
            # Always use portfolio value directly to ensure consistent allocation
            funds_to_use = self.portfolio_value
            logger.info(f"Using portfolio value: ${funds_to_use:.2f} with fixed margin percentage: {self.margin_percent*100}%")
            
            # Calculate position size using fixed percentage of portfolio
            margin_amount = funds_to_use * self.margin_percent
            notional = margin_amount * self.leverage
            logger.info(f"Position sizing (fixed): Portfolio: ${funds_to_use:.2f}, Margin: ${margin_amount:.2f} ({self.margin_percent*100}%), Leverage: {self.leverage}x")
            quantity = notional / self.entry_price
            
            # Place the stop limit order with a small offset for guaranteed execution
            execution_price = stop_price * 0.999  # Slightly lower price to ensure execution
            
            # Place the order
            order_result = self.api.place_order(
                pair=self.trading_pair,
                type_="sell",
                ordertype="stop-loss",  # Using stop-loss type for trailing stop
                price=execution_price,  # Limit price
                volume=str(quantity),   # Order volume
                leverage=str(self.leverage)  # Using strategy-specific leverage
            )
            
            # Store the order ID for future reference
            if 'txid' in order_result and len(order_result['txid']) > 0:
                self.trailing_stop_limit_order_id = order_result['txid'][0]
                logger.info(f"Placed trailing stop limit order at {stop_price:.4f}, execution price {execution_price:.4f}, ID: {self.trailing_stop_limit_order_id}")
            else:
                logger.error(f"Failed to place trailing stop limit order: {order_result}")
        
        except Exception as e:
            logger.error(f"Error placing trailing stop limit order: {e}")
    
    def _update_trailing_stop_limit_order(self, new_stop_price):
        """
        Cancel the existing trailing stop limit order and place a new one at the updated price
        
        Args:
            new_stop_price (float): New price for the trailing stop
        """
        try:
            # Only proceed if we have a valid order ID
            if self.trailing_stop_limit_order_id is None:
                logger.warning("Cannot update trailing stop: no order ID exists")
                return
            
            # Cancel the existing order
            cancel_result = self.api.cancel_order(self.trailing_stop_limit_order_id)
            logger.info(f"Cancelled previous trailing stop order: {self.trailing_stop_limit_order_id}")
            
            # Reset the order ID
            self.trailing_stop_limit_order_id = None
            
            # Place a new order at the updated price
            self._place_trailing_stop_limit_order(new_stop_price)
            
        except Exception as e:
            logger.error(f"Error updating trailing stop limit order: {e}")
            # If cancellation fails, we should still try to place a new order
            self._place_trailing_stop_limit_order(new_stop_price)
    
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
        # Subscribe to ticker first as it's the most important for price updates
        logger.info(f"Subscribing to ticker for {self.trading_pair}")
        self.ws.subscribe_ticker([self.trading_pair], self._handle_ticker_update)
        
        # Subscribe to OHLC for strategy updates
        logger.info(f"Subscribing to OHLC for {self.trading_pair}")
        self.ws.subscribe_ohlc([self.trading_pair], self._handle_ohlc_update, 1)
        
        # Subscribe to trades for real-time price updates
        logger.info(f"Subscribing to trades for {self.trading_pair}")
        self.ws.subscribe_trades([self.trading_pair], self._handle_trade_update)
        
        # Commenting out order book subscription as it generates too many messages
        # and overwhelms the more important ticker and trade messages
        # logger.info(f"Subscribing to order book for {self.trading_pair}")
        # self.ws.subscribe_book([self.trading_pair], self._handle_book_update, 10)
        
        # Subscribe to private data streams
        self.ws.subscribe_own_trades(self._handle_own_trades)
        self.ws.subscribe_open_orders(self._handle_open_orders)
        
        # Check current position
        self.check_position()
        
        # Fetch initial price data
        try:
            # Request at least 100 candles for better signal calculation
            # Interval = 1 minute for more granular data
            # Use original_pair for REST API calls since it expects a different format
            logger.info(f"Fetching OHLC data for {self.original_pair}")
            ohlc_data = self.api.get_ohlc(self.original_pair, 1)
            pair_key = list(ohlc_data.keys())[0]  # Get the pair key
            logger.info(f"Received OHLC data with pair key: {pair_key}")
            
            # Convert to DataFrame for easier processing
            df = parse_kraken_ohlc(ohlc_data[pair_key])
            
            # Load historical OHLC data into strategy
            for _, row in df.iterrows():
                self.strategy.update_ohlc(row['open'], row['high'], row['low'], row['close'])
                
                # Also add to the ohlc_data list for signal calculation
                self.ohlc_data.append({
                    'time': row['time'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                })
            
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
            
            # Add status update timing
            self.last_status_update = 0
            
            while self.running:
                # Main loop sleeps to avoid excessive CPU usage
                # Check orders and trailing stops periodically
                self._check_orders()
                
                # Check if we need to update signals
                if time.time() - self.last_signal_update >= SIGNAL_INTERVAL:
                    logger.info("Signal interval reached, updating signals...")
                    self._update_signals()
                elif int(time.time()) % 30 == 0:  # Force update every 30 seconds for testing
                    logger.info("Forcing signal update for testing...")
                    self._update_signals()
                
                # Directly log the portfolio value on each iteration
                logger.info(f"【PORTFOLIO STATUS】 Current Value: ${self.portfolio_value:.2f} | Profit: ${self.total_profit:.2f} ({self.total_profit_percent:.2f}%) | Trades: {self.trade_count}")
                
                # Check for portfolio request
                self._check_portfolio_request()
                
                # Display detailed status only every STATUS_UPDATE_INTERVAL seconds to avoid log spam
                if time.time() - self.last_status_update >= STATUS_UPDATE_INTERVAL:
                    position_str = "NONE" if self.position is None else ("LONG" if self.position == "long" else "SHORT")
                    logger.info("=" * 60)
                    # Use latest ticker data if available, even if current_price is None
                    if self.current_price is not None:
                        current_price_str = f"${self.current_price:.2f}"
                    elif self.trading_pair in self.ticker_data and 'close' in self.ticker_data[self.trading_pair] and self.ticker_data[self.trading_pair]['close'] > 0:
                        self.current_price = self.ticker_data[self.trading_pair]['close']
                        current_price_str = f"${self.current_price:.2f}"
                        logger.info(f"Updated current price from ticker data: {current_price_str}")
                    else:
                        # More aggressive debugging
                        if self.trading_pair in self.ticker_data:
                            logger.debug(f"Ticker data for {self.trading_pair}: {self.ticker_data[self.trading_pair]}")
                        else:
                            logger.debug(f"No ticker data for {self.trading_pair}. Available pairs: {list(self.ticker_data.keys())}")
                        current_price_str = "Unknown"
                    
                    logger.info(f"【MARKET】 PRICE: {current_price_str} | POSITION: {position_str}")
                    if self.position is not None and self.entry_price is not None and self.current_price is not None:
                        logger.info(f"ENTRY PRICE: ${self.entry_price:.2f}")
                        if self.position == "long":
                            profit = (self.current_price - self.entry_price) * self.trade_quantity
                        else:  # short
                            profit = (self.entry_price - self.current_price) * self.trade_quantity
                        percent = (profit / (self.portfolio_value * MARGIN_PERCENT)) * 100
                        logger.info(f"UNREALIZED P/L: ${profit:.2f} ({percent:.2f}%)")
                    elif self.position is not None and self.entry_price is not None:
                        # We have a position but no current price
                        logger.info(f"ENTRY PRICE: ${self.entry_price:.2f}")
                        logger.info("UNREALIZED P/L: Unknown (waiting for price data)")
                    logger.info(f"【ACCOUNT】 PORTFOLIO: ${self.portfolio_value:.2f} | PROFIT: ${self.total_profit:.2f} ({self.total_profit_percent:.2f}%) | TRADES: {self.trade_count}")
                    if self.trailing_stop_order is not None:
                        logger.info(f"TRAILING STOP: ${self.trailing_stop_order['price']:.2f}")
                    if self.breakeven_order is not None:
                        logger.info(f"BREAKEVEN EXIT: ${self.breakeven_order['price']:.2f}")
                    if self.liquidity_exit_order is not None:
                        logger.info(f"LIQUIDITY EXIT: ${self.liquidity_exit_order['price']:.2f}")
                    logger.info("=" * 60)
                    self.last_status_update = time.time()
                
                time.sleep(LOOP_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        
        except Exception as e:
            logger.error(f"Bot error: {e}")
        
        finally:
            self.stop()
