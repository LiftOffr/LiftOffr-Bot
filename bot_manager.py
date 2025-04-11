import logging
import time
from typing import Dict, List
import threading

from kraken_trading_bot import KrakenTradingBot
from config import (
    INITIAL_CAPITAL, MARGIN_PERCENT, TRADE_QUANTITY,
    TRADING_PAIR, LOOP_INTERVAL, STATUS_UPDATE_INTERVAL,
    LEVERAGE
)

logger = logging.getLogger(__name__)

class BotManager:
    """
    Manager for running multiple trading bots with different strategies
    concurrently while sharing portfolio resources.
    """
    def __init__(self):
        """
        Initialize the bot manager
        """
        self.bots: Dict[str, KrakenTradingBot] = {}
        self.bot_threads: Dict[str, threading.Thread] = {}
        self.running = False
        
        # Shared resources
        self.portfolio_value = INITIAL_CAPITAL
        self.total_profit = 0.0
        self.total_profit_percent = 0.0
        self.trade_count = 0
        
        # Track allocated capital and available funds
        self.allocated_capital = 0.0
        self.available_funds = INITIAL_CAPITAL
        
        # For thread-safe access to shared resources
        self.lock = threading.Lock()
        
    def add_bot(self, strategy_type: str, trading_pair: str = TRADING_PAIR, 
                trade_quantity: float = TRADE_QUANTITY, margin_percent: float = MARGIN_PERCENT,
                leverage: int = LEVERAGE) -> str:
        """
        Add a new trading bot with the specified strategy
        
        Args:
            strategy_type (str): Trading strategy type ('arima', 'adaptive', etc.)
            trading_pair (str): Trading pair to trade
            trade_quantity (float): Quantity to trade
            margin_percent (float): Percentage of portfolio used as margin (default from config)
            leverage (int): Leverage for trading (default from config)
        
        Returns:
            str: Bot ID (strategy_type-trading_pair)
        """
        bot_id = f"{strategy_type}-{trading_pair}"
        
        # Create the bot instance
        bot = KrakenTradingBot(
            trading_pair=trading_pair,
            trade_quantity=trade_quantity,
            strategy_type=strategy_type,
            margin_percent=margin_percent,
            leverage=leverage
        )
        
        # Set the shared portfolio
        bot.portfolio_value = self.portfolio_value
        bot.total_profit = self.total_profit
        bot.total_profit_percent = self.total_profit_percent
        bot.trade_count = self.trade_count
        bot.available_funds = self.available_funds
        
        # Set bot manager reference for callbacks
        bot.bot_manager = self
        
        # Store the bot instance
        self.bots[bot_id] = bot
        
        # Calculate margin allocation for this bot
        allocated_margin = self.portfolio_value * margin_percent
        logger.info(f"[BotManager] Added {bot_id} with margin allocation: ${allocated_margin:.2f} ({margin_percent*100}%)")
        
        logger.info(f"Added bot with ID {bot_id} (strategy: {strategy_type}, pair: {trading_pair}, margin: {margin_percent*100}%, leverage: {leverage}x)")
        return bot_id
    
    def start_bot(self, bot_id: str) -> bool:
        """
        Start a specific bot in a separate thread
        
        Args:
            bot_id (str): ID of the bot to start
        
        Returns:
            bool: True if bot was started successfully, False otherwise
        """
        if bot_id not in self.bots:
            logger.error(f"Bot with ID {bot_id} not found")
            return False
        
        if bot_id in self.bot_threads and self.bot_threads[bot_id].is_alive():
            logger.warning(f"Bot with ID {bot_id} is already running")
            return True
        
        # Create a thread for the bot
        bot_thread = threading.Thread(target=self._run_bot, args=(bot_id,))
        bot_thread.daemon = True
        
        # Store the thread
        self.bot_threads[bot_id] = bot_thread
        
        # Start the thread
        bot_thread.start()
        
        logger.info(f"Started bot with ID {bot_id}")
        return True
    
    def _run_bot(self, bot_id: str):
        """
        Target function for bot threads
        
        Args:
            bot_id (str): ID of the bot to run
        """
        bot = self.bots[bot_id]
        
        # Start the bot
        bot.start()
        
        # Main running loop
        last_status_update = 0
        previous_position = None
        
        while self.running:
            try:
                # Update bot state
                if hasattr(bot, '_update_signals'):
                    bot._update_signals()
                    
                if hasattr(bot, '_check_orders'):
                    bot._check_orders()
                
                # Update shared portfolio with lock to prevent race conditions
                with self.lock:
                    # Position change detection for capital allocation tracking
                    position_changed = previous_position != bot.position
                    
                    # Track when a position is opened (allocates capital)
                    if position_changed and bot.position is not None and previous_position is None:
                        # Calculate capital allocated to this position
                        position_margin = bot.portfolio_value * bot.margin_percent
                        self.allocated_capital += position_margin
                        self.available_funds -= position_margin
                        logger.info(f"Bot {bot_id} opened a {bot.position} position, allocating ${position_margin:.2f} (margin: {bot.margin_percent*100}%)")
                        logger.info(f"Allocated capital: ${self.allocated_capital:.2f}, Available funds: ${self.available_funds:.2f}")
                    
                    # Track when a position is closed (frees up capital)
                    elif position_changed and bot.position is None and previous_position is not None:
                        # Calculate capital freed up from this position
                        position_margin = bot.portfolio_value * bot.margin_percent
                        self.allocated_capital -= position_margin
                        if self.allocated_capital < 0:  # Safety check
                            self.allocated_capital = 0
                        
                        # Update available funds based on new portfolio value
                        self.portfolio_value = bot.portfolio_value
                        self.available_funds = self.portfolio_value - self.allocated_capital
                        
                        logger.info(f"Bot {bot_id} closed a {previous_position} position, freeing ${position_margin:.2f}")
                        logger.info(f"Portfolio value: ${self.portfolio_value:.2f}, Allocated capital: ${self.allocated_capital:.2f}, Available funds: ${self.available_funds:.2f}")
                    
                    # Update shared portfolio metrics
                    self.portfolio_value = bot.portfolio_value
                    self.total_profit = bot.total_profit
                    self.total_profit_percent = bot.total_profit_percent
                    self.trade_count = bot.trade_count
                    
                    # Always update the available funds in case portfolio value changed
                    self.available_funds = self.portfolio_value - self.allocated_capital
                    
                    # Update bot with latest available funds
                    bot.available_funds = self.available_funds
                    
                    # Save current position for next comparison
                    previous_position = bot.position
                
                # Print status periodically
                if time.time() - last_status_update > STATUS_UPDATE_INTERVAL:
                    if hasattr(bot, '_print_status'):
                        bot._print_status()
                    last_status_update = time.time()
                
                # Sleep for a bit
                time.sleep(LOOP_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in bot {bot_id}: {e}")
                time.sleep(LOOP_INTERVAL)
    
    def start_all(self) -> bool:
        """
        Start all bots
        
        Returns:
            bool: True if all bots were started successfully
        """
        self.running = True
        
        success = True
        for bot_id in self.bots:
            if not self.start_bot(bot_id):
                success = False
        
        return success
    
    def stop_all(self):
        """
        Stop all bots
        """
        self.running = False
        
        # Close WebSocket connections for all bots
        for bot_id, bot in self.bots.items():
            try:
                bot.ws.disconnect()
                logger.info(f"Stopped bot with ID {bot_id}")
            except Exception as e:
                logger.error(f"Error stopping bot {bot_id}: {e}")
        
        # Wait for all threads to terminate
        for bot_id, thread in self.bot_threads.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
                
        logger.info("All bots stopped")
    
    def track_position_change(self, bot_id: str, new_position: str, previous_position: str, margin_percent: float, funds_to_allocate: float = 0.0):
        """
        Track position changes to update available funds
        
        Args:
            bot_id (str): ID of the bot that changed position
            new_position (str): New position ("long", "short", or None)
            previous_position (str): Previous position ("long", "short", or None)
            margin_percent (float): Margin percentage for this strategy
            funds_to_allocate (float): Specific amount to allocate (if 0, calculated based on portfolio)
        """
        with self.lock:
            # Opening a new position - reduce available funds
            if new_position is not None and previous_position is None:
                # Calculate capital to allocate based on margin percentage
                if funds_to_allocate <= 0:
                    position_margin = self.portfolio_value * margin_percent
                else:
                    position_margin = funds_to_allocate
                    
                self.allocated_capital += position_margin
                self.available_funds -= position_margin
                
                logger.info(f"[BotManager] Bot {bot_id} opened a {new_position} position, allocating ${position_margin:.2f}")
                logger.info(f"[BotManager] Allocated capital: ${self.allocated_capital:.2f}, Available funds: ${self.available_funds:.2f}")
                
            # Closing a position - restore available funds
            elif new_position is None and previous_position is not None:
                # Calculate freed capital
                if funds_to_allocate <= 0:
                    position_margin = self.portfolio_value * margin_percent
                else:
                    position_margin = funds_to_allocate
                
                self.allocated_capital -= position_margin
                if self.allocated_capital < 0:  # Safety check
                    self.allocated_capital = 0
                
                # Update available funds
                self.available_funds = self.portfolio_value - self.allocated_capital
                
                logger.info(f"[BotManager] Bot {bot_id} closed a {previous_position} position, freeing ${position_margin:.2f}")
                logger.info(f"[BotManager] Allocated capital: ${self.allocated_capital:.2f}, Available funds: ${self.available_funds:.2f}")
            
            # Update all bots with current available funds
            for _, bot in self.bots.items():
                bot.available_funds = self.available_funds
    
    def update_portfolio(self, bot_id: str, new_portfolio_value: float, trade_profit: float = 0.0):
        """
        Update the shared portfolio after a trade is executed
        
        Args:
            bot_id (str): ID of the bot that executed the trade
            new_portfolio_value (float): New portfolio value after trade
            trade_profit (float): Profit from the trade
        """
        with self.lock:
            logger.info(f"[BotManager] Updating portfolio: Bot {bot_id} executed trade with profit: ${trade_profit:.2f}")
            logger.info(f"[BotManager] Previous portfolio value: ${self.portfolio_value:.2f}, New portfolio value: ${new_portfolio_value:.2f}")
            
            # Update portfolio value
            self.portfolio_value = new_portfolio_value
            self.total_profit += trade_profit
            self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100.0
            self.trade_count += 1
            
            # Recalculate available funds
            self.available_funds = self.portfolio_value - self.allocated_capital
            
            # Update all bots with new portfolio values
            for bot_id, bot in self.bots.items():
                bot.portfolio_value = self.portfolio_value
                bot.total_profit = self.total_profit
                bot.total_profit_percent = self.total_profit_percent
                bot.trade_count = self.trade_count
                bot.available_funds = self.available_funds
                
            logger.info(f"[BotManager] Updated portfolio: Value=${self.portfolio_value:.2f}, Profit=${self.total_profit:.2f} ({self.total_profit_percent:.2f}%), Trades={self.trade_count}")
            logger.info(f"[BotManager] Available funds: ${self.available_funds:.2f}, Allocated capital: ${self.allocated_capital:.2f}")
    
    def get_status(self) -> Dict:
        """
        Get status information for all bots
        
        Returns:
            dict: Status information
        """
        status = {
            "portfolio_value": self.portfolio_value,
            "total_profit": self.total_profit,
            "total_profit_percent": self.total_profit_percent,
            "trade_count": self.trade_count,
            "allocated_capital": self.allocated_capital,
            "available_funds": self.available_funds,
            "allocation_percentage": (self.allocated_capital / self.portfolio_value * 100) if self.portfolio_value > 0 else 0,
            "bots": {}
        }
        
        for bot_id, bot in self.bots.items():
            strategy_margin = bot.margin_percent * 100 if hasattr(bot, 'margin_percent') else 0
            strategy_leverage = bot.leverage if hasattr(bot, 'leverage') else 0
            
            status["bots"][bot_id] = {
                "trading_pair": bot.trading_pair,
                "strategy_type": bot.strategy.__class__.__name__,
                "position": bot.position,
                "entry_price": bot.entry_price,
                "current_price": bot.current_price,
                "margin_percent": f"{strategy_margin:.2f}%",
                "leverage": f"{strategy_leverage}x",
                "active": bot.position is not None
            }
        
        return status