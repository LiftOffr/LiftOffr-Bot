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
        
        # Store the bot instance
        self.bots[bot_id] = bot
        
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
        
        while self.running:
            try:
                # Update bot state
                if hasattr(bot, '_update_signals'):
                    bot._update_signals()
                    
                if hasattr(bot, '_check_orders'):
                    bot._check_orders()
                
                # Update shared portfolio with lock to prevent race conditions
                with self.lock:
                    self.portfolio_value = bot.portfolio_value
                    self.total_profit = bot.total_profit
                    self.total_profit_percent = bot.total_profit_percent
                    self.trade_count = bot.trade_count
                
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
            "bots": {}
        }
        
        for bot_id, bot in self.bots.items():
            status["bots"][bot_id] = {
                "trading_pair": bot.trading_pair,
                "strategy_type": bot.strategy.__class__.__name__,
                "position": bot.position,
                "entry_price": bot.entry_price,
                "current_price": bot.current_price
            }
        
        return status