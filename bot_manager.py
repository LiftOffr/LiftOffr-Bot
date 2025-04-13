import logging
import os
import time
from typing import Dict, List
import threading

from kraken_trading_bot import KrakenTradingBot
from config import (
    INITIAL_CAPITAL, MARGIN_PERCENT, TRADE_QUANTITY,
    TRADING_PAIR, LOOP_INTERVAL, STATUS_UPDATE_INTERVAL,
    LEVERAGE, ENABLE_CROSS_STRATEGY_EXITS, CROSS_STRATEGY_EXIT_THRESHOLD,
    CROSS_STRATEGY_EXIT_CONFIRMATION_COUNT, STRONGER_SIGNAL_DOMINANCE,
    SIGNAL_STRENGTH_ADVANTAGE, MIN_SIGNAL_STRENGTH
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
        
        # Strategy categories
        self.strategy_categories = {
            "those dudes": [],  # For existing ARIMA and Adaptive strategies
            "him all along": []  # For the integrated strategy
        }
        self.available_funds = INITIAL_CAPITAL
        
        # For thread-safe access to shared resources
        self.lock = threading.Lock()
        
        # Cross-strategy exit tracking
        self.cross_strategy_signals = {}  # Dict to track opposing signals from strategies
        self.cross_strategy_exits = {}    # Dict to track cross-strategy exit events
        self.enable_cross_exits = ENABLE_CROSS_STRATEGY_EXITS  # Config flag to enable/disable feature
        self.exit_threshold = CROSS_STRATEGY_EXIT_THRESHOLD    # Threshold for signal strength to trigger exit
        self.exit_confirmations = CROSS_STRATEGY_EXIT_CONFIRMATION_COUNT  # Required confirmations
        
        # Signal strength arbitration configuration
        self.stronger_signal_dominance = STRONGER_SIGNAL_DOMINANCE
        self.signal_strength_advantage = SIGNAL_STRENGTH_ADVANTAGE
        self.min_signal_strength = MIN_SIGNAL_STRENGTH
        
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
        
        # Assign the bot to appropriate category
        if strategy_type == 'integrated':
            self.strategy_categories["him all along"].append(bot_id)
            logger.info(f"[BotManager] Added {bot_id} to 'him all along' category")
        else:
            self.strategy_categories["those dudes"].append(bot_id)
            logger.info(f"[BotManager] Added {bot_id} to 'those dudes' category")
        
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
    
    def track_position_change(self, bot_id: str, new_position: str, previous_position: str, margin_percent: float, funds_to_allocate: float = 0.0, signal_strength: float = 0.0):
        """
        Track position changes to update available funds
        
        Args:
            bot_id (str): ID of the bot that changed position
            new_position (str): New position ("long", "short", or None)
            previous_position (str): Previous position ("long", "short", or None)
            margin_percent (float): Margin percentage for this strategy
            funds_to_allocate (float): Specific amount to allocate (if 0, calculated based on portfolio)
            signal_strength (float): Strength of the signal that triggered the position change (0.0 to 1.0)
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
    
    def register_strategy_signal(self, bot_id: str, signal_type: str, signal_strength: float = 1.0, 
                               trading_pair: str = None) -> None:
        """
        Register a strategy signal for cross-strategy coordination
        
        Args:
            bot_id (str): ID of the bot sending the signal
            signal_type (str): Type of signal ("buy", "sell", or "neutral")
            signal_strength (float): Strength of the signal (0.0 to 1.0)
            trading_pair (str): Trading pair the signal applies to (defaults to bot's pair)
        """
        if not self.enable_cross_exits:
            return  # Cross-strategy exits disabled
            
        with self.lock:
            # Get trading pair from bot if not specified
            if trading_pair is None and bot_id in self.bots:
                trading_pair = self.bots[bot_id].trading_pair
            
            # Skip if no trading pair available
            if not trading_pair:
                return
                
            # Initialize signal tracking for this pair if needed
            if trading_pair not in self.cross_strategy_signals:
                self.cross_strategy_signals[trading_pair] = {}
                
            # Update signal for this bot
            self.cross_strategy_signals[trading_pair][bot_id] = {
                "type": signal_type,
                "strength": signal_strength,
                "timestamp": time.time()
            }
            
            # Log the signal
            logger.info(f"[BotManager] Registered {signal_type.upper()} signal from {bot_id} with strength {signal_strength:.2f}")
    
    def check_cross_strategy_exit(self, bot_id: str, current_position: str) -> bool:
        """
        Check if a cross-strategy exit should be triggered based on signals from other strategies
        
        Args:
            bot_id (str): ID of the bot to check for
            current_position (str): Current position of the bot ("long", "short", or None)
            
        Returns:
            bool: True if a cross-strategy exit should be triggered
        """
        if not self.enable_cross_exits or not current_position or bot_id not in self.bots:
            return False
            
        trading_pair = self.bots[bot_id].trading_pair
        
        # Skip if no signals for this pair
        if trading_pair not in self.cross_strategy_signals:
            return False
            
        # Look for opposing signals from other strategies
        opposing_signal_type = "sell" if current_position == "long" else "buy"
        opposing_signals = []
        
        # Get current strategy's signal strength
        current_bot_signal_strength = 0.0
        current_bot_signal_data = None
        
        with self.lock:
            # Get current bot's signal strength if available
            if bot_id in self.cross_strategy_signals[trading_pair]:
                current_bot_signal_data = self.cross_strategy_signals[trading_pair][bot_id]
                # Only consider signals less than 3 minutes old
                if time.time() - current_bot_signal_data["timestamp"] <= 180:
                    current_bot_signal_strength = current_bot_signal_data["strength"]
            
            # Get all opposing signals from other strategies for this pair
            for other_bot_id, signal_data in self.cross_strategy_signals[trading_pair].items():
                # Skip self and old signals (more than 3 minutes old - reduced from 5 minutes for faster reaction)
                if other_bot_id == bot_id or time.time() - signal_data["timestamp"] > 180:
                    continue
                    
                # Check if signal type is opposing and strength exceeds minimum threshold
                if signal_data["type"] == opposing_signal_type and signal_data["strength"] >= self.min_signal_strength:
                    # If stronger signal dominance is enabled, check if this signal is strong enough
                    if self.stronger_signal_dominance:
                        # Calculate the strength advantage of this opposing signal
                        strength_advantage = signal_data["strength"] - current_bot_signal_strength
                        
                        # Only include signals that have sufficient advantage over the current position
                        if strength_advantage >= self.signal_strength_advantage:
                            opposing_signals.append({
                                "bot_id": other_bot_id,
                                "strength": signal_data["strength"],
                                "advantage": strength_advantage
                            })
                            logger.info(f"[BotManager] Signal arbitration: {other_bot_id} ({opposing_signal_type.upper()}, " 
                                        f"strength: {signal_data['strength']:.2f}) has advantage of {strength_advantage:.2f} " 
                                        f"over {bot_id} (strength: {current_bot_signal_strength:.2f})")
                    else:
                        # If stronger signal dominance is disabled, just use the minimum threshold
                        opposing_signals.append({
                            "bot_id": other_bot_id,
                            "strength": signal_data["strength"],
                            "advantage": signal_data["strength"] - current_bot_signal_strength
                        })
        
        # Check if we have enough opposing signals to trigger an exit
        if len(opposing_signals) >= self.exit_confirmations:
            # Get the strongest opposing signal for attribution
            strongest_signal = max(opposing_signals, key=lambda x: x["strength"])
            
            # Log the cross-strategy exit with more detailed information
            log_message = (f"[BotManager] Cross-strategy exit triggered for {bot_id} ({current_position}) "
                          f"based on opposing signals from {len(opposing_signals)} strategies. ")
            
            if self.stronger_signal_dominance:
                log_message += (f"Strongest signal from {strongest_signal['bot_id']} with strength {strongest_signal['strength']:.2f} "
                               f"(advantage: {strongest_signal['advantage']:.2f})")
            else:
                log_message += f"Strongest signal from {strongest_signal['bot_id']} with strength {strongest_signal['strength']:.2f}"
                
            logger.info(log_message)
            
            # Record the exit event for tracking and reporting
            exit_key = f"{bot_id}_{int(time.time())}"
            self.cross_strategy_exits[exit_key] = {
                "bot_id": bot_id,
                "position": current_position,
                "trading_pair": trading_pair,
                "exit_time": time.time(),
                "opposing_signals": opposing_signals,
                "exited_by": strongest_signal["bot_id"],
                "signal_strength": strongest_signal["strength"],
                "current_bot_strength": current_bot_signal_strength,
                "advantage": strongest_signal.get("advantage", 0.0)
            }
            
            return True
        
        return False
            
    def get_cross_strategy_stats(self) -> Dict:
        """
        Get statistics about cross-strategy exits
        
        Returns:
            dict: Cross-strategy exit statistics
        """
        with self.lock:
            stats = {
                "enabled": self.enable_cross_exits,
                "threshold": self.exit_threshold,
                "confirmations_required": self.exit_confirmations,
                "stronger_signal_dominance": self.stronger_signal_dominance,
                "min_signal_strength": self.min_signal_strength,
                "signal_strength_advantage": self.signal_strength_advantage,
                "exit_count": len(self.cross_strategy_exits),
                "exits_by_strategy": {},
                "recent_exits": []
            }
            
            # Calculate exits by strategy
            for exit_data in self.cross_strategy_exits.values():
                exited_by = exit_data["exited_by"]
                if exited_by not in stats["exits_by_strategy"]:
                    stats["exits_by_strategy"][exited_by] = 0
                stats["exits_by_strategy"][exited_by] += 1
            
            # Get 5 most recent exits
            recent_exits = sorted(
                self.cross_strategy_exits.values(),
                key=lambda x: x["exit_time"],
                reverse=True
            )[:5]
            
            for exit_data in recent_exits:
                exit_stats = {
                    "bot_id": exit_data["bot_id"],
                    "position": exit_data["position"],
                    "trading_pair": exit_data["trading_pair"],
                    "exit_time": exit_data["exit_time"],
                    "exited_by": exit_data["exited_by"]
                }
                
                # Add signal strength information if available
                if "signal_strength" in exit_data:
                    exit_stats["signal_strength"] = exit_data["signal_strength"]
                if "current_bot_strength" in exit_data:
                    exit_stats["current_bot_strength"] = exit_data["current_bot_strength"]
                if "advantage" in exit_data:
                    exit_stats["advantage"] = exit_data["advantage"]
                
                stats["recent_exits"].append(exit_stats)
            
            return stats
    
    def get_status(self) -> Dict:
        """
        Get status information for all bots
        
        Returns:
            dict: Status information
        """
        # Calculate current portfolio value and profit from trades.csv
        total_pnl = 0.0
        if os.path.exists("trades.csv"):
            try:
                with open("trades.csv", "r") as f:
                    # Skip header
                    next(f)
                    for line in f:
                        parts = line.strip().split(',')
                        # Check if we have a PnL value (column 6)
                        if len(parts) > 6 and parts[6] and parts[6] != "":
                            try:
                                pnl_value = float(parts[6])
                                total_pnl += pnl_value
                            except (ValueError, TypeError):
                                pass
                
                # Update portfolio value with total P&L from trades
                self.portfolio_value = INITIAL_CAPITAL + total_pnl
                self.total_profit = total_pnl
                
                # Update profit percentage
                if self.total_profit != 0:
                    self.total_profit_percent = (self.total_profit / INITIAL_CAPITAL) * 100
            except Exception as e:
                logger.error(f"Error calculating portfolio value from trades: {e}")
        
        status = {
            "portfolio_value": self.portfolio_value,
            "total_profit": self.total_profit,
            "total_profit_percent": self.total_profit_percent,
            "trade_count": self.trade_count,
            "allocated_capital": self.allocated_capital,
            "available_funds": self.available_funds,
            "allocation_percentage": (self.allocated_capital / self.portfolio_value * 100) if self.portfolio_value > 0 else 0,
            "cross_strategy_exits": {
                "enabled": self.enable_cross_exits,
                "exit_count": len(self.cross_strategy_exits),
                "stronger_signal_dominance": self.stronger_signal_dominance,
                "min_signal_strength": self.min_signal_strength,
                "signal_strength_advantage": self.signal_strength_advantage
            },
            "categories": {
                "those_dudes": len(self.strategy_categories.get("those dudes", [])),
                "him_all_along": len(self.strategy_categories.get("him all along", []))
            },
            "bots": {}
        }
        
        for bot_id, bot in self.bots.items():
            strategy_margin = bot.margin_percent * 100 if hasattr(bot, 'margin_percent') else 0
            strategy_leverage = bot.leverage if hasattr(bot, 'leverage') else 0
            
            # Determine strategy category
            category = "unknown"
            if bot_id in self.strategy_categories.get("those dudes", []):
                category = "those dudes"
            elif bot_id in self.strategy_categories.get("him all along", []):
                category = "him all along"
            
            # Check if bot.strategy has a category attribute
            if hasattr(bot.strategy, 'category'):
                category = bot.strategy.category
                
            status["bots"][bot_id] = {
                "trading_pair": bot.trading_pair,
                "strategy_type": bot.strategy.__class__.__name__,
                "category": category,
                "position": bot.position,
                "entry_price": bot.entry_price,
                "current_price": bot.current_price,
                "margin_percent": f"{strategy_margin:.2f}%",
                "leverage": f"{strategy_leverage}x",
                "active": bot.position is not None
            }
        
        return status