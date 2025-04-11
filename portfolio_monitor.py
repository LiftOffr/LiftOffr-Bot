"""
Portfolio Status Monitor

This module provides a beautiful, readable output of portfolio status
on a regular interval.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class PortfolioMonitor:
    """
    Monitor and periodically display portfolio status in a nice readable format
    """
    
    def __init__(self, bot_manager, update_interval: int = 60):
        """
        Initialize the portfolio monitor
        
        Args:
            bot_manager: Reference to the BotManager instance
            update_interval: Interval in seconds to update the status (default: 60)
        """
        self.bot_manager = bot_manager
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.last_trade_count = 0
        
    def start(self):
        """Start the portfolio monitor"""
        if self.thread and self.thread.is_alive():
            logger.warning("Portfolio monitor is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Portfolio monitor started with update interval: %d seconds", self.update_interval)
        
    def stop(self):
        """Stop the portfolio monitor"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
    def _run(self):
        """Main run loop for the portfolio monitor"""
        next_update = time.time() + self.update_interval
        
        while self.running:
            current_time = time.time()
            
            if current_time >= next_update:
                self._display_portfolio_status()
                next_update = current_time + self.update_interval
                
            # Sleep for a short time to prevent high CPU usage
            time.sleep(1)
            
    def _display_portfolio_status(self):
        """Display current portfolio status in a nice format"""
        try:
            status = self.bot_manager.get_status()
            
            # Set defaults for missing values
            trade_count = status.get('trade_count', 0)
            new_trade = trade_count > self.last_trade_count if hasattr(self, 'last_trade_count') else False
            self.last_trade_count = trade_count
            
            # Enhanced with colorful border for new trades
            border = "🟢" * 80 if new_trade else "=" * 80
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get values with defaults for missing data - convert to float to handle string values
            try:
                portfolio_value = float(status.get('portfolio_value', 0.0))
            except (ValueError, TypeError):
                portfolio_value = 0.0
                
            try:
                total_profit = float(status.get('total_profit', 0.0))
            except (ValueError, TypeError):
                total_profit = 0.0
                
            try:
                total_profit_percent = float(status.get('total_profit_percent', 0.0))
            except (ValueError, TypeError):
                total_profit_percent = 0.0
                
            try:
                allocated_capital = float(status.get('allocated_capital', 0.0))
            except (ValueError, TypeError):
                allocated_capital = 0.0
                
            try:
                allocation_percentage = float(status.get('allocation_percentage', 0.0))
            except (ValueError, TypeError):
                allocation_percentage = 0.0
                
            try:
                available_funds = float(status.get('available_funds', 0.0))
            except (ValueError, TypeError):
                available_funds = 0.0
            
            # Get initial portfolio value with fallback
            try:
                initial_value = float(getattr(self.bot_manager, 'initial_portfolio_value', 
                                        getattr(self.bot_manager, 'portfolio_value', 20000.0)))
            except (ValueError, TypeError):
                initial_value = 20000.0
            
            # Prepare the output
            output = [
                f"\n{border}",
                f"📊 KRAKEN TRADING BOT - PORTFOLIO STATUS [Updated: {timestamp}]",
                f"{border}",
                "",
                "💰 PORTFOLIO SUMMARY:",
                f"  Initial Value:     ${initial_value:.2f}",
                f"  Current Value:     ${portfolio_value:.2f}",
                f"  Total P&L:         ${total_profit:.2f} ({total_profit_percent:.2f}%)",
                f"  Allocated Capital: ${allocated_capital:.2f} ({allocation_percentage:.2f}%)",
                f"  Available Funds:   ${available_funds:.2f}",
                f"  Total Trades:      {trade_count}",
                "",
            ]
            
            # Add bot details
            output.append("🤖 ACTIVE STRATEGIES:")
            bots = status.get('bots', {})
            if not bots:
                output.append("  No active bots found.")
                output.append("")
            else:
                for bot_id, bot_status in bots.items():
                    # Get values safely and ensure they are the correct types
                    position = bot_status.get('position')
                    trading_pair = str(bot_status.get('trading_pair', 'Unknown'))
                    strategy_type = str(bot_status.get('strategy_type', 'Unknown'))
                    
                    try:
                        margin_percent = float(bot_status.get('margin_percent', 0))
                    except (ValueError, TypeError):
                        margin_percent = 0
                        
                    try:
                        leverage = int(bot_status.get('leverage', 0))
                    except (ValueError, TypeError):
                        leverage = 0
                    
                    # Highlight active positions
                    position_icon = "⚪" if position is None else "🟢" if position == 'long' else "🔴"
                    
                    if position is None:
                        position_text = "No Position"
                    else:
                        entry_price = bot_status.get('entry_price', 0)
                        position_text = f"{position.upper()} @ ${entry_price:.2f}"
                    
                    output.append(f"  {position_icon} {bot_id} ({trading_pair}):")
                    output.append(f"      Strategy: {strategy_type} | Position: {position_text}")
                    output.append(f"      Margin: {margin_percent:.1f}% | Leverage: {leverage}x")
                    
                    if position is not None:
                        current_price = bot_status.get('current_price', 0)
                        entry_price = bot_status.get('entry_price', 0)
                        if current_price > 0 and entry_price > 0:
                            profit_pct = (current_price / entry_price - 1) * 100
                            if position == 'short':
                                profit_pct = -profit_pct
                            output.append(f"      Current Price: ${current_price:.2f} | Unrealized P&L: {profit_pct:.2f}%")
                    output.append("")
            
            # Add market data
            output.append("📈 RECENT MARKET DATA:")
            # Try to get current price from any bot
            any_bot_id = next(iter(bots.keys()), None)
            if any_bot_id:
                bot_data = bots[any_bot_id]
                pair = bot_data.get('trading_pair', 'Unknown')
                price = bot_data.get('current_price', 0)
                output.append(f"  Current Price ({pair}): ${price:.2f}")
            else:
                output.append("  No price data available yet.")
                
            output.append(f"{border}\n")
            
            # Join all lines into a single string
            output_str = "\n".join(output)
            
            # Log the status with INFO level so it appears in the console
            logger.info(f"\n{output_str}")
            
            # Also print to stdout for terminal display
            print(output_str)
            
        except Exception as e:
            import traceback
            logger.error(f"Error displaying portfolio status: {e}")
            logger.debug(f"Error traceback: {traceback.format_exc()}")

# Function to attach monitor to an existing bot manager
def attach_portfolio_monitor(bot_manager, update_interval: int = 60):
    """
    Attach a portfolio monitor to a bot manager
    
    Args:
        bot_manager: The BotManager instance
        update_interval: Update interval in seconds
        
    Returns:
        PortfolioMonitor: The created monitor instance
    """
    monitor = PortfolioMonitor(bot_manager, update_interval)
    monitor.start()
    return monitor