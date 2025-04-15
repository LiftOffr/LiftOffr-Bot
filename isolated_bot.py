#!/usr/bin/env python3
"""
Isolated Trading Bot

This is a completely isolated trading bot that doesn't rely on 
any Flask components. It runs directly as a standalone script.
"""
import os
import sys
import json
import time
import random
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "sandbox_portfolio.json")
POSITIONS_FILE = os.path.join(DATA_DIR, "sandbox_positions.json")
TRADES_FILE = os.path.join(DATA_DIR, "sandbox_trades.json")
PORTFOLIO_HISTORY_FILE = os.path.join(DATA_DIR, "sandbox_portfolio_history.json")
INITIAL_CAPITAL = 20000.0

# Global price cache to reduce API calls
price_cache = {}
price_cache_time = {}
PRICE_CACHE_EXPIRY = 5  # seconds

def get_price(pair: str) -> Optional[float]:
    """
    Get current price with caching to reduce API calls.
    
    Args:
        pair: Trading pair
        
    Returns:
        Current price or None if unavailable
    """
    global price_cache, price_cache_time
    
    # Check cache first
    now = time.time()
    if pair in price_cache and (now - price_cache_time.get(pair, 0)) < PRICE_CACHE_EXPIRY:
        return price_cache[pair]
    
    try:
        # For demonstration, using simulated prices
        # In a real implementation, this would call the Kraken API
        # Simulating price movement ±2% from a base price
        base_prices = {
            "BTC/USD": 62350.0,
            "ETH/USD": 3050.0,
            "SOL/USD": 142.50,
            "ADA/USD": 0.45,
            "DOT/USD": 6.75,
            "LINK/USD": 15.30,
            "AVAX/USD": 35.25,
            "MATIC/USD": 0.65,
            "UNI/USD": 9.80,
            "ATOM/USD": 8.45
        }
        
        if pair in base_prices:
            movement = random.uniform(-0.02, 0.02)
            price = base_prices[pair] * (1 + movement)
            
            # Update cache
            price_cache[pair] = price
            price_cache_time[pair] = now
            
            logger.debug(f"Price for {pair}: ${price:.2f}")
            return price
        else:
            logger.warning(f"Price not available for pair: {pair}")
            return None
    except Exception as e:
        logger.error(f"Error fetching price for {pair}: {e}")
        return None

class IsolatedTradingBot:
    """Isolated trading bot with real-time price data"""
    
    def __init__(self):
        """Initialize the trading bot"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load or create portfolio and positions
        self.portfolio = self._load_portfolio()
        self.positions = self._load_positions()
        
        # Additional state
        self.last_update_time = time.time()
        self.last_trade_time = time.time() - 300  # Allow trading immediately
        self.trade_interval = 60  # seconds
        
        # Performance metrics
        self.total_trades = len(self._load_trades())
        self.profitable_trades = sum(1 for t in self._load_trades() if t.get("profit", 0) > 0)
        self.win_rate = self.profitable_trades / max(1, self.total_trades)
        
        logger.info(f"Initialized trading bot with {len(self.positions)} active positions")
        logger.info(f"Current portfolio value: ${self.portfolio['total_value']:.2f}")
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio from file or create new if it doesn't exist"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    portfolio = json.load(f)
                logger.info(f"Loaded portfolio from {PORTFOLIO_FILE}")
                return portfolio
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
        
        # Create new portfolio
        portfolio = {
            "initial_capital": INITIAL_CAPITAL,
            "available_capital": INITIAL_CAPITAL,
            "total_value": INITIAL_CAPITAL,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        self._save_portfolio(portfolio)
        logger.info(f"Created new portfolio with {INITIAL_CAPITAL:.2f} capital")
        return portfolio
    
    def _load_positions(self) -> List:
        """Load positions from file or create empty list if not exists"""
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    positions = json.load(f)
                logger.info(f"Loaded {len(positions)} positions from {POSITIONS_FILE}")
                return positions
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
        
        # Create empty positions
        positions = []
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=2)
        logger.info("Created empty positions file")
        return positions
    
    def _load_trades(self) -> List:
        """Load historical trades from file"""
        try:
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
                return trades
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
        
        # Create empty trades file
        trades = []
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        return trades
    
    def _load_portfolio_history(self) -> List:
        """Load portfolio history from file"""
        try:
            if os.path.exists(PORTFOLIO_HISTORY_FILE):
                with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
                return history
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
        
        # Create empty history
        history = []
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return history
    
    def _save_portfolio(self, portfolio=None) -> None:
        """Save portfolio to file"""
        if portfolio is None:
            portfolio = self.portfolio
        
        try:
            portfolio["updated_at"] = datetime.datetime.now().isoformat()
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(portfolio, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def _save_positions(self, positions=None) -> None:
        """Save positions to file"""
        if positions is None:
            positions = self.positions
        
        try:
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(positions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def _save_trades(self, trades) -> None:
        """Save trades to file"""
        try:
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _update_portfolio_history(self) -> None:
        """Update portfolio history with current value"""
        try:
            history = self._load_portfolio_history()
            
            # Add current portfolio snapshot
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_value": self.portfolio["total_value"],
                "available_capital": self.portfolio["available_capital"],
                "total_pnl": self.portfolio["total_pnl"],
                "total_pnl_pct": self.portfolio["total_pnl_pct"],
                "unrealized_pnl": self.portfolio["unrealized_pnl"],
                "unrealized_pnl_pct": self.portfolio["unrealized_pnl_pct"],
                "win_rate": self.portfolio["win_rate"],
                "total_trades": self.portfolio["total_trades"],
                "profitable_trades": self.portfolio["profitable_trades"]
            }
            
            history.append(entry)
            
            # Limit history to 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            
            with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def _log_trade(self, position: Dict, action: str) -> None:
        """
        Log a trade to the trades file.
        
        Args:
            position: Position data
            action: Trade action (OPEN/CLOSE)
        """
        try:
            trades = self._load_trades()
            
            trade_entry = {
                "id": len(trades) + 1,
                "pair": position["pair"],
                "action": action,
                "direction": position["direction"],
                "entry_price": position["entry_price"],
                "current_price": position.get("current_price", position["entry_price"]),
                "size": position["size"],
                "leverage": position["leverage"],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if action == "CLOSE":
                trade_entry["exit_price"] = position["current_price"]
                trade_entry["profit"] = position["unrealized_pnl"]
                trade_entry["profit_pct"] = position["unrealized_pnl_pct"]
                trade_entry["duration"] = (
                    datetime.datetime.now() - 
                    datetime.datetime.fromisoformat(position["entry_time"])
                ).total_seconds() / 3600.0  # in hours
            
            trades.append(trade_entry)
            self._save_trades(trades)
            
            # Update metrics
            self.portfolio["total_trades"] = len(trades)
            self.portfolio["profitable_trades"] = sum(1 for t in trades if t.get("profit", 0) > 0)
            self.portfolio["win_rate"] = self.portfolio["profitable_trades"] / max(1, self.portfolio["total_trades"])
            
            # Log trade
            if action == "OPEN":
                logger.info(f"Opened {position['direction']} position for {position['pair']} at ${position['entry_price']:.2f}")
            else:
                logger.info(f"Closed {position['direction']} position for {position['pair']} at ${position['current_price']:.2f} with P&L: ${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_pct']:.2f}%)")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def _update_portfolio_pnl(self) -> None:
        """Update portfolio with current P&L"""
        try:
            # Calculate unrealized P&L from active positions
            total_unrealized_pnl = sum(position.get("unrealized_pnl", 0) for position in self.positions)
            
            # Calculate total portfolio value
            self.portfolio["unrealized_pnl"] = total_unrealized_pnl
            self.portfolio["total_value"] = self.portfolio["available_capital"] + total_unrealized_pnl
            
            # Calculate percentages
            if self.portfolio["initial_capital"] > 0:
                self.portfolio["total_pnl"] = self.portfolio["total_value"] - self.portfolio["initial_capital"]
                self.portfolio["total_pnl_pct"] = (self.portfolio["total_pnl"] / self.portfolio["initial_capital"]) * 100
                self.portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / self.portfolio["initial_capital"]) * 100
            
            self._save_portfolio()
        except Exception as e:
            logger.error(f"Error updating portfolio P&L: {e}")
    
    def _update_position(self, position: Dict) -> None:
        """
        Update a position with current price.
        
        Args:
            position: Position to update
        """
        try:
            # Get current price
            current_price = get_price(position["pair"])
            if current_price is None:
                logger.warning(f"Could not update position for {position['pair']}: Price unavailable")
                return
            
            # Update position with current price
            position["current_price"] = current_price
            position["last_update_time"] = datetime.datetime.now().isoformat()
            
            # Calculate P&L
            entry_price = position["entry_price"]
            size = position["size"]
            leverage = position["leverage"]
            direction = position["direction"]
            
            # Calculate P&L based on direction
            if direction == "LONG":
                profit_factor = (current_price - entry_price) / entry_price
            else:  # SHORT
                profit_factor = (entry_price - current_price) / entry_price
            
            position["unrealized_pnl"] = size * leverage * profit_factor
            position["unrealized_pnl_pct"] = profit_factor * leverage * 100
            
            # Check for liquidation
            max_loss_pct = 100 / leverage
            if position["unrealized_pnl_pct"] <= -max_loss_pct * 0.95:  # Close at 95% of liquidation price
                logger.warning(f"Position {position['pair']} {direction} approaching liquidation threshold!")
                
                # In real implementation, this would force close the position
                # For now, we'll just make note of it
                position["liquidation_warning"] = True
            
            # Log significant price moves
            if abs(position["unrealized_pnl_pct"]) > 5:
                logger.info(f"Position {position['pair']} {direction} has {position['unrealized_pnl_pct']:.2f}% P&L")
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _try_open_position(self) -> Optional[Dict]:
        """
        Try to open a new position based on market conditions and ML predictions.
        
        Returns:
            New position data or None if no position was opened
        """
        try:
            # Check if we should make a new trade
            if time.time() - self.last_trade_time < self.trade_interval:
                return None
            
            # Get tradable pairs
            tradable_pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                              "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
            
            # Check if we already have positions for any pairs
            current_pairs = [p["pair"] for p in self.positions]
            available_pairs = [p for p in tradable_pairs if p not in current_pairs]
            
            if not available_pairs:
                logger.debug("No available pairs to trade")
                return None
            
            # Randomly choose a pair for demonstration
            pair = random.choice(available_pairs)
            
            # Get current price
            price = get_price(pair)
            if price is None:
                logger.warning(f"Could not get price for {pair}")
                return None
            
            # In a real implementation, ML models would be used here
            # For demonstration, we'll simulate ML predictions
            confidence = random.uniform(0.5, 0.95)
            direction = random.choice(["LONG", "SHORT"])
            
            # Calculate dynamic leverage based on confidence
            # Higher confidence = higher leverage, between 5x and 125x
            base_leverage = 5
            max_leverage = 125
            leverage = base_leverage + (max_leverage - base_leverage) * confidence
            
            # Calculate position size
            # Risk 0.2-2% of available capital based on confidence
            risk_pct = 0.2 + (2.0 - 0.2) * confidence
            max_capital_per_trade = self.portfolio["available_capital"] * (risk_pct / 100)
            
            # Ensure it's at least $10
            max_capital_per_trade = max(10, max_capital_per_trade)
            
            # Calculate size based on price and leverage
            size = max_capital_per_trade / price
            
            # Ensure we have enough capital
            required_capital = (size * price) / leverage
            if required_capital > self.portfolio["available_capital"]:
                logger.warning(f"Insufficient capital for {pair} trade: Need ${required_capital:.2f}, have ${self.portfolio['available_capital']:.2f}")
                return None
            
            # Create position
            position = {
                "pair": pair,
                "direction": direction,
                "entry_price": price,
                "current_price": price,
                "size": size,
                "leverage": leverage,
                "entry_time": datetime.datetime.now().isoformat(),
                "last_update_time": datetime.datetime.now().isoformat(),
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "confidence": confidence
            }
            
            # Update portfolio
            self.portfolio["available_capital"] -= required_capital
            self._save_portfolio()
            
            # Update last trade time
            self.last_trade_time = time.time()
            
            # Log the new position
            self._log_trade(position, "OPEN")
            
            logger.info(f"Opened {direction} position for {pair} with leverage {leverage:.1f}x and confidence {confidence:.2f}")
            return position
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _try_close_positions(self) -> List[Dict]:
        """
        Try to close positions based on market conditions and ML predictions.
        
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        try:
            for i, position in enumerate(self.positions):
                # In a real implementation, close decisions would be based on ML predictions
                # For demonstration, we'll close positions with 10% chance or if they hit profit/loss thresholds
                
                # Update position first
                self._update_position(position)
                
                # Decision to close:
                # 1. Random chance (10%)
                # 2. Profit > 10%
                # 3. Loss > 20%
                close_random = random.random() < 0.1
                close_profit = position["unrealized_pnl_pct"] > 10
                close_loss = position["unrealized_pnl_pct"] < -20
                liquidation_warning = position.get("liquidation_warning", False)
                
                should_close = close_random or close_profit or close_loss or liquidation_warning
                
                if should_close:
                    # Get updated price
                    current_price = get_price(position["pair"])
                    if current_price is None:
                        logger.warning(f"Could not close position for {position['pair']}: Price unavailable")
                        continue
                    
                    # Update final price
                    position["current_price"] = current_price
                    
                    # Log reason for closing
                    close_reason = "random" if close_random else (
                        "profit target" if close_profit else (
                        "stop loss" if close_loss else "liquidation risk"
                        )
                    )
                    logger.info(f"Closing {position['pair']} position due to {close_reason}")
                    
                    # Close the position and update portfolio
                    self.portfolio["available_capital"] += (
                        (position["size"] * position["current_price"]) / position["leverage"] +
                        position["unrealized_pnl"]
                    )
                    
                    # Log the closed trade
                    self._log_trade(position, "CLOSE")
                    
                    # Add to closed positions
                    closed_positions.append(position)
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
        
        # Remove closed positions
        self.positions = [p for p in self.positions if p not in closed_positions]
        self._save_positions()
        
        return closed_positions
    
    def _update_all_positions(self) -> None:
        """Update all positions with current prices"""
        for position in self.positions:
            self._update_position(position)
        
        # Save updated positions
        self._save_positions()
        
        # Update portfolio with new P&L values
        self._update_portfolio_pnl()
    
    def run(self) -> None:
        """Run the trading bot"""
        logger.info("Starting isolated trading bot...")
        
        try:
            # Main trading loop
            while True:
                # Update all positions
                self._update_all_positions()
                
                # Try to close positions
                closed_positions = self._try_close_positions()
                if closed_positions:
                    logger.info(f"Closed {len(closed_positions)} positions")
                
                # Try to open new positions
                new_position = self._try_open_position()
                if new_position:
                    self.positions.append(new_position)
                    self._save_positions()
                
                # Update portfolio history
                if time.time() - self.last_update_time > 60:  # Once per minute
                    self._update_portfolio_history()
                    self.last_update_time = time.time()
                
                # Display status
                if random.random() < 0.05:  # ~5% chance each iteration
                    self.get_status()
                
                # Sleep to prevent excessive CPU usage
                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Shutting down trading bot...")
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
        finally:
            # Final portfolio update
            self._update_all_positions()
            self._update_portfolio_history()
            logger.info("Trading bot shutdown complete")
    
    def get_status(self) -> Dict:
        """
        Get current bot status.
        
        Returns:
            Dictionary with current status information
        """
        try:
            # Update all positions first
            self._update_all_positions()
            
            # Prepare status info
            status = {
                "timestamp": datetime.datetime.now().isoformat(),
                "portfolio": {
                    "total_value": self.portfolio["total_value"],
                    "available_capital": self.portfolio["available_capital"],
                    "total_pnl": self.portfolio["total_pnl"],
                    "total_pnl_pct": self.portfolio["total_pnl_pct"],
                    "unrealized_pnl": self.portfolio["unrealized_pnl"],
                    "unrealized_pnl_pct": self.portfolio["unrealized_pnl_pct"],
                    "win_rate": self.portfolio["win_rate"]
                },
                "active_positions": len(self.positions),
                "trades_today": sum(1 for t in self._load_trades() if 
                                  datetime.datetime.fromisoformat(t["timestamp"]).date() == 
                                  datetime.datetime.now().date())
            }
            
            # Log status
            logger.info(f"Portfolio: ${status['portfolio']['total_value']:.2f} | " +
                      f"P&L: ${status['portfolio']['total_pnl']:.2f} ({status['portfolio']['total_pnl_pct']:.2f}%) | " +
                      f"Win Rate: {status['portfolio']['win_rate']*100:.1f}% | " +
                      f"Active Positions: {status['active_positions']}")
            
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}

def main():
    """Main function"""
    try:
        # Print startup banner
        print("\n" + "=" * 60)
        print(" ISOLATED TRADING BOT STARTING")
        print("=" * 60 + "\n")
        
        # Create and run the bot
        bot = IsolatedTradingBot()
        bot.run()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        print(f"Error running trading bot: {e}")
    finally:
        print("\n" + "=" * 60)
        print(" ISOLATED TRADING BOT SHUTDOWN")
        print("=" * 60)

if __name__ == "__main__":
    main()