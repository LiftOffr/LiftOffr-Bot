#!/usr/bin/env python3
"""
Trade Entry Manager with Available Capital Tracking

This module manages opening new trades while tracking available capital.
It ensures new trades only use a percentage of the remaining available capital,
not the total portfolio balance.
"""
import os
import json
import logging
import random
from datetime import datetime

# Import risk management functions
from risk_management import (
    prepare_trade_entry,
    calculate_risk_adjusted_leverage,
    calculate_position_size
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
STOP_LOSS_PCT = 4.0  # Fixed at 4%

class TradeEntryManager:
    """
    Manages trade entries with available capital tracking.
    
    This class handles opening new trades while ensuring they only use
    a percentage of the remaining available capital, not the total portfolio.
    """
    
    def __init__(self):
        """Initialize the trade entry manager"""
        self.portfolio = self._load_portfolio()
        self.positions = self._load_positions()
        self.trades = self._load_trades()
        
        # Calculate available capital
        self.available_capital = self._calculate_available_capital()
        logger.info(f"Trade entry manager initialized with ${self.available_capital:.2f} available capital")
    
    def _load_portfolio(self):
        """Load portfolio data"""
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        return {"balance": 20000.0, "available_capital": 20000.0}
    
    def _load_positions(self):
        """Load current positions"""
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def _load_trades(self):
        """Load current trades"""
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        return []
    
    def _calculate_available_capital(self):
        """Calculate available capital after existing positions"""
        # Get portfolio balance
        balance = self.portfolio.get("balance", 20000.0)
        
        # Calculate capital used by existing positions
        used_capital = sum(position.get("position_size", 0) for position in self.positions)
        
        # Available capital is balance minus used capital
        available_capital = balance - used_capital
        
        # Ensure available capital is not negative
        if available_capital < 0:
            available_capital = 0
            logger.warning("Available capital is negative, setting to zero")
        
        return available_capital
    
    def _save_portfolio(self):
        """Save updated portfolio"""
        # Update available capital in portfolio
        self.portfolio["available_capital"] = self.available_capital
        self.portfolio["updated_at"] = datetime.now().isoformat()
        
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def _save_positions(self):
        """Save updated positions"""
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def _save_trades(self):
        """Save updated trades"""
        with open(TRADES_FILE, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def can_open_trade(self, position_size):
        """
        Check if a trade can be opened with the given position size.
        
        Args:
            position_size (float): Proposed position size
            
        Returns:
            bool: True if trade can be opened, False otherwise
        """
        return position_size <= self.available_capital and self.available_capital > 0
    
    def generate_trade_id(self):
        """Generate a unique trade ID"""
        # Get highest current trade ID number
        highest_id = 0
        for trade in self.trades:
            trade_id = trade.get("id", "")
            if trade_id.startswith("trade_"):
                try:
                    id_num = int(trade_id.split("_")[1])
                    highest_id = max(highest_id, id_num)
                except (ValueError, IndexError):
                    pass
        
        # Generate new ID
        new_id = highest_id + 1
        return f"trade_{new_id}"
    
    def open_trade(self, pair, entry_price, confidence, model="Adaptive", category=None):
        """
        Open a new trade with the given parameters, using only available capital.
        
        Args:
            pair (str): Trading pair
            entry_price (float): Entry price
            confidence (float): Model confidence (0.0-1.0)
            model (str): Model name
            category (str): Strategy category
            
        Returns:
            dict: New position if trade was opened, None otherwise
        """
        # Refresh available capital
        self.available_capital = self._calculate_available_capital()
        
        if self.available_capital <= 0:
            logger.warning(f"Cannot open trade for {pair} - no available capital")
            return None
        
        # Prepare trade with risk management using AVAILABLE capital
        trade_params = prepare_trade_entry(
            portfolio_balance=self.available_capital,  # Use available capital only!
            pair=pair,
            confidence=confidence,
            entry_price=entry_price
        )
        
        # Extract parameters
        direction = trade_params['direction']
        leverage = trade_params['leverage']
        position_size = trade_params['position_size']
        stop_loss_pct = trade_params['stop_loss_pct']
        take_profit_pct = trade_params['take_profit_pct']
        liquidation_price = trade_params['liquidation_price']
        
        # Check if we have enough available capital
        if not self.can_open_trade(position_size):
            logger.warning(f"Cannot open trade for {pair} - insufficient available capital (${self.available_capital:.2f})")
            
            # Try with a smaller position size (50% of available)
            max_position = self.available_capital * 0.5
            if max_position > 100:  # Minimum viable position size
                position_size = max_position
                logger.info(f"Adjusting position size to ${position_size:.2f} to fit available capital")
            else:
                logger.warning(f"Available capital too low for a viable trade")
                return None
        
        # Generate trade ID
        trade_id = self.generate_trade_id()
        
        # Select category if not provided
        if not category:
            category = random.choice(["those dudes", "him all along"])
        
        # Create position
        position = {
            "pair": pair,
            "entry_price": entry_price,
            "current_price": entry_price,
            "position_size": position_size,
            "direction": direction,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "unrealized_pnl_pct": 0.0,
            "unrealized_pnl_amount": 0.0,
            "current_value": position_size,
            "confidence": confidence,
            "model": model,
            "category": category,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "liquidation_price": liquidation_price,
            "open_trade_id": trade_id
        }
        
        # Create trade record
        trade = {
            "id": trade_id,
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "position_size": position_size,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "status": "open",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "exit_price": None,
            "exit_time": None,
            "confidence": confidence,
            "model": model,
            "category": category,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "liquidation_price": liquidation_price
        }
        
        # Update available capital
        self.available_capital -= position_size
        
        # Update data structures
        self.positions.append(position)
        self.trades.append(trade)
        
        # Save changes
        self._save_portfolio()
        self._save_positions()
        self._save_trades()
        
        logger.info(f"Opened {direction} trade for {pair} at ${entry_price} with {leverage}x leverage")
        logger.info(f"Position size: ${position_size:.2f}, Available capital remaining: ${self.available_capital:.2f}")
        
        return position
    
    def get_available_capital(self):
        """Get current available capital"""
        return self._calculate_available_capital()
    
if __name__ == "__main__":
    # Example usage
    trade_manager = TradeEntryManager()
    
    # Get available capital
    available = trade_manager.get_available_capital()
    print(f"Available capital: ${available:.2f}")
    
    # Try to open a trade
    if available > 1000:
        new_position = trade_manager.open_trade(
            pair="ETH/USD",
            entry_price=1600.0,
            confidence=0.85,
            model="Adaptive",
            category="those dudes"
        )
        
        if new_position:
            print(f"Opened new trade. New available capital: ${trade_manager.get_available_capital():.2f}")