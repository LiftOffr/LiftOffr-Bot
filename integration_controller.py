#!/usr/bin/env python3
"""
Integration Controller

This module coordinates the integration of real-time market data and liquidation
handling with the existing trading bot infrastructure. It serves as the central
controller for real-time data flow and risk management.
"""

import os
import json
import time
import logging
import threading
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
MARKET_DATA_DIR = f"{DATA_DIR}/market_data"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD",
    "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD",
    "UNI/USD", "ATOM/USD"
]

# Create data directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MARKET_DATA_DIR, exist_ok=True)

class IntegrationController:
    """
    Main controller for integrating real-time market data with the trading bot.
    
    This class:
    1. Fetches real-time price data from Kraken API
    2. Calculates accurate liquidation prices
    3. Updates position prices and unrealized PnL
    4. Manages liquidation checks
    """
    
    def __init__(self, pairs: Optional[List[str]] = None):
        """
        Initialize the integration controller.
        
        Args:
            pairs: List of trading pairs to monitor
        """
        self.pairs = pairs or DEFAULT_PAIRS
        self.running = False
        self.price_thread = None
        self.update_thread = None
        self.latest_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        self.last_update_time = 0
        
        # Maintenance margin requirements (simplified)
        # In reality, these vary by exchange and can depend on position size
        self.maintenance_margins = {
            # Format: leverage: maintenance_margin_percentage (as decimal)
            1: 0.0,      # No margin required for 1x leverage
            2: 0.025,    # 2.5% maintenance margin for 2x leverage
            3: 0.05,     # 5% for 3x
            5: 0.10,     # 10% for 5x
            10: 0.15,    # 15% for 10x
            25: 0.25,    # 25% for 25x
            50: 0.35,    # 35% for 50x
            75: 0.45,    # 45% for 75x
            100: 0.50,   # 50% for 100x
            125: 0.60    # 60% for 125x
        }
    
    def start(self):
        """Start the integration controller threads."""
        if self.running:
            logger.warning("Integration controller already running")
            return
        
        self.running = True
        
        # Start price update thread
        self.price_thread = threading.Thread(target=self._price_update_loop)
        self.price_thread.daemon = True
        self.price_thread.start()
        
        # Start position update thread
        self.update_thread = threading.Thread(target=self._position_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Integration controller started")
    
    def stop(self):
        """Stop the integration controller threads."""
        self.running = False
        
        # Wait for threads to terminate
        if self.price_thread and self.price_thread.is_alive():
            self.price_thread.join(timeout=2.0)
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        
        logger.info("Integration controller stopped")
    
    def _price_update_loop(self):
        """Background thread for updating prices."""
        while self.running:
            try:
                self.update_prices()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _position_update_loop(self):
        """Background thread for updating positions."""
        while self.running:
            try:
                # Update positions with latest prices
                self.update_position_prices()
                
                # Check for liquidations
                self.check_liquidations()
                
                # Update portfolio value
                self.update_portfolio()
                
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in position update loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def update_prices(self):
        """Update prices for all pairs."""
        current_time = time.time()
        
        # Simulated price updates for demonstration
        # In production, this would fetch from Kraken API
        for pair in self.pairs:
            base_price = self._get_base_price(pair)
            
            # Apply small random price movement (up to ±1%)
            price_change = random.uniform(-0.01, 0.01)
            current_price = base_price * (1 + price_change)
            
            # Save current price
            self.latest_prices[pair] = current_price
            
            # Save to price history
            if pair not in self.price_history:
                self.price_history[pair] = []
            
            self.price_history[pair].append({
                "timestamp": current_time,
                "price": current_price
            })
            
            # Keep only last 1000 price points per pair
            if len(self.price_history[pair]) > 1000:
                self.price_history[pair] = self.price_history[pair][-1000:]
        
        # Save to disk occasionally (every minute)
        if current_time - self.last_update_time > 60:
            self._save_market_data()
            self.last_update_time = current_time
    
    def _get_base_price(self, pair: str) -> float:
        """
        Get base price for a pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            base_price: Base price for the pair
        """
        # Reference prices (April 2023 approximate values)
        base_prices = {
            "BTC/USD": 29875.0,
            "ETH/USD": 1940.0,
            "SOL/USD": 84.75,
            "ADA/USD": 0.383,
            "DOT/USD": 6.15,
            "LINK/USD": 13.85,
            "AVAX/USD": 17.95,
            "MATIC/USD": 0.98,
            "UNI/USD": 5.75,
            "ATOM/USD": 11.65
        }
        
        return base_prices.get(pair, 100.0)  # Default to $100 if pair not found
    
    def _save_market_data(self):
        """Save market data to disk."""
        try:
            # Save latest prices
            prices_file = f"{MARKET_DATA_DIR}/latest_prices.json"
            with open(prices_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "prices": self.latest_prices
                }, f, indent=2)
            
            # Save price history for each pair
            for pair in self.price_history:
                pair_file = f"{MARKET_DATA_DIR}/{pair.replace('/', '_')}_prices.json"
                with open(pair_file, 'w') as f:
                    json.dump(self.price_history[pair][-100:], f, indent=2)  # Save last 100 points
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    def update_position_prices(self):
        """
        Update all positions with current prices and calculate unrealized PnL.
        """
        try:
            # Load positions
            if not os.path.exists(POSITIONS_FILE):
                return
            
            with open(POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            
            if not positions:
                return
            
            # Update each position
            for i, position in enumerate(positions):
                pair = position.get("pair")
                direction = position.get("direction", "Long")
                entry_price = position.get("entry_price", 0)
                size = position.get("size", 0)
                leverage = position.get("leverage", 1)
                
                if pair in self.latest_prices:
                    current_price = self.latest_prices[pair]
                    
                    # Update position with current price
                    position["current_price"] = current_price
                    
                    # Calculate unrealized PnL percentage
                    if direction.lower() == "long":
                        pnl_pct = (current_price / entry_price - 1) * leverage
                    else:
                        pnl_pct = (1 - current_price / entry_price) * leverage
                    
                    position["unrealized_pnl"] = pnl_pct
                    
                    # Calculate unrealized PnL amount
                    notional_value = size * entry_price
                    position["unrealized_pnl_amount"] = notional_value * pnl_pct
                    
                    # Ensure liquidation price is set
                    if "liquidation_price" not in position:
                        position["liquidation_price"] = self.calculate_liquidation_price(
                            entry_price, leverage, direction
                        )
            
            # Save updated positions
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(positions, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating position prices: {e}")
    
    def check_liquidations(self):
        """
        Check all positions for liquidation conditions.
        """
        try:
            # Load positions
            if not os.path.exists(POSITIONS_FILE):
                return
            
            with open(POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            
            if not positions:
                return
            
            # Check each position for liquidation
            active_positions = []
            liquidated_positions = []
            
            for position in positions:
                pair = position.get("pair")
                direction = position.get("direction", "Long")
                current_price = position.get("current_price")
                liquidation_price = position.get("liquidation_price")
                
                if not current_price or not liquidation_price:
                    active_positions.append(position)
                    continue
                
                # Check if position is liquidated
                is_liquidated = False
                
                if direction.lower() == "long" and current_price <= liquidation_price:
                    is_liquidated = True
                elif direction.lower() == "short" and current_price >= liquidation_price:
                    is_liquidated = True
                
                if is_liquidated:
                    # Mark as liquidated
                    position["liquidated"] = True
                    position["liquidation_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                    
                    # Calculate loss amount (usually 100% of margin, minus fees)
                    entry_price = position.get("entry_price", 0)
                    size = position.get("size", 0)
                    leverage = position.get("leverage", 1)
                    
                    # Calculate margin and full loss
                    notional_value = size * entry_price
                    margin = notional_value / leverage
                    position["liquidation_loss"] = -margin
                    
                    # Log liquidation
                    logger.warning(
                        f"Position liquidated: {pair} {direction} {leverage}x - "
                        f"Loss: ${margin:.2f}"
                    )
                    
                    # Record for handling
                    liquidated_positions.append(position)
                else:
                    active_positions.append(position)
            
            # If any positions were liquidated, handle them
            if liquidated_positions:
                self._handle_liquidations(active_positions, liquidated_positions)
            
        except Exception as e:
            logger.error(f"Error checking liquidations: {e}")
    
    def _handle_liquidations(self, active_positions, liquidated_positions):
        """
        Handle liquidated positions.
        
        Args:
            active_positions: List of positions still active
            liquidated_positions: List of liquidated positions
        """
        try:
            # Save active positions
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(active_positions, f, indent=2)
            
            # Add liquidated positions to trades history
            if not os.path.exists(TRADES_FILE):
                trades = []
            else:
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            
            # Convert liquidated positions to closed trades
            for position in liquidated_positions:
                trade = {
                    "pair": position.get("pair"),
                    "direction": position.get("direction"),
                    "size": position.get("size"),
                    "entry_price": position.get("entry_price"),
                    "exit_price": position.get("current_price"),
                    "leverage": position.get("leverage"),
                    "strategy": position.get("strategy"),
                    "entry_time": position.get("entry_time"),
                    "exit_time": position.get("liquidation_time"),
                    "profit_loss": position.get("liquidation_loss", 0),
                    "profit_loss_percentage": -100.0,  # 100% loss on liquidation
                    "liquidated": True
                }
                trades.append(trade)
            
            # Save updated trades
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)
            
            # Update portfolio
            self.update_portfolio()
            
        except Exception as e:
            logger.error(f"Error handling liquidations: {e}")
    
    def update_portfolio(self):
        """
        Update portfolio value with current unrealized PnL.
        """
        try:
            # Load portfolio
            if not os.path.exists(PORTFOLIO_FILE):
                # Initialize with default portfolio
                portfolio = {
                    "balance": 20000.0,
                    "unrealized_pnl_usd": 0.0,
                    "total_value": 20000.0,
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                }
            else:
                with open(PORTFOLIO_FILE, 'r') as f:
                    portfolio = json.load(f)
            
            # Get all open positions
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    positions = json.load(f)
            else:
                positions = []
            
            # Calculate total unrealized PnL
            unrealized_pnl = 0.0
            for position in positions:
                if "unrealized_pnl_amount" in position:
                    unrealized_pnl += position["unrealized_pnl_amount"]
            
            # Update portfolio values
            portfolio["unrealized_pnl_usd"] = unrealized_pnl
            portfolio["total_value"] = portfolio.get("balance", 20000.0) + unrealized_pnl
            portfolio["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            
            # Save updated portfolio
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            # Update portfolio history
            self._update_portfolio_history(portfolio)
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    def _update_portfolio_history(self, portfolio):
        """
        Update portfolio history with current value.
        
        Args:
            portfolio: Current portfolio state
        """
        try:
            # Load history
            if not os.path.exists(PORTFOLIO_HISTORY_FILE):
                history = []
            else:
                with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            
            # Add current snapshot
            history.append({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "portfolio_value": portfolio["total_value"]
            })
            
            # Keep only last 1000 points to avoid huge files
            if len(history) > 1000:
                history = history[-1000:]
            
            # Save updated history
            with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def calculate_liquidation_price(self, entry_price: float, leverage: float, direction: str) -> float:
        """
        Calculate liquidation price for a position.
        
        Args:
            entry_price: Position entry price
            leverage: Position leverage
            direction: "Long" or "Short"
            
        Returns:
            liquidation_price: Price at which the position would be liquidated
        """
        # Get maintenance margin requirement
        maintenance_margin = self._get_maintenance_margin(leverage)
        
        # Calculate liquidation price
        if direction.lower() == "long":
            # Long position liquidation:
            # Liquidation happens when: equity ≤ maintenance_margin
            # equity = margin + position_value - initial_position_value
            # position_value = initial_position_value * (current_price / entry_price)
            # At liquidation: margin * (1 - maintenance_margin) = initial_position_value - position_value
            # Solving for liquidation_price:
            liquidation_price = entry_price * (1 - (1 - maintenance_margin) / leverage)
        else:
            # Short position liquidation:
            # Similar logic but inverse price movement
            liquidation_price = entry_price * (1 + (1 - maintenance_margin) / leverage)
        
        return liquidation_price
    
    def _get_maintenance_margin(self, leverage: float) -> float:
        """
        Get maintenance margin requirement for a leverage level.
        
        Args:
            leverage: Position leverage
            
        Returns:
            maintenance_margin: Maintenance margin as a decimal (0.0-1.0)
        """
        # Find the closest leverage tier
        leverage_tiers = sorted(list(self.maintenance_margins.keys()))
        
        # Find the applicable tier
        applicable_tier = leverage_tiers[0]
        for tier in leverage_tiers:
            if leverage >= tier:
                applicable_tier = tier
            else:
                break
        
        # Return the maintenance margin for that tier
        return self.maintenance_margins.get(applicable_tier, 0.5)  # Default to 50% if not found