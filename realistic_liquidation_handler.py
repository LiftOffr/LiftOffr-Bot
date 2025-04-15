#!/usr/bin/env python3
"""
Realistic Liquidation Handler

This module provides enhanced liquidation handling for the trading bot:
1. Real-time monitoring of positions for liquidation conditions
2. Accurate liquidation price calculations based on Kraken's rules
3. Realistic fees and penalties for liquidation events
4. Smart liquidation risk management
"""

import os
import json
import time
import logging
import datetime
import random
from typing import Dict, List, Optional, Tuple, Any
from kraken_realtime_price_fetcher import KrakenRealtimePriceFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
LIQUIDATIONS_LOG_FILE = f"{DATA_DIR}/liquidations.json"

class RealisticLiquidationHandler:
    """
    Handles realistic liquidation events for leveraged positions
    
    This class monitors positions for liquidation conditions and properly
    processes liquidation events when they occur.
    """
    
    def __init__(self, price_fetcher: Optional[KrakenRealtimePriceFetcher] = None):
        """
        Initialize the liquidation handler
        
        Args:
            price_fetcher: Optional price fetcher for real-time prices
        """
        self.price_fetcher = price_fetcher or KrakenRealtimePriceFetcher()
        self.positions = self._load_positions()
        self.trades = self._load_trades()
        self.portfolio = self._load_portfolio()
        self.liquidations = self._load_liquidations()
        
        # Liquidation parameters (typical values for crypto exchanges)
        self.maintenance_margin = 0.01  # 1%
        self.liquidation_fee = 0.0075   # 0.75%
        self.min_time_between_checks = 5  # seconds
        self.last_check_time = 0
        
        # Register for price updates
        if self.price_fetcher:
            self.price_fetcher.register_update_callback(self._on_price_update)
    
    def _load_json(self, filepath: str, default: Any = None) -> Any:
        """
        Load JSON data from file or return default if file doesn't exist
        
        Args:
            filepath: Path to JSON file
            default: Default value if file doesn't exist or is invalid
            
        Returns:
            Loaded data or default value
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default
    
    def _save_json(self, filepath: str, data: Any) -> None:
        """
        Save JSON data to file
        
        Args:
            filepath: Path to JSON file
            data: Data to save
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_positions(self) -> List[Dict]:
        """
        Load positions from file
        
        Returns:
            List of position dictionaries
        """
        return self._load_json(POSITIONS_FILE, [])
    
    def _load_trades(self) -> List[Dict]:
        """
        Load trades from file
        
        Returns:
            List of trade dictionaries
        """
        return self._load_json(TRADES_FILE, [])
    
    def _load_portfolio(self) -> Dict:
        """
        Load portfolio from file
        
        Returns:
            Portfolio dictionary
        """
        return self._load_json(PORTFOLIO_FILE, {"balance": 20000.0})
    
    def _load_liquidations(self) -> List[Dict]:
        """
        Load liquidation history from file
        
        Returns:
            List of liquidation events
        """
        return self._load_json(LIQUIDATIONS_LOG_FILE, [])
    
    def _save_positions(self) -> None:
        """Save positions to file"""
        self._save_json(POSITIONS_FILE, self.positions)
    
    def _save_trades(self) -> None:
        """Save trades to file"""
        self._save_json(TRADES_FILE, self.trades)
    
    def _save_portfolio(self) -> None:
        """Save portfolio to file"""
        self._save_json(PORTFOLIO_FILE, self.portfolio)
    
    def _save_liquidations(self) -> None:
        """Save liquidation history to file"""
        self._save_json(LIQUIDATIONS_LOG_FILE, self.liquidations)
    
    def _on_price_update(self, pair: str, price: float) -> None:
        """
        Handle price updates from the price fetcher
        
        Args:
            pair: Trading pair
            price: Current price
        """
        current_time = time.time()
        if current_time - self.last_check_time < self.min_time_between_checks:
            return  # Avoid excessive checks
        
        self.last_check_time = current_time
        self.check_liquidations()
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        direction: str,
        maintenance_margin: Optional[float] = None
    ) -> float:
        """
        Calculate liquidation price for a position
        
        Args:
            entry_price: Entry price
            leverage: Leverage used
            direction: 'Long' or 'Short'
            maintenance_margin: Optional maintenance margin (default: self.maintenance_margin)
            
        Returns:
            Liquidation price
        """
        if maintenance_margin is None:
            maintenance_margin = self.maintenance_margin
        
        if direction.lower() == "long":
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
        else:  # Short
            liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
        
        return liquidation_price
    
    def check_liquidations(self) -> None:
        """
        Check all open positions for liquidation conditions
        
        This function should be called regularly to monitor positions,
        or when price updates are received.
        """
        if not self.positions:
            return
        
        # Get latest prices for all pairs
        all_prices = self.price_fetcher.get_all_latest_prices()
        if not all_prices:
            logger.warning("No price data available for liquidation check")
            return
        
        # Check each position
        liquidated_positions = []
        for position in self.positions:
            pair = position["pair"]
            direction = position["direction"]
            entry_price = position["entry_price"]
            leverage = position["leverage"]
            liquidation_price = position.get("liquidation_price")
            
            # Calculate liquidation price if not already set
            if not liquidation_price:
                liquidation_price = self.calculate_liquidation_price(
                    entry_price, leverage, direction
                )
                position["liquidation_price"] = liquidation_price
            
            # Check if current price is available
            if pair not in all_prices:
                continue
            
            current_price = all_prices[pair]
            
            # Check for liquidation
            liquidated = False
            if direction.lower() == "long" and current_price <= liquidation_price:
                liquidated = True
            elif direction.lower() == "short" and current_price >= liquidation_price:
                liquidated = True
            
            if liquidated:
                logger.warning(f"LIQUIDATION: {pair} {direction} position liquidated at {current_price}")
                liquidated_positions.append((position, current_price))
        
        # Process liquidations if any
        if liquidated_positions:
            self._process_liquidations(liquidated_positions)
    
    def _process_liquidations(self, liquidated_positions: List[Tuple[Dict, float]]) -> None:
        """
        Process liquidation events
        
        Args:
            liquidated_positions: List of (position, liquidation_price) tuples
        """
        for position, liquidation_price in liquidated_positions:
            self._handle_liquidation(position, liquidation_price)
    
    def _handle_liquidation(self, position: Dict, liquidation_price: float) -> None:
        """
        Handle a liquidation event
        
        Args:
            position: Position being liquidated
            liquidation_price: Price at which liquidation occurred
        """
        pair = position["pair"]
        strategy = position["strategy"]
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]
        leverage = position["leverage"]
        margin = size / leverage
        entry_time = position["entry_time"]
        
        # Calculate liquidation fee
        liquidation_fee = margin * self.liquidation_fee
        
        # Create trade record for liquidation
        liquidation_time = datetime.datetime.now().isoformat()
        trade = {
            "timestamp": liquidation_time,
            "pair": pair,
            "strategy": strategy,
            "type": "Liquidation",
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": liquidation_price,
            "size": size,
            "leverage": leverage,
            "fees": liquidation_fee,
            "slippage": 0,
            "pnl_percentage": -100.0,  # 100% loss
            "pnl_amount": -margin,
            "exit_reason": "Liquidation"
        }
        
        # Calculate duration
        entry_time_dt = datetime.datetime.fromisoformat(entry_time.replace('Z', '+00:00') if entry_time.endswith('Z') else entry_time)
        exit_time_dt = datetime.datetime.fromisoformat(liquidation_time.replace('Z', '+00:00') if liquidation_time.endswith('Z') else liquidation_time)
        duration_seconds = (exit_time_dt - entry_time_dt).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        trade["duration"] = f"{hours}h {minutes}m"
        
        # Create liquidation event record
        liquidation_event = {
            "timestamp": liquidation_time,
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "liquidation_price": liquidation_price,
            "size": size,
            "leverage": leverage,
            "margin": margin,
            "liquidation_fee": liquidation_fee,
            "total_loss": margin + liquidation_fee,
            "duration": trade["duration"],
            "strategy": strategy
        }
        
        # Update portfolio balance
        balance = self.portfolio.get("balance", 20000.0)
        new_balance = balance - margin - liquidation_fee
        self.portfolio["balance"] = new_balance
        self.portfolio["last_updated"] = liquidation_time
        
        # Update trades and portfolio records
        self.trades.append(trade)
        if "trades" in self.portfolio:
            self.portfolio["trades"].append({
                "pair": pair,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": liquidation_price,
                "entry_time": entry_time,
                "exit_time": liquidation_time,
                "pnl_percentage": -100.0,
                "pnl_amount": -margin,
                "exit_reason": "Liquidation",
                "position_size": size,
                "leverage": leverage,
                "confidence": position.get("confidence", 0.0),
                "strategy": strategy
            })
        
        # Calculate total trades and winning trades
        total_trades = self.portfolio.get("total_trades", 0) + 1
        winning_trades = self.portfolio.get("winning_trades", 0)
        self.portfolio["total_trades"] = total_trades
        self.portfolio["winning_trades"] = winning_trades
        
        # Add liquidation to history
        self.liquidations.append(liquidation_event)
        
        # Remove the position
        self.positions = [p for p in self.positions if not (
            p["pair"] == pair and 
            p["strategy"] == strategy and
            p["direction"] == direction
        )]
        
        # Save updated data
        self._save_positions()
        self._save_trades()
        self._save_portfolio()
        self._save_liquidations()
        
        logger.warning(
            f"Liquidation processed: {direction} {pair} at {liquidation_price}. "
            f"Loss: {margin + liquidation_fee:.2f} USD"
        )
    
    def start(self):
        """Start the liquidation handler"""
        # Start the price fetcher if not already running
        if not getattr(self.price_fetcher, 'running', False):
            self.price_fetcher.start()
        
        logger.info("Started liquidation handler")
    
    def stop(self):
        """Stop the liquidation handler"""
        # Unregister from price updates
        if self.price_fetcher:
            self.price_fetcher.unregister_update_callback(self._on_price_update)
        
        logger.info("Stopped liquidation handler")


# Usage example
if __name__ == "__main__":
    import signal
    
    # Set up signal handling
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        handler.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start the liquidation handler
    price_fetcher = KrakenRealtimePriceFetcher()
    handler = RealisticLiquidationHandler(price_fetcher)
    handler.start()
    
    logger.info("Liquidation handler running. Press Ctrl+C to exit")
    
    # Keep the main thread running
    while True:
        try:
            time.sleep(1)
            # Periodically check for liquidations
            handler.check_liquidations()
        except KeyboardInterrupt:
            break
    
    handler.stop()