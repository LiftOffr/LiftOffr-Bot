#!/usr/bin/env python3
"""
Realtime Trading Integration

This module integrates the real-time price data and liquidation handling
with the trading bot system. It ensures accurate price data is used for
trading decisions and provides realistic simulation of market conditions.
"""

import os
import json
import time
import logging
import datetime
import threading
import signal
import random
from typing import Dict, List, Optional, Any, Tuple

from kraken_realtime_price_fetcher import KrakenRealtimePriceFetcher
from realistic_liquidation_handler import RealisticLiquidationHandler

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
MARKET_DATA_DIR = f"{DATA_DIR}/market_data"
PRICE_SNAPSHOT_FILE = f"{MARKET_DATA_DIR}/current_prices.json"
ORDER_BOOK_SNAPSHOT_FILE = f"{MARKET_DATA_DIR}/order_book_snapshot.json"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MARKET_DATA_DIR, exist_ok=True)

class RealtimeTradingIntegration:
    """
    Integrates real-time market data with the trading bot
    
    This class:
    1. Maintains real-time price data from Kraken
    2. Handles liquidation events based on real-time prices
    3. Updates position data with current prices
    4. Records price snapshots for analysis
    """
    
    def __init__(self):
        """Initialize the integration"""
        self.price_fetcher = KrakenRealtimePriceFetcher()
        self.liquidation_handler = RealisticLiquidationHandler(self.price_fetcher)
        self.positions = self._load_positions()
        self.running = False
        self.snapshot_interval = 60  # seconds
        self.last_snapshot_time = 0
        self.update_interval = 5  # seconds
        
        # Register for price updates
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
    
    def _save_positions(self) -> None:
        """Save positions to file"""
        self._save_json(POSITIONS_FILE, self.positions)
    
    def _on_price_update(self, pair: str, price: float) -> None:
        """
        Handle price updates from the price fetcher
        
        This callback is triggered whenever a new price is received.
        It updates position data and triggers liquidation checks.
        
        Args:
            pair: Trading pair
            price: Current price
        """
        # Update positions with new price
        positions_updated = False
        
        for position in self.positions:
            if position["pair"] == pair:
                position["current_price"] = price
                
                # Recalculate unrealized PnL
                entry_price = position["entry_price"]
                leverage = position["leverage"]
                
                if position["direction"].lower() == "long":
                    pnl_percentage = (price / entry_price - 1) * leverage
                else:  # Short
                    pnl_percentage = (1 - price / entry_price) * leverage
                
                # Calculate unrealized PnL in USD
                position_size = position["size"]
                margin = position_size / leverage
                unrealized_pnl = margin * pnl_percentage
                
                position["unrealized_pnl"] = unrealized_pnl
                position["unrealized_pnl_pct"] = pnl_percentage * 100  # as percentage
                positions_updated = True
        
        # Save positions if updated
        if positions_updated:
            self._save_positions()
        
        # Take price snapshots periodically
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            self._take_price_snapshot()
            self.last_snapshot_time = current_time
    
    def _take_price_snapshot(self) -> None:
        """Take a snapshot of current prices and save to file"""
        prices = self.price_fetcher.get_all_latest_prices()
        if prices:
            snapshot = {
                "timestamp": datetime.datetime.now().isoformat(),
                "prices": prices
            }
            self._save_json(PRICE_SNAPSHOT_FILE, snapshot)
            logger.debug(f"Saved price snapshot with {len(prices)} pairs")
    
    def simulate_order_book(self) -> None:
        """
        Simulate an order book snapshot based on current prices
        
        This creates a realistic order book snapshot that can be used
        for slippage calculations and liquidity analysis.
        """
        prices = self.price_fetcher.get_all_latest_prices()
        if not prices:
            return
        
        order_book = {}
        
        for pair, price in prices.items():
            # Create synthetic order book with realistic bid-ask spread
            # and depth based on typical Kraken liquidity
            
            # Parameters for synthetic book
            base_spread = 0.0005  # 0.05% base spread
            spread_multiplier = 1.0
            
            # Different spreads for different assets based on liquidity
            if "BTC" in pair:
                spread_multiplier = 0.8  # Tighter spread for BTC
            elif "ETH" in pair:
                spread_multiplier = 0.9  # Tighter spread for ETH
            elif "SOL" in pair:
                spread_multiplier = 1.2  # Wider spread for less liquid assets
            
            spread = price * base_spread * spread_multiplier
            mid_price = price
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2
            
            # Create synthetic orders
            bids = []
            asks = []
            
            # Generate bids (below current price)
            for i in range(20):
                price_level = bid_price * (1 - 0.0001 * (i + 1) * (i + 1))
                size = 0.1 + 0.3 * random.random()  # Random size between 0.1 and 0.4
                bids.append({
                    "price": price_level,
                    "size": size,
                    "order_id": f"order_{random.randint(10000, 99999)}"
                })
            
            # Generate asks (above current price)
            for i in range(20):
                price_level = ask_price * (1 + 0.0001 * (i + 1) * (i + 1))
                size = 0.1 + 0.3 * random.random()  # Random size between 0.1 and 0.4
                asks.append({
                    "price": price_level,
                    "size": size,
                    "order_id": f"order_{random.randint(10000, 99999)}"
                })
            
            order_book[pair] = {
                "bids": bids,
                "asks": asks
            }
        
        # Save order book snapshot
        snapshot = {
            "timestamp": datetime.datetime.now().isoformat(),
            "order_book": order_book
        }
        self._save_json(ORDER_BOOK_SNAPSHOT_FILE, snapshot)
        logger.debug(f"Saved order book snapshot with {len(order_book)} pairs")
    
    def update_portfolio_with_prices(self) -> None:
        """
        Update portfolio data with current unrealized P&L
        
        This calculates the total unrealized P&L across all positions
        and updates the portfolio data accordingly.
        """
        portfolio = self._load_json(PORTFOLIO_FILE, {"balance": 20000.0})
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = 0.0
        
        for position in self.positions:
            if "unrealized_pnl" in position:
                total_unrealized_pnl += position["unrealized_pnl"]
        
        # Update portfolio
        portfolio["unrealized_pnl_usd"] = total_unrealized_pnl
        
        # Calculate as percentage of initial capital (assuming 20000)
        initial_capital = 20000.0
        portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / initial_capital) * 100
        
        # Update total open positions count
        portfolio["open_positions_count"] = len(self.positions)
        
        # Add timestamp
        portfolio["last_price_update"] = datetime.datetime.now().isoformat()
        
        # Save updated portfolio
        self._save_json(PORTFOLIO_FILE, portfolio)
    
    def start(self):
        """Start the integration"""
        if self.running:
            logger.warning("Integration already running")
            return
        
        self.running = True
        
        # Start the price fetcher
        self.price_fetcher.start()
        
        # Start the liquidation handler
        self.liquidation_handler.start()
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started realtime trading integration")
    
    def stop(self):
        """Stop the integration"""
        self.running = False
        
        # Stop the liquidation handler
        self.liquidation_handler.stop()
        
        # Stop the price fetcher
        self.price_fetcher.stop()
        
        logger.info("Stopped realtime trading integration")
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                # Reload positions in case they've been modified externally
                self.positions = self._load_positions()
                
                # Update portfolio with current prices
                self.update_portfolio_with_prices()
                
                # Simulate order book periodically
                self.simulate_order_book()
                
                # Sleep for the update interval
                for _ in range(self.update_interval):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")


# Usage example
if __name__ == "__main__":
    # Set up signal handling
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        integration.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start the integration
    integration = RealtimeTradingIntegration()
    integration.start()
    
    logger.info("Realtime trading integration running. Press Ctrl+C to exit")
    
    # Keep the main thread running
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    
    integration.stop()