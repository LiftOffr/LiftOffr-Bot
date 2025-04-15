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
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
PRICE_SNAPSHOT_FILE = f"{MARKET_DATA_DIR}/current_prices.json"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MARKET_DATA_DIR, exist_ok=True)

class IntegrationController:
    """
    Coordinates real-time market data with the trading system
    
    This class:
    1. Fetches real-time price data using REST API
    2. Monitors positions for liquidation risks
    3. Updates positions and portfolio with accurate prices
    4. Tracks unrealized P&L
    """
    
    def __init__(self, pairs: Optional[List[str]] = None):
        """
        Initialize the integration controller
        
        Args:
            pairs: List of trading pairs to monitor
        """
        self.pairs = pairs or [
            "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", 
            "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD", 
            "UNI/USD", "ATOM/USD"
        ]
        self.latest_prices = {}
        self.running = False
        self.update_interval = 30  # seconds
        self.price_snapshot_interval = 60  # seconds
        self.liquidation_check_interval = 15  # seconds
        self.last_snapshot_time = 0
        self.last_liquidation_check = 0
        
        # Maintenance margin requirements (standard for crypto exchanges)
        self.maintenance_margin = 0.01  # 1%
        self.liquidation_fee = 0.0075  # 0.75%
        
        # Load initial data
        self.positions = self._load_positions()
        self.portfolio = self._load_portfolio()
        self.trades = self._load_trades()
        
        # Initialize with REST API for key pairs
        self._initialize_prices()
    
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
    
    def _save_positions(self) -> None:
        """Save positions to file"""
        self._save_json(POSITIONS_FILE, self.positions)
    
    def _save_trades(self) -> None:
        """Save trades to file"""
        self._save_json(TRADES_FILE, self.trades)
    
    def _save_portfolio(self) -> None:
        """Save portfolio to file"""
        self._save_json(PORTFOLIO_FILE, self.portfolio)
    
    def _save_portfolio_history(self, history: List[Dict]) -> None:
        """Save portfolio history to file"""
        self._save_json(PORTFOLIO_HISTORY_FILE, history)
    
    def _initialize_prices(self) -> None:
        """Initialize prices for all pairs using REST API"""
        for pair in self.pairs:
            try:
                price = self._fetch_current_price(pair)
                if price:
                    self.latest_prices[pair] = price
                    logger.info(f"Initialized {pair} price: {price}")
                else:
                    # Fallback to typical prices if fetch fails
                    fallback_prices = {
                        "SOL/USD": 160.0,
                        "BTC/USD": 70000.0,
                        "ETH/USD": 3500.0,
                        "ADA/USD": 0.45,
                        "DOT/USD": 7.0,
                        "LINK/USD": 18.0,
                        "AVAX/USD": 35.0,
                        "MATIC/USD": 0.70,
                        "UNI/USD": 8.0,
                        "ATOM/USD": 6.5
                    }
                    
                    if pair in fallback_prices:
                        self.latest_prices[pair] = fallback_prices[pair]
                        logger.warning(f"Using fallback price for {pair}: {fallback_prices[pair]}")
                    else:
                        self.latest_prices[pair] = 100.0
                        logger.warning(f"Using default price for {pair}: 100.0")
            except Exception as e:
                logger.error(f"Error initializing {pair} price: {e}")
    
    def _fetch_current_price(self, pair: str) -> Optional[float]:
        """
        Fetch current price from Kraken API
        
        Args:
            pair: Trading pair (e.g., "SOL/USD")
            
        Returns:
            Current price or None if fetch failed
        """
        import requests
        
        try:
            # Convert pair to Kraken API format (e.g., "SOL/USD" -> "SOLUSD")
            kraken_pair = pair.replace("/", "")
            
            # Fetch price from Kraken API
            url = "https://api.kraken.com/0/public/Ticker"
            response = requests.get(url, params={"pair": kraken_pair})
            data = response.json()
            
            if response.status_code != 200 or "error" in data and data["error"]:
                logger.error(f"Kraken API error: {data.get('error', response.status_code)}")
                return None
            
            # Extract price from the response (c[0] is the last trade price)
            if "result" in data and kraken_pair in data["result"]:
                price = float(data["result"][kraken_pair]["c"][0])
                logger.debug(f"Fetched {pair} price: {price}")
                return price
            
            logger.warning(f"No price data for {pair}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {pair} price: {e}")
            return None
    
    def update_prices(self) -> None:
        """Update prices for all trading pairs"""
        for pair in self.pairs:
            try:
                price = self._fetch_current_price(pair)
                if price:
                    self.latest_prices[pair] = price
                    logger.debug(f"Updated {pair} price: {price}")
            except Exception as e:
                logger.error(f"Error updating {pair} price: {e}")
    
    def update_position_prices(self) -> None:
        """Update all positions with current prices"""
        # Reload positions
        self.positions = self._load_positions()
        
        positions_updated = False
        
        for position in self.positions:
            pair = position["pair"]
            if pair in self.latest_prices:
                # Update price
                position["current_price"] = self.latest_prices[pair]
                
                # Recalculate unrealized PnL
                entry_price = position["entry_price"]
                current_price = position["current_price"]
                leverage = position["leverage"]
                
                if position["direction"].lower() == "long":
                    pnl_percentage = (current_price / entry_price - 1) * leverage
                else:  # Short
                    pnl_percentage = (1 - current_price / entry_price) * leverage
                
                # Calculate unrealized PnL in USD
                position_size = position["size"]
                margin = position_size / leverage
                unrealized_pnl = margin * pnl_percentage
                
                # Calculate liquidation price if not already set
                if "liquidation_price" not in position:
                    position["liquidation_price"] = self.calculate_liquidation_price(
                        entry_price, leverage, position["direction"]
                    )
                
                position["unrealized_pnl"] = pnl_percentage
                position["unrealized_pnl_amount"] = unrealized_pnl
                position["unrealized_pnl_pct"] = pnl_percentage * 100  # as percentage
                positions_updated = True
        
        # Save positions if updated
        if positions_updated:
            self._save_positions()
    
    def update_portfolio(self) -> None:
        """Update portfolio with current position values and unrealized P&L"""
        # Reload portfolio
        self.portfolio = self._load_portfolio()
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = 0.0
        total_position_value = 0.0
        
        for position in self.positions:
            if "unrealized_pnl_amount" in position:
                total_unrealized_pnl += position["unrealized_pnl_amount"]
                
                # Calculate position value
                leverage = position["leverage"]
                size = position["size"]
                current_price = position.get("current_price", position["entry_price"])
                position_value = size * current_price
                margin_value = position_value / leverage
                total_position_value += margin_value
        
        # Update portfolio
        self.portfolio["unrealized_pnl_usd"] = total_unrealized_pnl
        
        # Calculate as percentage of initial capital
        initial_capital = 20000.0
        self.portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / initial_capital) * 100
        
        # Update balance + unrealized P&L = equity
        self.portfolio["equity"] = self.portfolio.get("balance", 20000.0) + total_unrealized_pnl
        
        # Update margin usage
        self.portfolio["margin_used"] = total_position_value
        self.portfolio["margin_used_pct"] = (total_position_value / self.portfolio.get("balance", 20000.0)) * 100
        
        # Update open positions count
        self.portfolio["open_positions_count"] = len(self.positions)
        
        # Add timestamp
        self.portfolio["last_price_update"] = datetime.now().isoformat()
        
        # Save updated portfolio
        self._save_portfolio()
        
        # Update portfolio history
        self.update_portfolio_history()
    
    def update_portfolio_history(self) -> None:
        """Update portfolio history with the current equity value"""
        # Load existing history
        history = self._load_json(PORTFOLIO_HISTORY_FILE, [])
        
        # Add current equity
        current_equity = self.portfolio.get("equity", self.portfolio.get("balance", 20000.0))
        current_time = datetime.now().isoformat()
        
        # Add new entry
        history.append({
            "timestamp": current_time,
            "portfolio_value": current_equity
        })
        
        # Keep only the last 1000 entries to avoid huge files
        if len(history) > 1000:
            history = history[-1000:]
        
        # Save updated history
        self._save_portfolio_history(history)
    
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
        """Check all positions for liquidation conditions"""
        if not self.positions:
            return
        
        # Check each position for liquidation
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
            if pair not in self.latest_prices:
                continue
            
            current_price = self.latest_prices[pair]
            
            # Check for liquidation
            liquidated = False
            if direction.lower() == "long" and current_price <= liquidation_price:
                liquidated = True
            elif direction.lower() == "short" and current_price >= liquidation_price:
                liquidated = True
            
            if liquidated:
                logger.warning(f"LIQUIDATION: {pair} {direction} position liquidated at {current_price}")
                liquidated_positions.append((position, current_price))
        
        # Process liquidations
        for position, liquidation_price in liquidated_positions:
            self._process_liquidation(position, liquidation_price)
    
    def _process_liquidation(self, position: Dict, liquidation_price: float) -> None:
        """
        Process a liquidation event
        
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
        liquidation_time = datetime.now().isoformat()
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
        entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00') if entry_time.endswith('Z') else entry_time)
        exit_time_dt = datetime.fromisoformat(liquidation_time.replace('Z', '+00:00') if liquidation_time.endswith('Z') else liquidation_time)
        duration_seconds = (exit_time_dt - entry_time_dt).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        trade["duration"] = f"{hours}h {minutes}m"
        
        # Update portfolio balance
        balance = self.portfolio.get("balance", 20000.0)
        new_balance = balance - margin - liquidation_fee
        self.portfolio["balance"] = new_balance
        self.portfolio["last_updated"] = liquidation_time
        
        # Update trades list
        self.trades.append(trade)
        self._save_trades()
        
        # Update portfolio records
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
        
        # Save portfolio data
        self._save_portfolio()
        
        # Remove the liquidated position
        self.positions = [p for p in self.positions if not (
            p["pair"] == pair and 
            p["strategy"] == strategy and
            p["direction"] == direction
        )]
        self._save_positions()
        
        logger.warning(
            f"Liquidation processed: {direction} {pair} at {liquidation_price}. "
            f"Loss: {margin + liquidation_fee:.2f} USD"
        )
    
    def take_price_snapshot(self) -> None:
        """Save current prices to snapshot file"""
        if not self.latest_prices:
            return
            
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "prices": self.latest_prices
        }
        self._save_json(PRICE_SNAPSHOT_FILE, snapshot)
        logger.debug(f"Saved price snapshot with {len(self.latest_prices)} pairs")
    
    def start(self) -> None:
        """Start the integration controller"""
        if self.running:
            logger.warning("Integration controller already running")
            return
            
        self.running = True
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started integration controller")
    
    def stop(self) -> None:
        """Stop the integration controller"""
        self.running = False
        logger.info("Stopped integration controller")
    
    def _update_loop(self) -> None:
        """Main update loop"""
        while self.running:
            try:
                # Update prices from API
                self.update_prices()
                
                # Update positions with current prices
                self.update_position_prices()
                
                # Check for liquidations
                current_time = time.time()
                if current_time - self.last_liquidation_check >= self.liquidation_check_interval:
                    self.check_liquidations()
                    self.last_liquidation_check = current_time
                
                # Update portfolio with current unrealized P&L
                self.update_portfolio()
                
                # Take price snapshot periodically
                if current_time - self.last_snapshot_time >= self.price_snapshot_interval:
                    self.take_price_snapshot()
                    self.last_snapshot_time = current_time
                
                # Sleep until next update
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # Wait before retrying on error


# Usage example
if __name__ == "__main__":
    import signal
    
    # Set up signal handling
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        controller.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start the controller
    controller = IntegrationController()
    controller.start()
    
    logger.info("Integration controller running. Press Ctrl+C to exit")
    
    # Keep the main thread running
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break
    
    controller.stop()