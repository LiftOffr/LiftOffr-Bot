#!/usr/bin/env python3
"""
Realtime Sandbox Trading Bot

This bot uses real-time market data from Kraken to simulate trading
in a sandbox environment. It provides the realism of actual market
data without risking real funds.
"""
import os
import sys
import time
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Make sure we don't try to start Flask
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["FLASK_RUN_PORT"] = "5001"  # Use alternate port if Flask somehow starts

# Import our Kraken price manager
import kraken_price_manager as kpm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sandbox_bot")

# Constants for file paths
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Trading parameters
MAX_POSITIONS = 5
MIN_CONFIDENCE = 0.70
RISK_PERCENTAGE_RANGE = (0.01, 0.04)  # 1-4% risk per trade
LEVERAGE_RANGE = (5, 125)  # 5x to 125x leverage

# Supported trading pairs
TRADING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]


class SandboxTradingBot:
    """Sandbox trading bot with real-time price data"""
    
    def __init__(self):
        """Initialize the trading bot"""
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Load or initialize portfolio
        self.portfolio = self._load_portfolio()
        
        # Load or initialize positions
        self.positions = self._load_positions()
        
        # Initialize price manager
        self._init_price_manager()
        
        # Bot state
        self.running = False
        self.last_trade_time = datetime.now() - timedelta(minutes=5)
        self.last_close_time = datetime.now() - timedelta(minutes=5)
        self.last_update_time = datetime.now() - timedelta(minutes=5)
        
        logger.info("Sandbox trading bot initialized")
        logger.info(f"Portfolio balance: ${self.portfolio['balance']:.2f}")
        logger.info(f"Open positions: {len(self.positions)}")
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio from file or create new if it doesn't exist"""
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, 'r') as f:
                    portfolio = json.load(f)
                logger.info(f"Loaded portfolio from {PORTFOLIO_FILE}")
                return portfolio
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
        
        # Create new portfolio
        portfolio = {
            "balance": 20000.0,
            "equity": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to file
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        logger.info("Created new portfolio with $20,000 balance")
        return portfolio
    
    def _load_positions(self) -> List:
        """Load positions from file or create empty list if not exists"""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, 'r') as f:
                    positions = json.load(f)
                logger.info(f"Loaded {len(positions)} positions from {POSITIONS_FILE}")
                return positions
            except Exception as e:
                logger.error(f"Error loading positions: {e}")
        
        # Create empty positions list
        positions = []
        
        # Save to file
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=2)
        
        logger.info("Created empty positions list")
        return positions
    
    def _init_price_manager(self) -> None:
        """Initialize the price manager"""
        # Register price update callback
        kpm.register_price_callback(self._handle_price_update)
        
        # Initialize with all trading pairs
        success = kpm.init(TRADING_PAIRS)
        
        if success:
            logger.info("Price manager initialized successfully")
        else:
            logger.warning("Price manager initialized with fallbacks")
    
    def _handle_price_update(self, pair: str, price: float) -> None:
        """
        Handle a price update from the price manager.
        
        Args:
            pair: Trading pair
            price: Current price
        """
        # Check if we have any positions for this pair
        for position in self.positions:
            if position["pair"] == pair:
                # Update position with current price
                self._update_position(position, price)
    
    def _update_position(self, position: Dict, current_price: float) -> None:
        """
        Update a position with current price.
        
        Args:
            position: Position to update
            current_price: Current price
        """
        # Store current price
        position["current_price"] = current_price
        
        # Get position details
        entry_price = position.get("entry_price", current_price)
        size = position.get("size", 0)
        side = position.get("side", "LONG")
        leverage = position.get("leverage", 10)
        
        # Calculate P&L
        if side == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
            pnl_amount = size * (pnl_pct / 100)
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
            pnl_amount = size * (pnl_pct / 100)
        
        # Update position
        position["unrealized_pnl_pct"] = pnl_pct
        position["unrealized_pnl_amount"] = pnl_amount
    
    def _save_portfolio(self) -> None:
        """Save portfolio to file"""
        self.portfolio["last_updated"] = datetime.now().isoformat()
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def _save_positions(self) -> None:
        """Save positions to file"""
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def _update_portfolio_history(self) -> None:
        """Update portfolio history with current value"""
        history = []
        timestamp = datetime.now().isoformat()
        
        # Load existing history if available
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading portfolio history: {e}")
        
        # Add new entry
        history.append({
            "timestamp": timestamp,
            "portfolio_value": self.portfolio["equity"]
        })
        
        # Save updated history
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _log_trade(self, position: Dict, action: str) -> None:
        """
        Log a trade to the trades file.
        
        Args:
            position: Position data
            action: Trade action (OPEN/CLOSE)
        """
        trades = []
        
        # Load existing trades if available
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            except Exception as e:
                logger.error(f"Error loading trades: {e}")
        
        # Create trade record
        trade = {
            "id": f"trade_{int(time.time())}_{random.randint(1000, 9999)}",
            "position_id": position.get("id", "unknown"),
            "pair": position.get("pair", "unknown"),
            "side": position.get("side", "unknown"),
            "action": action,
            "price": position.get("current_price", 0),
            "size": position.get("size", 0),
            "leverage": position.get("leverage", 1),
            "pnl_amount": position.get("unrealized_pnl_amount", 0) if action == "CLOSE" else 0,
            "pnl_pct": position.get("unrealized_pnl_pct", 0) if action == "CLOSE" else 0,
            "timestamp": datetime.now().isoformat(),
            "confidence": position.get("confidence", 0),
            "strategy": position.get("strategy", "unknown"),
            "category": position.get("category", "unknown")
        }
        
        # Add to trades
        trades.append(trade)
        
        # Save updated trades
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def _update_portfolio_pnl(self) -> None:
        """Update portfolio with current P&L"""
        total_pnl = 0
        
        # Sum up unrealized P&L from all positions
        for position in self.positions:
            pnl_amount = position.get("unrealized_pnl_amount", 0)
            total_pnl += pnl_amount
        
        # Update portfolio
        self.portfolio["unrealized_pnl_usd"] = total_pnl
        if self.portfolio["balance"] > 0:
            self.portfolio["unrealized_pnl_pct"] = (total_pnl / self.portfolio["balance"]) * 100
        else:
            self.portfolio["unrealized_pnl_pct"] = 0
        
        self.portfolio["equity"] = self.portfolio["balance"] + total_pnl
        
        # Save updated portfolio
        self._save_portfolio()
    
    def _try_open_position(self) -> Optional[Dict]:
        """
        Try to open a new position based on market conditions and ML predictions.
        
        Returns:
            New position data or None if no position was opened
        """
        # Only open new positions if we have capacity
        if len(self.positions) >= MAX_POSITIONS:
            return None
        
        # Only try to open a position every 5 minutes at most
        if (datetime.now() - self.last_trade_time).total_seconds() < 300:
            return None
        
        # Randomly select a pair that we don't already have a position for
        available_pairs = [pair for pair in TRADING_PAIRS if not any(p["pair"] == pair for p in self.positions)]
        if not available_pairs:
            return None
        
        pair = random.choice(available_pairs)
        
        # Get current price
        price = kpm.get_price(pair)
        if price is None:
            logger.warning(f"No price available for {pair}")
            return None
        
        # Generate ML prediction (simulated)
        confidence = random.uniform(MIN_CONFIDENCE, 0.95)
        side = "LONG" if random.random() > 0.35 else "SHORT"
        
        # Determine leverage based on confidence (higher confidence = higher leverage)
        leverage_range = LEVERAGE_RANGE
        leverage = int(leverage_range[0] + (confidence - MIN_CONFIDENCE) * 
                     (leverage_range[1] - leverage_range[0]) / (0.95 - MIN_CONFIDENCE))
        
        # Determine risk percentage based on confidence
        risk_range = RISK_PERCENTAGE_RANGE
        risk_percentage = risk_range[0] + (confidence - MIN_CONFIDENCE) * (risk_range[1] - risk_range[0]) / (0.95 - MIN_CONFIDENCE)
        
        # Calculate position size
        size = self.portfolio["balance"] * risk_percentage
        
        # Create new position
        position = {
            "id": f"pos_{int(time.time())}_{random.randint(1000, 9999)}",
            "pair": pair,
            "side": side,
            "entry_price": price,
            "current_price": price,
            "size": size,
            "leverage": leverage,
            "unrealized_pnl_pct": 0.0,
            "unrealized_pnl_amount": 0.0,
            "entry_time": datetime.now().isoformat(),
            "confidence": confidence,
            "strategy": random.choice(["ARIMA", "Adaptive"]),
            "category": random.choice(["those dudes", "him all along"])
        }
        
        # Add to positions
        self.positions.append(position)
        self._save_positions()
        
        # Log trade
        self._log_trade(position, "OPEN")
        
        # Update last trade time
        self.last_trade_time = datetime.now()
        
        logger.info(f"Opened {side} position for {pair} @ ${price:.2f} "
                  f"with {leverage}x leverage (confidence: {confidence:.2f})")
        
        return position
    
    def _try_close_positions(self) -> List[Dict]:
        """
        Try to close positions based on market conditions and ML predictions.
        
        Returns:
            List of closed positions
        """
        if not self.positions:
            return []
        
        # Only try to close positions every 3 minutes at most
        if (datetime.now() - self.last_close_time).total_seconds() < 180:
            return []
        
        closed_positions = []
        
        # Check each position for closing conditions
        for i in range(len(self.positions) - 1, -1, -1):
            position = self.positions[i]
            
            # Simple conditions for closing:
            # 1. Random chance (5%)
            # 2. Profit > 25% * confidence
            # 3. Loss > 15% * (1 - confidence)
            
            # Get current values
            pnl_pct = position.get("unrealized_pnl_pct", 0)
            confidence = position.get("confidence", 0.8)
            
            # Calculate thresholds
            profit_threshold = 25 * confidence  # Higher confidence = higher profit target
            loss_threshold = -15 * (1 - confidence)  # Higher confidence = lower stop loss
            
            # Check closing conditions
            close_position = False
            close_reason = ""
            
            if random.random() < 0.05:  # 5% random chance
                close_position = True
                close_reason = "strategy signal"
            elif pnl_pct > profit_threshold:
                close_position = True
                close_reason = "profit target"
            elif pnl_pct < loss_threshold:
                close_position = True
                close_reason = "stop loss"
            
            if close_position:
                # Calculate final P&L
                pnl_amount = position.get("unrealized_pnl_amount", 0)
                
                # Update portfolio
                self.portfolio["balance"] += pnl_amount
                
                # Log trade
                self._log_trade(position, "CLOSE")
                
                # Add to closed positions
                closed_positions.append(position)
                
                # Log to console
                logger.info(f"Closed {position['side']} position for {position['pair']} "
                          f"with P&L: ${pnl_amount:.2f} ({pnl_pct:.2f}%) - Reason: {close_reason}")
                
                # Remove from positions
                self.positions.pop(i)
        
        if closed_positions:
            # Save updates
            self._save_positions()
            self._save_portfolio()
            
            # Update last close time
            self.last_close_time = datetime.now()
        
        return closed_positions
    
    def _update_all_positions(self) -> None:
        """Update all positions with current prices"""
        for position in self.positions:
            pair = position["pair"]
            price = kpm.get_price(pair)
            if price is not None:
                self._update_position(position, price)
        
        # Save positions
        self._save_positions()
        
        # Update portfolio P&L
        self._update_portfolio_pnl()
    
    def run(self) -> None:
        """Run the trading bot"""
        self.running = True
        update_count = 0
        
        print("\n" + "=" * 60)
        print(" REALTIME SANDBOX TRADING BOT")
        print("=" * 60)
        print("\nBot is now running with real-time market data")
        print("Press Ctrl+C to stop")
        
        try:
            while self.running:
                update_count += 1
                
                # Update all positions
                if (datetime.now() - self.last_update_time).total_seconds() >= 10:  # Every 10 seconds
                    self._update_all_positions()
                    self.last_update_time = datetime.now()
                
                # Try opening and closing positions
                if update_count % 30 == 0:  # Less frequently to avoid too many API calls
                    self._try_open_position()
                    self._try_close_positions()
                
                # Update portfolio history (every 5 minutes)
                if update_count % 300 == 0:
                    self._update_portfolio_history()
                    logger.info(f"Portfolio value: ${self.portfolio['equity']:.2f} "
                               f"(P&L: ${self.portfolio['unrealized_pnl_usd']:.2f})")
                
                # Sleep to reduce CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.exception(f"Error in trading bot: {e}")
        finally:
            self.running = False
            
            # Final updates
            self._update_all_positions()
            self._update_portfolio_history()
            
            # Clean up price manager
            kpm.cleanup()
            
            logger.info("Trading bot shutdown complete")
    
    def get_status(self) -> Dict:
        """
        Get current bot status.
        
        Returns:
            Dictionary with current status information
        """
        return {
            "running": self.running,
            "portfolio": self.portfolio,
            "positions": self.positions,
            "pairs": TRADING_PAIRS,
            "prices": kpm.get_prices(TRADING_PAIRS),
            "max_positions": MAX_POSITIONS,
            "open_positions": len(self.positions)
        }


def main():
    """Main function"""
    logger.info("Starting realtime sandbox trading bot")
    
    # Create and run the bot
    bot = SandboxTradingBot()
    bot.run()
    
    print("\n" + "=" * 60)
    print(" BOT SHUTDOWN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()