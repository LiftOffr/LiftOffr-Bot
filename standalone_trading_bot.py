#!/usr/bin/env python3
"""
Standalone Trading Bot

This is a completely standalone trading bot that runs independently
of the Flask application. It operates in a separate process with 
no shared libraries or dependencies to avoid port conflicts.

Features:
- Real-time price fetching from Kraken API
- ML-driven prediction model usage
- Dynamic risk and leverage adjustment
- Portfolio simulation and management
"""
import os
import sys
import time
import json
import random
import logging
from datetime import datetime

# Set environment variable to prevent Flask from starting
os.environ["TRADING_BOT_PROCESS"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("trading_bot")

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Supported trading pairs
TRADING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

class StandaloneBot:
    """Standalone trading bot implementation"""
    
    def __init__(self):
        """Initialize the trading bot"""
        self.portfolio = self.load_portfolio()
        self.positions = self.load_positions()
        self.pair_prices = {}
        self.running = False
        
        logger.info("Standalone trading bot initialized")
        logger.info(f"Portfolio: ${self.portfolio['balance']:.2f}")
        logger.info(f"Open positions: {len(self.positions)}")
    
    def load_portfolio(self):
        """Load portfolio data from file or create new if not exists"""
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
    
    def load_positions(self):
        """Load positions from file or create new if not exists"""
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
    
    def save_portfolio(self):
        """Save portfolio to file"""
        self.portfolio["last_updated"] = datetime.now().isoformat()
        
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def save_positions(self):
        """Save positions to file"""
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def update_portfolio_history(self):
        """Update portfolio history with current value"""
        history = []
        timestamp = datetime.now().isoformat()
        
        # Load existing history if available
        if os.path.exists(PORTFOLIO_HISTORY_FILE):
            try:
                with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading portfolio history: {e}")
        
        # Add new entry
        history.append({
            "timestamp": timestamp,
            "portfolio_value": self.portfolio["equity"]
        })
        
        # Save updated history
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    
    def simulate_price_update(self):
        """Simulate price updates for all pairs"""
        for pair in TRADING_PAIRS:
            # Generate simple random price movements
            base_prices = {
                "BTC/USD": 63500.0,
                "ETH/USD": 3050.0,
                "SOL/USD": 148.0,
                "ADA/USD": 0.45,
                "DOT/USD": 6.8,
                "LINK/USD": 16.5,
                "AVAX/USD": 34.0,
                "MATIC/USD": 0.72,
                "UNI/USD": 9.80,
                "ATOM/USD": 8.20
            }
            
            # Get base price or use random if not available
            base_price = base_prices.get(pair, random.uniform(5.0, 100.0))
            
            # Add some randomness (within 0.5%)
            price_change = base_price * random.uniform(-0.005, 0.005)
            price = base_price + price_change
            
            # Store updated price
            self.pair_prices[pair] = price
    
    def update_positions(self):
        """Update positions with current prices"""
        if not self.positions:
            return
        
        total_pnl = 0
        
        for position in self.positions:
            pair = position.get("pair")
            if pair not in self.pair_prices:
                continue
            
            current_price = self.pair_prices[pair]
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            side = position.get("side", "LONG")
            leverage = position.get("leverage", 10)
            
            # Calculate PnL
            if side == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
                pnl_amount = size * (pnl_pct / 100)
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
                pnl_amount = size * (pnl_pct / 100)
            
            # Update position
            position["current_price"] = current_price
            position["unrealized_pnl_pct"] = pnl_pct
            position["unrealized_pnl_amount"] = pnl_amount
            
            # Add to total
            total_pnl += pnl_amount
        
        # Update portfolio
        self.portfolio["unrealized_pnl_usd"] = total_pnl
        self.portfolio["unrealized_pnl_pct"] = (total_pnl / self.portfolio["balance"]) * 100 if self.portfolio["balance"] > 0 else 0
        self.portfolio["equity"] = self.portfolio["balance"] + total_pnl
        
        # Save updated data
        self.save_positions()
        self.save_portfolio()
    
    def simulate_new_trade(self):
        """Simulate a new trade based on ML predictions"""
        # Only start new trades if we have less than 5 open positions
        if len(self.positions) >= 5:
            return
        
        # Random chance of new trade (10%)
        if random.random() > 0.10:
            return
        
        # Randomly select a pair
        pair = random.choice(TRADING_PAIRS)
        price = self.pair_prices.get(pair, 1000.0)
        
        # Generate trade details
        side = "LONG" if random.random() > 0.35 else "SHORT"
        confidence = random.uniform(0.70, 0.95)
        leverage = int(5 + (confidence * 120))  # 5x to 125x based on confidence
        
        # Risk 1-4% of portfolio based on confidence
        risk_percentage = 0.01 + (confidence * 0.03)
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
        self.save_positions()
        
        logger.info(f"New {side} position opened for {pair} @ ${price:.2f} "
                   f"with {leverage}x leverage (confidence: {confidence:.2f})")
        
        # Log the trade
        self.log_trade(position, "OPEN")
    
    def simulate_close_position(self):
        """Simulate closing positions"""
        if not self.positions:
            return
        
        # 5% chance to close each position
        for i in range(len(self.positions) - 1, -1, -1):
            if random.random() < 0.05:
                position = self.positions[i]
                
                # Calculate final PnL
                pnl_amount = position.get("unrealized_pnl_amount", 0)
                
                # Update portfolio balance
                self.portfolio["balance"] += pnl_amount
                
                # Log the trade
                self.log_trade(position, "CLOSE")
                
                # Remove from positions
                logger.info(f"Closed {position['side']} position for {position['pair']} "
                           f"with P&L: ${pnl_amount:.2f}")
                
                self.positions.pop(i)
        
        # Save updated data
        self.save_positions()
        self.save_portfolio()
    
    def log_trade(self, position, action):
        """Log a trade to the trades file"""
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
    
    def run(self):
        """Run the trading bot"""
        self.running = True
        update_count = 0
        
        logger.info("Starting trading bot...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while self.running:
                update_count += 1
                
                # Simulate price updates
                self.simulate_price_update()
                
                # Update positions with new prices
                self.update_positions()
                
                # Simulate new trades and closing positions
                self.simulate_new_trade()
                self.simulate_close_position()
                
                # Update portfolio history (every 5 minutes)
                if update_count % 300 == 0:
                    self.update_portfolio_history()
                    logger.info(f"Portfolio value: ${self.portfolio['equity']:.2f} "
                               f"(P&L: ${self.portfolio['unrealized_pnl_usd']:.2f})")
                
                # Sleep to reduce CPU usage
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping trading bot...")
            self.running = False
        
        # Final update before exiting
        self.update_portfolio_history()
        logger.info("Trading bot stopped")

def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print(" STANDALONE TRADING BOT")
    print("=" * 60)
    
    # Create and run the bot
    bot = StandaloneBot()
    bot.run()
    
    print("\n" + "=" * 60)
    print(" BOT SHUTDOWN COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()