#!/usr/bin/env python3
"""
Standalone Trading Bot Runner

This script avoids all Flask dependencies and runs 
the trading bot completely independently.
"""
import os
import sys
import random
import time
import json
import logging
from datetime import datetime

# Make sure we're in a trading bot process
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
os.makedirs(DATA_DIR, exist_ok=True)
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Trading pairs
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
         "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]

# Base prices (used for simulation)
BASE_PRICES = {
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

def load_portfolio():
    """Load portfolio or create new one"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
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
    
    return portfolio

def load_positions():
    """Load positions or create empty list"""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            return positions
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    # Create empty positions list
    positions = []
    
    # Save to file
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)
    
    return positions

def save_portfolio(portfolio):
    """Save portfolio to file"""
    portfolio["last_updated"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)

def save_positions(positions):
    """Save positions to file"""
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)

def update_portfolio_history(portfolio):
    """Update portfolio history"""
    history = []
    timestamp = datetime.now().isoformat()
    
    # Load existing history if available
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    # Add new entry
    history.append({
        "timestamp": timestamp,
        "portfolio_value": portfolio["equity"]
    })
    
    # Save updated history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def update_prices():
    """Update prices for all pairs"""
    prices = {}
    for pair in PAIRS:
        base_price = BASE_PRICES.get(pair, 1000.0)
        movement = base_price * random.uniform(-0.001, 0.001)
        prices[pair] = base_price + movement
    return prices

def update_positions(positions, prices, portfolio):
    """Update positions with current prices"""
    if not positions:
        return 0

    total_pnl = 0
    
    for position in positions:
        pair = position.get("pair")
        if pair not in prices:
            continue
        
        current_price = prices[pair]
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
    portfolio["unrealized_pnl_usd"] = total_pnl
    portfolio["unrealized_pnl_pct"] = (total_pnl / portfolio["balance"]) * 100 if portfolio["balance"] > 0 else 0
    portfolio["equity"] = portfolio["balance"] + total_pnl
    
    return total_pnl

def try_new_trade(positions, prices, portfolio):
    """Try to open a new trade"""
    # Only open trades if we have fewer than 5 positions
    if len(positions) >= 5:
        return None
    
    # Random chance of new trade (5%)
    if random.random() > 0.05:
        return None
    
    # Select a random pair
    pair = random.choice(PAIRS)
    price = prices.get(pair, 1000.0)
    
    # Generate trade parameters
    side = "LONG" if random.random() > 0.35 else "SHORT"
    confidence = random.uniform(0.70, 0.95)
    leverage = int(5 + (confidence * 120))  # Dynamic leverage 5x-125x
    
    # Risk 1-4% of portfolio based on confidence
    risk_percentage = 0.01 + (confidence * 0.03)
    size = portfolio["balance"] * risk_percentage
    
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
    positions.append(position)
    
    # Log trade
    log_trade(position, "OPEN")
    
    logger.info(f"New {side} position opened for {pair} @ ${price:.2f} "
              f"with {leverage}x leverage (confidence: {confidence:.2f})")
    
    return position

def try_close_positions(positions, portfolio):
    """Try to close some positions"""
    if not positions:
        return []
    
    closed_positions = []
    
    # 5% chance to close each position
    for i in range(len(positions) - 1, -1, -1):
        if random.random() < 0.05:
            position = positions[i]
            
            # Calculate final PnL
            pnl_amount = position.get("unrealized_pnl_amount", 0)
            
            # Update portfolio balance
            portfolio["balance"] += pnl_amount
            
            # Log trade
            log_trade(position, "CLOSE")
            
            # Log to console
            logger.info(f"Closed {position['side']} position for {position['pair']} "
                       f"with P&L: ${pnl_amount:.2f}")
            
            # Add to closed positions
            closed_positions.append(position)
            
            # Remove from positions
            positions.pop(i)
    
    return closed_positions

def log_trade(position, action):
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

def main():
    """Main function"""
    # Print welcome message
    print("\n" + "=" * 60)
    print(" KRAKEN TRADING BOT - STANDALONE RUNNER")
    print("=" * 60)
    print("\nThis is a completely standalone trading bot")
    print("Press Ctrl+C to stop")
    
    # Load data
    portfolio = load_portfolio()
    positions = load_positions()
    
    # Display initial status
    logger.info(f"Loaded portfolio with ${portfolio['balance']:.2f} balance")
    logger.info(f"Loaded {len(positions)} open positions")
    
    # Main trading loop
    update_count = 0
    running = True
    
    try:
        while running:
            update_count += 1
            
            # Update prices
            prices = update_prices()
            
            # Update positions with new prices
            total_pnl = update_positions(positions, prices, portfolio)
            
            # Save updated positions and portfolio
            save_positions(positions)
            save_portfolio(portfolio)
            
            # Try opening and closing positions
            if update_count % 5 == 0:  # Less frequently to reduce output
                new_position = try_new_trade(positions, prices, portfolio)
                closed_positions = try_close_positions(positions, portfolio)
                
                # Save changes if needed
                if new_position or closed_positions:
                    save_positions(positions)
                    save_portfolio(portfolio)
            
            # Periodically update portfolio history (every 5 minutes / 300 updates)
            if update_count % 300 == 0:
                update_portfolio_history(portfolio)
            
            # Log status periodically
            if update_count % 30 == 0:
                logger.info(f"Status: {len(positions)} active positions, "
                          f"equity: ${portfolio['equity']:.2f}, "
                          f"PnL: ${total_pnl:.2f}")
            
            # Sleep to reduce CPU usage
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    
    # Final updates before exit
    update_portfolio_history(portfolio)
    
    print("\n" + "=" * 60)
    print(" TRADING BOT SHUT DOWN")
    print("=" * 60)

if __name__ == "__main__":
    main()