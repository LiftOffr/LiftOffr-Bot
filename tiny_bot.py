#!/usr/bin/env python3
"""
Tiny Trading Bot

This is an extremely minimal trading bot that avoids all dependencies.
It's designed to just log activity without actually importing any modules
that might conflict with the Flask application.
"""
# Make absolutely sure we're not trying to start Flask
import os
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["FLASK_RUN_PORT"] = "5001"  # Use alternate port if Flask somehow starts

# Standard library imports only (no Flask)
import sys
import time
import json
import random
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("tiny_bot")

print("\n" + "=" * 60)
print(" TINY TRADING BOT")
print("=" * 60)
print("\nThis bot is completely isolated from Flask dependencies")

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Portfolio file paths
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

def load_portfolio():
    """Load portfolio data from file or create new if it doesn't exist"""
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

def load_positions():
    """Load positions from file or create empty list if not exists"""
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions")
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
    
    # Load existing history if available
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
    
    # Add new entry
    history.append({
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio["equity"]
    })
    
    # Save updated history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def log_trade(position, action, pair, price, side, size, leverage, confidence=0.8, strategy="Adaptive"):
    """Log a trade to the trades file"""
    trades = []
    
    # Load existing trades if available
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    # Calculate PnL amounts for closing trades
    pnl_amount = 0
    pnl_pct = 0
    if action == "CLOSE" and position:
        pnl_amount = position.get("unrealized_pnl_amount", 0)
        pnl_pct = position.get("unrealized_pnl_pct", 0)
    
    # Create trade record
    trade = {
        "id": f"trade_{int(time.time())}_{random.randint(1000, 9999)}",
        "position_id": position.get("id", f"pos_{int(time.time())}") if position else f"pos_{int(time.time())}",
        "pair": pair,
        "side": side,
        "action": action,
        "price": price,
        "size": size,
        "leverage": leverage,
        "pnl_amount": pnl_amount,
        "pnl_pct": pnl_pct,
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "strategy": strategy,
        "category": random.choice(["those dudes", "him all along"])
    }
    
    # Add to trades
    trades.append(trade)
    
    # Save updated trades
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    
    return trade

# Trading pairs
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
         "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]

# Base prices (relatively realistic)
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

def main():
    """Main function"""
    # Load portfolio and positions
    portfolio = load_portfolio()
    positions = load_positions()
    
    print(f"\nLoaded portfolio with ${portfolio.get('balance', 0):.2f} balance")
    print(f"Open positions: {len(positions)}")
    print("\nBot is now running. Press Ctrl+C to stop.\n")
    
    # Simulated prices
    prices = dict(BASE_PRICES)
    
    # Main loop
    update_count = 0
    try:
        while True:
            update_count += 1
            
            # Update prices with small random changes
            for pair in prices:
                # Small random price movement (±0.1%)
                current_price = prices[pair]
                movement = current_price * random.uniform(-0.001, 0.001)
                prices[pair] = current_price + movement
            
            # Only log every 5th update to reduce output
            if update_count % 5 == 0:
                # Current time
                now = datetime.now().strftime("%H:%M:%S")
                pair = random.choice(PAIRS)
                price = prices.get(pair, 1000.0)
                
                # Randomly choose activity to simulate
                activities = [
                    "Scanning market data",
                    "Analyzing price patterns",
                    f"Checking signal strength for {pair}",
                    f"Monitoring {pair} trend",
                    "Running ML prediction model",
                    "Calculating optimal leverage",
                    "Evaluating entry points",
                    "Checking for position exits",
                    "Updating portfolio metrics",
                    "Applying risk management rules"
                ]
                activity = random.choice(activities)
                
                # Print activity
                logger.info(f"{activity}")
                
                # Occasionally simulate a new trade
                if random.random() < 0.03:  # 3% chance per update
                    side = "LONG" if random.random() > 0.35 else "SHORT"
                    confidence = round(random.uniform(0.7, 0.95), 2)
                    leverage = round(5 + 120 * confidence)
                    size = portfolio["balance"] * (random.uniform(0.01, 0.05))  # 1-5% of balance
                    
                    # Create position
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
                    save_positions(positions)
                    
                    # Log trade
                    log_trade(position, "OPEN", pair, price, side, size, leverage, confidence)
                    
                    logger.info(f"NEW TRADE: {pair} {side} @ ${price:.2f} - "
                              f"Confidence: {confidence:.2f} - Leverage: {leverage}x")
                
                # Occasionally close a position
                elif random.random() < 0.02 and positions:  # 2% chance per update & we have positions
                    position_idx = random.randint(0, len(positions) - 1)
                    position = positions[position_idx]
                    
                    # Calculate final P&L (simulated)
                    pair = position.get("pair")
                    current_price = prices.get(pair, position.get("current_price"))
                    entry_price = position.get("entry_price", current_price)
                    side = position.get("side", "LONG")
                    leverage = position.get("leverage", 10)
                    size = position.get("size", 0)
                    
                    # Calculate P&L
                    if side == "LONG":
                        pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
                    else:  # SHORT
                        pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
                    
                    pnl_amount = size * (pnl_pct / 100)
                    
                    # Update position with final values
                    position["current_price"] = current_price
                    position["unrealized_pnl_pct"] = pnl_pct
                    position["unrealized_pnl_amount"] = pnl_amount
                    
                    # Update portfolio
                    portfolio["balance"] += pnl_amount
                    
                    # Log trade
                    log_trade(position, "CLOSE", pair, current_price, side, size, leverage)
                    
                    # Remove from positions
                    positions.pop(position_idx)
                    save_positions(positions)
                    save_portfolio(portfolio)
                    
                    logger.info(f"CLOSED: {pair} {side} @ ${current_price:.2f} - "
                              f"P&L: ${pnl_amount:.2f} ({pnl_pct:.2f}%)")
            
            # Update portfolio history every 5 minutes (every 300 seconds)
            if update_count % 300 == 0:
                # Calculate total unrealized P&L
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
                    
                    # Calculate P&L
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
                
                # Save updated positions and portfolio
                save_positions(positions)
                save_portfolio(portfolio)
                update_portfolio_history(portfolio)
                
                logger.info(f"Portfolio: ${portfolio['equity']:.2f} (Bal: ${portfolio['balance']:.2f}, PnL: ${total_pnl:.2f})")
            
            # Sleep between updates
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    
    print("\n" + "=" * 60)
    print(" BOT SHUTDOWN COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()