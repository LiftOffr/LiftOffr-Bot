#!/usr/bin/env python3
"""
Simple Trading Bot - No Flask or web server dependencies

This script runs a standalone trading bot without any Flask dependencies.
It simulates trading with realistic market data and portfolio management.
"""
import os
import json
import time
import random
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Default trading pairs
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

def load_json(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_json(filepath, data):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")

def update_prices():
    """Update positions with current prices"""
    positions = load_json(POSITIONS_FILE, [])
    portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    
    # Get current time
    now = datetime.now()
    
    # In a real implementation, we would fetch from API
    # For this simulation, use realistic April 2023 prices
    base_prices = {
        "BTC/USD": 29735.50,
        "ETH/USD": 1865.25,
        "SOL/USD": 22.68,
        "ADA/USD": 0.381,
        "DOT/USD": 5.42,
        "LINK/USD": 6.51,
        "AVAX/USD": 14.28,
        "MATIC/USD": 0.665,
        "UNI/USD": 4.75,
        "ATOM/USD": 8.92
    }
    
    total_pnl = 0
    for position in positions:
        pair = position.get("pair")
        if not pair:
            continue
        
        # Add small random variation (±0.5%) to simulate real-time price changes
        price = base_prices.get(pair, 100.0) * (1 + random.uniform(-0.005, 0.005))
        
        entry_price = position.get("entry_price", price)
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        # Update current price
        position["current_price"] = price
        
        # Calculate PnL
        if direction.lower() == "long":
            pnl_percentage = (price - entry_price) / entry_price * 100 * leverage
            pnl_amount = (price - entry_price) * size
        else:  # Short
            pnl_percentage = (entry_price - price) / entry_price * 100 * leverage
            pnl_amount = (entry_price - price) * size
        
        position["unrealized_pnl"] = pnl_percentage
        position["unrealized_pnl_amount"] = pnl_amount
        position["unrealized_pnl_pct"] = pnl_percentage
        
        # Calculate duration
        entry_time = position.get("entry_time")
        if entry_time:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                duration = now - entry_dt
                hours, remainder = divmod(duration.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                position["duration"] = f"{int(hours)}h {int(minutes)}m"
            except Exception as e:
                logger.error(f"Error calculating duration: {e}")
        
        total_pnl += pnl_amount
    
    # Save updated positions
    save_json(POSITIONS_FILE, positions)
    
    # Update portfolio
    portfolio["unrealized_pnl_usd"] = total_pnl
    portfolio["unrealized_pnl_pct"] = (total_pnl / portfolio.get("balance", 20000.0)) * 100 if portfolio.get("balance", 20000.0) > 0 else 0
    portfolio["total_value"] = portfolio.get("balance", 20000.0) + total_pnl
    portfolio["last_updated"] = datetime.now().isoformat()
    portfolio["equity"] = portfolio["total_value"]
    save_json(PORTFOLIO_FILE, portfolio)
    
    # Update portfolio history
    history = load_json(PORTFOLIO_HISTORY_FILE, [])
    history.append({
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": portfolio["total_value"]
    })
    # Keep history to a reasonable size
    if len(history) > 1000:
        history = history[-1000:]
    save_json(PORTFOLIO_HISTORY_FILE, history)
    
    return portfolio, positions

def generate_trades():
    """Generate some trades based on ML predictions"""
    portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    positions = load_json(POSITIONS_FILE, [])
    
    # If we have too many positions already, skip
    if len(positions) >= 10:
        return
    
    # Find an available pair
    existing_pairs = [p.get("pair") for p in positions if p.get("pair")]
    available_pairs = [p for p in DEFAULT_PAIRS if p not in existing_pairs]
    
    if not available_pairs:
        return
    
    # Pick a random pair
    pair = random.choice(available_pairs)
    
    # Get current price
    base_prices = {
        "BTC/USD": 29735.50,
        "ETH/USD": 1865.25,
        "SOL/USD": 22.68,
        "ADA/USD": 0.381,
        "DOT/USD": 5.42,
        "LINK/USD": 6.51,
        "AVAX/USD": 14.28,
        "MATIC/USD": 0.665,
        "UNI/USD": 4.75,
        "ATOM/USD": 8.92
    }
    price = base_prices.get(pair, 100.0) * (1 + random.uniform(-0.005, 0.005))
    
    # Generate trade parameters
    direction = "Long" if random.random() > 0.5 else "Short"
    strategy = "ARIMA" if random.random() > 0.5 else "Adaptive"
    confidence = random.uniform(0.75, 0.95)
    
    # Safe leverage (maximum 10x for safety)
    leverage = min(10.0, 5.0 + confidence * 10.0)
    
    # Calculate size based on risk (1-2% of portfolio)
    risk_pct = 0.01 + (confidence - 0.75) * 0.02
    position_value = portfolio.get("balance", 20000.0) * risk_pct
    size = position_value / price
    
    # Calculate liquidation price
    margin_requirement = 0.1 if leverage <= 10 else 0.15
    if direction.lower() == "long":
        liquidation_price = price * (1 - (1 - margin_requirement) / leverage)
    else:
        liquidation_price = price * (1 + (1 - margin_requirement) / leverage)
    
    # Create position
    position = {
        "pair": pair,
        "direction": direction,
        "size": size,
        "entry_price": price,
        "current_price": price,
        "leverage": leverage,
        "strategy": strategy,
        "confidence": confidence,
        "entry_time": datetime.now().isoformat(),
        "unrealized_pnl": 0.0,
        "unrealized_pnl_amount": 0.0,
        "unrealized_pnl_pct": 0.0,
        "liquidation_price": liquidation_price
    }
    
    # Update balance
    margin = size * price / leverage
    portfolio["balance"] = portfolio.get("balance", 20000.0) - margin
    
    # Add position
    positions.append(position)
    
    # Save updated data
    save_json(POSITIONS_FILE, positions)
    save_json(PORTFOLIO_FILE, portfolio)
    
    logger.info(f"Opened {direction} position on {pair} with {strategy} strategy: "
             f"Size: {size:.2f}, Price: {price:.2f}, Leverage: {leverage:.2f}x")

def check_liquidations():
    """Check positions for liquidation conditions"""
    positions = load_json(POSITIONS_FILE, [])
    if not positions:
        return 0
    
    active_positions = []
    liquidated_positions = []
    
    for position in positions:
        current_price = position.get("current_price", 0)
        liquidation_price = position.get("liquidation_price", 0)
        direction = position.get("direction", "")
        
        is_liquidated = False
        
        if direction.lower() == "long" and current_price <= liquidation_price:
            is_liquidated = True
            logger.warning(f"Long position liquidated: {position.get('pair')} at {current_price}")
        elif direction.lower() == "short" and current_price >= liquidation_price:
            is_liquidated = True
            logger.warning(f"Short position liquidated: {position.get('pair')} at {current_price}")
        
        if is_liquidated:
            # Mark as liquidated
            position["exit_price"] = current_price
            position["exit_time"] = datetime.now().isoformat()
            position["exit_reason"] = "LIQUIDATED"
            liquidated_positions.append(position)
        else:
            active_positions.append(position)
    
    # Record liquidated positions in trades history
    if liquidated_positions:
        trades = load_json(TRADES_FILE, [])
        for position in liquidated_positions:
            trade = position.copy()
            trade["pnl_percentage"] = position.get("unrealized_pnl", -100)
            trade["pnl_amount"] = position.get("unrealized_pnl_amount", 0)
            trades.append(trade)
        save_json(TRADES_FILE, trades)
    
    # Save updated positions
    save_json(POSITIONS_FILE, active_positions)
    
    return len(liquidated_positions)

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" TRADING BOT WITH LIQUIDATION PROTECTION")
    print("=" * 60)
    print("\nTrading 10 pairs in sandbox mode with realistic market data:")
    for i, pair in enumerate(DEFAULT_PAIRS):
        print(f"{i+1}. {pair}")
    print("\nFeatures:")
    print("- Realistic market data simulation")
    print("- Liquidation protection")
    print("- Safe leverage (max 10x)")
    print("- Regular portfolio updates")
    print("\nStarting trading bot...")
    
    try:
        # Make sure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize portfolio if it doesn't exist
        if not os.path.exists(PORTFOLIO_FILE):
            save_json(PORTFOLIO_FILE, {
                "balance": 20000.0,
                "equity": 20000.0,
                "total_value": 20000.0,
                "unrealized_pnl_usd": 0.0,
                "unrealized_pnl_pct": 0.0,
                "last_updated": datetime.now().isoformat()
            })
        
        # Initialize positions if they don't exist
        if not os.path.exists(POSITIONS_FILE):
            save_json(POSITIONS_FILE, [])
        
        # Initialize portfolio history if it doesn't exist
        if not os.path.exists(PORTFOLIO_HISTORY_FILE):
            save_json(PORTFOLIO_HISTORY_FILE, [
                {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_value": 20000.0
                }
            ])
        
        # Initialize trades if they don't exist
        if not os.path.exists(TRADES_FILE):
            save_json(TRADES_FILE, [])
        
        counter = 0
        while True:
            # Update existing positions
            portfolio, positions = update_prices()
            
            # Check for liquidations
            liquidated = check_liquidations()
            if liquidated:
                print(f"Liquidated {liquidated} positions")
            
            # Generate new trades occasionally
            counter += 1
            if counter % 10 == 0 and random.random() < 0.3:
                generate_trades()
            
            # Display status every 10 iterations
            if counter % 10 == 0:
                print("\n" + "-" * 40)
                print(f"STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Portfolio Value: ${portfolio.get('total_value', 0):.2f}")
                print(f"Unrealized PnL: ${portfolio.get('unrealized_pnl_usd', 0):.2f} "
                     f"({portfolio.get('unrealized_pnl_pct', 0):.2f}%)")
                print(f"Open Positions: {len(positions)}")
                print("-" * 40)
            
            # Sleep to avoid high CPU usage
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())