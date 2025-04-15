#!/usr/bin/env python3
"""
Direct Runner

This script directly runs the trading bot using code that is completely
separate from Flask. It doesn't import any Flask-related modules to prevent
port conflicts.
"""
import os
import sys
import json
import time
import random
import logging
import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "sandbox_portfolio.json")
POSITIONS_FILE = os.path.join(DATA_DIR, "sandbox_positions.json")
TRADES_FILE = os.path.join(DATA_DIR, "sandbox_trades.json")
PORTFOLIO_HISTORY_FILE = os.path.join(DATA_DIR, "sandbox_portfolio_history.json")
INITIAL_CAPITAL = 20000.0

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global price cache to reduce API calls
price_cache = {}
price_cache_time = {}
PRICE_CACHE_EXPIRY = 5  # seconds

def get_price(pair: str) -> Optional[float]:
    """Get current price for trading pair"""
    global price_cache, price_cache_time
    
    # Check cache first
    now = time.time()
    if pair in price_cache and (now - price_cache_time.get(pair, 0)) < PRICE_CACHE_EXPIRY:
        return price_cache[pair]
    
    # For now, using static prices with small random adjustments
    # In a real implementation, this would call the Kraken API
    base_prices = {
        "BTC/USD": 62350.0,
        "ETH/USD": 3050.0,
        "SOL/USD": 142.50,
        "ADA/USD": 0.45,
        "DOT/USD": 6.75,
        "LINK/USD": 15.30,
        "AVAX/USD": 35.25,
        "MATIC/USD": 0.65,
        "UNI/USD": 9.80,
        "ATOM/USD": 8.45
    }
    
    if pair in base_prices:
        # Add ±1% random movement
        movement = random.uniform(-0.01, 0.01)
        price = base_prices[pair] * (1 + movement)
        
        # Update cache
        price_cache[pair] = price
        price_cache_time[pair] = now
        
        return price
    
    return None

def load_portfolio() -> Dict:
    """Load portfolio or create new one if it doesn't exist"""
    try:
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
                
            # Ensure all required fields exist for backwards compatibility
            if "available_capital" not in portfolio and "balance" in portfolio:
                portfolio["available_capital"] = portfolio["balance"]
            if "initial_capital" not in portfolio:
                portfolio["initial_capital"] = INITIAL_CAPITAL
            if "win_rate" not in portfolio:
                portfolio["win_rate"] = 0.0
            if "total_trades" not in portfolio:
                portfolio["total_trades"] = 0
            if "profitable_trades" not in portfolio:
                portfolio["profitable_trades"] = 0
                
            logger.info(f"Loaded portfolio from {PORTFOLIO_FILE}")
            return portfolio
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
    
    # Create new portfolio
    portfolio = {
        "initial_capital": INITIAL_CAPITAL,
        "available_capital": INITIAL_CAPITAL,
        "balance": INITIAL_CAPITAL,
        "equity": INITIAL_CAPITAL,
        "total_value": INITIAL_CAPITAL,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "profitable_trades": 0,
        "updated_at": datetime.datetime.now().isoformat()
    }
    
    # Save new portfolio
    save_portfolio(portfolio)
    logger.info(f"Created new portfolio with {INITIAL_CAPITAL:.2f} capital")
    return portfolio

def load_positions() -> List:
    """Load positions or create empty list if it doesn't exist"""
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions from {POSITIONS_FILE}")
            return positions
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
    
    # Create empty positions
    positions = []
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)
    logger.info("Created empty positions file")
    return positions

def save_portfolio(portfolio: Dict) -> None:
    """Save portfolio to file"""
    try:
        portfolio["updated_at"] = datetime.datetime.now().isoformat()
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")

def save_positions(positions: List) -> None:
    """Save positions to file"""
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving positions: {e}")

def update_portfolio_history(portfolio: Dict) -> None:
    """Update portfolio history with current value"""
    try:
        # Load existing history or create new
        history = []
        if os.path.exists(PORTFOLIO_HISTORY_FILE):
            with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        
        # Add current portfolio snapshot
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_value": portfolio["total_value"],
            "available_capital": portfolio["available_capital"],
            "total_pnl": portfolio["total_pnl"],
            "total_pnl_pct": portfolio["total_pnl_pct"],
            "unrealized_pnl": portfolio["unrealized_pnl"],
            "unrealized_pnl_pct": portfolio["unrealized_pnl_pct"],
            "win_rate": portfolio["win_rate"],
            "total_trades": portfolio["total_trades"],
            "profitable_trades": portfolio["profitable_trades"]
        }
        
        history.append(entry)
        
        # Limit history to 1000 entries
        if len(history) > 1000:
            history = history[-1000:]
        
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating portfolio history: {e}")

def simulate_trading():
    """Run a simple trading simulation"""
    logger.info("Starting simplified trading simulation")
    
    # Load portfolio and positions
    portfolio = load_portfolio()
    positions = load_positions()
    
    try:
        # Simple trading loop
        for _ in range(5):  # Just run a few iterations for demonstration
            # 1. Update existing positions
            total_unrealized_pnl = 0
            for position in positions:
                # Get current price
                current_price = get_price(position["pair"])
                if current_price:
                    # Update position with current price
                    position["current_price"] = current_price
                    position["last_update_time"] = datetime.datetime.now().isoformat()
                    
                    # Calculate P&L
                    entry_price = position["entry_price"]
                    size = position["size"]
                    leverage = position["leverage"]
                    direction = position["direction"]
                    
                    # Calculate P&L based on direction
                    if direction == "LONG":
                        profit_factor = (current_price - entry_price) / entry_price
                    else:  # SHORT
                        profit_factor = (entry_price - current_price) / entry_price
                    
                    # Update position P&L
                    position["unrealized_pnl"] = size * leverage * profit_factor
                    position["unrealized_pnl_pct"] = profit_factor * leverage * 100
                    
                    # Add to total P&L
                    total_unrealized_pnl += position["unrealized_pnl"]
            
            # 2. Update portfolio P&L
            portfolio["unrealized_pnl"] = total_unrealized_pnl
            portfolio["total_value"] = portfolio["available_capital"] + total_unrealized_pnl
            
            if portfolio["initial_capital"] > 0:
                portfolio["total_pnl"] = portfolio["total_value"] - portfolio["initial_capital"]
                portfolio["total_pnl_pct"] = (portfolio["total_pnl"] / portfolio["initial_capital"]) * 100
                portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / portfolio["initial_capital"]) * 100
            
            # 3. Try to open a new position
            if len(positions) < 3:  # Limit to 3 positions for demonstration
                # Check if we have capital available
                if portfolio["available_capital"] > 1000:
                    # Choose a random pair
                    tradable_pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                                     "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
                    
                    # Skip pairs that already have positions
                    current_pairs = [p["pair"] for p in positions]
                    available_pairs = [p for p in tradable_pairs if p not in current_pairs]
                    
                    if available_pairs:
                        # Choose a random pair
                        pair = random.choice(available_pairs)
                        price = get_price(pair)
                        
                        if price:
                            # Create new position
                            new_position = {
                                "pair": pair,
                                "direction": random.choice(["LONG", "SHORT"]),
                                "entry_price": price,
                                "current_price": price,
                                "size": 0.5,  # Fixed size for demonstration
                                "leverage": random.randint(5, 20),  # Random leverage between 5x and 20x
                                "entry_time": datetime.datetime.now().isoformat(),
                                "last_update_time": datetime.datetime.now().isoformat(),
                                "unrealized_pnl": 0.0,
                                "unrealized_pnl_pct": 0.0
                            }
                            
                            # Calculate required capital
                            required_capital = (new_position["size"] * price) / new_position["leverage"]
                            if required_capital <= portfolio["available_capital"]:
                                # Add to positions and update portfolio
                                positions.append(new_position)
                                portfolio["available_capital"] -= required_capital
                                
                                logger.info(f"Opened {new_position['direction']} position for {pair} at ${price:.2f}")
            
            # 4. Try to close a position
            if positions and random.random() < 0.3:  # 30% chance to close a position
                position = random.choice(positions)
                
                logger.info(f"Closing {position['direction']} position for {position['pair']} at ${position['current_price']:.2f}")
                
                # Return capital + profit to portfolio
                portfolio["available_capital"] += (
                    (position["size"] * position["current_price"]) / position["leverage"] +
                    position["unrealized_pnl"]
                )
                
                # Update trade stats
                portfolio["total_trades"] += 1
                if position["unrealized_pnl"] > 0:
                    portfolio["profitable_trades"] += 1
                
                # Update win rate
                portfolio["win_rate"] = portfolio["profitable_trades"] / max(1, portfolio["total_trades"])
                
                # Remove position
                positions.remove(position)
            
            # Save updated data
            save_portfolio(portfolio)
            save_positions(positions)
            update_portfolio_history(portfolio)
            
            # Log status
            logger.info(f"Portfolio: ${portfolio['total_value']:.2f} | " +
                      f"P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%) | " +
                      f"Active Positions: {len(positions)}")
            
            # Wait a moment
            time.sleep(2)
    
    except KeyboardInterrupt:
        logger.info("Trading simulation stopped by user")
    except Exception as e:
        logger.error(f"Error in trading simulation: {e}")
    finally:
        # Final save
        save_portfolio(portfolio)
        save_positions(positions)
        update_portfolio_history(portfolio)
        
        logger.info("Trading simulation complete")

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" SIMPLE DIRECT TRADING RUNNER")
    print("=" * 60 + "\n")
    
    try:
        simulate_trading()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\n" + "=" * 60)
        print(" TRADING RUNNER COMPLETE")
        print("=" * 60)

if __name__ == "__main__":
    main()