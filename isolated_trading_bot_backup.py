#!/usr/bin/env python3
"""
Isolated Trading Bot

A completely standalone trading bot that doesn't use Flask or any web components,
avoiding port conflicts and import issues.
"""
import os
import sys
import time
import json
import random
import logging
import datetime
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
INITIAL_CAPITAL = 20000.0
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "sandbox_portfolio.json")
POSITIONS_FILE = os.path.join(DATA_DIR, "sandbox_positions.json")
TRADES_FILE = os.path.join(DATA_DIR, "sandbox_trades.json")
PORTFOLIO_HISTORY_FILE = os.path.join(DATA_DIR, "sandbox_portfolio_history.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Price cache to reduce API calls
price_cache = {}
price_cache_time = {}
PRICE_CACHE_EXPIRY = 5  # seconds

def get_price(pair: str) -> Optional[float]:
    """Get current price for trading pair from Kraken API with retry mechanism"""
    global price_cache, price_cache_time
    
    # Check cache first (shorter expiry - 3 seconds - to ensure more real-time data)
    now = time.time()
    if pair in price_cache and (now - price_cache_time.get(pair, 0)) < 3.0:
        return price_cache[pair]
    
    # Try multiple times to ensure we get real-time data
    for attempt in range(3):
        try:
            # Get price from Kraken API
            from kraken_api_client import get_price_for_pair
            
            # Get the current price
            price = get_price_for_pair(pair, sandbox=True)
            
            if price is not None:
                # Update cache
                price_cache[pair] = price
                price_cache_time[pair] = now
                logger.info(f"Price for {pair} from Kraken API: ${price:.2f}")
                return price
            
            # If price is None, wait briefly and retry
            if attempt < 2:
                logger.warning(f"Retrying price fetch for {pair} (attempt {attempt+1}/3)")
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error fetching price for {pair} (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(0.5)
    
    # After all retries failed, use a fallback (only in extreme cases)
    logger.error(f"All attempts to get real-time price for {pair} failed")
    
    # If we have a cached price that's not too old (within 60 seconds), use it
    if pair in price_cache and (now - price_cache_time.get(pair, 0)) < 60.0:
        logger.warning(f"Using slightly stale cached price for {pair}")
        return price_cache[pair]
    
    # As an absolute last resort, use a fallback price
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
        logger.warning(f"Using emergency fallback price for {pair}")
        price = base_prices[pair]
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

def add_trade_record(trade_data: Dict) -> None:
    """Add a trade record to the trades file"""
    try:
        # Load existing trades or create new list
        trades = []
        if os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
        
        # Add timestamp if not present
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add trade
        trades.append(trade_data)
        
        # Save trades
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
    except Exception as e:
        logger.error(f"Error adding trade record: {e}")

def simulate_trading():
    """Run an enhanced trading simulation using real-time market data"""
    logger.info("Starting enhanced trading simulation with real market data")
    
    # Load portfolio and positions
    portfolio = load_portfolio()
    positions = load_positions()
    
    try:
        # Trading loop
        for _ in range(10):  # Run for a limited number of iterations
            # 1. Update existing positions with current prices
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
            
            # 3. Try to open new positions
            if len(positions) < 6:  # Limit to 6 positions
                # Check if we have capital available
                if portfolio["available_capital"] > 1000:
                    # All trading pairs that we support
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
                            # Calculate confidence and leverage based on ML model prediction
                            # For now, using random simulation values
                            confidence = random.uniform(0.6, 0.95)
                            
                            # Scale leverage based on confidence (5x-20x)
                            # Higher confidence = higher leverage
                            max_leverage = 20
                            min_leverage = 5
                            leverage = min_leverage + (max_leverage - min_leverage) * confidence
                            leverage = round(leverage)
                            
                            # Direction based on confidence
                            # Higher confidence > 0.8 = LONG, otherwise 50/50
                            if confidence > 0.8:
                                direction = "LONG"
                            else:
                                direction = random.choice(["LONG", "SHORT"])
                            
                            # Position size based on confidence and available capital
                            # Using size relative to capital and confidence
                            max_capital_pct = 0.1  # Max 10% of capital per position
                            size_pct = max_capital_pct * confidence  # Scale by confidence
                            allocation = portfolio["available_capital"] * size_pct
                            size = allocation * leverage / price  # Convert to asset amount
                            
                            # Create new position
                            new_position = {
                                "pair": pair,
                                "direction": direction,
                                "entry_price": price,
                                "current_price": price,
                                "size": size,
                                "leverage": leverage,
                                "confidence": confidence,
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
                                
                                logger.info(f"Opened {direction} position for {pair} at ${price:.2f} with {leverage}x leverage (confidence: {confidence:.2f})")
                                
                                # Add trade record
                                trade_data = {
                                    "type": "OPEN",
                                    "pair": pair,
                                    "direction": direction,
                                    "price": price,
                                    "size": size,
                                    "leverage": leverage,
                                    "confidence": confidence,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                add_trade_record(trade_data)
            
            # 4. Check for positions to close based on profit/loss
            positions_to_close = []
            for position in positions:
                # Calculate profit or loss percentage
                profit_pct = position["unrealized_pnl_pct"]
                
                # Close position based on profit/loss thresholds
                if profit_pct >= 10:  # Take profit at 10%
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} with profit: {profit_pct:.2f}%")
                elif profit_pct <= -5:  # Stop loss at -5%
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} with loss: {profit_pct:.2f}%")
                elif random.random() < 0.1:  # Random close with 10% probability
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} at ${position['current_price']:.2f} (random)")
            
            # Process positions to close
            for position in positions_to_close:
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
                
                # Add trade record
                trade_data = {
                    "type": "CLOSE",
                    "pair": position["pair"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": position["current_price"],
                    "size": position["size"],
                    "leverage": position["leverage"],
                    "pnl": position["unrealized_pnl"],
                    "pnl_pct": position["unrealized_pnl_pct"],
                    "timestamp": datetime.datetime.now().isoformat()
                }
                add_trade_record(trade_data)
                
                # Remove position
                positions.remove(position)
            
            # Save updated data
            save_portfolio(portfolio)
            save_positions(positions)
            update_portfolio_history(portfolio)
            
            # Log portfolio status
            logger.info(f"Portfolio: ${portfolio['total_value']:.2f} | " +
                      f"P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%) | " +
                      f"Active Positions: {len(positions)}")
            
            # Wait a moment before next iteration
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

def reset_portfolio():
    """Reset portfolio and positions to initial state"""
    logger.info("Resetting portfolio and positions to initial state")
    
    # Create fresh portfolio
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
    
    # Save portfolio
    save_portfolio(portfolio)
    
    # Reset positions to empty list
    positions = []
    save_positions(positions)
    
    # Reset trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'w') as f:
            json.dump([], f, indent=2)
    
    # Reset portfolio history
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
            json.dump([], f, indent=2)
    
    logger.info(f"Portfolio reset to ${INITIAL_CAPITAL:.2f}, all positions closed")
    return portfolio, positions

def simulate_continuous_trading():
    """Run a continuous trading simulation with real-time market data"""
    logger.info("Starting continuous trading simulation with real-time market data")
    
    # Reset portfolio to start fresh
    portfolio, positions = reset_portfolio()
    
    try:
        # Infinite trading loop
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"Trading iteration {iteration}")
            
            # 1. Update existing positions with current prices
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
                    position["unrealized_pnl"] = size * current_price * profit_factor
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
            
            # 3. Try to open new positions if we have less than 5 active positions
            if len(positions) < 5:
                # Check if we have capital available (minimum $1000)
                if portfolio["available_capital"] > 1000:
                    # All trading pairs that we support
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
                            # Calculate confidence and leverage based on ML model prediction
                            # For now, using random simulation values
                            confidence = random.uniform(0.6, 0.95)
                            
                            # Scale leverage based on confidence (5x-20x)
                            # Higher confidence = higher leverage
                            max_leverage = 20
                            min_leverage = 5
                            leverage = min_leverage + (max_leverage - min_leverage) * confidence
                            leverage = round(leverage)
                            
                            # Direction based on confidence
                            # Higher confidence > 0.8 = LONG, otherwise 50/50
                            if confidence > 0.8:
                                direction = "LONG"
                            else:
                                direction = random.choice(["LONG", "SHORT"])
                            
                            # Position size based on confidence and available capital
                            # Using size relative to capital and confidence
                            max_capital_pct = 0.15  # Max 15% of capital per position
                            size_pct = max_capital_pct * confidence  # Scale by confidence
                            allocation = portfolio["available_capital"] * size_pct
                            
                            # Calculate size in coins
                            size = allocation / price  # Non-leveraged size
                            leveraged_exposure = size * price * leverage  # Total exposure
                            
                            # Create new position
                            new_position = {
                                "pair": pair,
                                "direction": direction,
                                "entry_price": price,
                                "current_price": price,
                                "size": size,
                                "leverage": leverage,
                                "confidence": confidence,
                                "leveraged_exposure": leveraged_exposure,
                                "allocated_capital": allocation,
                                "entry_time": datetime.datetime.now().isoformat(),
                                "last_update_time": datetime.datetime.now().isoformat(),
                                "unrealized_pnl": 0.0,
                                "unrealized_pnl_pct": 0.0
                            }
                            
                            # Calculate required capital (margin)
                            required_capital = allocation
                            
                            if required_capital <= portfolio["available_capital"]:
                                # Add to positions and update portfolio
                                positions.append(new_position)
                                portfolio["available_capital"] -= required_capital
                                
                                logger.info(f"Opened {direction} position for {pair} at ${price:.2f} with {leverage}x leverage (confidence: {confidence:.2f})")
                                
                                # Add trade record
                                trade_data = {
                                    "type": "OPEN",
                                    "pair": pair,
                                    "direction": direction,
                                    "price": price,
                                    "size": size,
                                    "leverage": leverage,
                                    "confidence": confidence,
                                    "allocated_capital": allocation,
                                    "timestamp": datetime.datetime.now().isoformat()
                                }
                                add_trade_record(trade_data)
            
            # 4. Check positions to close based on profit/loss
            positions_to_close = []
            for position in positions:
                # Calculate profit or loss percentage
                profit_pct = position["unrealized_pnl_pct"]
                
                # Close position based on profit/loss thresholds or random chance
                # Take profit at 8% for high leverage (>15x), 15% for medium leverage, 20% for low leverage
                take_profit_threshold = 20 if position["leverage"] < 10 else (15 if position["leverage"] < 15 else 8)
                
                # Stop loss at -5% for high leverage (>15x), -8% for medium leverage, -10% for low leverage
                stop_loss_threshold = -10 if position["leverage"] < 10 else (-8 if position["leverage"] < 15 else -5)
                
                if profit_pct >= take_profit_threshold:  # Take profit
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} with profit: {profit_pct:.2f}% (Take Profit)")
                elif profit_pct <= stop_loss_threshold:  # Stop loss
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} with loss: {profit_pct:.2f}% (Stop Loss)")
                elif random.random() < 0.05:  # Random close with 5% probability
                    positions_to_close.append(position)
                    logger.info(f"Closing {position['direction']} position for {position['pair']} at ${position['current_price']:.2f} (Random/Signal Change)")
            
            # Process positions to close
            for position in positions_to_close:
                # Calculate actual profit/loss for this position
                pnl = position["unrealized_pnl"]
                
                # Return allocated capital + profit to portfolio
                returned_capital = position["allocated_capital"] + pnl
                portfolio["available_capital"] += returned_capital
                
                # Update trade stats
                portfolio["total_trades"] += 1
                if pnl > 0:
                    portfolio["profitable_trades"] += 1
                
                # Update win rate
                portfolio["win_rate"] = portfolio["profitable_trades"] / max(1, portfolio["total_trades"])
                
                # Add trade record
                trade_data = {
                    "type": "CLOSE",
                    "pair": position["pair"],
                    "direction": position["direction"],
                    "entry_price": position["entry_price"],
                    "exit_price": position["current_price"],
                    "size": position["size"],
                    "leverage": position["leverage"],
                    "pnl": pnl,
                    "pnl_pct": position["unrealized_pnl_pct"],
                    "allocated_capital": position["allocated_capital"],
                    "returned_capital": returned_capital,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                add_trade_record(trade_data)
                
                # Remove position
                positions.remove(position)
            
            # 5. Save updated data
            save_portfolio(portfolio)
            save_positions(positions)
            update_portfolio_history(portfolio)
            
            # 6. Log portfolio status
            logger.info(f"Portfolio: ${portfolio['total_value']:.2f} | " +
                      f"P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%) | " +
                      f"Active Positions: {len(positions)}")
            
            # 7. Wait before next iteration (5 seconds)
            time.sleep(5)
    
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
    print(" ISOLATED TRADING BOT WITH REAL MARKET DATA")
    print("=" * 60 + "\n")
    
    try:
        # Run continuous trading simulation with real-time market data
        simulate_continuous_trading()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\n" + "=" * 60)
        print(" TRADING BOT COMPLETED")
        print("=" * 60)

if __name__ == "__main__":
    main()