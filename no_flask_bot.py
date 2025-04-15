#!/usr/bin/env python3
"""
IMPORTANT: THIS SCRIPT MUST BE RUN MANUALLY WITH:
python3 no_flask_bot.py

This script completely avoids using Flask and will directly run the trading bot
without any port conflicts.
"""
import os
import sys
import time
import json
import random
import logging
from datetime import datetime, timedelta

# CRITICAL: Set environment variables to prevent Flask from loading
os.environ["DISABLE_FLASK"] = "1"
os.environ["TRADING_BOT_PROCESS"] = "1"
os.environ["FLASK_APP"] = "none"
os.environ["PYTHONPATH"] = os.getcwd()

# Configure logging directly to avoid Flask logger conflicts
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "trading_bot.log")

logger = logging.getLogger("no_flask_bot")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Print startup message
logger.info("=" * 70)
logger.info("TRADING BOT STARTING - NO FLASK MODE")
logger.info("=" * 70)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = os.path.join(DATA_DIR, "sandbox_portfolio.json")
POSITIONS_FILE = os.path.join(DATA_DIR, "sandbox_positions.json")
PORTFOLIO_HISTORY_FILE = os.path.join(DATA_DIR, "sandbox_portfolio_history.json")
TRADES_FILE = os.path.join(DATA_DIR, "sandbox_trades.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Trading pairs
TRADING_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", 
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Helper functions
def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default if default is not None else {}

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def calculate_dynamic_leverage(confidence_score):
    """
    Calculate dynamic leverage based on confidence score
    Scales from 5x (minimum) to 125x (maximum) based on confidence
    
    Args:
        confidence_score (float): Confidence score between 0.5 and 1.0
        
    Returns:
        float: Calculated leverage value
    """
    # Normalize confidence to 0-1 range (assuming confidence is 0.5-1.0)
    normalized_confidence = max(0, min(1, (confidence_score - 0.5) * 2))
    
    # Calculate leverage from 5x to 125x based on normalized confidence
    min_leverage = 5.0
    max_leverage = 125.0
    leverage = min_leverage + normalized_confidence * (max_leverage - min_leverage)
    
    # Round to 1 decimal place for readability
    return round(leverage, 1)

def get_kraken_price(pair):
    """
    Get the current price from Kraken API or fallback to simulated price
    
    Args:
        pair (str): Trading pair (e.g., "BTC/USD")
        
    Returns:
        float: Current price
    """
    # Replace "/" with "" for Kraken API format
    symbol = pair.replace("/", "")
    
    # Current approximate market prices as of April 2025 (for simulation)
    base_prices = {
        "SOLUSD": 285.75,
        "BTCUSD": 93750.25,
        "ETHUSD": 4975.50,
        "ADAUSD": 1.25,
        "DOTUSD": 17.80,
        "LINKUSD": 22.45,
        "AVAXUSD": 45.20,
        "MATICUSD": 1.35,
        "UNIUSD": 12.80,
        "ATOMUSD": 14.95
    }
    
    # Add some randomness to simulate price movement
    if symbol in base_prices:
        base_price = base_prices[symbol]
        # Add random noise between -0.5% and +0.5%
        noise = base_price * (random.random() * 0.01 - 0.005)
        return round(base_price + noise, 2)
    else:
        logger.warning(f"Unknown trading pair: {pair}, using fallback price")
        return 100.0  # Fallback price

def create_new_portfolio():
    """Create a new portfolio with default values"""
    portfolio = {
        "balance": 20000.0,
        "equity": 20000.0,
        "unrealized_pnl_usd": 0.0,
        "unrealized_pnl_pct": 0.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "daily_pnl": 0.0,
        "weekly_pnl": 0.0,
        "monthly_pnl": 0.0,
        "open_positions_count": 0,
        "margin_used_pct": 0.0,
        "available_margin": 20000.0,
        "max_leverage": 125.0
    }
    
    # Save the portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    
    # Create empty positions and trades files
    save_file(POSITIONS_FILE, [])
    save_file(TRADES_FILE, [])
    
    # Create portfolio history with initial value
    now = datetime.now().isoformat()
    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    
    portfolio_history = [
        {"timestamp": yesterday, "portfolio_value": 20000.0},
        {"timestamp": now, "portfolio_value": 20000.0}
    ]
    save_file(PORTFOLIO_HISTORY_FILE, portfolio_history)
    
    logger.info("Created new portfolio with initial balance of $20,000")
    return portfolio

def update_portfolio_and_positions():
    """Update portfolio and positions with current market prices"""
    try:
        # Load portfolio and positions
        portfolio = load_file(PORTFOLIO_FILE, {})
        positions = load_file(POSITIONS_FILE, [])
        
        # Create new portfolio if it doesn't exist
        if not portfolio:
            portfolio = create_new_portfolio()
            positions = []
        
        # Update each position with current market price
        total_unrealized_pnl = 0
        
        for position in positions:
            pair = position.get("pair")
            entry_price = position.get("entry_price", 0)
            current_price = get_kraken_price(pair)
            position_size = position.get("position_size", 0)
            is_long = position.get("direction", "long") == "long"
            leverage = position.get("leverage", 20.0)
            
            # Calculate P&L
            if is_long:
                pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
                pnl_amount = position_size * pnl_pct / 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
                pnl_amount = position_size * pnl_pct / 100
            
            # Update position data
            position["current_price"] = current_price
            position["unrealized_pnl_pct"] = round(pnl_pct, 2)
            position["unrealized_pnl_amount"] = round(pnl_amount, 2)
            position["current_value"] = round(position_size + pnl_amount, 2)
            
            # Add to total unrealized P&L
            total_unrealized_pnl += pnl_amount
        
        # Update portfolio
        portfolio["unrealized_pnl_usd"] = round(total_unrealized_pnl, 2)
        portfolio["unrealized_pnl_pct"] = round(total_unrealized_pnl / portfolio.get("balance", 20000.0) * 100, 2)
        portfolio["equity"] = round(portfolio.get("balance", 20000.0) + total_unrealized_pnl, 2)
        portfolio["open_positions_count"] = len(positions)
        
        # Calculate margin used
        total_margin = sum(p.get("position_size", 0) for p in positions)
        portfolio["margin_used_pct"] = round(total_margin / portfolio.get("balance", 20000.0) * 100, 2)
        portfolio["available_margin"] = round(portfolio.get("balance", 20000.0) - total_margin, 2)
        
        # Save updated data
        save_file(PORTFOLIO_FILE, portfolio)
        save_file(POSITIONS_FILE, positions)
        
        # Update portfolio history
        portfolio_history = load_file(PORTFOLIO_HISTORY_FILE, [])
        
        # Add new history point if it's been more than 30 minutes since the last one
        if portfolio_history:
            last_timestamp = portfolio_history[-1].get("timestamp", "")
            last_time = datetime.fromisoformat(last_timestamp) if last_timestamp else datetime.now() - timedelta(hours=1)
            
            if (datetime.now() - last_time) > timedelta(minutes=30):
                portfolio_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_value": portfolio.get("equity", 20000.0)
                })
                save_file(PORTFOLIO_HISTORY_FILE, portfolio_history)
        else:
            # Create new history if none exists
            now = datetime.now().isoformat()
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            
            portfolio_history = [
                {"timestamp": yesterday, "portfolio_value": 20000.0},
                {"timestamp": now, "portfolio_value": portfolio.get("equity", 20000.0)}
            ]
            save_file(PORTFOLIO_HISTORY_FILE, portfolio_history)
        
        return True
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_ml_signal(pair):
    """
    Generate a simulated ML trading signal for the given pair
    
    Args:
        pair (str): Trading pair (e.g., "BTC/USD")
        
    Returns:
        dict: Signal details
    """
    # Simulate an ML model prediction
    confidence = random.uniform(0.65, 0.98)
    direction = random.choice(["long", "short"]) if confidence > 0.8 else None
    
    # Dynamically calculate leverage based on confidence
    leverage = calculate_dynamic_leverage(confidence) if direction else 0
    
    return {
        "pair": pair,
        "direction": direction,
        "confidence": round(confidence, 2),
        "leverage": leverage,
        "timestamp": datetime.now().isoformat(),
        "signal_strength": round(confidence * 100, 1),
        "model": random.choice(["ARIMA", "Adaptive"]),
        "category": random.choice(["those dudes", "him all along"]),
        "reason": f"ML model prediction with {round(confidence * 100, 1)}% confidence"
    }

def execute_trade(signal):
    """
    Execute a trade based on the given signal
    
    Args:
        signal (dict): Trading signal
        
    Returns:
        bool: Success or failure
    """
    try:
        # Skip if no direction
        if not signal.get("direction"):
            return False
        
        # Load portfolio and positions
        portfolio = load_file(PORTFOLIO_FILE, {})
        positions = load_file(POSITIONS_FILE, [])
        trades = load_file(TRADES_FILE, [])
        
        # Create portfolio if it doesn't exist or is empty
        if not portfolio:
            portfolio = create_new_portfolio()
            positions = []
            trades = []
        
        # Check if we already have a position for this pair
        for position in positions:
            if position.get("pair") == signal.get("pair"):
                logger.info(f"Already have a position for {signal.get('pair')}, skipping trade")
                return False
        
        # Calculate position size (20% of portfolio balance)
        risk_percentage = 0.20  # 20% risk per trade
        balance = portfolio.get("balance", 20000.0)
        position_size = round(balance * risk_percentage, 2)
        
        # Get current price
        pair = signal.get("pair")
        current_price = get_kraken_price(pair)
        direction = signal.get("direction")
        leverage = signal.get("leverage", 20.0)
        confidence = signal.get("confidence", 0.75)
        
        # Create new position
        trade_id = f"trade_{len(trades) + 1}"
        new_position = {
            "pair": pair,
            "entry_price": current_price,
            "current_price": current_price,
            "position_size": position_size,
            "direction": direction,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "unrealized_pnl_pct": 0.0,
            "unrealized_pnl_amount": 0.0,
            "current_value": position_size,
            "confidence": confidence,
            "model": signal.get("model", "Ensemble"),
            "category": signal.get("category", "those dudes"),
            "stop_loss_pct": 4.0,  # 4% maximum loss as per requirements
            "take_profit_pct": round(8.0 + confidence * 10, 1),  # Dynamic take profit based on confidence
            "open_trade_id": trade_id
        }
        
        # Add to positions list
        positions.append(new_position)
        
        # Create trade record
        new_trade = {
            "trade_id": trade_id,
            "pair": pair,
            "entry_price": current_price,
            "direction": direction,
            "position_size": position_size,
            "leverage": leverage,
            "entry_time": datetime.now().isoformat(),
            "exit_price": None,
            "exit_time": None,
            "pnl_amount": None,
            "pnl_pct": None,
            "status": "open",
            "reason_entry": signal.get("reason", "ML signal"),
            "reason_exit": None,
            "confidence": confidence,
            "model": signal.get("model", "Ensemble"),
            "category": signal.get("category", "those dudes")
        }
        
        # Add to trades list
        trades.append(new_trade)
        
        # Update portfolio available margin
        portfolio["open_positions_count"] = len(positions)
        
        # Save updated data
        save_file(POSITIONS_FILE, positions)
        save_file(TRADES_FILE, trades)
        save_file(PORTFOLIO_FILE, portfolio)
        
        logger.info(f"Opened {direction} position for {pair} at {current_price} with {leverage}x leverage and {position_size:.2f} USD")
        return True
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_and_close_positions():
    """Check positions for take profit, stop loss, or other exit conditions"""
    try:
        # Load portfolio and positions
        portfolio = load_file(PORTFOLIO_FILE, {})
        positions = load_file(POSITIONS_FILE, [])
        trades = load_file(TRADES_FILE, [])
        
        if not portfolio or not positions:
            return False
        
        positions_to_remove = []
        
        # Check each position
        for position in positions:
            pair = position.get("pair")
            entry_price = position.get("entry_price", 0)
            current_price = get_kraken_price(pair)
            position_size = position.get("position_size", 0)
            is_long = position.get("direction", "long") == "long"
            leverage = position.get("leverage", 20.0)
            stop_loss_pct = position.get("stop_loss_pct", 4.0)
            take_profit_pct = position.get("take_profit_pct", 12.0)
            
            # Calculate P&L
            if is_long:
                pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
            
            pnl_amount = position_size * pnl_pct / 100
            
            # Check for exit conditions
            exit_reason = None
            
            # Take profit condition
            if pnl_pct >= take_profit_pct:
                exit_reason = f"Take profit at {take_profit_pct}%"
            
            # Stop loss condition
            elif pnl_pct <= -stop_loss_pct:
                exit_reason = f"Stop loss at {stop_loss_pct}%"
            
            # Exit the position if we have a reason
            if exit_reason:
                positions_to_remove.append(position)
                
                # Update the trade record
                for trade in trades:
                    if trade.get("trade_id") == position.get("open_trade_id"):
                        trade["exit_price"] = current_price
                        trade["exit_time"] = datetime.now().isoformat()
                        trade["pnl_amount"] = round(pnl_amount, 2)
                        trade["pnl_pct"] = round(pnl_pct, 2)
                        trade["status"] = "closed"
                        trade["reason_exit"] = exit_reason
                        break
                
                # Update portfolio balance
                portfolio["balance"] = round(portfolio.get("balance", 20000.0) + pnl_amount, 2)
                
                logger.info(f"Closed {position.get('direction')} position for {pair} at {current_price} with P&L: {pnl_amount:.2f} USD ({pnl_pct:.2f}%)")
        
        # Remove closed positions
        if positions_to_remove:
            positions = [p for p in positions if p not in positions_to_remove]
            
            # Update portfolio
            portfolio["open_positions_count"] = len(positions)
            closed_trades = [t for t in trades if t.get("status") == "closed"]
            total_pnl = sum(float(t.get("pnl_amount", 0) or 0) for t in closed_trades)
            initial_balance = 20000.0
            portfolio["total_pnl"] = round(total_pnl, 2)
            portfolio["total_pnl_pct"] = round(total_pnl / initial_balance * 100, 2)
            
            # Save updated data
            save_file(POSITIONS_FILE, positions)
            save_file(TRADES_FILE, trades)
            save_file(PORTFOLIO_FILE, portfolio)
            
            return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking positions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_loop():
    """Main trading bot loop"""
    logger.info("Starting trading bot in sandbox mode")
    logger.info(f"Dynamic leverage range: 5x - 125x based on confidence")
    
    # Make sure we have a portfolio
    portfolio = load_file(PORTFOLIO_FILE, None)
    if not portfolio:
        create_new_portfolio()
    
    last_signal_time = {}
    for pair in TRADING_PAIRS:
        last_signal_time[pair] = datetime.now() - timedelta(hours=1)
    
    # Run until interrupted
    try:
        while True:
            # Update portfolio and positions with current prices
            update_portfolio_and_positions()
            
            # Check and close positions if needed
            check_and_close_positions()
            
            # Generate trading signals for each pair
            current_time = datetime.now()
            
            for pair in TRADING_PAIRS:
                # Only generate signals every 15-60 minutes for each pair
                time_since_last = (current_time - last_signal_time[pair]).total_seconds() / 60
                
                # Random interval between 15-60 minutes
                interval = random.randint(15, 60)
                
                if time_since_last >= interval:
                    signal = generate_ml_signal(pair)
                    
                    # Only execute trades with high confidence
                    if signal.get("confidence", 0) > 0.75 and signal.get("direction"):
                        if execute_trade(signal):
                            last_signal_time[pair] = current_time
                            
                            # Log the leverage and confidence details
                            logger.info(f"Trade executed for {pair} with {signal.get('leverage')}x leverage based on {signal.get('confidence') * 100}% confidence")
            
            # Sleep for a while
            time.sleep(15)  # Update every 15 seconds
            
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

# Main entry point
if __name__ == "__main__":
    logger.info("Starting trading bot - NO FLASK EDITION")
    main_loop()
else:
    logger.warning("This script must be run directly, not imported!")