#!/usr/bin/env python3
"""
Trading Bot Server

This script starts a minimal Flask server on a different port (8080)
to show trading bot information without conflicting with the main dashboard.
"""
import os
import sys
import json
import logging
import threading
import random
import time
from datetime import datetime
from flask import Flask, jsonify, render_template

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create the Flask app explicitly for the bot server
bot_app = Flask(__name__, static_folder='static', template_folder='templates')
bot_app.secret_key = os.environ.get("BOT_SERVER_SECRET", "trading_bot_secret_key")

# Global state
running = False
current_prices = {}
trading_pairs = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Sandbox prices
SANDBOX_PRICES = {
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

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")

def update_prices():
    """Update price data periodically"""
    global current_prices, running
    
    while running:
        try:
            for pair in trading_pairs:
                # Use sandbox prices with small variations
                if pair in SANDBOX_PRICES:
                    base_price = SANDBOX_PRICES[pair]
                    # Add small random price movement (±0.5%)
                    price_change_pct = (random.random() - 0.5) * 0.01
                    price = base_price * (1 + price_change_pct)
                    current_prices[pair] = price
            
            # Log price updates occasionally
            if random.random() < 0.01:  # Log about 1% of updates
                pair = random.choice(trading_pairs)
                logger.debug(f"Price update for {pair}: ${current_prices.get(pair, 0):.2f}")
            
            # Update portfolio with new prices
            update_portfolio()
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
            time.sleep(5)

def update_portfolio():
    """Update portfolio with current prices"""
    # Load data
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0,
        "equity": 20000.0,
        "total_value": 20000.0,
        "unrealized_pnl_usd": 0.0,
        "unrealized_pnl_pct": 0.0,
        "last_updated": datetime.now().isoformat()
    })
    
    positions = load_file(POSITIONS_FILE, [])
    
    # Get current time
    now = datetime.now()
    
    # Calculate total unrealized PnL
    total_pnl = 0.0
    for position in positions:
        pair = position.get("pair")
        if not pair or pair not in current_prices:
            continue
        
        current_price = current_prices[pair]
        entry_price = position.get("entry_price", current_price)
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        # Update current price in position
        position["current_price"] = current_price
        
        # Calculate PnL
        if direction.lower() == "long":
            pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
            pnl_amount = (current_price - entry_price) * size
        else:  # Short
            pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
            pnl_amount = (entry_price - current_price) * size
        
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
    
    # Update portfolio
    portfolio["unrealized_pnl_usd"] = total_pnl
    portfolio["unrealized_pnl_pct"] = (total_pnl / portfolio.get("balance", 20000.0)) * 100 if portfolio.get("balance", 20000.0) > 0 else 0
    portfolio["total_value"] = portfolio.get("balance", 20000.0) + total_pnl
    portfolio["last_updated"] = now.isoformat()
    portfolio["equity"] = portfolio["total_value"]
    
    # Save updated data
    save_file(POSITIONS_FILE, positions)
    save_file(PORTFOLIO_FILE, portfolio)
    
    # Update portfolio history
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    history.append({
        "timestamp": now.isoformat(),
        "portfolio_value": portfolio["total_value"]
    })
    
    # Keep history to a reasonable size
    if len(history) > 1000:
        history = history[-1000:]
    
    save_file(PORTFOLIO_HISTORY_FILE, history)

def generate_trade():
    """Generate a simulated trade"""
    # Load data
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0,
        "equity": 20000.0,
        "total_value": 20000.0
    })
    
    positions = load_file(POSITIONS_FILE, [])
    
    # Calculate available pairs (ones without open positions)
    existing_pairs = [p.get("pair") for p in positions if p.get("pair")]
    available_pairs = [p for p in trading_pairs if p not in existing_pairs]
    
    if not available_pairs:
        logger.debug("No available pairs for new trades")
        return None
    
    # Pick a random pair from available ones
    pair = random.choice(available_pairs)
    
    # Make sure we have a price for this pair
    if pair not in current_prices:
        logger.warning(f"No price data available for {pair}")
        return None
    
    current_price = current_prices[pair]
    balance = portfolio.get("balance", 20000.0)
    
    # Simulate confidence level
    confidence = 0.7 + random.random() * 0.25  # 0.7 to 0.95
    
    # Calculate leverage based on confidence
    leverage_map = {
        0.5: 5,    # 50% confidence -> 5x leverage
        0.6: 10,   # 60% confidence -> 10x leverage
        0.7: 20,   # 70% confidence -> 20x leverage
        0.8: 40,   # 80% confidence -> 40x leverage
        0.9: 80,   # 90% confidence -> 80x leverage
        0.95: 100, # 95% confidence -> 100x leverage
        1.0: 125   # 100% confidence -> 125x leverage
    }
    
    # Find appropriate leverage
    leverage = 5  # Default
    for threshold, lev in sorted(leverage_map.items()):
        if confidence >= threshold:
            leverage = lev
    
    # Randomly choose direction with slight bias towards long
    direction = "Long" if random.random() > 0.4 else "Short"
    
    # Calculate risk percentage based on confidence
    risk_percentage = 0.01 + (confidence * 0.19)  # 1% to 20%
    
    # Calculate position size
    margin = balance * risk_percentage
    position_size = (margin * leverage) / current_price
    
    # Calculate liquidation price
    liquidation_price = current_price * (1 - 0.9/leverage) if direction == "Long" else current_price * (1 + 0.9/leverage)
    
    # Create new position
    position = {
        "pair": pair,
        "direction": direction,
        "size": position_size,
        "entry_price": current_price,
        "current_price": current_price,
        "leverage": leverage,
        "strategy": random.choice(["ARIMA", "Adaptive"]),
        "category": random.choice(["those dudes", "him all along"]),
        "confidence": confidence,
        "entry_time": datetime.now().isoformat(),
        "unrealized_pnl": 0.0,
        "unrealized_pnl_amount": 0.0,
        "unrealized_pnl_pct": 0.0,
        "liquidation_price": liquidation_price,
        "risk_percentage": risk_percentage
    }
    
    # Update balance
    portfolio["balance"] = balance - margin
    
    # Add position
    positions.append(position)
    
    # Save updated data
    save_file(POSITIONS_FILE, positions)
    save_file(PORTFOLIO_FILE, portfolio)
    
    logger.info(f"Generated {direction} trade for {pair} with leverage {leverage}x")
    logger.info(f"Position size: {position_size:.6f}, Entry price: ${current_price:.2f}")
    
    return position

def simulated_trading():
    """Simulate trading with generated trades and management"""
    global running
    
    # Initialize files if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(PORTFOLIO_FILE):
        save_file(PORTFOLIO_FILE, {
            "balance": 20000.0,
            "equity": 20000.0,
            "total_value": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        })
    
    if not os.path.exists(POSITIONS_FILE):
        save_file(POSITIONS_FILE, [])
    
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        save_file(PORTFOLIO_HISTORY_FILE, [
            {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": 20000.0
            }
        ])
    
    if not os.path.exists(TRADES_FILE):
        save_file(TRADES_FILE, [])
    
    trade_interval = 15  # seconds between trade evaluations
    manage_interval = 30  # seconds between position management checks
    
    last_trade_time = 0
    last_manage_time = 0
    
    while running:
        try:
            current_time = time.time()
            
            # Generate trades
            if current_time - last_trade_time >= trade_interval:
                positions = load_file(POSITIONS_FILE, [])
                # Only trade if we have capacity
                if len(positions) < len(trading_pairs):
                    generate_trade()
                last_trade_time = current_time
            
            # Manage positions
            if current_time - last_manage_time >= manage_interval:
                # Auto-manage positions (close profitable/unprofitable positions)
                manage_positions()
                last_manage_time = current_time
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in simulated trading: {e}")
            time.sleep(5)

def manage_positions():
    """Auto-manage positions"""
    positions = load_file(POSITIONS_FILE, [])
    if not positions:
        return
    
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0,
        "equity": 20000.0
    })
    
    # Check for positions to close
    active_positions = []
    closed_positions = []
    
    for position in positions:
        unrealized_pnl = position.get("unrealized_pnl", 0)
        
        # Take profit at high levels or cut losses
        if unrealized_pnl > 50 or unrealized_pnl < -20:
            # Calculate PnL
            pair = position.get("pair")
            if not pair or pair not in current_prices:
                active_positions.append(position)
                continue
                
            current_price = current_prices[pair]
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            leverage = position.get("leverage", 1)
            direction = position.get("direction", "Long")
            
            # Calculate final PnL
            if direction.lower() == "long":
                pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
                pnl_amount = (current_price - entry_price) * size
            else:  # Short
                pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
                pnl_amount = (entry_price - current_price) * size
            
            # Update position with exit info
            position["exit_price"] = current_price
            position["exit_time"] = datetime.now().isoformat()
            position["exit_reason"] = "TAKE_PROFIT" if unrealized_pnl > 0 else "STOP_LOSS"
            position["pnl_percentage"] = pnl_percentage
            position["pnl_amount"] = pnl_amount
            
            # Return margin + profit to balance
            margin = size * entry_price / leverage
            portfolio["balance"] = portfolio.get("balance", 20000.0) + margin + pnl_amount
            
            # Add to closed positions
            closed_positions.append(position)
            
            logger.info(f"Closed {direction} position for {pair} with PnL: {pnl_percentage:.2f}% (${pnl_amount:.2f})")
        else:
            # Keep position active
            active_positions.append(position)
    
    # If any positions were closed, update data
    if closed_positions:
        # Update positions file
        save_file(POSITIONS_FILE, active_positions)
        
        # Update portfolio file
        save_file(PORTFOLIO_FILE, portfolio)
        
        # Add to trades history
        trades = load_file(TRADES_FILE, [])
        trades.extend(closed_positions)
        save_file(TRADES_FILE, trades)

@bot_app.route('/api/status')
def api_status():
    """API endpoint for status"""
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0,
        "equity": 20000.0,
        "total_value": 20000.0
    })
    
    positions = load_file(POSITIONS_FILE, [])
    
    # Calculate some stats
    total_positions = len(positions)
    total_pnl = portfolio.get("unrealized_pnl_usd", 0)
    total_pnl_pct = portfolio.get("unrealized_pnl_pct", 0)
    
    # Format status
    status = {
        "running": running,
        "total_positions": total_positions,
        "portfolio_value": portfolio.get("total_value", 20000.0),
        "unrealized_pnl": total_pnl,
        "unrealized_pnl_pct": total_pnl_pct,
        "last_update": datetime.now().isoformat()
    }
    
    return jsonify(status)

@bot_app.route('/api/positions')
def api_positions():
    """API endpoint for positions"""
    positions = load_file(POSITIONS_FILE, [])
    return jsonify(positions)

@bot_app.route('/api/prices')
def api_prices():
    """API endpoint for current prices"""
    return jsonify(current_prices)

@bot_app.route('/api/trades')
def api_trades():
    """API endpoint for trade history"""
    trades = load_file(TRADES_FILE, [])
    return jsonify(trades)

@bot_app.route('/bot')
def bot_dashboard():
    """Simple bot dashboard"""
    return render_template('bot_dashboard.html')

@bot_app.route('/')
def index():
    """Main bot page - redirect to bot dashboard"""
    return bot_dashboard()

def start_bot():
    """Start the trading bot"""
    global running
    
    # Check if already running
    if running:
        logger.warning("Bot is already running")
        return False
    
    # Set running flag
    running = True
    
    # Start price update thread
    price_thread = threading.Thread(target=update_prices)
    price_thread.daemon = True
    price_thread.start()
    
    # Start trading thread
    trading_thread = threading.Thread(target=simulated_trading)
    trading_thread.daemon = True
    trading_thread.start()
    
    logger.info("Bot started successfully")
    return True

def stop_bot():
    """Stop the trading bot"""
    global running
    
    # Check if already stopped
    if not running:
        logger.warning("Bot is already stopped")
        return False
    
    # Clear running flag
    running = False
    
    logger.info("Bot stopped successfully")
    return True

def main():
    """Main function"""
    # Start the bot
    start_bot()
    
    # Run the Flask app on a different port
    try:
        port = 8080
        logger.info(f"Starting bot server on port {port}...")
        bot_app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
    finally:
        # Make sure to stop the bot
        stop_bot()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())