#!/usr/bin/env python3
"""
Show Trading Status

This script displays the current state of the trading bot:
- Portfolio status
- Open positions
- Recent trades
- ML model status
"""
import os
import json
import logging
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories and files
DATA_DIR = "data"
MODEL_WEIGHTS_DIR = "model_weights"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"

def load_json_file(file_path):
    """Load a JSON file"""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return {}

def display_portfolio():
    """Display portfolio information"""
    portfolio = load_json_file(PORTFOLIO_PATH)
    
    if not portfolio:
        logger.warning("Portfolio information not available")
        return
    
    print("\n==== PORTFOLIO STATUS ====")
    print(f"Balance: ${portfolio.get('balance', 0):.2f}")
    print(f"Initial Balance: ${portfolio.get('initial_balance', 0):.2f}")
    
    # Calculate profit/loss
    initial_balance = portfolio.get('initial_balance', 0)
    current_balance = portfolio.get('balance', 0)
    if initial_balance > 0:
        profit_loss = current_balance - initial_balance
        profit_loss_pct = (profit_loss / initial_balance) * 100
        print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
    
    # Calculate unrealized profit/loss
    positions = load_json_file(POSITIONS_PATH)
    unrealized_pnl = 0
    
    for position_id, position in positions.items():
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', entry_price)
        size = position.get('size', 0)
        long = position.get('long', True)
        
        if long:
            position_pnl = (current_price - entry_price) * size
        else:
            position_pnl = (entry_price - current_price) * size
        
        unrealized_pnl += position_pnl
    
    print(f"Unrealized P/L: ${unrealized_pnl:.2f}")
    if current_balance > 0:
        unrealized_pnl_pct = (unrealized_pnl / current_balance) * 100
        print(f"Unrealized P/L: {unrealized_pnl_pct:.2f}%")

def display_positions():
    """Display open positions"""
    positions = load_json_file(POSITIONS_PATH)
    
    if not positions:
        print("\n==== OPEN POSITIONS ====")
        print("No open positions")
        return
    
    print("\n==== OPEN POSITIONS ====")
    print(f"Number of Open Positions: {len(positions)}")
    
    for position_id, position in positions.items():
        symbol = position.get('symbol', 'Unknown')
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', entry_price)
        size = position.get('size', 0)
        long = position.get('long', True)
        entry_time = position.get('entry_time', '')
        
        direction = "LONG" if long else "SHORT"
        
        if long:
            position_pnl = (current_price - entry_price) * size
            position_pnl_pct = ((current_price / entry_price) - 1) * 100
        else:
            position_pnl = (entry_price - current_price) * size
            position_pnl_pct = ((entry_price / current_price) - 1) * 100
        
        print(f"\nPosition ID: {position_id}")
        print(f"Symbol: {symbol}")
        print(f"Direction: {direction}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Size: {size}")
        print(f"Entry Time: {entry_time}")
        print(f"P/L: ${position_pnl:.2f} ({position_pnl_pct:.2f}%)")

def display_recent_trades():
    """Display recent trades"""
    trades = load_json_file(TRADES_PATH)
    
    if not trades:
        print("\n==== RECENT TRADES ====")
        print("No trades recorded")
        return
    
    # Convert trades to a list and sort by exit_time or entry_time
    trades_list = []
    for trade_id, trade in trades.items():
        trade['id'] = trade_id
        trades_list.append(trade)
    
    # Sort by exit_time or entry_time (if exit_time not available)
    trades_list.sort(
        key=lambda x: x.get('exit_time', x.get('entry_time', '')),
        reverse=True
    )
    
    print("\n==== RECENT TRADES ====")
    print(f"Number of Trades: {len(trades_list)}")
    
    # Display the 5 most recent trades
    recent_trades = trades_list[:5]
    for trade in recent_trades:
        symbol = trade.get('symbol', 'Unknown')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        size = trade.get('size', 0)
        long = trade.get('long', True)
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        profit_loss = trade.get('profit_loss', 0)
        
        direction = "LONG" if long else "SHORT"
        status = "CLOSED" if exit_price > 0 else "OPEN"
        
        print(f"\nTrade ID: {trade.get('id', 'Unknown')}")
        print(f"Symbol: {symbol}")
        print(f"Direction: {direction}")
        print(f"Status: {status}")
        print(f"Entry Price: ${entry_price:.2f}")
        
        if status == "CLOSED":
            print(f"Exit Price: ${exit_price:.2f}")
            print(f"P/L: ${profit_loss:.2f}")
            
            if entry_price > 0:
                pnl_pct = (profit_loss / (entry_price * size)) * 100
                print(f"P/L: {pnl_pct:.2f}%")
        
        print(f"Entry Time: {entry_time}")
        if exit_time:
            print(f"Exit Time: {exit_time}")

def display_ml_status():
    """Display ML model status summary"""
    ml_config = load_json_file(ML_CONFIG_PATH)
    
    if not ml_config:
        print("\n==== ML MODEL STATUS ====")
        print("ML configuration not available")
        return
    
    model_files = glob.glob(f"{MODEL_WEIGHTS_DIR}/*.h5")
    model_files_base = [os.path.basename(f) for f in model_files]
    
    print("\n==== ML MODEL STATUS ====")
    
    models = ml_config.get('models', {})
    print(f"Number of Configured Models: {len(models)}")
    
    for pair, model_info in models.items():
        model_type = model_info.get('model_type', 'Unknown')
        model_path = model_info.get('model_path', '')
        model_filename = os.path.basename(model_path)
        confidence_threshold = model_info.get('confidence_threshold', 0)
        min_leverage = model_info.get('min_leverage', 0)
        max_leverage = model_info.get('max_leverage', 0)
        
        status = "AVAILABLE" if model_filename in model_files_base else "MISSING"
        
        print(f"\nPair: {pair}")
        print(f"Model Type: {model_type}")
        print(f"Status: {status}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Leverage Range: {min_leverage}x - {max_leverage}x")

def check_trading_bot_status():
    """Check if trading bot is running"""
    try:
        # Check for .bot_pid.txt file
        if os.path.exists('.bot_pid.txt'):
            with open('.bot_pid.txt', 'r') as f:
                pid = f.read().strip()
            
            # Check if process is running
            os.kill(int(pid), 0)  # This will raise an exception if the process is not running
            return True
    except (FileNotFoundError, ProcessLookupError, ValueError, OSError):
        return False
    
    return False

def main():
    """Main function"""
    print("\n" + "=" * 50)
    print("KRAKEN TRADING BOT STATUS")
    print("=" * 50)
    
    # Check if trading bot is running
    bot_running = check_trading_bot_status()
    print(f"Trading Bot Status: {'RUNNING' if bot_running else 'STOPPED'}")
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display portfolio information
    display_portfolio()
    
    # Display open positions
    display_positions()
    
    # Display recent trades
    display_recent_trades()
    
    # Display ML model status
    display_ml_status()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error displaying trading status: {e}")
        raise
