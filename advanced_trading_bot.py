#!/usr/bin/env python3
"""
Advanced Trading Bot with Risk Management and Liquidation Handling

This script runs a trading bot that:
1. Uses WebSockets for real-time price updates
2. Implements proper risk management for leverage calculation
3. Handles liquidations correctly
4. Updates positions and PnL calculations accurately
"""
import os
import sys
import json
import time
import random
import logging
from datetime import datetime
import threading

# Import our WebSocket client and risk management
from kraken_websocket_client import KrakenWebSocketClient
from risk_management import (
    check_liquidation, 
    calculate_risk_adjusted_leverage,
    calculate_position_size, 
    prepare_trade_entry
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
CLOSED_TRADES_FILE = f"{DATA_DIR}/sandbox_closed_trades.json"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global WebSocket client
ws_client = None

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

def price_update_handler(pair, price):
    """
    Handle price updates from WebSocket.
    This function is called every time a new price update is received.
    """
    logger.debug(f"WebSocket price update: {pair} = ${price}")
    
    # Get current positions
    positions = load_file(POSITIONS_FILE, [])
    if not positions:
        return
    
    # Check if any position uses this pair
    position_updated = False
    updated_positions = []
    
    for position in positions:
        if position.get('pair') == pair:
            # Update the position with the new price
            updated_position = update_position_with_price(position, price)
            updated_positions.append(updated_position)
            position_updated = True
        else:
            updated_positions.append(position)
    
    # If we updated any positions, save them and update the portfolio
    if position_updated:
        save_file(POSITIONS_FILE, updated_positions)
        update_portfolio(updated_positions)

def update_position_with_price(position, current_price):
    """
    Update a single position with the current price, handling liquidations.
    
    Args:
        position (dict): Position information
        current_price (float): Current price
        
    Returns:
        dict: Updated position
    """
    # Make a copy of the position
    updated_position = position.copy()
    
    # Get position details
    entry_price = position.get('entry_price', 0)
    position_size = position.get('position_size', 0)
    direction = position.get('direction', '').lower()
    leverage = position.get('leverage', 1.0)
    pair = position.get('pair')
    
    # Ensure position_size is positive
    if position_size < 0:
        position_size = abs(position_size)
        updated_position['position_size'] = position_size
    
    # Update current price
    updated_position['current_price'] = current_price
    
    # Check for liquidation
    is_liquidated, liquidation_price, liquidation_pnl = check_liquidation(position, current_price)
    
    if is_liquidated:
        # Handle liquidation
        logger.warning(f"LIQUIDATION: {pair} {direction} position liquidated at ${current_price}")
        
        # Close the position
        close_position(position, current_price, "liquidation", liquidation_pnl)
        
        # Create a copy with liquidation info for the client
        updated_position['status'] = "liquidated"
        updated_position['exit_price'] = current_price
        updated_position['exit_time'] = datetime.now().isoformat()
        updated_position['unrealized_pnl_amount'] = liquidation_pnl
        updated_position['unrealized_pnl_pct'] = -95.0  # Approximately 95% loss on liquidation
        updated_position['current_value'] = round(position_size + liquidation_pnl, 2)
        
        return updated_position
    
    # Calculate unrealized PnL if not liquidated
    if entry_price and position_size and entry_price > 0:
        # For long positions:
        # - Price goes up: profit
        # - Price goes down: loss
        if direction == 'long':
            price_change_pct = ((current_price / entry_price) - 1) * 100
            pnl_pct = price_change_pct * leverage
            pnl_amount = (position_size * price_change_pct / 100) * leverage
        # For short positions:
        # - Price goes up: loss
        # - Price goes down: profit
        else:  # short
            price_change_pct = ((entry_price / current_price) - 1) * 100
            pnl_pct = price_change_pct * leverage
            pnl_amount = (position_size * price_change_pct / 100) * leverage
        
        updated_position['unrealized_pnl_amount'] = round(pnl_amount, 2)
        updated_position['unrealized_pnl_pct'] = round(pnl_pct, 2)
        updated_position['current_value'] = round(position_size + pnl_amount, 2)
        
        # Update trade record with the same PnL information
        trade_id = position.get('open_trade_id')
        if trade_id:
            update_trade_with_pnl(trade_id, current_price, pnl_amount, pnl_pct)
        
        logger.info(f"Updated {pair} position: Price=${current_price}, PnL=${pnl_amount:.2f} ({pnl_pct:.2f}%)")
    
    return updated_position

def close_position(position, exit_price, exit_reason="manual", pnl=None):
    """
    Close a position and record it in closed trades.
    
    Args:
        position (dict): Position to close
        exit_price (float): Exit price
        exit_reason (str): Reason for closing (manual, take_profit, stop_loss, liquidation)
        pnl (float): Optional PnL value to use
    """
    # Get position details
    pair = position.get('pair')
    entry_price = position.get('entry_price')
    position_size = position.get('position_size')
    direction = position.get('direction')
    leverage = position.get('leverage')
    entry_time = position.get('entry_time')
    trade_id = position.get('open_trade_id')
    
    # Calculate PnL if not provided
    if pnl is None:
        if direction == 'long':
            price_change_pct = ((exit_price / entry_price) - 1) * 100
            pnl_pct = price_change_pct * leverage
            pnl = (position_size * price_change_pct / 100) * leverage
        else:  # short
            price_change_pct = ((entry_price / exit_price) - 1) * 100
            pnl_pct = price_change_pct * leverage
            pnl = (position_size * price_change_pct / 100) * leverage
    else:
        # Calculate PnL percentage from absolute PnL
        pnl_pct = (pnl / position_size) * 100
    
    # Create closed trade record
    closed_trade = {
        "id": trade_id,
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "position_size": position_size,
        "leverage": leverage,
        "entry_time": entry_time,
        "exit_time": datetime.now().isoformat(),
        "exit_reason": exit_reason,
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "duration_seconds": (datetime.now() - datetime.fromisoformat(entry_time)).total_seconds()
    }
    
    # Add to closed trades
    closed_trades = load_file(CLOSED_TRADES_FILE, [])
    closed_trades.append(closed_trade)
    save_file(CLOSED_TRADES_FILE, closed_trades)
    
    # Update original trade record
    update_trade_status(trade_id, "closed", exit_price, pnl, pnl_pct, exit_reason)
    
    # Update portfolio balance
    update_portfolio_balance(pnl)
    
    logger.info(f"Closed {direction} position for {pair} at ${exit_price}, PnL: ${pnl:.2f} ({pnl_pct:.2f}%), Reason: {exit_reason}")

def update_trade_status(trade_id, status, exit_price=None, pnl=None, pnl_pct=None, exit_reason=None):
    """Update trade status in trades file"""
    trades = load_file(TRADES_FILE, [])
    
    for trade in trades:
        if trade.get('id') == trade_id:
            trade['status'] = status
            if exit_price:
                trade['exit_price'] = exit_price
            if pnl is not None:
                trade['pnl'] = round(pnl, 2)
            if pnl_pct is not None:
                trade['pnl_pct'] = round(pnl_pct, 2)
            if exit_reason:
                trade['exit_reason'] = exit_reason
            trade['exit_time'] = datetime.now().isoformat()
            break
    
    save_file(TRADES_FILE, trades)

def update_portfolio_balance(pnl_amount):
    """Update portfolio balance with realized PnL"""
    portfolio = load_file(PORTFOLIO_FILE, {"initial_capital": 20000.0, "balance": 20000.0})
    
    # Update balance with realized PnL
    current_balance = portfolio.get('balance', 20000.0)
    new_balance = current_balance + pnl_amount
    
    # Ensure balance doesn't go negative
    if new_balance <= 0:
        new_balance = 100.0  # Reset to small amount if wiped out
        logger.warning(f"Portfolio balance reduced to ${new_balance} after ${pnl_amount} loss")
    
    portfolio['balance'] = round(new_balance, 2)
    portfolio['updated_at'] = datetime.now().isoformat()
    
    save_file(PORTFOLIO_FILE, portfolio)
    
    logger.info(f"Updated portfolio balance: ${portfolio['balance']} after ${pnl_amount:.2f} PnL")

def update_trade_with_pnl(trade_id, current_price, pnl_amount, pnl_pct):
    """Update the corresponding trade record with PnL information"""
    trades = load_file(TRADES_FILE, [])
    
    updated = False
    for trade in trades:
        if trade.get('id') == trade_id and trade.get('status') == 'open':
            # Update PnL
            trade['pnl'] = round(pnl_amount, 2)
            trade['pnl_pct'] = round(pnl_pct, 2)
            trade['current_price'] = current_price
            updated = True
    
    # Save updated trades if any were changed
    if updated:
        save_file(TRADES_FILE, trades)

def update_portfolio(positions):
    """Update portfolio with current position values"""
    portfolio = load_file(PORTFOLIO_FILE, {"initial_capital": 20000.0, "balance": 20000.0, "equity": 20000.0})
    
    # Calculate total unrealized PnL
    total_pnl = 0
    active_positions = []
    
    for position in positions:
        # Only include active positions in calculations
        if position.get('status') != 'liquidated':
            active_positions.append(position)
            if "unrealized_pnl_amount" in position:
                total_pnl += position.get("unrealized_pnl_amount", 0)
    
    # Update portfolio metrics
    initial_balance = portfolio.get("initial_capital", 20000.0)
    current_balance = portfolio.get("balance", initial_balance)
    
    # Fix negative balance values if present
    if current_balance < 0:
        current_balance = initial_balance
        portfolio["balance"] = current_balance
    
    # Set unrealized PnL values
    unrealized_pnl = total_pnl  # This is the total unrealized PnL in USD
    unrealized_pnl_pct = round((unrealized_pnl / current_balance) * 100, 2) if current_balance > 0 else 0
    
    # Update the portfolio fields
    portfolio["unrealized_pnl"] = unrealized_pnl
    portfolio["unrealized_pnl_usd"] = round(unrealized_pnl, 2)
    portfolio["unrealized_pnl_pct"] = unrealized_pnl_pct
    
    # Calculate total value and equity (balance + unrealized PnL)
    total_value = current_balance + unrealized_pnl
    portfolio["equity"] = round(total_value, 2)
    portfolio["total_value"] = round(total_value, 2)
    
    # Calculate total PnL (from initial capital)
    portfolio["total_pnl"] = round(total_value - initial_balance, 2)
    portfolio["total_pnl_pct"] = round(((total_value / initial_balance) - 1) * 100, 2)
    
    # Position metrics
    portfolio["open_positions_count"] = len(active_positions)
    
    # Calculate total margin used and available capital
    total_margin_used = sum(p.get("position_size", 0) for p in active_positions)
    available_capital = current_balance - total_margin_used
    margin_used_pct = (total_margin_used / current_balance) * 100 if current_balance > 0 else 0
    
    portfolio["available_capital"] = round(available_capital, 2)
    portfolio["margin_used_pct"] = round(margin_used_pct, 2)
    portfolio["available_margin"] = round(available_capital, 2)
    
    # Update timestamp
    portfolio["updated_at"] = datetime.now().isoformat()
    
    # Save updated portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    logger.info(f"Updated portfolio: Balance=${current_balance}, Equity=${portfolio['equity']}, Unrealized PnL=${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)")
    
    # Update portfolio history
    update_portfolio_history(portfolio)
    
    return portfolio

def update_portfolio_history(portfolio):
    """Add a new data point to portfolio history"""
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Add current equity value as a new data point
    now = datetime.now().isoformat()
    
    if len(history) > 0:
        # Only add a new point if it's been at least 5 minutes since the last update
        last_timestamp = history[-1].get("timestamp", "")
        if last_timestamp:
            last_time = datetime.fromisoformat(last_timestamp)
            time_diff = (datetime.now() - last_time).total_seconds() / 60  # in minutes
            
            if time_diff < 5:
                return  # Skip if less than 5 minutes
    
    new_point = {
        "timestamp": now,
        "portfolio_value": portfolio.get("equity", 20000.0)
    }
    
    history.append(new_point)
    
    # Keep only the last 1000 data points to avoid file growing too large
    if len(history) > 1000:
        history = history[-1000:]
    
    save_file(PORTFOLIO_HISTORY_FILE, history)
    logger.info(f"Added portfolio history point: {new_point['portfolio_value']:.2f} at {now}")

def get_all_pairs():
    """Get all trading pairs from positions"""
    positions = load_file(POSITIONS_FILE, [])
    pairs = set()
    
    for position in positions:
        pair = position.get('pair')
        if pair:
            pairs.add(pair)
    
    return list(pairs)

def initialize_websocket():
    """Initialize the WebSocket client"""
    global ws_client
    
    # Get all trading pairs from positions
    pairs = get_all_pairs()
    if not pairs:
        logger.warning("No trading pairs found in positions")
        return False
    
    logger.info(f"Initializing WebSocket for pairs: {pairs}")
    
    # Create WebSocket client
    ws_client = KrakenWebSocketClient(pairs=pairs, callback=price_update_handler)
    
    # Connect to WebSocket
    if ws_client.connect():
        logger.info("Connected to Kraken WebSocket API")
        return True
    else:
        logger.error("Failed to connect to Kraken WebSocket API")
        return False

def main():
    """Main function to run the trading bot"""
    logger.info("=" * 50)
    logger.info("Starting advanced trading bot with risk management and liquidation handling")
    logger.info("=" * 50)
    
    try:
        # Initialize WebSocket client
        if not initialize_websocket():
            logger.error("Failed to initialize WebSocket client, falling back to REST API")
            # Continue with REST API as fallback
        
        # Main loop - even if WebSocket fails, we'll use REST API as fallback
        while True:
            # If WebSocket is not connected or not initialized, try to initialize it
            if ws_client is None or not ws_client.connected:
                initialize_websocket()
            
            # Load latest positions
            positions = load_file(POSITIONS_FILE, [])
            
            # If WebSocket is not working, update prices manually
            if ws_client is None or not ws_client.connected:
                logger.warning("WebSocket not connected, updating prices via REST API")
                # Fall back to REST API
                # This part is handled by simple_trading_bot.py
            
            # Update portfolio with current position values
            update_portfolio(positions)
            
            # Wait before next update (30-60 seconds)
            sleep_time = random.randint(30, 60)
            logger.info(f"Waiting {sleep_time} seconds until next update...")
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in trading bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ws_client:
            ws_client.disconnect()

if __name__ == "__main__":
    main()