#!/usr/bin/env python3
"""
Command-line portfolio summary tool
"""
import json
import os
import sys
from datetime import datetime

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default

def format_currency(amount):
    """Format currency with 2 decimal places"""
    return f"${amount:,.2f}"

def format_percentage(percentage):
    """Format percentage with 2 decimal places"""
    return f"{percentage:.2f}%"

def print_separator(char="=", length=80):
    """Print a separator line"""
    print(char * length)

def print_section_header(title):
    """Print a section header"""
    print_separator("-", 80)
    print(f"{title.upper()}")
    print_separator("-", 80)

def summarize_portfolio():
    """Summarize portfolio information"""
    # Load data
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    positions = load_file(POSITIONS_FILE, [])
    trades = load_file(TRADES_FILE, [])
    
    # Calculate metrics
    initial_capital = 20000.0
    current_balance = portfolio.get("balance", initial_capital)
    total_pnl = current_balance - initial_capital
    total_pnl_pct = ((current_balance / initial_capital) - 1) * 100
    
    # Calculate unrealized P&L
    unrealized_pnl = 0.0
    if positions:
        for position in positions:
            if "unrealized_pnl" in position:
                unrealized_pnl += position["unrealized_pnl"]
    
    # Print summary
    print_separator()
    print(f"PORTFOLIO SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    print(f"Starting Capital: {format_currency(initial_capital)}")
    print(f"Current Balance: {format_currency(current_balance)}")
    print(f"Total P&L: {format_currency(total_pnl)} ({format_percentage(total_pnl_pct)})")
    print(f"Unrealized P&L: {format_currency(unrealized_pnl)}")
    print(f"Open Positions: {len(positions)}")
    print(f"Total Completed Trades: {len(trades)}")
    print()
    
    # Print positions
    if positions:
        print_section_header("Open Positions")
        for i, position in enumerate(positions, 1):
            print(f"{i}. {position['pair']} - {position['direction']} @ {position['entry_price']:.4f}")
            print(f"   Size: {position['size']:.2f}, Leverage: {position['leverage']:.1f}x")
            print(f"   Unrealized P&L: {format_currency(position['unrealized_pnl'])} ({position['unrealized_pnl_pct']:.2f}%)")
            print(f"   Duration: {position['duration']}, Strategy: {position['strategy']}")
            print()
    
    # Print recent trades
    if trades:
        print_section_header("Recent Trades (Last 5)")
        recent_trades = trades[-5:] if len(trades) >= 5 else trades
        for i, trade in enumerate(reversed(recent_trades), 1):
            direction = trade.get("direction", "Unknown")
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            timestamp = trade.get("timestamp", "Unknown")
            
            print(f"{i}. {trade.get('pair', 'Unknown')} - {direction} @ {entry_price:.4f}")
            if exit_price > 0:
                pnl = (exit_price - entry_price) * trade.get("size", 0)
                print(f"   Exit: {exit_price:.4f}, P&L: {format_currency(pnl)}")
            print(f"   Time: {timestamp}, Strategy: {trade.get('strategy', 'Unknown')}")
            print()

if __name__ == "__main__":
    summarize_portfolio()