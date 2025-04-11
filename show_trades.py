#!/usr/bin/env python3

import csv
import os
import datetime
import pandas as pd
from typing import List, Dict, Any

# Constants from config
INITIAL_PORTFOLIO_VALUE = 20000.0

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    return f"+{value:.2f}%" if value >= 0 else f"{value:.2f}%"

def load_trades_from_csv() -> List[Dict[str, Any]]:
    """Load trades from the CSV file"""
    trades = []
    
    try:
        if os.path.exists("trades.csv"):
            with open("trades.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert timestamp to datetime
                    if "time" in row and row["time"]:
                        try:
                            timestamp = float(row["time"])
                            row["datetime"] = datetime.datetime.fromtimestamp(timestamp)
                        except (ValueError, TypeError):
                            row["datetime"] = None
                    
                    # Convert numeric fields
                    for field in ["price", "volume", "cost", "fee"]:
                        if field in row and row[field]:
                            try:
                                row[field] = float(row[field])
                            except (ValueError, TypeError):
                                row[field] = 0.0
                    
                    trades.append(row)
    except Exception as e:
        print(f"Error loading trades: {e}")
        return []
    
    return trades

def pair_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair buy and sell trades to create complete trade records"""
    paired_trades = []
    buy_trades = {}
    
    # Sort trades by time
    sorted_trades = sorted(trades, key=lambda x: x.get("time", 0))
    
    for trade in sorted_trades:
        trade_type = trade.get("type", "")
        trade_pair = trade.get("pair", "")
        trade_volume = trade.get("volume", 0)
        
        if trade_type == "buy":
            # Store buy trade keyed by pair and volume for matching
            key = f"{trade_pair}_{trade_volume}"
            buy_trades[key] = trade
        elif trade_type == "sell" and trade_volume > 0:
            # Look for matching buy trade
            key = f"{trade_pair}_{trade_volume}"
            if key in buy_trades:
                buy_trade = buy_trades[key]
                
                # Create paired trade record
                entry_price = buy_trade.get("price", 0)
                exit_price = trade.get("price", 0)
                volume = trade_volume
                
                # Calculate P&L
                gross_pnl = (exit_price - entry_price) * volume
                fees = buy_trade.get("fee", 0) + trade.get("fee", 0)
                net_pnl = gross_pnl - fees
                
                # Calculate percentage gain/loss
                if entry_price > 0:
                    pct_gain = (exit_price / entry_price - 1) * 100
                else:
                    pct_gain = 0
                
                paired_trade = {
                    "entry_time": buy_trade.get("datetime", None),
                    "exit_time": trade.get("datetime", None),
                    "pair": trade_pair,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "volume": volume,
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "pct_gain": pct_gain,
                    "trade_type": "long",  # Assuming all are long trades for now
                    "win": net_pnl > 0
                }
                
                paired_trades.append(paired_trade)
                
                # Remove the used buy trade
                del buy_trades[key]
    
    # Add remaining open trades
    for key, buy_trade in buy_trades.items():
        paired_trades.append({
            "entry_time": buy_trade.get("datetime", None),
            "exit_time": None,  # Still open
            "pair": buy_trade.get("pair", ""),
            "entry_price": buy_trade.get("price", 0),
            "exit_price": None,  # Still open
            "volume": buy_trade.get("volume", 0),
            "gross_pnl": None,  # Still open
            "fees": buy_trade.get("fee", 0),
            "net_pnl": None,  # Still open
            "pct_gain": None,  # Still open
            "trade_type": "long",  # Assuming all are long trades for now
            "win": None,  # Still open
            "status": "open"
        })
    
    return paired_trades

def get_exit_reason(exit_time):
    """Get the reason for exiting the market from the logs"""
    try:
        with open("trade_notifications.log", "r") as f:
            lines = f.readlines()
        
        if exit_time is None:
            return "Position still open"
            
        # Convert exit_time to string format for comparison
        exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M")
        
        # Search for action around the exit time
        for line in lines:
            if exit_time_str in line and "【ACTION】" in line and "SELL" in line:
                # Extract reason from log line
                if "Trailing Stop" in line or "Volatility Stop" in line:
                    return "Trailing stop triggered"
                elif "Signal Reversal" in line:
                    return "Signal reversal"
                elif "Take Profit" in line:
                    return "Take profit target reached"
                elif "Market Closed" in line:
                    return "Market closed"
                else:
                    return "Sell signal generated"
                    
        return "Unknown (check logs)"
    except Exception as e:
        return f"Error finding exit reason: {e}"

def display_trades():
    """Display paired trade information"""
    trades = load_trades_from_csv()
    paired_trades = pair_trades(trades)
    
    print("\n" + "=" * 100)
    print("KRAKEN TRADING BOT - DETAILED TRADE HISTORY")
    print("=" * 100)
    
    if not paired_trades:
        print("No trades found in the records.")
        return
    
    # Print trade details
    for i, trade in enumerate(paired_trades):
        trade_status = trade.get("status", "closed")
        is_win = trade.get("win")
        
        print(f"\nTRADE #{i+1}: {'🟢 WIN' if is_win is True else '🔴 LOSS' if is_win is False else '⚪ OPEN'}")
        print(f"Pair: {trade.get('pair', 'N/A')}")
        print(f"Type: {trade.get('trade_type', 'N/A').upper()}")
        
        # Entry details
        entry_time = trade.get("entry_time")
        entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A"
        print(f"Entry Time: {entry_time_str}")
        print(f"Entry Price: {format_currency(trade.get('entry_price', 0))}")
        
        # Exit details
        if trade_status == "open":
            print("Exit Time: Still open")
            print("Exit Price: Still open")
            print("Exit Reason: N/A (position still open)")
        else:
            exit_time = trade.get("exit_time")
            exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S") if exit_time else "N/A"
            print(f"Exit Time: {exit_time_str}")
            print(f"Exit Price: {format_currency(trade.get('exit_price', 0))}")
            print(f"Exit Reason: {get_exit_reason(exit_time)}")
        
        # Performance details
        print(f"Quantity: {trade.get('volume', 0)} SOL")
        
        if trade_status != "open":
            gross_pnl = trade.get('gross_pnl', 0)
            fees = trade.get('fees', 0)
            net_pnl = trade.get('net_pnl', 0)
            pct_gain = trade.get('pct_gain', 0)
            
            pnl_style = "+" if net_pnl >= 0 else ""
            print(f"Gross P&L: {pnl_style}{format_currency(gross_pnl)}")
            print(f"Fees: {format_currency(fees)}")
            print(f"Net P&L: {pnl_style}{format_currency(net_pnl)} ({format_percentage(pct_gain)})")
        else:
            print("P&L: Position still open")
        
        # Trade condition
        trade_conditions = "Signal-based entry with trailing stop for risk management"
        print(f"Trade Conditions: {trade_conditions}")
        print("-" * 100)
    
    # Print summary
    closed_trades = [t for t in paired_trades if t.get("status") != "open"]
    winning_trades = [t for t in closed_trades if t.get("win") is True]
    
    print("\nTRADE SUMMARY")
    print(f"Total Trades: {len(paired_trades)}")
    print(f"Closed Trades: {len(closed_trades)}")
    print(f"Open Positions: {len(paired_trades) - len(closed_trades)}")
    
    if closed_trades:
        win_rate = len(winning_trades) / len(closed_trades) * 100
        print(f"Win Rate: {format_percentage(win_rate)}")
        
        total_pnl = sum(t.get("net_pnl", 0) for t in closed_trades)
        pnl_style = "+" if total_pnl >= 0 else ""
        pnl_percent = (total_pnl / INITIAL_PORTFOLIO_VALUE) * 100
        print(f"Total P&L: {pnl_style}{format_currency(total_pnl)} ({format_percentage(pnl_percent)})")
        
        avg_win = sum(t.get("net_pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        losing_trades = [t for t in closed_trades if t.get("win") is False]
        avg_loss = sum(t.get("net_pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"Average Win: {format_currency(avg_win)}")
        print(f"Average Loss: {format_currency(avg_loss)}")
        
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    display_trades()