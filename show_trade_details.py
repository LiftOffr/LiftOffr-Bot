#!/usr/bin/env python3
import os
import csv
import json
import datetime
from statistics import mean
from collections import defaultdict
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    return f"{value:.2f}%"

def parse_datetime(date_str):
    """Parse different datetime formats to datetime object"""
    if not date_str:
        return datetime.datetime.now()
    
    # Handle Unix timestamp
    if date_str.replace('.', '', 1).isdigit():
        return datetime.datetime.fromtimestamp(float(date_str))
    
    # Try different date formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If all formats fail, return current time
    logger.warning(f"Could not parse datetime: {date_str}")
    return datetime.datetime.now()

def get_trades_history():
    """Get trades from trades.csv file"""
    trades = []
    
    try:
        if os.path.exists("trades.csv"):
            with open("trades.csv", "r") as f:
                reader = csv.reader(f)
                headers = next(reader, [])  # Get headers or empty list
                
                if not headers:
                    logger.warning("CSV file has no headers")
                    return []
                
                # Create a mapping of column names to indices
                column_map = {h.lower(): i for i, h in enumerate(headers) if h}
                
                for row in reader:
                    if not row or len(row) < 3:  # Skip empty rows
                        continue
                    
                    trade = {}
                    
                    # Detect format type
                    if 'time' in column_map and row[column_map['time']].replace('.', '', 1).isdigit():
                        # Old format from Binance: time,type,pair,price,volume,cost,fee,txid
                        try:
                            timestamp = float(row[column_map['time']])
                            trade['timestamp'] = datetime.datetime.fromtimestamp(timestamp)
                            trade['type'] = row[column_map['type']].upper()
                            trade['pair'] = row[column_map['pair']]
                            trade['price'] = float(row[column_map['price']]) if row[column_map['price']] else 0
                            trade['volume'] = float(row[column_map['volume']]) if row[column_map['volume']] else 0
                            trade['strategy'] = 'DEMO'  # Mark historical trades as demo
                        except (ValueError, KeyError, IndexError) as e:
                            logger.error(f"Error parsing old format trade: {e}")
                            continue
                            
                    elif row[0].startswith('20') and '/' in row[1]:
                        # Newer custom format: date,pair,type,volume,price,empty,empty,strategy
                        try:
                            trade['timestamp'] = parse_datetime(row[0])
                            trade['pair'] = row[1]
                            trade['type'] = row[2].upper()
                            trade['volume'] = float(row[3]) if row[3] else 0
                            trade['price'] = float(row[4]) if row[4] else 0
                            
                            # Check if we have a strategy in the 8th column
                            if len(row) > 7 and row[7]:
                                strategy_value = row[7].lower().strip()
                                if 'arima' in strategy_value:
                                    trade['strategy'] = 'ARIMA'
                                elif 'adaptive' in strategy_value:
                                    trade['strategy'] = 'ADAPTIVE'
                                else:
                                    # Default based on trade type
                                    trade['strategy'] = 'ARIMA' if trade['type'] == 'SELL' else 'ADAPTIVE'
                            else:
                                # Default based on trade type
                                trade['strategy'] = 'ARIMA' if trade['type'] == 'SELL' else 'ADAPTIVE'
                        except (ValueError, IndexError) as e:
                            logger.error(f"Error parsing custom format trade: {e}")
                            continue
                    
                    else:
                        # Try to detect through column names
                        try:
                            # Get timestamp
                            if 'timestamp' in column_map:
                                trade['timestamp'] = parse_datetime(row[column_map['timestamp']])
                            elif 'time' in column_map:
                                trade['timestamp'] = parse_datetime(row[column_map['time']])
                            else:
                                trade['timestamp'] = datetime.datetime.now()
                            
                            # Get pair/symbol
                            if 'pair' in column_map:
                                trade['pair'] = row[column_map['pair']]
                            elif 'symbol' in column_map:
                                trade['pair'] = row[column_map['symbol']]
                            else:
                                trade['pair'] = 'SOL/USD'
                            
                            # Get trade type (buy/sell)
                            if 'type' in column_map:
                                trade['type'] = row[column_map['type']].upper()
                            elif 'side' in column_map:
                                trade['type'] = row[column_map['side']].upper()
                            else:
                                trade['type'] = 'BUY'  # Default to buy
                            
                            # Get volume/amount
                            if 'volume' in column_map:
                                trade['volume'] = float(row[column_map['volume']]) if row[column_map['volume']] else 0
                            elif 'amount' in column_map:
                                trade['volume'] = float(row[column_map['amount']]) if row[column_map['amount']] else 0
                            elif 'quantity' in column_map:
                                trade['volume'] = float(row[column_map['quantity']]) if row[column_map['quantity']] else 0
                            else:
                                trade['volume'] = 0
                            
                            # Get price
                            if 'price' in column_map:
                                trade['price'] = float(row[column_map['price']]) if row[column_map['price']] else 0
                            else:
                                trade['price'] = 0
                            
                            # Get strategy
                            if 'strategy' in column_map:
                                strategy_value = row[column_map['strategy']].lower() if row[column_map['strategy']] else ''
                                if 'arima' in strategy_value:
                                    trade['strategy'] = 'ARIMA'
                                elif 'adaptive' in strategy_value:
                                    trade['strategy'] = 'ADAPTIVE'
                                else:
                                    trade['strategy'] = 'DEMO'
                            else:
                                trade['strategy'] = 'DEMO'
                        except (ValueError, IndexError) as e:
                            logger.error(f"Error parsing generic format trade: {e}")
                            continue
                    
                    trades.append(trade)
                    
                # Sort trades by timestamp
                trades = sorted(trades, key=lambda x: x.get('timestamp', datetime.datetime.now()))
        
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        return []
    
    return trades

def pair_trades(trades):
    """
    Pair buy and sell trades to calculate P&L for each completed trade cycle
    Returns a list of paired trades with calculated P&L
    
    In a multi-strategy environment, strategies can interact with each other's positions:
    - ARIMA can sell positions opened by ADAPTIVE
    - Adaptive can sell positions opened by ARIMA
    """
    if not trades:
        return []
    
    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', datetime.datetime.now()))
    
    paired_trades = []
    open_long_positions = {}  # Track open long positions by pair
    open_short_positions = {}  # Track open short positions by pair
    
    for trade in sorted_trades:
        pair = trade.get('pair', 'SOL/USD')
        trade_type = trade.get('type', '').upper()
        strategy = trade.get('strategy', 'UNKNOWN')
        
        if trade_type == 'BUY':
            # Check if we have an open short position for this pair that can be closed
            if pair in open_short_positions and open_short_positions[pair]:
                # Sort by oldest first
                open_short_positions[pair].sort(key=lambda x: x.get('timestamp', datetime.datetime.now()))
                
                # Get the oldest short position
                open_trade = open_short_positions[pair].pop(0)
                
                # Calculate P&L
                entry_price = open_trade.get('price', 0)
                exit_price = trade.get('price', 0)
                volume = min(open_trade.get('volume', 0), trade.get('volume', 0))
                
                if entry_price > 0 and exit_price > 0:
                    # For a short position, profit when exit_price < entry_price
                    pnl = (entry_price - exit_price) * volume
                    pnl_percent = ((entry_price / exit_price) - 1) * 100
                    
                    # Use the exit trade's strategy for attribution
                    paired_trades.append({
                        'entry_trade': open_trade,
                        'exit_trade': trade,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'strategy': strategy
                    })
                
                # If we have no more short positions for this pair, clean up
                if not open_short_positions[pair]:
                    del open_short_positions[pair]
            else:
                # Opening a long position
                if pair not in open_long_positions:
                    open_long_positions[pair] = []
                open_long_positions[pair].append(trade)
        
        elif trade_type == 'SELL':
            # Check if we have an open long position for this pair that can be closed
            if pair in open_long_positions and open_long_positions[pair]:
                # Sort by oldest first
                open_long_positions[pair].sort(key=lambda x: x.get('timestamp', datetime.datetime.now()))
                
                # Get the oldest long position
                open_trade = open_long_positions[pair].pop(0)
                
                # Calculate P&L
                entry_price = open_trade.get('price', 0)
                exit_price = trade.get('price', 0)
                volume = min(open_trade.get('volume', 0), trade.get('volume', 0))
                
                if entry_price > 0 and exit_price > 0:
                    # For a long position, profit when exit_price > entry_price
                    pnl = (exit_price - entry_price) * volume
                    pnl_percent = ((exit_price / entry_price) - 1) * 100
                    
                    # Use the exit trade's strategy for attribution 
                    paired_trades.append({
                        'entry_trade': open_trade,
                        'exit_trade': trade,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'strategy': strategy
                    })
                
                # If we have no more long positions for this pair, clean up
                if not open_long_positions[pair]:
                    del open_long_positions[pair]
            else:
                # Opening a short position
                if pair not in open_short_positions:
                    open_short_positions[pair] = []
                open_short_positions[pair].append(trade)
    
    return paired_trades

def display_trade_details():
    """Display detailed information about each trade"""
    trades = get_trades_history()
    paired_trades = pair_trades(trades)
    
    if not paired_trades:
        print("No completed trade cycles found.")
        return
    
    print("=" * 80)
    print("DETAILED TRADE ANALYSIS")
    print("=" * 80)
    
    total_pnl = 0
    
    for i, trade in enumerate(paired_trades, 1):
        # Extract trade details
        entry_trade = trade['entry_trade']
        exit_trade = trade['exit_trade']
        pnl = trade['pnl']
        pnl_percent = trade['pnl_percent']
        strategy = trade['strategy']
        
        # Update total PnL
        total_pnl += pnl
        
        # Format trade information
        entry_time = entry_trade['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        exit_time = exit_trade['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        entry_type = entry_trade['type']
        exit_type = exit_trade['type']
        pair = entry_trade['pair']
        volume = min(entry_trade['volume'], exit_trade['volume'])
        entry_price = entry_trade['price']
        exit_price = exit_trade['price']
        entry_strategy = entry_trade.get('strategy', 'UNKNOWN')
        exit_strategy = exit_trade.get('strategy', 'UNKNOWN')
        
        # Print trade details
        print(f"TRADE #{i}")
        print(f"Pair: {pair}")
        print(f"Entry: {entry_type} at {format_currency(entry_price)} on {entry_time} - Strategy: {entry_strategy}")
        print(f"Exit: {exit_type} at {format_currency(exit_price)} on {exit_time} - Strategy: {exit_strategy}")
        print(f"Volume: {volume:.2f}")
        print(f"P&L: {format_currency(pnl)} ({format_percentage(pnl_percent)})")
        print(f"Attribution: {strategy}")
        print("-" * 80)
    
    # Print summary
    print("SUMMARY")
    print(f"Total Trades: {len(paired_trades)}")
    print(f"Total P&L: {format_currency(total_pnl)} ({format_percentage(total_pnl/20000*100)})")
    print("=" * 80)

if __name__ == "__main__":
    display_trade_details()