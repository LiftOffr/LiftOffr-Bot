#!/usr/bin/env python3
import os
import csv
import json
import time
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    try:
        return f"{float(value):+.2f}%"
    except (ValueError, TypeError):
        return "+0.00%"

def get_current_price(trading_pair="SOL/USD"):
    """Get current price from the trading bot logs"""
    try:
        # Try to find price in trade_notifications.log
        current_price = 0
        
        if os.path.exists("trade_notifications.log"):
            with open("trade_notifications.log", "r") as f:
                for line in reversed(f.readlines()):
                    if "Current Price" in line and trading_pair in line:
                        price_str = line.split("$")[-1].strip()
                        try:
                            current_price = float(price_str)
                            break
                        except ValueError:
                            continue
        
        # If price wasn't found in logs, try to find in stdout logs
        if current_price == 0:
            if os.path.exists("nohup.out"):
                with open("nohup.out", "r") as f:
                    for line in reversed(f.readlines()):
                        if "TICKER" in line and trading_pair in line:
                            try:
                                current_price = float(line.split("$")[1].split()[0])
                                break
                            except (ValueError, IndexError):
                                continue
        
        return current_price
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        return 0

def get_current_position():
    """Get current position information"""
    position_info = {
        'type': None,
        'entry_price': 0,
        'current_price': 0,
        'size': 0,
        'unrealized_pnl': 0,
        'unrealized_pnl_percent': 0
    }
    
    try:
        # Try to get position from logs
        current_price = get_current_price()
        position_info['current_price'] = current_price
        
        # Check log files for position information
        if os.path.exists("trade_notifications.log"):
            with open("trade_notifications.log", "r") as f:
                lines = f.readlines()
                
                # Find the most recent position information
                for line in reversed(lines):
                    if "Current Position:" in line:
                        # Parse position type
                        if "LONG" in line:
                            position_info['type'] = "LONG"
                            break
                        elif "SHORT" in line:
                            position_info['type'] = "SHORT"
                            break
                        elif "No Position" in line or "NO POSITION" in line:
                            position_info['type'] = None
                            break
                
                # Look for entry price in recent logs
                if position_info['type']:
                    for line in reversed(lines):
                        if "Entry Price:" in line or "entry price:" in line:
                            try:
                                price_str = line.split("$")[1].split()[0].strip()
                                position_info['entry_price'] = float(price_str)
                                break
                            except (ValueError, IndexError):
                                continue
                
                # Look for position size in recent logs
                if position_info['type']:
                    for line in reversed(lines):
                        if "Position Size:" in line or "size:" in line:
                            try:
                                size_str = line.split("Size:")[1].split()[0].strip()
                                position_info['size'] = float(size_str)
                                break
                            except (ValueError, IndexError):
                                continue
        
        # Calculate unrealized P&L if we have a position
        if position_info['type'] and position_info['entry_price'] > 0 and position_info['size'] > 0:
            if position_info['type'] == "LONG":
                position_info['unrealized_pnl'] = (current_price - position_info['entry_price']) * position_info['size']
                if position_info['entry_price'] > 0:
                    position_info['unrealized_pnl_percent'] = ((current_price / position_info['entry_price']) - 1) * 100
            elif position_info['type'] == "SHORT":
                position_info['unrealized_pnl'] = (position_info['entry_price'] - current_price) * position_info['size']
                if current_price > 0:
                    position_info['unrealized_pnl_percent'] = ((position_info['entry_price'] / current_price) - 1) * 100
    
    except Exception as e:
        logger.error(f"Error getting current position: {e}")
    
    return position_info

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
    
    # Default to current time if parsing fails
    return datetime.datetime.now()

def get_trades_history():
    """Get recent trade history - handles different CSV formats"""
    trades = []
    
    try:
        # Check if trades.csv exists
        if not os.path.exists('trades.csv'):
            return []
        
        with open('trades.csv', 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)  # Skip header row
            
            if not headers:
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
                        
                        # Get order type
                        if 'type' in column_map:
                            trade['type'] = row[column_map['type']].upper()
                        elif 'order_type' in column_map:
                            trade['type'] = row[column_map['order_type']].upper()
                        else:
                            trade['type'] = 'UNKNOWN'
                        
                        # Get price
                        if 'price' in column_map:
                            trade['price'] = float(row[column_map['price']]) if row[column_map['price']] else 0
                        else:
                            trade['price'] = 0
                        
                        # Get volume/quantity
                        if 'volume' in column_map:
                            trade['volume'] = float(row[column_map['volume']]) if row[column_map['volume']] else 0
                        elif 'quantity' in column_map:
                            trade['volume'] = float(row[column_map['quantity']]) if row[column_map['quantity']] else 0
                        else:
                            trade['volume'] = 0
                        
                        # Try to determine strategy
                        if 'strategy' in column_map and row[column_map['strategy']]:
                            strategy_value = row[column_map['strategy']].lower()
                            if 'arima' in strategy_value:
                                trade['strategy'] = 'ARIMA'
                            elif 'adaptive' in strategy_value:
                                trade['strategy'] = 'ADAPTIVE'
                            else:
                                trade['strategy'] = 'DEMO'
                        else:
                            # Default strategy based on trade type
                            trade['strategy'] = 'ARIMA' if trade['type'] == 'SELL' else 'ADAPTIVE'
                            
                    except (ValueError, KeyError, IndexError) as e:
                        logger.error(f"Error parsing generic format trade: {e}")
                        continue
                
                # Add the trade to our list if it has required fields
                if 'timestamp' in trade and 'type' in trade and 'price' in trade and 'volume' in trade:
                    trades.append(trade)
        
        return trades
    except Exception as e:
        logger.error(f"Error reading trades: {e}")
        return []

def pair_trades(trades):
    """
    Pair buy and sell trades to calculate P&L for each completed trade cycle
    Returns a list of paired trades with calculated P&L
    """
    if not trades:
        return []
    
    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', datetime.datetime.now()))
    
    paired_trades = []
    open_positions = {}  # Track open positions by pair and strategy
    
    for trade in sorted_trades:
        pair = trade.get('pair', 'SOL/USD')
        trade_type = trade.get('type', '').upper()
        strategy = trade.get('strategy', 'UNKNOWN')
        position_key = f"{pair}_{strategy}"
        
        if trade_type == 'BUY':
            # Opening a long position or closing a short position
            if position_key in open_positions and open_positions[position_key].get('type') == 'SELL':
                # Closing a short position - pair with the open position
                open_trade = open_positions[position_key]
                
                # Calculate P&L
                entry_price = open_trade.get('price', 0)
                exit_price = trade.get('price', 0)
                volume = min(open_trade.get('volume', 0), trade.get('volume', 0))
                
                if entry_price > 0 and exit_price > 0:
                    # For a short position, profit when exit_price < entry_price
                    pnl = (entry_price - exit_price) * volume
                    pnl_percent = ((entry_price / exit_price) - 1) * 100
                    
                    paired_trades.append({
                        'entry_trade': open_trade,
                        'exit_trade': trade,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'strategy': strategy
                    })
                
                # Remove the position from open positions
                del open_positions[position_key]
            else:
                # Opening a long position
                open_positions[position_key] = trade
        
        elif trade_type == 'SELL':
            # Opening a short position or closing a long position
            if position_key in open_positions and open_positions[position_key].get('type') == 'BUY':
                # Closing a long position - pair with the open position
                open_trade = open_positions[position_key]
                
                # Calculate P&L
                entry_price = open_trade.get('price', 0)
                exit_price = trade.get('price', 0)
                volume = min(open_trade.get('volume', 0), trade.get('volume', 0))
                
                if entry_price > 0 and exit_price > 0:
                    # For a long position, profit when exit_price > entry_price
                    pnl = (exit_price - entry_price) * volume
                    pnl_percent = ((exit_price / entry_price) - 1) * 100
                    
                    paired_trades.append({
                        'entry_trade': open_trade,
                        'exit_trade': trade,
                        'pnl': pnl,
                        'pnl_percent': pnl_percent,
                        'strategy': strategy
                    })
                
                # Remove the position from open positions
                del open_positions[position_key]
            else:
                # Opening a short position
                open_positions[position_key] = trade
    
    return paired_trades

def calculate_trade_metrics(trades):
    """
    Calculate comprehensive trade metrics including strategy-specific performance
    """
    if not trades:
        return {
            'total_trades': 0,
            'profitable_trades': 0,
            'profitable_percent': 0,
            'losing_trades': 0,
            'losing_percent': 0,
            'total_pnl': 0.0,
            'total_pnl_percent': 0.0,
            'strategies': {}
        }
    
    # Pair trades to calculate P&L for completed trade cycles
    paired_trades = pair_trades(trades)
    
    metrics = {
        'total_trades': len(paired_trades),
        'profitable_trades': 0,
        'profitable_percent': 0,
        'losing_trades': 0,
        'losing_percent': 0,
        'total_pnl': 0.0,
        'total_pnl_percent': 0.0,
        'strategies': {
            'ADAPTIVE': {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            },
            'ARIMA': {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            },
            'DEMO': {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            }
        }
    }
    
    # Calculate metrics for each paired trade
    for trade in paired_trades:
        pnl = trade.get('pnl', 0)
        strategy = trade.get('strategy', 'UNKNOWN').upper()
        
        # Skip DEMO trades if we have real trades
        if strategy == 'DEMO' and (metrics['strategies']['ADAPTIVE']['total_trades'] > 0 or 
                                 metrics['strategies']['ARIMA']['total_trades'] > 0):
            continue
        
        # Update overall metrics
        metrics['total_pnl'] += pnl
        
        if pnl > 0:
            metrics['profitable_trades'] += 1
        else:
            metrics['losing_trades'] += 1
        
        # Update strategy-specific metrics
        if strategy not in metrics['strategies']:
            metrics['strategies'][strategy] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0
            }
        
        metrics['strategies'][strategy]['total_trades'] += 1
        metrics['strategies'][strategy]['total_pnl'] += pnl
        
        if pnl > 0:
            metrics['strategies'][strategy]['profitable_trades'] += 1
        else:
            metrics['strategies'][strategy]['losing_trades'] += 1
    
    # Calculate percentages
    if metrics['total_trades'] > 0:
        metrics['profitable_percent'] = (metrics['profitable_trades'] / metrics['total_trades']) * 100
        metrics['losing_percent'] = (metrics['losing_trades'] / metrics['total_trades']) * 100
    
    # Calculate strategy percentages
    for strategy in metrics['strategies']:
        if metrics['strategies'][strategy]['total_trades'] > 0:
            profitable = metrics['strategies'][strategy]['profitable_trades']
            total = metrics['strategies'][strategy]['total_trades']
            metrics['strategies'][strategy]['profitable_percent'] = (profitable / total) * 100
    
    # Calculate total P&L percent based on initial portfolio value of $20000
    metrics['total_pnl_percent'] = (metrics['total_pnl'] / 20000) * 100
    
    return metrics

def get_portfolio_metrics():
    """Calculate comprehensive portfolio metrics"""
    metrics = {}
    
    try:
        # Get all trades
        trades = get_trades_history()
        trade_metrics = calculate_trade_metrics(trades)
        
        # Get current position
        position = get_current_position()
        
        # Calculate portfolio metrics
        initial_value = 20000  # Default initial portfolio value
        current_value = initial_value + trade_metrics['total_pnl'] + position['unrealized_pnl']
        
        metrics = {
            'initial_value': initial_value,
            'current_value': current_value,
            'closed_trades_pnl': trade_metrics['total_pnl'],
            'closed_trades_pnl_percent': trade_metrics['total_pnl_percent'],
            'current_position_pnl': position['unrealized_pnl'],
            'current_position_pnl_percent': position['unrealized_pnl_percent'],
            'total_pnl': trade_metrics['total_pnl'] + position['unrealized_pnl'],
            'total_pnl_percent': ((current_value / initial_value) - 1) * 100,
            'trade_metrics': trade_metrics,
            'current_position': position
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        metrics = {
            'initial_value': 20000,
            'current_value': 20000,
            'closed_trades_pnl': 0,
            'closed_trades_pnl_percent': 0,
            'current_position_pnl': 0,
            'current_position_pnl_percent': 0,
            'total_pnl': 0,
            'total_pnl_percent': 0,
            'trade_metrics': {
                'total_trades': 0,
                'profitable_trades': 0,
                'profitable_percent': 0,
                'losing_trades': 0,
                'losing_percent': 0,
                'strategies': {}
            },
            'current_position': {
                'type': None,
                'entry_price': 0,
                'current_price': 0,
                'size': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_percent': 0
            }
        }
    
    return metrics

def generate_pnl_chart(trades):
    """
    Generate an ASCII chart representing the P&L over time
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        str: ASCII chart showing P&L over time
    """
    if not trades:
        return "No trades to display"
    
    # Pair trades to calculate P&L
    paired_trades = pair_trades(trades)
    
    if not paired_trades:
        return "No completed trade cycles to display"
    
    # Extract dates and P&L values
    dates = [t['exit_trade']['timestamp'] for t in paired_trades]
    pnl_values = [t['pnl'] for t in paired_trades]
    strategies = [t['strategy'] for t in paired_trades]
    
    # Calculate cumulative P&L
    cumulative_pnl = []
    running_total = 0
    for pnl in pnl_values:
        running_total += pnl
        cumulative_pnl.append(running_total)
    
    # Determine chart height and width
    width = min(80, len(cumulative_pnl) + 5)  # +5 for padding
    height = 15
    
    if not cumulative_pnl:
        return "No P&L data to display"
    
    # Calculate min and max for scaling
    min_pnl = min(0, min(cumulative_pnl))  # Include 0 in range
    max_pnl = max(0, max(cumulative_pnl))  # Include 0 in range
    
    # Prevent division by zero
    pnl_range = max_pnl - min_pnl
    if pnl_range == 0:
        pnl_range = 1
    
    # Scale points to fit chart height
    scaled_points = [int((pnl - min_pnl) / pnl_range * (height-1)) for pnl in cumulative_pnl]
    
    # Create empty chart
    chart = [' ' * width for _ in range(height)]
    
    # Draw horizontal axis (zero line)
    zero_line = int((0 - min_pnl) / pnl_range * (height-1))
    if 0 <= zero_line < height:
        chart[zero_line] = '-' * width
    
    # Draw data points
    for i, point in enumerate(scaled_points):
        # Skip if out of chart bounds
        if i >= width or point >= height:
            continue
            
        # Decide on symbol based on positive/negative and strategy
        if cumulative_pnl[i] >= 0:
            symbol = '+'  # Plus for profit
        else:
            symbol = '-'  # Minus for loss
            
        # Place the point
        chart[height - 1 - point] = chart[height - 1 - point][:i] + symbol + chart[height - 1 - point][i+1:]
    
    # Add y-axis labels
    chart_with_labels = []
    for i, line in enumerate(chart):
        # Add P&L value to left of chart
        pnl_value = max_pnl - (i * pnl_range / (height-1))
        label = f"{pnl_value:6.2f} |"
        chart_with_labels.append(label + line)
    
    # Add title and summary
    title = "TOTAL P&L OVER TIME"
    final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
    pnl_summary = f"Final P&L: {format_currency(final_pnl)} ({format_percentage(final_pnl/20000*100)})"
    
    # Build final output
    output = [
        f"{title:^{width+9}}",  # +9 to account for label width
        f"{pnl_summary:^{width+9}}",
        "-" * (width + 9),
        *chart_with_labels,
        "-" * (width + 9),
    ]
    
    return "\n".join(output)

def display_status():
    """Display current trading status in the terminal"""
    # Get portfolio metrics
    metrics = get_portfolio_metrics()
    
    # Get trades for chart
    trades = get_trades_history()
    
    # Generate P&L chart
    pnl_chart = generate_pnl_chart(trades)
    
    # Build header
    output = [
        "=" * 80,
        "KRAKEN TRADING BOT - CURRENT STATUS (SANDBOX MODE)",
        "=" * 80,
        f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"SOL/USD Current Price: {format_currency(metrics['current_position']['current_price'])}",
        "",
        "--- PORTFOLIO SUMMARY ---",
        f"Initial Value: {format_currency(metrics['initial_value'])}",
        f"Current Value: {format_currency(metrics['current_value'])}",
        f"Closed Trades P&L: {format_currency(metrics['closed_trades_pnl'])} ({format_percentage(metrics['closed_trades_pnl_percent'])})",
        f"Current Position P&L: {format_currency(metrics['current_position_pnl'])} ({format_percentage(metrics['current_position_pnl_percent'])})",
        f"Total P&L: {format_currency(metrics['total_pnl'])} ({format_percentage(metrics['total_pnl_percent'])})",
        "",
    ]
    
    # Add current position information
    if metrics['current_position']['type']:
        strategy = "unknown Strategy"  # Default
        
        # Try to determine position strategy from logs
        if os.path.exists("trade_notifications.log"):
            with open("trade_notifications.log", "r") as f:
                for line in reversed(f.readlines()):
                    if "Strategy:" in line:
                        if "ARIMA" in line:
                            strategy = "ARIMA Strategy"
                            break
                        elif "Adaptive" in line:
                            strategy = "Adaptive Strategy"
                            break
                        
        output.extend([
            "--- CURRENT POSITION ---",
            f"Type: {metrics['current_position']['type']} ({strategy})",
            f"Entry Price: {format_currency(metrics['current_position']['entry_price'])}",
            f"Size: {metrics['current_position']['size']:.2f} SOL",
            f"Current Price: {format_currency(metrics['current_position']['current_price'])}",
            f"Unrealized P&L: {format_currency(metrics['current_position_pnl'])} ({format_percentage(metrics['current_position_pnl_percent'])})",
            ""
        ])
    else:
        output.extend([
            "--- CURRENT POSITION ---",
            "Type: No Position",
            f"Current Price: {format_currency(metrics['current_position']['current_price'])}",
            ""
        ])
    
    # Add trade history summary
    output.extend([
        "--- TRADE HISTORY ---",
        f"Total Trades: {metrics['trade_metrics']['total_trades']}",
        f"Profitable Trades: {metrics['trade_metrics']['profitable_trades']} ({format_percentage(metrics['trade_metrics']['profitable_percent'])})",
        f"Losing Trades: {metrics['trade_metrics']['losing_trades']} ({format_percentage(metrics['trade_metrics']['losing_percent'])})",
        f"Total P&L: {format_currency(metrics['closed_trades_pnl'])} ({format_percentage(metrics['closed_trades_pnl_percent'])})",
        ""
    ])
    
    # Add strategy performance
    output.append("--- STRATEGY PERFORMANCE ---")
    
    for strategy, data in metrics['trade_metrics']['strategies'].items():
        if data['total_trades'] > 0:
            win_rate = data['profitable_percent'] if 'profitable_percent' in data else \
                      (data['profitable_trades'] / data['total_trades'] * 100 if data['total_trades'] > 0 else 0)
            
            pnl_contribution = (data['total_pnl'] / metrics['closed_trades_pnl'] * 100) if metrics['closed_trades_pnl'] != 0 else 0
            
            output.append(f"🟢 {strategy} Strategy:")
            output.append(f"  Total Trades: {data['total_trades']} ({data['profitable_trades']} profitable, {data['losing_trades']} losing)")
            output.append(f"  Win Rate: {format_percentage(win_rate)}")
            output.append(f"  Total P&L: {format_currency(data['total_pnl'])}")
            output.append(f"  PnL Contribution: {format_percentage(pnl_contribution)}")
            output.append("")
    
    # Add recent trades
    output.append("--- RECENT TRADES ---")
    
    # Get all trades and sort by timestamp descending
    sorted_trades = sorted(
        get_trades_history(), 
        key=lambda x: x.get('timestamp', datetime.datetime.now()), 
        reverse=True
    )
    
    # Show the most recent 5 trades
    for i, trade in enumerate(sorted_trades[:5]):
        timestamp = trade.get('timestamp', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        pair = trade.get('pair', 'UNKNOWN')
        type_str = trade.get('type', 'UNKNOWN')
        price = format_currency(trade.get('price', 0))
        volume = trade.get('volume', 0)
        strategy = trade.get('strategy', 'UNKNOWN')
        
        output.append(f"{i+1}. {timestamp} | {pair} | {type_str} | {volume:.2f} SOL @ {price} | {strategy}")
    
    # Add the P&L chart
    output.extend([
        "",
        "--- P&L CHART ---",
        pnl_chart,
        "=" * 80
    ])
    
    # Print the output
    print("\n".join(output))

def main():
    """Main function to display current status"""
    display_status()

if __name__ == "__main__":
    main()