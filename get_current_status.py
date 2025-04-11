#!/usr/bin/env python3

import os
import datetime
import json
import pandas as pd
from decimal import Decimal
import requests
import sys

# Default portfolio value in sandbox mode
INITIAL_PORTFOLIO_VALUE = 20000.00
TRADES_CSV = 'trades.csv'

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    return f"${float(value):.2f}"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{float(value):.2f}%"

def get_current_price(trading_pair="SOL/USD"):
    """Get current price from the trading bot logs"""
    try:
        # Read last few lines of the log file to find the latest price
        with open("trade_notifications.log", "r") as f:
            lines = f.readlines()
            
        # Look for ticker information
        for line in reversed(lines):
            if "【TICKER】" in line and trading_pair in line:
                # Extract the price
                parts = line.split('=')
                if len(parts) > 1:
                    price_str = parts[1].split('|')[0].strip()
                    try:
                        return float(price_str)
                    except ValueError:
                        continue
        
        # If no price found in logs, estimate from the last action
        for line in reversed(lines):
            if "【ACTION】" in line:
                # Try to extract the ATR line which usually has current price info
                if "ATR:" in line:
                    parts = line.split('|')
                    if len(parts) > 1:
                        atr_part = parts[1].strip()
                        try:
                            # ATR is typically a small value like $0.1550
                            atr_value = float(atr_part.split('$')[1].strip())
                            # Look for nearby lines with price information
                            for check_line in reversed(lines):
                                if "Current:" in check_line:
                                    current_parts = check_line.split("Current:")
                                    if len(current_parts) > 1:
                                        price_str = current_parts[1].split('→')[0].strip().replace('$', '')
                                        try:
                                            return float(price_str)
                                        except ValueError:
                                            continue
                        except (ValueError, IndexError):
                            continue
        
        # Fallback to a default value from configuration or recent analysis
        for line in reversed(lines):
            if "Forecast:" in line and "Current:" in line:
                current_part = line.split("Current:")[1].split("→")[0].strip()
                try:
                    return float(current_part.replace('$', ''))
                except ValueError:
                    continue
        
        # Final fallback
        return 113.00  # Recent approximate price seen in logs
    except Exception as e:
        print(f"Error getting current price: {e}")
        return 113.00  # Safe fallback

def get_current_position():
    """Get current position information"""
    try:
        position = {
            "symbol": "SOL/USD",
            "position_type": "none",
            "entry_price": 0.0,
            "current_price": get_current_price(),
            "quantity": 0.0,
            "pnl": 0.0,
            "pnl_percent": 0.0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Check logs for current position
        try:
            with open("trade_notifications.log", "r") as f:
                lines = f.readlines()
            
            # Look for position information
            for line in reversed(lines):
                if "【POSITION】" in line:
                    # Extract the position information
                    if "Size:" in line:
                        size_part = line.split("Size:")[1].split('|')[0].strip()
                        try:
                            position["quantity"] = float(size_part)
                            # Now that we know we're in a position, determine type
                            # Check previous lines to find the action type
                            for prev_line in reversed(lines[:lines.index(line)]):
                                if "BUY" in prev_line:
                                    position["position_type"] = "long"
                                    break
                                elif "SELL" in prev_line:
                                    position["position_type"] = "short"
                                    break
                            break
                        except ValueError:
                            continue
                
                # Look for entry price
                elif "【ORDER】" in line:
                    if "buy" in line.lower() and position["position_type"] == "long":
                        if "order at" in line:
                            price_str = line.split("order at")[1].strip()
                            try:
                                position["entry_price"] = float(price_str.replace('$', ''))
                            except ValueError:
                                continue
                    elif "sell" in line.lower() and position["position_type"] == "short":
                        if "order at" in line:
                            price_str = line.split("order at")[1].strip()
                            try:
                                position["entry_price"] = float(price_str.replace('$', ''))
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Error reading position from logs: {e}")
        
        # Calculate PnL if in a position
        if position["position_type"] != "none" and position["entry_price"] > 0:
            if position["position_type"] == "long":
                position["pnl"] = (position["current_price"] - position["entry_price"]) * position["quantity"]
                position["pnl_percent"] = ((position["current_price"] / position["entry_price"]) - 1) * 100
            elif position["position_type"] == "short":
                position["pnl"] = (position["entry_price"] - position["current_price"]) * position["quantity"]
                position["pnl_percent"] = ((position["entry_price"] / position["current_price"]) - 1) * 100
        
        return position
    except Exception as e:
        print(f"Error getting current position: {e}")
        return {"symbol": "SOL/USD", "position_type": "none", "current_price": get_current_price()}

def get_trades_history():
    """Get recent trade history"""
    try:
        # Start with empty trades list
        trades = []
        
        # Check if trades CSV exists
        if os.path.exists(TRADES_CSV):
            # Read trades from CSV
            try:
                trades_df = pd.read_csv(TRADES_CSV)
                trades = trades_df.to_dict('records')
            except Exception as e:
                print(f"Error reading trades CSV: {e}")
        
        # If no trades found or CSV doesn't exist, look in logs
        if not trades:
            try:
                with open("trade_notifications.log", "r") as f:
                    lines = f.readlines()
                
                # Find trade entries
                for i, line in enumerate(lines):
                    if "【ORDER】" in line:
                        trade = {
                            "time": datetime.datetime.now().timestamp(),
                            "type": "buy" if "BUY" in line else "sell",
                            "pair": "SOL/USD",
                            "price": 0.0,
                            "volume": 0.0,
                            "cost": 0.0,
                            "fee": 0.0
                        }
                        
                        # Extract price
                        if "order at" in line:
                            price_str = line.split("order at")[1].strip()
                            try:
                                trade["price"] = float(price_str.replace('$', ''))
                            except ValueError:
                                continue
                        
                        # Look for position size in nearby lines
                        for j in range(i, min(i+5, len(lines))):
                            if "【POSITION】" in lines[j] and "Size:" in lines[j]:
                                size_part = lines[j].split("Size:")[1].split('|')[0].strip()
                                try:
                                    trade["volume"] = float(size_part)
                                    trade["cost"] = trade["price"] * trade["volume"]
                                    # Estimate fee at 0.16% (Kraken's typical fee)
                                    trade["fee"] = trade["cost"] * 0.0016
                                    trades.append(trade)
                                    break
                                except ValueError:
                                    continue
            except Exception as e:
                print(f"Error reading trades from logs: {e}")
        
        return trades
    except Exception as e:
        print(f"Error getting trades history: {e}")
        return []

def get_portfolio_metrics():
    """Calculate portfolio metrics"""
    try:
        # Start with initial portfolio value
        metrics = {
            "initial_value": INITIAL_PORTFOLIO_VALUE,
            "current_value": INITIAL_PORTFOLIO_VALUE,
            "pnl": 0.0,
            "pnl_percent": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "strategies": {
                "adaptive": {"total": 0, "profitable": 0, "losing": 0, "win_rate": 0.0},
                "arima": {"total": 0, "profitable": 0, "losing": 0, "win_rate": 0.0}
            },
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Get current position
            position = get_current_position()
            
            # Add PnL from current position if in a position
            if position["position_type"] != "none":
                metrics["current_value"] += position["pnl"]
            
            # Get closed trades
            trades = get_trades_history()
            metrics["total_trades"] = len(trades)
            
            # Calculate metrics from closed trades
            if metrics["total_trades"] > 0:
                total_pnl = 0.0
                for trade in trades:
                    # Simple estimation - assuming trades are paired (buy followed by sell)
                    if isinstance(trade, dict) and 'type' in trade and trade["type"] == "sell":
                        # This is a closing trade, find the corresponding opening trade
                        for open_trade in trades:
                            if isinstance(open_trade, dict) and 'type' in open_trade and 'time' in open_trade and \
                               open_trade["type"] == "buy" and open_trade.get("time", 0) < trade.get("time", 0):
                                # Calculate PnL
                                price_diff = trade.get("price", 0) - open_trade.get("price", 0)
                                volume = trade.get("volume", 0)
                                trade_pnl = price_diff * volume
                                total_pnl += trade_pnl
                                
                                # Determine strategy type from trade notes or logs
                                strategy = "adaptive"  # Default to adaptive
                                if "strategy" in trade:
                                    strategy = trade.get("strategy", "adaptive").lower()
                                elif "arima" in str(trade.get("notes", "")).lower():
                                    strategy = "arima"
                                
                                # Update strategy-specific metrics
                                metrics["strategies"][strategy]["total"] += 1
                                
                                if trade_pnl > 0:
                                    metrics["profitable_trades"] += 1
                                    metrics["strategies"][strategy]["profitable"] += 1
                                else:
                                    metrics["losing_trades"] += 1
                                    metrics["strategies"][strategy]["losing"] += 1
                                break
                
                metrics["pnl"] = total_pnl
                metrics["pnl_percent"] = (total_pnl / INITIAL_PORTFOLIO_VALUE) * 100
                
                # Calculate win rates
                if metrics["total_trades"] > 0:
                    metrics["win_rate"] = (metrics["profitable_trades"] / metrics["total_trades"]) * 100
                
                # Calculate strategy-specific win rates
                for strategy in metrics["strategies"]:
                    if metrics["strategies"][strategy]["total"] > 0:
                        metrics["strategies"][strategy]["win_rate"] = (
                            metrics["strategies"][strategy]["profitable"] / 
                            metrics["strategies"][strategy]["total"] * 100
                        )
            
            # Add current position's unrealized PnL
            position_pnl = 0.0
            if position["position_type"] != "none":
                position_pnl = position.get("pnl", 0.0)
            
            metrics["current_value"] = INITIAL_PORTFOLIO_VALUE + metrics["pnl"] + position_pnl
            metrics["pnl_percent"] = ((metrics["current_value"] / INITIAL_PORTFOLIO_VALUE) - 1) * 100
            
        except KeyError as e:
            print(f"Missing key in portfolio calculation: {e}")
        except TypeError as e:
            print(f"Type error in portfolio calculation: {e}")
            
        return metrics
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return {
            "initial_value": INITIAL_PORTFOLIO_VALUE,
            "current_value": INITIAL_PORTFOLIO_VALUE,
            "pnl": 0.0,
            "pnl_percent": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "losing_trades": 0,
            "strategies": {
                "adaptive": {"total": 0, "profitable": 0, "losing": 0, "win_rate": 0.0},
                "arima": {"total": 0, "profitable": 0, "losing": 0, "win_rate": 0.0}
            },
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def display_status():
    """Display current trading status in the terminal"""
    print("\n" + "=" * 80)
    print("KRAKEN TRADING BOT - CURRENT STATUS")
    print("=" * 80)
    
    # Get current time
    now = datetime.datetime.now()
    print(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get current price
    current_price = get_current_price()
    print(f"SOL/USD Current Price: {format_currency(current_price)}")
    
    # Get portfolio metrics
    metrics = get_portfolio_metrics()
    trades = get_trades_history()
    position = get_current_position()
    
    print("\n--- PORTFOLIO SUMMARY ---")
    print(f"Initial Value: {format_currency(metrics['initial_value'])}")
    print(f"Current Value: {format_currency(metrics['current_value'])}")
    
    # Calculate and display P&L for closed trades
    closed_pnl = metrics['pnl']
    closed_pnl_percent = (closed_pnl / metrics['initial_value']) * 100 if metrics['initial_value'] > 0 else 0
    closed_pnl_style = "\033[92m" if closed_pnl >= 0 else "\033[91m"
    print(f"Closed Trades P&L: {closed_pnl_style}{format_currency(closed_pnl)} ({format_percentage(closed_pnl_percent)})\033[0m")
    
    # Calculate and display P&L for open positions
    open_pnl = position['pnl'] if position['position_type'] != "none" else 0
    open_pnl_percent = position['pnl_percent'] if position['position_type'] != "none" else 0
    open_pnl_style = "\033[92m" if open_pnl >= 0 else "\033[91m"
    print(f"Open Position P&L: {open_pnl_style}{format_currency(open_pnl)} ({format_percentage(open_pnl_percent)})\033[0m")
    
    # Calculate and display total P&L (closed + open)
    total_pnl = closed_pnl + open_pnl
    total_pnl_percent = ((metrics['current_value'] / metrics['initial_value']) - 1) * 100 if metrics['initial_value'] > 0 else 0
    total_pnl_style = "\033[92m" if total_pnl >= 0 else "\033[91m"
    print(f"Total P&L: {total_pnl_style}{format_currency(total_pnl)} ({format_percentage(total_pnl_percent)})\033[0m")
    
    # Trade statistics for all strategies
    print(f"Total Trades: {metrics['total_trades']}")
    if metrics['total_trades'] > 0:
        win_rate_style = "\033[92m" if metrics['win_rate'] >= 50 else "\033[93m" if metrics['win_rate'] >= 30 else "\033[91m"
        print(f"Win Rate: {win_rate_style}{format_percentage(metrics['win_rate'])}\033[0m")
        print(f"Profitable Trades: \033[92m{metrics['profitable_trades']}\033[0m")
        print(f"Losing Trades: \033[91m{metrics['losing_trades']}\033[0m")
        
        # Per-strategy breakdown
        print("\n--- STRATEGY BREAKDOWN ---")
        for strategy_name, strategy_data in metrics['strategies'].items():
            if strategy_data['total'] > 0:
                strategy_win_rate_style = "\033[92m" if strategy_data['win_rate'] >= 50 else "\033[93m" if strategy_data['win_rate'] >= 30 else "\033[91m"
                print(f"{strategy_name.upper()} Strategy:")
                print(f"  Total Trades: {strategy_data['total']}")
                print(f"  Win Rate: {strategy_win_rate_style}{format_percentage(strategy_data['win_rate'])}\033[0m")
                print(f"  Profitable Trades: \033[92m{strategy_data['profitable']}\033[0m")
                print(f"  Losing Trades: \033[91m{strategy_data['losing']}\033[0m")
    
    # Calculate allocation and available funds
    allocated_capital = 0
    if position['position_type'] != "none":
        # Rough estimation based on the position size and leverage
        allocated_capital = position['entry_price'] * position['quantity']
    available_funds = metrics['current_value'] - allocated_capital
    print(f"Allocated Capital: {format_currency(allocated_capital)}")
    print(f"Available Funds: {format_currency(available_funds)}")
    
    # Get current position
    print("\n--- CURRENT POSITION ---")
    if position['position_type'] == "none":
        print("No active position")
    else:
        position_style = "\033[92m" if position['position_type'] == "long" else "\033[91m"
        print(f"Type: {position_style}{position['position_type'].upper()}\033[0m")
        print(f"Symbol: {position['symbol']}")
        print(f"Entry Price: {format_currency(position['entry_price'])}")
        print(f"Current Price: {format_currency(position['current_price'])}")
        print(f"Quantity: {position['quantity']}")
        pnl_style = "\033[92m" if position['pnl'] >= 0 else "\033[91m"
        print(f"Unrealized P&L: {pnl_style}{format_currency(position['pnl'])} ({format_percentage(position['pnl_percent'])})\033[0m")
    
    # Get trading signals
    print("\n--- MOST RECENT SIGNALS ---")
    try:
        log_lines = []
        with open("trade_notifications.log", "r") as f:
            log_lines = f.readlines()
        
        signals = []
        for line in reversed(log_lines):
            if "【SIGNAL】" in line:
                signals.append(line.strip())
                if len(signals) >= 3:
                    break
        
        if signals:
            for signal in reversed(signals):
                signal_color = "\033[92m" if "BULLISH" in signal else "\033[91m" if "BEARISH" in signal else "\033[93m"
                print(f"{signal_color}{signal}\033[0m")
        else:
            print("No recent signals found")
            
        # Get most recent actions
        print("\n--- MOST RECENT ACTIONS ---")
        actions = []
        for line in reversed(log_lines):
            if "【ACTION】" in line:
                actions.append(line.strip())
                if len(actions) >= 3:
                    break
        
        if actions:
            for action in reversed(actions):
                action_color = "\033[92m" if "BUY" in action else "\033[91m" if "SELL" in action else "\033[93m"
                print(f"{action_color}{action}\033[0m")
        else:
            print("No recent actions found")
            
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    display_status()