#!/usr/bin/env python3

import csv
import os
import datetime
from typing import List, Dict, Any, Optional, Tuple

# Constants
INITIAL_CAPITAL = 20000.0

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    return f"+{value:.2f}%" if value >= 0 else f"{value:.2f}%"

def parse_datetime(date_str):
    """Parse different datetime formats to datetime object"""
    if not date_str:
        return None
    
    if isinstance(date_str, (float, int)):
        # Unix timestamp
        return datetime.datetime.fromtimestamp(date_str)
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    
    # If all formats fail, return None
    return None

def load_trades_from_csv() -> List[Dict[str, Any]]:
    """Load trades from the CSV file"""
    trades = []
    
    try:
        if os.path.exists("trades.csv"):
            with open("trades.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    for field in ["price", "amount", "value", "pnl", "pnl_percentage"]:
                        if field in row and row[field] and row[field] != "":
                            try:
                                row[field] = float(row[field])
                            except (ValueError, TypeError):
                                row[field] = 0.0
                        else:
                            row[field] = 0.0
                    
                    # Parse datetime
                    if "timestamp" in row and row["timestamp"]:
                        row["datetime"] = parse_datetime(row["timestamp"])
                    else:
                        row["datetime"] = None
                        
                    trades.append(row)
    except Exception as e:
        print(f"Error loading trades: {e}")
        return []
    
    return trades

def pair_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process trades and create trade history
    For the Kraken bot, trades are already logged with PnL information for completed trades
    """
    processed_trades = []
    buy_positions = {}
    
    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda x: x.get("datetime", datetime.datetime.min))
    
    # Process each trade
    for trade in sorted_trades:
        side = trade.get("side", "").upper()
        pair = trade.get("pair", "")
        trade_amount = trade.get("amount", 0.0)
        trade_price = trade.get("price", 0.0)
        trade_time = trade.get("datetime")
        strategy = trade.get("strategy", "Unknown")
        
        # If PnL is directly provided in the trade
        pnl = trade.get("pnl", 0.0)
        pnl_percentage = trade.get("pnl_percentage", 0.0)
        
        # Create the processed trade record
        if side == "BUY":
            # This is an entry/opening trade
            # Store it for reference
            position_key = f"{pair}_{trade_amount}_{strategy}"
            buy_positions[position_key] = trade
            
            processed_trade = {
                "entry_time": trade_time,
                "exit_time": None,
                "pair": pair,
                "strategy": strategy,
                "entry_price": trade_price,
                "exit_price": None,
                "amount": trade_amount,
                "pnl": None,
                "pnl_percentage": None,
                "status": "open",
                "position_type": "long"
            }
            processed_trades.append(processed_trade)
            
        elif side == "SELL":
            # Determine if this is a closing trade for an existing position
            # or a standalone/short position
            position_key = f"{pair}_{trade_amount}_{strategy}"
            
            if position_key in buy_positions:
                # This is closing a long position
                buy_trade = buy_positions[position_key]
                entry_price = buy_trade.get("price", 0.0)
                
                # Calculate P&L if not provided in the trade
                if pnl is None or pnl == 0.0:
                    pnl = (trade_price - entry_price) * trade_amount
                    if entry_price > 0:
                        pnl_percentage = ((trade_price / entry_price) - 1) * 100
                    else:
                        pnl_percentage = 0.0
                
                processed_trade = {
                    "entry_time": buy_trade.get("datetime"),
                    "exit_time": trade_time,
                    "pair": pair,
                    "strategy": strategy,
                    "entry_price": entry_price,
                    "exit_price": trade_price,
                    "amount": trade_amount,
                    "pnl": pnl,
                    "pnl_percentage": pnl_percentage,
                    "status": "closed",
                    "position_type": "long"
                }
                processed_trades.append(processed_trade)
                
                # Remove the used buy position
                del buy_positions[position_key]
                
                # Also remove the open trade from processed_trades
                open_trades = [t for t in processed_trades if 
                              t.get("status") == "open" and 
                              t.get("pair") == pair and 
                              t.get("amount") == trade_amount and
                              t.get("strategy") == strategy]
                
                for open_trade in open_trades:
                    if open_trade in processed_trades:
                        processed_trades.remove(open_trade)
            else:
                # This is a standalone sell (short position or closing a trade not in our records)
                processed_trade = {
                    "entry_time": None,
                    "exit_time": trade_time,
                    "pair": pair,
                    "strategy": strategy,
                    "entry_price": None,
                    "exit_price": trade_price,
                    "amount": trade_amount,
                    "pnl": pnl,
                    "pnl_percentage": pnl_percentage,
                    "status": "isolated_sell",
                    "position_type": "short"
                }
                processed_trades.append(processed_trade)
                
    # Keep remaining open positions (BUY trades without matching SELLs)
    # They're already in processed_trades from the BUY section
    
    return processed_trades

def calculate_trade_metrics(paired_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate detailed trade metrics based on paired trades
    """
    # Filter to closed trades with valid P&L
    closed_trades = [t for t in paired_trades if t.get("status") == "closed" and t.get("pnl") is not None]
    
    # Initialize metrics
    metrics = {
        "total_trades": len(closed_trades),
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "total_pnl": 0.0,
        "total_pnl_percentage": 0.0,
        "win_rate": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "profit_factor": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "average_holding_time": datetime.timedelta(0),
        "current_portfolio_value": INITIAL_CAPITAL,
        "strategy_performance": {}
    }
    
    if not closed_trades:
        return metrics
    
    # Calculate basic metrics
    winning_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in closed_trades if t.get("pnl", 0) < 0]
    breakeven_trades = [t for t in closed_trades if t.get("pnl", 0) == 0]
    
    total_pnl = sum(t.get("pnl", 0) for t in closed_trades)
    total_profit = sum(t.get("pnl", 0) for t in winning_trades)
    total_loss = sum(t.get("pnl", 0) for t in losing_trades)
    
    # Calculate win rate and other metrics
    metrics["winning_trades"] = len(winning_trades)
    metrics["losing_trades"] = len(losing_trades)
    metrics["breakeven_trades"] = len(breakeven_trades)
    metrics["total_pnl"] = total_pnl
    metrics["win_rate"] = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
    metrics["average_win"] = total_profit / len(winning_trades) if winning_trades else 0
    metrics["average_loss"] = total_loss / len(losing_trades) if losing_trades else 0
    metrics["profit_factor"] = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
    metrics["largest_win"] = max([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0
    metrics["largest_loss"] = min([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0
    
    # Calculate average holding time
    holding_times = []
    for trade in closed_trades:
        if trade.get("entry_time") and trade.get("exit_time"):
            try:
                holding_time = trade.get("exit_time") - trade.get("entry_time")
                holding_times.append(holding_time)
            except (TypeError, ValueError):
                # Skip if we can't calculate holding time
                pass
    
    if holding_times:
        metrics["average_holding_time"] = sum(holding_times, datetime.timedelta(0)) / len(holding_times)
    
    # Calculate per-strategy metrics
    strategy_performance = {}
    for trade in closed_trades:
        strategy = trade.get("strategy", "Unknown")
        if strategy not in strategy_performance:
            strategy_performance[strategy] = {
                "trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0
            }
        
        strategy_performance[strategy]["trades"] += 1
        strategy_performance[strategy]["total_pnl"] += trade.get("pnl", 0)
        
        if trade.get("pnl", 0) > 0:
            strategy_performance[strategy]["winning_trades"] += 1
        elif trade.get("pnl", 0) < 0:
            strategy_performance[strategy]["losing_trades"] += 1
    
    # Calculate win rate for each strategy
    for strategy, stats in strategy_performance.items():
        stats["win_rate"] = (stats["winning_trades"] / stats["trades"]) * 100 if stats["trades"] > 0 else 0
    
    metrics["strategy_performance"] = strategy_performance
    
    # Calculate final portfolio value
    metrics["current_portfolio_value"] = INITIAL_CAPITAL + total_pnl
    metrics["total_pnl_percentage"] = (total_pnl / INITIAL_CAPITAL) * 100
    
    return metrics

def print_portfolio_report():
    """
    Generate and print a comprehensive portfolio report
    """
    print("\n" + "=" * 80)
    print("KRAKEN TRADING BOT - PORTFOLIO REPORT")
    print("=" * 80)
    
    # Load and pair trades
    raw_trades = load_trades_from_csv()
    paired_trades = pair_trades(raw_trades)
    metrics = calculate_trade_metrics(paired_trades)
    
    # Print portfolio summary
    print("\n📊 PORTFOLIO SUMMARY:")
    print(f"  Initial Capital:     {format_currency(INITIAL_CAPITAL)}")
    print(f"  Current Value:       {format_currency(metrics['current_portfolio_value'])}")
    print(f"  Total P&L:           {format_currency(metrics['total_pnl'])} ({format_percentage(metrics['total_pnl_percentage'])})")
    
    # Print trade statistics
    print("\n📈 TRADE STATISTICS:")
    print(f"  Total Trades:        {metrics['total_trades']}")
    print(f"  Winning Trades:      {metrics['winning_trades']} ({format_percentage(metrics['win_rate'])})")
    print(f"  Losing Trades:       {metrics['losing_trades']}")
    print(f"  Average Win:         {format_currency(metrics['average_win'])}")
    print(f"  Average Loss:        {format_currency(metrics['average_loss'])}")
    print(f"  Largest Win:         {format_currency(metrics['largest_win'])}")
    print(f"  Largest Loss:        {format_currency(metrics['largest_loss'])}")
    
    if metrics['average_holding_time'] > datetime.timedelta(0):
        hours, remainder = divmod(metrics['average_holding_time'].total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"  Avg Holding Time:    {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Print strategy performance
    print("\n🤖 STRATEGY PERFORMANCE:")
    for strategy, stats in metrics["strategy_performance"].items():
        print(f"  {strategy}:")
        print(f"    Trades: {stats['trades']} | Win Rate: {format_percentage(stats['win_rate'])}")
        print(f"    P&L: {format_currency(stats['total_pnl'])}")
    
    # Print recent trades
    print("\n🔄 RECENT TRADES:")
    # Sort by exit time (most recent first), with open trades first
    sorted_trades = sorted(
        paired_trades, 
        key=lambda x: (x.get("status") != "open", 
                       x.get("exit_time", datetime.datetime.max) if x.get("exit_time") else datetime.datetime.max), 
        reverse=True
    )
    
    # Show the 5 most recent trades
    for i, trade in enumerate(sorted_trades[:5]):
        status = trade.get("status", "")
        if status == "open":
            entry_time = trade.get("entry_time")
            entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A"
            print(f"  ⏳ OPEN - {trade.get('pair', 'Unknown')} entry at {format_currency(trade.get('entry_price', 0))} on {entry_time_str}")
        else:
            pnl = trade.get("pnl", 0)
            pnl_str = format_currency(pnl)
            exit_time = trade.get("exit_time")
            exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S") if exit_time else "N/A"
            
            if pnl > 0:
                print(f"  🟢 WIN - {trade.get('pair', 'Unknown')} {pnl_str} on {exit_time_str}")
            elif pnl < 0:
                print(f"  🔴 LOSS - {trade.get('pair', 'Unknown')} {pnl_str} on {exit_time_str}")
            else:
                print(f"  ⚪ EVEN - {trade.get('pair', 'Unknown')} {pnl_str} on {exit_time_str}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_portfolio_report()