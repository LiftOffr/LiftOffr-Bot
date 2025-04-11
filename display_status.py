import os
import csv
import json
import datetime
from statistics import mean
from collections import defaultdict
import pandas as pd

def format_currency(value):
    """Format a number as currency with 2 decimal places"""
    return f"${value:.2f}"

def format_percentage(value):
    """Format a number as percentage with 2 decimal places"""
    return f"{value:.2f}%"

def get_current_price(trading_pair="SOL/USD"):
    """Get current price from logs"""
    try:
        with open("nohup.out", "r") as f:
            lines = f.readlines()
            
        # Look for the most recent ticker lines
        for line in reversed(lines):
            if "【TICKER】" in line and trading_pair in line:
                parts = line.split("|")
                price_part = parts[0].split("=")[1].strip()
                return float(price_part.replace("$", ""))
        
        return None
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_current_position():
    """Get current position information"""
    try:
        with open("nohup.out", "r") as f:
            lines = f.readlines()
        
        # Look for position information
        position_info = {"arima": None, "adaptive": None}
        
        for line in reversed(lines):
            if "PORTFOLIO STATUS" in line:
                # Found the portfolio status section
                for i in range(lines.index(line), min(lines.index(line) + 20, len(lines))):
                    status_line = lines[i]
                    
                    if "arima-SOLUSD" in status_line and position_info["arima"] is None:
                        # Extract position info for ARIMA strategy
                        position_type = "LONG" if "LONG" in lines[i+1] else "SHORT" if "SHORT" in lines[i+1] else "No Position"
                        position_info["arima"] = {
                            "type": position_type,
                            "margin": 0.0,
                            "leverage": 0
                        }
                        
                        if position_type != "No Position":
                            # Try to extract entry price
                            entry_price_match = status_line.split("@")
                            if len(entry_price_match) > 1:
                                position_info["arima"]["entry_price"] = float(entry_price_match[1].strip().replace("$", ""))
                    
                    if "adaptive-SOLUSD" in status_line and position_info["adaptive"] is None:
                        # Extract position info for Adaptive strategy
                        position_type = "LONG" if "LONG" in lines[i+1] else "SHORT" if "SHORT" in lines[i+1] else "No Position"
                        position_info["adaptive"] = {
                            "type": position_type,
                            "margin": 0.0,
                            "leverage": 0
                        }
                        
                        if position_type != "No Position":
                            # Try to extract entry price
                            entry_price_match = status_line.split("@")
                            if len(entry_price_match) > 1:
                                position_info["adaptive"]["entry_price"] = float(entry_price_match[1].strip().replace("$", ""))
                
                # If we found the portfolio status section, break out
                break
        
        return position_info
    except Exception as e:
        print(f"Error getting current position: {e}")
        return {"arima": None, "adaptive": None}

def get_trades_history():
    """Get trades from trades.csv file"""
    try:
        trades = []
        if os.path.exists("trades.csv"):
            with open("trades.csv", "r") as f:
                # Read all lines
                lines = f.readlines()
                
                # Check which format we have
                if len(lines) > 0 and "time,type,pair,price,volume,cost,fee,txid" in lines[0]:
                    # This is the standard format with CSV header
                    reader = csv.DictReader(f)
                    trades = []
                    
                    # Reset file pointer to beginning
                    f.seek(0)
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        try:
                            # Process the first format (time, type, pair, price, volume, cost, fee, txid)
                            if 'time' in row and 'type' in row and 'price' in row and 'volume' in row:
                                # First 3 rows are sample trades
                                trade = {
                                    "timestamp": row.get('time'),
                                    "type": row.get('type'),
                                    "pair": row.get('pair'),
                                    "price": float(row.get('price', 0)),
                                    "size": float(row.get('volume', 0)),
                                    "profit_loss": 0.0,  # We'll calculate this later
                                    "profit_loss_percent": 0.0,
                                    "strategy": "sample"
                                }
                                trades.append(trade)
                        except Exception as e:
                            print(f"Error processing row: {e}")
                            continue
                else:
                    # Process the alternative format where each line is a trade entry
                    # Example: 2025-04-11 00:36:21,SOL/USD,SELL,1105.998938,113.02,,,short
                    trades = []
                    
                    for line in lines[1:]:  # Skip header row
                        try:
                            # Split the line
                            parts = line.strip().split(',')
                            
                            if len(parts) >= 5:
                                # Extract parts based on format
                                timestamp = parts[0]
                                pair = parts[1]
                                trade_type = parts[2]
                                quantity = float(parts[3])
                                price = float(parts[4])
                                strategy = parts[7] if len(parts) > 7 else "unknown"
                                
                                # Create trade record
                                trade = {
                                    "timestamp": timestamp,
                                    "pair": pair,
                                    "type": trade_type,
                                    "size": quantity,
                                    "price": price,
                                    "profit_loss": 0.0,  # We don't have this information
                                    "profit_loss_percent": 0.0,
                                    "strategy": strategy
                                }
                                trades.append(trade)
                        except Exception as e:
                            # Skip lines that don't match the expected format
                            continue
        
        return trades
    except Exception as e:
        print(f"Error reading trades: {e}")
        return []

def get_portfolio_metrics():
    """Calculate portfolio metrics"""
    try:
        trades = get_trades_history()
        
        if not trades:
            return {
                "initial_value": 20000.0,  # Default initial value
                "current_value": 20000.0,
                "total_profit_loss": 0.0,
                "total_profit_loss_percent": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "strategy_performance": {
                    "arima": {
                        "profit_loss": 0.0,
                        "profit_loss_percent": 0.0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0
                    },
                    "adaptive": {
                        "profit_loss": 0.0,
                        "profit_loss_percent": 0.0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0.0
                    }
                }
            }
        
        # Get the latest portfolio value
        initial_value = 20000.0  # Default initial value
        current_value = initial_value + sum(trade["profit_loss"] for trade in trades)
        total_profit_loss = current_value - initial_value
        total_profit_loss_percent = (total_profit_loss / initial_value) * 100
        
        # Count winning and losing trades
        winning_trades = sum(1 for trade in trades if trade["profit_loss"] > 0)
        losing_trades = sum(1 for trade in trades if trade["profit_loss"] < 0)
        win_rate = (winning_trades / len(trades)) * 100 if trades else 0
        
        # Strategy performance
        strategy_trades = defaultdict(list)
        for trade in trades:
            strategy = trade.get("strategy", "").lower()
            if "arima" in strategy:
                strategy_trades["arima"].append(trade)
            elif "adaptive" in strategy:
                strategy_trades["adaptive"].append(trade)
        
        strategy_performance = {}
        for strategy, strategy_trade_list in strategy_trades.items():
            strategy_profit_loss = sum(trade["profit_loss"] for trade in strategy_trade_list)
            strategy_profit_loss_percent = (strategy_profit_loss / initial_value) * 100
            strategy_winning_trades = sum(1 for trade in strategy_trade_list if trade["profit_loss"] > 0)
            strategy_losing_trades = sum(1 for trade in strategy_trade_list if trade["profit_loss"] < 0)
            strategy_win_rate = (strategy_winning_trades / len(strategy_trade_list)) * 100 if strategy_trade_list else 0
            
            strategy_performance[strategy] = {
                "profit_loss": strategy_profit_loss,
                "profit_loss_percent": strategy_profit_loss_percent,
                "winning_trades": strategy_winning_trades,
                "losing_trades": strategy_losing_trades,
                "win_rate": strategy_win_rate
            }
        
        # Fill in missing strategies
        for strategy in ["arima", "adaptive"]:
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "profit_loss": 0.0,
                    "profit_loss_percent": 0.0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0
                }
        
        return {
            "initial_value": initial_value,
            "current_value": current_value,
            "total_profit_loss": total_profit_loss,
            "total_profit_loss_percent": total_profit_loss_percent,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "strategy_performance": strategy_performance
        }
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return None

def get_portfolio_status():
    """Get portfolio allocation status from log files"""
    try:
        with open("nohup.out", "r") as f:
            lines = f.readlines()
        
        # Look for the most recent portfolio status
        for line_index, line in enumerate(reversed(lines)):
            if "PORTFOLIO STATUS" in line:
                # Extract information from the following lines
                for i in range(1, 15):  # Check the next 15 lines
                    if line_index - i >= 0:  # Make sure we don't go out of bounds
                        status_line = lines[-(line_index - i)]
                        
                        if "Initial Value:" in status_line:
                            initial_value = float(status_line.split(':')[1].strip().replace('$', '').split()[0])
                        elif "Current Value:" in status_line:
                            current_value = float(status_line.split(':')[1].strip().replace('$', '').split()[0])
                        elif "Total P&L:" in status_line:
                            pnl = status_line.split(':')[1].strip()
                            pnl_amount = float(pnl.split()[0].replace('$', ''))
                            pnl_percent = float(pnl.split('(')[1].split('%')[0])
                        elif "Allocated Capital:" in status_line:
                            allocated = status_line.split(':')[1].strip()
                            allocated_amount = float(allocated.split()[0].replace('$', ''))
                            allocated_percent = float(allocated.split('(')[1].split('%')[0])
                        elif "Available Funds:" in status_line:
                            available = float(status_line.split(':')[1].strip().replace('$', ''))
                
                return {
                    "initial_value": initial_value,
                    "current_value": current_value,
                    "pnl_amount": pnl_amount,
                    "pnl_percent": pnl_percent,
                    "allocated_amount": allocated_amount,
                    "allocated_percent": allocated_percent,
                    "available": available
                }
        
        return None
    except Exception as e:
        print(f"Error getting portfolio status: {e}")
        return None

def get_strategy_status():
    """Get the current status of each trading strategy"""
    try:
        with open("nohup.out", "r") as f:
            lines = f.readlines()
        
        # Look for the most recent strategy status indicators
        strategies = {
            "arima": {"position": None, "entry_price": None, "current_price": None, "unrealized_pnl": None},
            "adaptive": {"position": None, "entry_price": None, "current_price": None, "unrealized_pnl": None}
        }
        
        for line_index, line in enumerate(reversed(lines)):
            if "PORTFOLIO STATUS" in line:
                # Found the portfolio status section, now parse the strategy info
                for i in range(line_index):
                    if i < len(lines) - (line_index - i):
                        status_line = lines[-(line_index - i)]
                        
                        # ARIMA strategy info
                        if "arima-SOLUSD" in status_line and strategies["arima"]["position"] is None:
                            position_line = status_line.strip()
                            if "LONG" in position_line:
                                strategies["arima"]["position"] = "LONG"
                                price_part = position_line.split("@")[1].strip()
                                strategies["arima"]["entry_price"] = float(price_part.replace("$", ""))
                            elif "SHORT" in position_line:
                                strategies["arima"]["position"] = "SHORT"
                                price_part = position_line.split("@")[1].strip()
                                strategies["arima"]["entry_price"] = float(price_part.replace("$", ""))
                            else:
                                strategies["arima"]["position"] = "No Position"
                            
                            # Try to extract current price and unrealized PnL
                            for j in range(1, 5):
                                if i+j < len(lines) - (line_index - (i+j)):
                                    next_line = lines[-(line_index - (i+j))].strip()
                                    if "Current Price:" in next_line and "Unrealized P&L:" in next_line:
                                        price_part = next_line.split("|")[0].strip()
                                        strategies["arima"]["current_price"] = float(price_part.split(":")[1].strip().replace("$", ""))
                                        
                                        pnl_part = next_line.split("|")[1].strip()
                                        strategies["arima"]["unrealized_pnl"] = float(pnl_part.split(":")[1].strip().replace("%", ""))
                                        break
                        
                        # Adaptive strategy info
                        if "adaptive-SOLUSD" in status_line and strategies["adaptive"]["position"] is None:
                            position_line = status_line.strip()
                            if "LONG" in position_line:
                                strategies["adaptive"]["position"] = "LONG"
                                price_part = position_line.split("@")[1].strip()
                                strategies["adaptive"]["entry_price"] = float(price_part.replace("$", ""))
                            elif "SHORT" in position_line:
                                strategies["adaptive"]["position"] = "SHORT"
                                price_part = position_line.split("@")[1].strip()
                                strategies["adaptive"]["entry_price"] = float(price_part.replace("$", ""))
                            else:
                                strategies["adaptive"]["position"] = "No Position"
                            
                            # Try to extract current price and unrealized PnL
                            for j in range(1, 5):
                                if i+j < len(lines) - (line_index - (i+j)):
                                    next_line = lines[-(line_index - (i+j))].strip()
                                    if "Current Price:" in next_line and "Unrealized P&L:" in next_line:
                                        price_part = next_line.split("|")[0].strip()
                                        strategies["adaptive"]["current_price"] = float(price_part.split(":")[1].strip().replace("$", ""))
                                        
                                        pnl_part = next_line.split("|")[1].strip()
                                        strategies["adaptive"]["unrealized_pnl"] = float(pnl_part.split(":")[1].strip().replace("%", ""))
                                        break
                
                # If we found at least some strategy info, we can stop looking
                if strategies["arima"]["position"] is not None or strategies["adaptive"]["position"] is not None:
                    break
        
        return strategies
    except Exception as e:
        print(f"Error getting strategy status: {e}")
        return {
            "arima": {"position": None, "entry_price": None, "current_price": None, "unrealized_pnl": None},
            "adaptive": {"position": None, "entry_price": None, "current_price": None, "unrealized_pnl": None}
        }

def display_status():
    """Display current trading status in the terminal"""
    # Get the current state of the trading system
    current_price = get_current_price()
    position_info = get_current_position()
    portfolio_metrics = get_portfolio_metrics()
    portfolio_status = get_portfolio_status()
    strategy_status = get_strategy_status()
    
    # Print the status report
    print("\n" + "="*80)
    print(f"📊 KRAKEN TRADING BOT STATUS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Portfolio Summary
    print("\n💰 PORTFOLIO SUMMARY:")
    if portfolio_status:
        print(f"  Initial Value:     {format_currency(portfolio_status['initial_value'])}")
        print(f"  Current Value:     {format_currency(portfolio_status['current_value'])}")
        print(f"  Total P&L:         {format_currency(portfolio_status['pnl_amount'])} ({format_percentage(portfolio_status['pnl_percent'])})")
        print(f"  Allocated Capital: {format_currency(portfolio_status['allocated_amount'])} ({format_percentage(portfolio_status['allocated_percent'])})")
        print(f"  Available Funds:   {format_currency(portfolio_status['available'])}")
    elif portfolio_metrics:
        print(f"  Initial Value:     {format_currency(portfolio_metrics['initial_value'])}")
        print(f"  Current Value:     {format_currency(portfolio_metrics['current_value'])}")
        print(f"  Total P&L:         {format_currency(portfolio_metrics['total_profit_loss'])} ({format_percentage(portfolio_metrics['total_profit_loss_percent'])})")
    
    # Trade Performance
    print("\n📈 TRADE PERFORMANCE:")
    if portfolio_metrics:
        total_trades = portfolio_metrics['winning_trades'] + portfolio_metrics['losing_trades']
        print(f"  Total Trades:      {total_trades}")
        print(f"  Winning Trades:    {portfolio_metrics['winning_trades']} ({format_percentage(portfolio_metrics['win_rate'])})")
        print(f"  Losing Trades:     {portfolio_metrics['losing_trades']} ({format_percentage(100 - portfolio_metrics['win_rate'])})")
        
        # Strategy Performance
        print("\n🤖 STRATEGY PERFORMANCE:")
        
        # ARIMA Strategy
        arima_perf = portfolio_metrics['strategy_performance']['arima']
        arima_total = arima_perf['winning_trades'] + arima_perf['losing_trades']
        print(f"  ARIMA Strategy:")
        print(f"    Total Trades:    {arima_total}")
        print(f"    P&L:             {format_currency(arima_perf['profit_loss'])} ({format_percentage(arima_perf['profit_loss_percent'])})")
        if arima_total > 0:
            print(f"    Win Rate:        {format_percentage(arima_perf['win_rate'])}")
        
        # Adaptive Strategy
        adaptive_perf = portfolio_metrics['strategy_performance']['adaptive']
        adaptive_total = adaptive_perf['winning_trades'] + adaptive_perf['losing_trades']
        print(f"  Adaptive Strategy:")
        print(f"    Total Trades:    {adaptive_total}")
        print(f"    P&L:             {format_currency(adaptive_perf['profit_loss'])} ({format_percentage(adaptive_perf['profit_loss_percent'])})")
        if adaptive_total > 0:
            print(f"    Win Rate:        {format_percentage(adaptive_perf['win_rate'])}")
    
    # Current Market Data
    print("\n📉 MARKET DATA:")
    print(f"  Current Price (SOL/USD): {format_currency(current_price) if current_price else 'N/A'}")
    
    # Current Positions
    print("\n📊 CURRENT POSITIONS:")
    
    # ARIMA Strategy Position
    arima_position = strategy_status.get("arima", {})
    position_type = arima_position.get("position", "N/A")
    print(f"  ARIMA Strategy: {position_type}")
    if position_type not in ["No Position", "N/A", None]:
        entry_price = arima_position.get("entry_price", "N/A")
        current_price = arima_position.get("current_price", "N/A")
        unrealized_pnl = arima_position.get("unrealized_pnl", "N/A")
        
        if entry_price != "N/A":
            print(f"    Entry Price:     {format_currency(entry_price)}")
        if current_price != "N/A":
            print(f"    Current Price:   {format_currency(current_price)}")
        if unrealized_pnl != "N/A":
            print(f"    Unrealized P&L:  {format_percentage(unrealized_pnl)}")
    
    # Adaptive Strategy Position
    adaptive_position = strategy_status.get("adaptive", {})
    position_type = adaptive_position.get("position", "N/A")
    print(f"  Adaptive Strategy: {position_type}")
    if position_type not in ["No Position", "N/A", None]:
        entry_price = adaptive_position.get("entry_price", "N/A")
        current_price = adaptive_position.get("current_price", "N/A")
        unrealized_pnl = adaptive_position.get("unrealized_pnl", "N/A")
        
        if entry_price != "N/A":
            print(f"    Entry Price:     {format_currency(entry_price)}")
        if current_price != "N/A":
            print(f"    Current Price:   {format_currency(current_price)}")
        if unrealized_pnl != "N/A":
            print(f"    Unrealized P&L:  {format_percentage(unrealized_pnl)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    display_status()