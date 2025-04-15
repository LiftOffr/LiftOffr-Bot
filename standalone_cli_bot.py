#!/usr/bin/env python3
"""
Standalone CLI Trading Bot

This script implements a standalone CLI trading bot without any Flask,
servers, or imports that could conflict with other components.
"""
import os
import sys
import json
import time
import random
from datetime import datetime

# Create data directory if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Constants for file paths
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

# Trading pairs
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                 "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]

# Initialize portfolio
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump({
            "balance": 20000.0,
            "equity": 20000.0,
            "total_value": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
else:
    # Read existing portfolio
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
        print(f"\nLoaded portfolio with balance: ${portfolio.get('balance', 0):.2f}")

# Initialize positions
if not os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "w") as f:
        json.dump([], f, indent=2)

# Initialize portfolio history
if not os.path.exists(PORTFOLIO_HISTORY_FILE):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump([
            {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": 20000.0
            }
        ], f, indent=2)

# Initialize trades
if not os.path.exists(TRADES_FILE):
    with open(TRADES_FILE, "w") as f:
        json.dump([], f, indent=2)

def get_current_prices():
    """Simulate getting current prices"""
    # In a real implementation, this would call the Kraken API
    return {
        "BTC/USD": random.uniform(55000, 65000),
        "ETH/USD": random.uniform(2800, 3200),
        "SOL/USD": random.uniform(140, 160),
        "ADA/USD": random.uniform(0.4, 0.5),
        "DOT/USD": random.uniform(6, 7),
        "LINK/USD": random.uniform(13, 15),
        "AVAX/USD": random.uniform(30, 35),
        "MATIC/USD": random.uniform(0.6, 0.7),
        "UNI/USD": random.uniform(9, 11),
        "ATOM/USD": random.uniform(7, 9)
    }

def update_portfolio(prices):
    """Update portfolio with current positions and prices"""
    try:
        # Load portfolio
        with open(PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)
        
        # Load positions
        with open(POSITIONS_FILE, "r") as f:
            positions = json.load(f)
        
        # Calculate unrealized P&L
        total_unrealized_pnl = 0.0
        for position in positions:
            symbol = position.get("symbol")
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            side = position.get("side", "long")
            leverage = position.get("leverage", 1)
            
            # Calculate P&L
            price_diff = current_price - entry_price
            if side.lower() == "short":
                price_diff = -price_diff
                
            pnl = price_diff * size * leverage
            position["unrealized_pnl"] = pnl
            position["current_price"] = current_price
            position["last_updated"] = datetime.now().isoformat()
            
            total_unrealized_pnl += pnl
        
        # Update portfolio
        equity = portfolio["balance"] + total_unrealized_pnl
        portfolio["equity"] = equity
        portfolio["total_value"] = equity
        portfolio["unrealized_pnl_usd"] = total_unrealized_pnl
        
        # Calculate percentage P&L if balance is not zero
        if portfolio["balance"] > 0:
            portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / portfolio["balance"]) * 100
        else:
            portfolio["unrealized_pnl_pct"] = 0
            
        portfolio["last_updated"] = datetime.now().isoformat()
        
        # Save updated portfolio
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
            
        # Save updated positions
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
            
        # Add to portfolio history
        with open(PORTFOLIO_HISTORY_FILE, "r") as f:
            history = json.load(f)
            
        history.append({
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": equity
        })
        
        with open(PORTFOLIO_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error updating portfolio: {e}")
        return False

def simulate_trade(current_prices):
    """Simulate a trade with dynamic ML-based risk parameters"""
    # Randomly choose a pair
    pair = random.choice(TRADING_PAIRS)
    
    # Determine direction (biased toward long)
    side = "long" if random.random() > 0.35 else "short"
    
    # Generate ML confidence (higher values = higher confidence)
    confidence = random.uniform(0.68, 0.96)
    
    # Calculate leverage based on confidence (5x-125x)
    # Higher confidence = higher leverage
    max_leverage = 125
    min_leverage = 5
    leverage = round(min_leverage + confidence * (max_leverage - min_leverage))
    
    # Get current price
    price = current_prices[pair]
    
    # Calculate position size (risk 5-20% of portfolio based on confidence)
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
        
    balance = portfolio.get("balance", 20000)
    min_risk = 0.05  # 5%
    max_risk = 0.20  # 20%
    risk_percentage = min_risk + confidence * (max_risk - min_risk)
    
    # Ensure we don't risk more than 20% total across all positions
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    
    total_risk = sum(p.get("risk_percentage", 0) for p in positions)
    available_risk = max(0, 0.5 - total_risk)  # Limit to 50% total risk
    risk_percentage = min(risk_percentage, available_risk)
    
    # If we don't have risk capacity, cancel the trade
    if risk_percentage <= 0.01:  # Minimum 1% risk
        return None
        
    # Calculate position size
    position_value = balance * risk_percentage
    units = position_value / price
    
    # Calculate stop loss based on ATR-like value (simulated)
    atr_percent = random.uniform(0.01, 0.04)  # 1-4% volatility
    max_loss_percent = 0.04  # 4% maximum loss regardless of ATR
    stop_loss_percent = min(atr_percent * 1.5, max_loss_percent)
    
    # Calculate stop loss price
    if side == "long":
        stop_price = price * (1 - stop_loss_percent / leverage)
    else:
        stop_price = price * (1 + stop_loss_percent / leverage)
    
    # Calculate take profit (2-3x the risk)
    risk_reward = random.uniform(2.0, 3.0)
    if side == "long":
        take_profit = price * (1 + (stop_loss_percent * risk_reward) / leverage)
    else:
        take_profit = price * (1 - (stop_loss_percent * risk_reward) / leverage)
    
    # Create trade object
    trade = {
        "symbol": pair,
        "side": side,
        "entry_price": price,
        "size": units,
        "value": position_value,
        "leverage": leverage,
        "stop_loss": stop_price,
        "take_profit": take_profit,
        "confidence": confidence,
        "risk_percentage": risk_percentage,
        "timestamp": datetime.now().isoformat(),
        "status": "open"
    }
    
    # Add position
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    
    positions.append({
        "symbol": pair,
        "side": side,
        "entry_price": price,
        "current_price": price,
        "size": units,
        "leverage": leverage,
        "unrealized_pnl": 0,
        "risk_percentage": risk_percentage,
        "confidence": confidence,
        "stop_loss": stop_price,
        "take_profit": take_profit,
        "timestamp": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    })
    
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)
    
    # Add to trades
    with open(TRADES_FILE, "r") as f:
        trades = json.load(f)
    
    trades.append(trade)
    
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)
    
    return trade

def check_for_exits(current_prices):
    """Check if any positions should be closed"""
    # Load positions
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    
    if not positions:
        return []
        
    closed_positions = []
    remaining_positions = []
    
    for position in positions:
        symbol = position.get("symbol")
        if symbol not in current_prices:
            remaining_positions.append(position)
            continue
            
        current_price = current_prices[symbol]
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        side = position.get("side", "long").lower()
        
        # Check if stop loss or take profit hit
        stopped_out = False
        profit_taken = False
        
        if side == "long":
            if current_price <= stop_loss:
                stopped_out = True
            elif current_price >= take_profit:
                profit_taken = True
        else:  # short
            if current_price >= stop_loss:
                stopped_out = True
            elif current_price <= take_profit:
                profit_taken = True
        
        # Random chance to exit for other signals (simulated ML signal change)
        other_signal = random.random() < 0.02  # 2% chance per check
        
        if stopped_out or profit_taken or other_signal:
            # Calculate P&L
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            leverage = position.get("leverage", 1)
            
            # Calculate price difference
            price_diff = current_price - entry_price
            if side == "short":
                price_diff = -price_diff
                
            pnl = price_diff * size * leverage
            
            # Close position
            position["exit_price"] = current_price
            position["realized_pnl"] = pnl
            position["exit_reason"] = "stop_loss" if stopped_out else "take_profit" if profit_taken else "signal_change"
            position["exit_time"] = datetime.now().isoformat()
            
            closed_positions.append(position)
            
            # Update trades record
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
            
            # Find matching trade
            for trade in trades:
                if (trade.get("symbol") == symbol and 
                    trade.get("entry_price") == entry_price and
                    trade.get("size") == size and
                    trade.get("status") == "open"):
                    
                    trade["exit_price"] = current_price
                    trade["realized_pnl"] = pnl
                    trade["exit_reason"] = position["exit_reason"]
                    trade["exit_time"] = position["exit_time"]
                    trade["status"] = "closed"
                    
                    # Calculate percentage gain/loss
                    trade["pnl_percentage"] = (pnl / trade.get("value", 1)) * 100
                    break
            
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)
            
            # Update portfolio balance
            with open(PORTFOLIO_FILE, "r") as f:
                portfolio = json.load(f)
            
            portfolio["balance"] += pnl
            
            with open(PORTFOLIO_FILE, "w") as f:
                json.dump(portfolio, f, indent=2)
        else:
            remaining_positions.append(position)
    
    # Update positions file with remaining positions
    with open(POSITIONS_FILE, "w") as f:
        json.dump(remaining_positions, f, indent=2)
    
    return closed_positions

def print_portfolio_status():
    """Print current portfolio status"""
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)
        
        with open(POSITIONS_FILE, "r") as f:
            positions = json.load(f)
        
        balance = portfolio.get("balance", 0)
        equity = portfolio.get("equity", 0)
        unrealized_pnl = portfolio.get("unrealized_pnl_usd", 0)
        unrealized_pnl_pct = portfolio.get("unrealized_pnl_pct", 0)
        
        print("\n" + "=" * 50)
        print(f"PORTFOLIO STATUS: ${equity:.2f}")
        print("=" * 50)
        print(f"Balance: ${balance:.2f}")
        print(f"Unrealized P&L: ${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)")
        print(f"Open Positions: {len(positions)}")
        
        if positions:
            print("\nOPEN POSITIONS:")
            print("-" * 50)
            for pos in positions:
                symbol = pos.get("symbol", "UNKNOWN")
                side = pos.get("side", "long").upper()
                entry = pos.get("entry_price", 0)
                current = pos.get("current_price", 0)
                pnl = pos.get("unrealized_pnl", 0)
                lev = pos.get("leverage", 1)
                
                if side == "LONG":
                    pct_change = ((current - entry) / entry) * 100
                else:
                    pct_change = ((entry - current) / entry) * 100
                
                print(f"{symbol} {side} {lev}x: ${pnl:.2f} ({pct_change:.2f}%)")
        
        print("-" * 50)
    except Exception as e:
        print(f"Error printing status: {e}")

# Main function to run the trading bot
def main():
    print("\n" + "=" * 60)
    print(" KRAKEN TRADING BOT - STANDALONE MODE")
    print("=" * 60)
    print("\nRunning in entirely isolated process\n")
    
    update_interval = 5  # seconds
    trade_interval = 300  # 5 minutes between potential new trades
    stats_interval = 60  # 1 minute between status updates
    
    last_trade_time = 0
    last_stats_time = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Get current prices
            prices = get_current_prices()
            
            # Update portfolio and positions
            update_portfolio(prices)
            
            # Check for position exits
            closed_positions = check_for_exits(prices)
            
            # Log closed positions
            for position in closed_positions:
                symbol = position.get("symbol", "UNKNOWN")
                side = position.get("side", "long").upper()
                pnl = position.get("realized_pnl", 0)
                reason = position.get("exit_reason", "unknown")
                
                print(f"CLOSED: {symbol} {side} with P&L ${pnl:.2f} - Reason: {reason}")
            
            # Potentially enter new trades
            if current_time - last_trade_time >= trade_interval:
                # Don't always trade (simulate ML only finding good opportunities sometimes)
                if random.random() < 0.3:  # 30% chance of finding a trade
                    new_trade = simulate_trade(prices)
                    if new_trade:
                        symbol = new_trade.get("symbol")
                        side = new_trade.get("side", "").upper()
                        conf = new_trade.get("confidence", 0) * 100
                        lev = new_trade.get("leverage", 1)
                        
                        print(f"NEW TRADE: {symbol} {side} - Confidence: {conf:.1f}% - Leverage: {lev}x")
                
                last_trade_time = current_time
            
            # Print status at regular intervals
            if current_time - last_stats_time >= stats_interval:
                print_portfolio_status()
                last_stats_time = current_time
            
            # Sleep until next update
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        print(f"Error in trading bot: {e}")
    
    print("\n" + "=" * 60)
    print(" TRADING BOT SHUTDOWN COMPLETE")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    # This should prevent any Flask import conflicts since we're directly in __main__
    # without importing any external modules that might import Flask
    sys.exit(main())