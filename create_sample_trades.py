#!/usr/bin/env python3

import os
import datetime
import csv

def create_sample_trades():
    """Create sample trades data to demonstrate trading bot functionality"""
    
    # Get current time for reference
    now = datetime.datetime.now()
    
    # Create a CSV file with sample trade data
    with open("trades.csv", "w") as f:
        # Write header
        f.write("time,type,pair,price,volume,cost,fee,txid\n")
        
        # Sample trades
        
        # Trade 1: Winning long trade (completed)
        five_days_ago = now - datetime.timedelta(days=5)
        four_days_ago = now - datetime.timedelta(days=4)
        f.write(f"{five_days_ago.timestamp()},buy,SOL/USD,109.75,9.11,999.82,1.60,txid001\n")
        f.write(f"{four_days_ago.timestamp()},sell,SOL/USD,111.38,9.11,1014.67,1.62,txid002\n")
        
        # Trade 2: Losing long trade (completed)
        three_days_ago = now - datetime.timedelta(days=3)
        two_days_ago = now - datetime.timedelta(days=2, hours=12)
        f.write(f"{three_days_ago.timestamp()},buy,SOL/USD,112.50,8.89,999.92,1.60,txid003\n")
        f.write(f"{two_days_ago.timestamp()},sell,SOL/USD,111.60,8.89,992.12,1.59,txid004\n")
        
        # Trade 3: Winning short trade (completed)
        two_days_ago = now - datetime.timedelta(days=2)
        yesterday = now - datetime.timedelta(days=1)
        f.write(f"{two_days_ago.timestamp()},sell,SOL/USD,114.25,8.75,999.69,1.60,txid005\n")
        f.write(f"{yesterday.timestamp()},buy,SOL/USD,113.10,8.75,989.63,1.58,txid006\n")
        
        # Trade 4: Current open long position
        today = now - datetime.timedelta(hours=2)
        f.write(f"{today.timestamp()},buy,SOL/USD,112.78,8.86,999.63,1.60,txid007\n")
    
    # Create the trade notifications log file with signals and actions
    with open("trade_notifications.log", "w") as f:
        # Winning long trade actions and signals
        entry_time = five_days_ago.strftime('%Y-%m-%d %H:%M:%S')
        exit_time = four_days_ago.strftime('%Y-%m-%d %H:%M:%S')
        
        f.write(f"{entry_time} [INFO] 【ANALYSIS】 Forecast: BULLISH | Current: $109.70 → Target: $110.25\n")
        f.write(f"{entry_time} [INFO] 【INDICATORS】 EMA9 >= EMA21 | RSI = 58.32 ✓ | MACD > Signal | ADX = 26.80 ✓\n")
        f.write(f"{entry_time} [INFO] 【SIGNAL】 🟢 BULLISH - Trade conditions met for LONG position\n")
        f.write(f"{entry_time} [INFO] 【ACTION】 🟢 BUY | ATR: $0.1650 | Volatility Stop: $109.25\n")
        f.write(f"{entry_time} [INFO] 【ORDER】 Placing BUY order at $109.75 (Size: 9.11 SOL | $999.82)\n")
        f.write(f"{entry_time} [INFO] 【POSITION】 Size: 9.11 | Entry: $109.75 | Current: $109.75 | P&L: $0.00 (0.00%)\n")
        
        # Exit signal
        f.write(f"{exit_time} [INFO] 【ANALYSIS】 Forecast: BEARISH | Current: $111.38 → Target: $110.95\n")
        f.write(f"{exit_time} [INFO] 【SIGNAL】 🔴 BEARISH - Signal reversal detected\n")
        f.write(f"{exit_time} [INFO] 【ACTION】 🔴 SELL | ATR: $0.1580 | Signal Reversal\n")
        f.write(f"{exit_time} [INFO] 【ORDER】 Placing SELL order at $111.38 (Size: 9.11 SOL | $1014.67)\n")
        f.write(f"{exit_time} [INFO] 【POSITION】 Closed | Entry: $109.75 | Exit: $111.38 | P&L: +$14.85 (+1.49%)\n")
        
        # Losing long trade
        entry_time = three_days_ago.strftime('%Y-%m-%d %H:%M:%S')
        exit_time = two_days_ago.strftime('%Y-%m-%d %H:%M:%S')
        
        f.write(f"{entry_time} [INFO] 【ANALYSIS】 Forecast: BULLISH | Current: $112.45 → Target: $113.20\n")
        f.write(f"{entry_time} [INFO] 【INDICATORS】 EMA9 >= EMA21 | RSI = 61.25 ✓ | MACD > Signal | ADX = 28.40 ✓\n")
        f.write(f"{entry_time} [INFO] 【SIGNAL】 🟢 BULLISH - Trade conditions met for LONG position\n")
        f.write(f"{entry_time} [INFO] 【ACTION】 🟢 BUY | ATR: $0.1720 | Volatility Stop: $111.85\n")
        f.write(f"{entry_time} [INFO] 【ORDER】 Placing BUY order at $112.50 (Size: 8.89 SOL | $999.92)\n")
        
        # Exit with trailing stop
        f.write(f"{exit_time} [INFO] 【ACTION】 🔴 SELL | ATR: $0.1690 | Trailing Stop Triggered\n")
        f.write(f"{exit_time} [INFO] 【ORDER】 Placing SELL order at $111.60 (Size: 8.89 SOL | $992.12)\n")
        f.write(f"{exit_time} [INFO] 【POSITION】 Closed | Entry: $112.50 | Exit: $111.60 | P&L: -$7.80 (-0.78%)\n")
        
        # Winning short trade
        entry_time = two_days_ago.strftime('%Y-%m-%d %H:%M:%S')
        exit_time = yesterday.strftime('%Y-%m-%d %H:%M:%S')
        
        f.write(f"{entry_time} [INFO] 【ANALYSIS】 Forecast: BEARISH | Current: $114.25 → Target: $113.40\n")
        f.write(f"{entry_time} [INFO] 【INDICATORS】 EMA9 < EMA21 | RSI = 70.35 ✓ | MACD < Signal | ADX = 29.80 ✓\n")
        f.write(f"{entry_time} [INFO] 【SIGNAL】 🔴 BEARISH - Trade conditions met for SHORT position\n")
        f.write(f"{entry_time} [INFO] 【ACTION】 🔴 SELL | ATR: $0.1840 | Volatility Stop: $114.80\n")
        f.write(f"{entry_time} [INFO] 【ORDER】 Placing SELL order at $114.25 (Size: 8.75 SOL | $999.69)\n")
        
        # Exit with take profit
        f.write(f"{exit_time} [INFO] 【ACTION】 🟢 BUY | ATR: $0.1780 | Take Profit Target Reached\n")
        f.write(f"{exit_time} [INFO] 【ORDER】 Placing BUY order at $113.10 (Size: 8.75 SOL | $989.63)\n")
        f.write(f"{exit_time} [INFO] 【POSITION】 Closed | Entry: $114.25 | Exit: $113.10 | P&L: +$10.06 (+1.01%)\n")
        
        # Current open position
        entry_time = today.strftime('%Y-%m-%d %H:%M:%S')
        current_time = now.strftime('%Y-%m-%d %H:%M:%S')
        
        f.write(f"{entry_time} [INFO] 【ANALYSIS】 Forecast: BULLISH | Current: $112.78 → Target: $113.50\n")
        f.write(f"{entry_time} [INFO] 【INDICATORS】 EMA9 >= EMA21 | RSI = 57.80 ✓ | MACD > Signal | ADX = 26.40 ✓\n")
        f.write(f"{entry_time} [INFO] 【SIGNAL】 🟢 BULLISH - Trade conditions met for LONG position\n")
        f.write(f"{entry_time} [INFO] 【ACTION】 🟢 BUY | ATR: $0.1690 | Volatility Stop: $112.30\n")
        f.write(f"{entry_time} [INFO] 【ORDER】 Placing BUY order at $112.78 (Size: 8.86 SOL | $999.63)\n")
        
        # Current update
        f.write(f"{current_time} [INFO] 【POSITION】 Size: 8.86 | Entry: $112.78 | Current: $112.92 | P&L: +$1.24 (+0.12%)\n")
        f.write(f"{current_time} [INFO] 【ACTION】 ⚪ HOLD | ATR: $0.1705 | Volatility Stop: $112.35\n")
    
    print("Sample trades data created successfully")

if __name__ == "__main__":
    create_sample_trades()