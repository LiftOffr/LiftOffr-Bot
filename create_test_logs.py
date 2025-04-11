#!/usr/bin/env python3

import os
import datetime

def create_test_logs():
    """Create test log file for demonstrating the status command"""
    
    # Create the trade notifications log file
    with open("trade_notifications.log", "w") as f:
        # Add some sample data with timestamps
        now = datetime.datetime.now()
        
        # Add ticker data
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【TICKER】 SOL/USD = $112.78 | Bid: $112.77 | Ask: $112.78\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【TICKER】 SOL/USD = $112.78 | Bid: $112.77 | Ask: $112.78\n")
        
        # Add signal data
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] ============================================================\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【ANALYSIS】 Forecast: BULLISH | Current: $112.78 → Target: $112.92\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【INDICATORS】 EMA9 >= EMA21 | RSI = 58.12 ✓ | MACD > Signal | ADX = 27.35 ✓\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【VOLATILITY】 Volatility = 0.0012 ✓ (threshold: 0.006)\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【BANDS】 EMA9 >= EMA21 (112.64 vs 112.52) | RSI = 58.12 ✓ | MACD > Signal (0.0010 vs 0.0005) | ADX = 27.35 ✓ | Volatility = 0.0012 ✓ (threshold: 0.006) | Price < Upper BB (112.78 vs 113.05) | Price > Lower BB (112.78 vs 112.15) | Price vs KC Middle: 112.78 vs 112.52\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【SIGNAL】 🟢 BULLISH - Trade conditions met for LONG position\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] ============================================================\n")
        
        # Add action data
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【ACTION】 🟢 BUY | ATR: $0.1423 | Volatility Stop: $112.35\n")
        
        # Add position data
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【ORDER】 Placing BUY order at $112.78 (Size: 8.86 SOL | $1000.00)\n")
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【POSITION】 Size: 8.86 | Entry: $112.78 | Current: $112.78 | P&L: $0.00 (0.00%)\n")
        
        # Add signal data for more recent analysis (slightly bearish)
        five_min_later = now + datetime.timedelta(minutes=5)
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] ============================================================\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【ANALYSIS】 Forecast: NEUTRAL | Current: $112.92 → Target: $112.89\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【INDICATORS】 EMA9 >= EMA21 | RSI = 59.43 ✓ | MACD < Signal | ADX = 27.12 ✓\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【VOLATILITY】 Volatility = 0.0013 ✓ (threshold: 0.006)\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【BANDS】 EMA9 >= EMA21 (112.70 vs 112.55) | RSI = 59.43 ✓ | MACD < Signal (0.0008 vs 0.0009) | ADX = 27.12 ✓ | Volatility = 0.0013 ✓ (threshold: 0.006) | Price < Upper BB (112.92 vs 113.10) | Price > Lower BB (112.92 vs 112.20) | Price vs KC Middle: 112.92 vs 112.55\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【SIGNAL】 ⚪ NEUTRAL - No clear trade signal detected\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] ============================================================\n")
        
        # Add updated position data
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【ACTION】 ⚪ HOLD | ATR: $0.1442 | Volatility Stop: $112.40\n")
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【POSITION】 Size: 8.86 | Entry: $112.78 | Current: $112.92 | P&L: $1.24 (+0.12%)\n")
        
        # Add ticker update
        f.write(f"{five_min_later.strftime('%Y-%m-%d %H:%M:%S')} [INFO] 【TICKER】 SOL/USD = $112.92 | Bid: $112.91 | Ask: $112.92\n")
    
    # Create a CSV file with sample trade data
    with open("trades.csv", "w") as f:
        # Write header
        f.write("time,type,pair,price,volume,cost,fee,txid\n")
        
        # Add sample trade records
        yesterday = now - datetime.timedelta(days=1)
        two_days_ago = now - datetime.timedelta(days=2)
        
        # Previous trades (buy then sell)
        f.write(f"{two_days_ago.timestamp()},buy,SOL/USD,111.25,8.99,999.94,1.60,txid123\n")
        f.write(f"{yesterday.timestamp()},sell,SOL/USD,112.38,8.99,1010.28,1.62,txid124\n")
        
        # Most recent trade (buy)
        f.write(f"{now.timestamp()},buy,SOL/USD,112.78,8.86,999.63,1.60,txid125\n")
    
    print("Test log files created successfully")

if __name__ == "__main__":
    create_test_logs()