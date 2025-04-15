#!/usr/bin/env python3
"""
Tiny Trading Bot

This is an extremely minimal trading bot that avoids all dependencies.
It's designed to just log activity without actually importing any modules
that might conflict with the Flask application.
"""
import os
import sys
import time
import json
import random
from datetime import datetime

print("\n" + "=" * 60)
print(" TINY TRADING BOT")
print("=" * 60)
print("\nThis bot is completely isolated from Flask dependencies")

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Portfolio file paths
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

# Initialize portfolio if it doesn't exist
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump({
            "balance": 20000.0,
            "equity": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    print("\nCreated new portfolio with $20,000 balance")
else:
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
    print(f"\nLoaded portfolio with ${portfolio.get('balance', 0):.2f} balance")

# Define trading pairs
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
         "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]

# Simulate bot activity
print("\nBot is now running. Press Ctrl+C to stop.\n")

update_count = 0
try:
    while True:
        update_count += 1
        
        # Only print on every 5th update to reduce output
        if update_count % 5 == 0:
            # Current time
            now = datetime.now().strftime("%H:%M:%S")
            pair = random.choice(PAIRS)
            price = round(random.uniform(100, 60000), 2)
            
            # Randomly choose activity to simulate
            activities = [
                "Scanning market data",
                "Analyzing price patterns",
                f"Checking signal strength for {pair}",
                f"Monitoring {pair} trend",
                "Running ML prediction model",
                "Calculating optimal leverage",
                "Evaluating entry points",
                "Checking for position exits",
                "Updating portfolio metrics",
                "Applying risk management rules"
            ]
            activity = random.choice(activities)
            
            # Print activity
            print(f"[{now}] {activity}")
            
            # Occasionally simulate a new trade
            if random.random() < 0.05:  # 5% chance per update
                side = "LONG" if random.random() > 0.35 else "SHORT"
                confidence = round(random.uniform(0.7, 0.95), 2)
                leverage = round(5 + 120 * confidence)
                
                print(f"[{now}] NEW TRADE: {pair} {side} @ ${price} - "
                      f"Confidence: {confidence:.2f} - Leverage: {leverage}x")
        
        # Sleep between updates
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nBot stopped by user")

print("\n" + "=" * 60)
print(" BOT SHUTDOWN COMPLETE")
print("=" * 60)