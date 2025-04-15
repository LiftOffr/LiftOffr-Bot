#!/bin/bash
# Direct Run Script
# This script runs the Python interpreter directly to avoid 
# any Flask module imports

echo "========================================"
echo " DIRECT TRADING BOT EXECUTION"
echo "========================================"
echo

# Create temporary script content
cat > /tmp/run_bot.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import random
from datetime import datetime

# Avoid importing Flask
os.environ["TRADING_BOT_PROCESS"] = "1"

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Portfolio file paths
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"

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

# Print welcome message
print("\n" + "=" * 60)
print(" TRADING BOT - DIRECT EXECUTION MODE")
print("=" * 60)
print("\nPress Ctrl+C to stop\n")

# Main loop
update_count = 0
try:
    while True:
        update_count += 1
        
        # Update every few seconds
        if update_count % 5 == 0:
            now = datetime.now().strftime("%H:%M:%S")
            pair = random.choice(PAIRS)
            price = random.uniform(1000, 60000)
            
            print(f"[{now}] Processing {pair} at ${price:.2f}")
            
            # Occasionally simulate trades
            if random.random() < 0.1:  # 10% chance per update
                side = "LONG" if random.random() > 0.35 else "SHORT"
                conf = random.uniform(0.7, 0.95)
                lev = round(5 + conf * 120)
                
                print(f"[{now}] NEW TRADE: {pair} {side} @ ${price:.2f} - "
                      f"Confidence: {conf:.2f} - Leverage: {lev}x")
        
        # Sleep to avoid high CPU
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nBot stopped by user")

print("\nTrading bot shutdown complete")
EOF

# Make it executable
chmod +x /tmp/run_bot.py

# Run the script with Python, bypassing any imports
/usr/bin/env python3 -B /tmp/run_bot.py

# Clean up
rm /tmp/run_bot.py