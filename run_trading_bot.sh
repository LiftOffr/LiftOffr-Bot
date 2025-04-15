#!/bin/bash
# Simple script to run the trading bot in a separate process

echo "=============================================="
echo "  KRAKEN TRADING BOT LAUNCHER (BASH SCRIPT)  "
echo "=============================================="
echo
echo "This launcher completely bypasses Flask"
echo

# Create a temporary isolated environment
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Create the isolated Python script
ISOLATED_SCRIPT="$TEMP_DIR/isolated_bot.py"

cat > "$ISOLATED_SCRIPT" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import random
from datetime import datetime

# Create data directory
os.makedirs("data", exist_ok=True)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

# Initialize portfolio if needed
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump({
            "balance": 20000.0,
            "equity": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)
    print("Created new portfolio with $20,000 initial balance")
else:
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
    print(f"Loaded existing portfolio: ${portfolio.get('balance', 0):.2f}")

# Initialize positions if needed
if not os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "w") as f:
        json.dump([], f, indent=2)
    print("No open positions")
else:
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    print(f"Loaded {len(positions)} open positions")

# Initialize trades if needed
if not os.path.exists(TRADES_FILE):
    with open(TRADES_FILE, "w") as f:
        json.dump([], f, indent=2)
    print("No trade history")
else:
    with open(TRADES_FILE, "r") as f:
        trades = json.load(f)
    print(f"Loaded {len(trades)} historical trades")

# Initialize portfolio history if needed
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([{
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": 20000.0
        }], f, indent=2)
    print("Created new portfolio history")
else:
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    print(f"Loaded portfolio history with {len(history)} data points")

def get_prices():
    """Get simulated current prices"""
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

def print_status():
    """Print portfolio status"""
    with open(PORTFOLIO_FILE, "r") as f:
        portfolio = json.load(f)
    
    with open(POSITIONS_FILE, "r") as f:
        positions = json.load(f)
    
    print("\n" + "=" * 50)
    print(f"PORTFOLIO STATUS: ${portfolio.get('equity', 0):.2f}")
    print("=" * 50)
    print(f"Balance: ${portfolio.get('balance', 0):.2f}")
    print(f"Unrealized P&L: ${portfolio.get('unrealized_pnl_usd', 0):.2f}")
    print(f"Open Positions: {len(positions)}")
    
    if positions:
        for pos in positions:
            symbol = pos.get("symbol", "Unknown")
            side = pos.get("side", "long").upper()
            pnl = pos.get("unrealized_pnl", 0)
            leverage = pos.get("leverage", 1)
            print(f"- {symbol} {side} {leverage}x: ${pnl:.2f}")
    
    print("-" * 50)

# Main trading loop
print("\n" + "=" * 60)
print(" KRAKEN TRADING BOT - ISOLATED MODE")
print("=" * 60)
print("\nPress Ctrl+C to stop the bot")

update_interval = 5  # seconds
status_interval = 60  # 1 minute
trade_interval = 300  # 5 minutes

last_update = 0
last_status = 0
last_trade = 0

try:
    while True:
        now = time.time()
        
        # Update process
        if now - last_update >= update_interval:
            prices = get_prices()
            # Simulating bot activity
            print(".", end="", flush=True)
            last_update = now
        
        # Print status
        if now - last_status >= status_interval:
            print_status()
            last_status = now
        
        # Sleep to avoid high CPU usage
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nBot stopped by user")
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 60)
print(" TRADING BOT SHUTDOWN")
print("=" * 60)
EOF

# Make the script executable
chmod +x "$ISOLATED_SCRIPT"

# Run the script in its own Python interpreter
echo "Launching isolated trading bot..."
echo

python3 "$ISOLATED_SCRIPT"

# Clean up
echo "Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"
echo "Done."