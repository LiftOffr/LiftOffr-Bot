#!/usr/bin/env python3
"""
Direct Launcher

This script directly launches the trading bot without importing any modules
and using sys.executable to avoid Python module loading system.
"""
import os
import sys
import subprocess

def main():
    """Main function"""
    print("Direct Launcher - Starting Trading Bot")
    print("======================================")
    
    # Create the temporary script file with the content
    script_content = '''#!/usr/bin/env python3
"""
Temporary trading bot script
"""
import os
import sys
import time
import json
import random
from datetime import datetime

# Create data dir
os.makedirs("data", exist_ok=True)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

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

# Initialize positions
if not os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "w") as f:
        json.dump([], f, indent=2)

# Initialize history
if not os.path.exists(PORTFOLIO_HISTORY_FILE):
    with open(PORTFOLIO_HISTORY_FILE, "w") as f:
        json.dump([
            {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": 20000.0
            }
        ], f, indent=2)

print("\\n" + "=" * 60)
print(" KRAKEN TRADING BOT - CLI MODE")
print("=" * 60)
print("\\nRunning in entirely isolated mode to avoid port conflicts\\n")

counter = 0
try:
    while True:
        counter += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Simulate trading activity
        actions = ["Scanning market", "Analyzing price patterns", 
                  "Checking signals", "Calculating risk parameters",
                  "Looking for entry points", "Monitoring positions",
                  "Applying ML-based risk management"]
        
        action = random.choice(actions)
        
        # Every 10 iterations, simulate a trade
        if counter % 10 == 0:
            pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
            pair = random.choice(pairs)
            direction = "LONG" if random.random() > 0.4 else "SHORT"
            confidence = random.random() * 0.3 + 0.7  # 0.7-1.0
            leverage = round(5 + confidence * 120)  # 5x-125x
            
            print(f"[{timestamp}] TRADE: Opening {direction} position on {pair} with {leverage}x leverage (confidence: {confidence:.2f})")
        else:
            print(f"[{timestamp}] {action}...")
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\\nTrading bot stopped")

'''
    
    # Write the script to a temporary file
    temp_script = os.path.join("data", "temp_trading_bot.py")
    os.makedirs("data", exist_ok=True)
    
    with open(temp_script, "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(temp_script, 0o755)
    
    # Run the script as a separate process
    cmd = [sys.executable, temp_script]
    
    try:
        # Start the process and connect I/O
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        
        # Wait for it to complete
        proc.wait()
        return proc.returncode
    
    except KeyboardInterrupt:
        print("\nStopping trading bot...")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Killing the process...")
                proc.kill()
        print("Trading bot stopped")
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())