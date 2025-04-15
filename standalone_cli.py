#!/usr/bin/env python3
"""
Standalone CLI application for trading

This contains NO Flask or web-related imports
"""
import sys
import time
import json
import os
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_random_data():
    """Generate random trading data for simulation"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": timestamp,
        "price": round(random.uniform(20000, 40000), 2),
        "volume": round(random.uniform(1, 10), 4),
        "side": random.choice(["buy", "sell"])
    }

def main():
    """Main CLI function"""
    print("=" * 60)
    print("KRAKEN TRADING BOT CLI - SANDBOX MODE")
    print("=" * 60)
    print("\nThis is a standalone CLI application with NO Flask dependencies")
    print("Press Ctrl+C to exit\n")
    
    # Create data dir if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    try:
        counter = 0
        while True:
            # Generate some random data
            data = generate_random_data()
            
            # Print to console
            print(f"[{data['timestamp']}] Price: ${data['price']} | Volume: {data['volume']} | Side: {data['side']}")
            
            # Save periodically to a file
            if counter % 5 == 0:
                with open("data/cli_output.json", "w") as f:
                    json.dump({"last_update": data['timestamp'], "data": data}, f, indent=2)
                logger.info(f"Updated data file at {counter} iterations")
            
            counter += 1
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nExiting CLI application")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())