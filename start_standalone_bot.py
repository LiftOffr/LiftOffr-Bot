#!/usr/bin/env python3
"""
Simple script to start the standalone trading bot without Flask dependencies
"""
import subprocess
import sys
import os

def main():
    """Main function that launches the standalone trading bot directly"""
    print("Starting standalone trading bot...")
    
    try:
        # Run the standalone bot directly without any import that might 
        # trigger Flask initialization
        subprocess.run([
            "python", 
            "standalone_trading_bot.py", 
            "--sandbox"
        ], check=True)
        return 0
    except KeyboardInterrupt:
        print("\nStopping trading bot...")
        return 0
    except Exception as e:
        print(f"Error running trading bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())