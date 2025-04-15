#!/usr/bin/env python3
"""
Completely isolated trading bot runner

This script runs the standalone trading bot in a completely separate process
without importing any potentially conflicting modules.
"""
import os
import sys
import subprocess
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading bot")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=None,
                        help="Trading pairs to use (default: all supported pairs)")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Build command
    cmd = [sys.executable, "standalone_trading_bot.py", "--sandbox"]
    
    # Add pairs if specified
    if args.pairs:
        for pair in args.pairs:
            cmd.append("--pairs")
            cmd.append(pair)
    
    print(f"Starting trading bot with command: {' '.join(cmd)}")
    
    try:
        # Run the bot directly without importing it
        subprocess.run(cmd, check=True)
        return 0
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
        return 0
    except Exception as e:
        print(f"Error running trading bot: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())