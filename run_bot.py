#!/usr/bin/env python3
"""
Run the trading bot in a separate process from the Flask application.

This script serves as a completely standalone entry point for the trading bot
that doesn't depend on or conflict with Flask.
"""
import os
import sys
import time
import logging
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading bot")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                                 "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"],
                        help="Trading pairs to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" KRAKEN TRADING BOT WITH ML-BASED RISK MANAGEMENT")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check for Kraken API credentials
    if not args.sandbox and (not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET")):
        logger.warning("Kraken API credentials not found, forcing sandbox mode")
        args.sandbox = True
    
    # Start trading bot in a separate process
    print(f"\nStarting trading bot in {'sandbox' if args.sandbox else 'live'} mode")
    print(f"Trading pairs: {', '.join(args.pairs)}")
    print("\nPress Ctrl+C to stop the bot at any time\n")
    
    # Build command and environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["NO_FLASK"] = "1"  # Set a special flag to prevent Flask imports
    
    # Use a process isolation runner that doesn't import Flask
    cmd = [
        sys.executable,
        "non_flask_trading_bot.py",
        "--sandbox" if args.sandbox else ""
    ]
    
    # Add pairs if provided
    if args.pairs:
        cmd.append("--pairs")
        cmd.extend(args.pairs)
    
    process = None
    try:
        # Run the bot subprocess
        process = subprocess.Popen(
            [arg for arg in cmd if arg],  # Filter out empty strings
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping trading bot...")
        if process:
            process.terminate()
            process.wait()
        print("Trading bot stopped")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())