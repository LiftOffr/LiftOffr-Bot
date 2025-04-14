#!/usr/bin/env python3
"""
Run Optimized ML Trading Bot

This script provides a convenient wrapper to run the ML-enhanced trading bot
with optimized settings for production use.
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_optimized_ml_trading.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run optimized ML trading bot')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                       help='Assets to trade (default: SOL/USD ETH/USD BTC/USD)')
    
    parser.add_argument('--live', action='store_true',
                       help='Enable live trading (use with caution)')
    
    parser.add_argument('--reset', action='store_true',
                       help='Reset portfolio before trading')
    
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization before trading')
    
    parser.add_argument('--capital', type=float, default=20000.0,
                       help='Initial capital in USD (default: 20000.0)')
    
    parser.add_argument('--interval', type=int, default=60,
                       help='Trading interval in seconds (default: 60)')
    
    parser.add_argument('--iterations', type=int, default=None,
                       help='Maximum number of trading iterations')
    
    return parser.parse_args()

def build_command(args):
    """Build command to run start_ml_trading.py"""
    cmd = [sys.executable, "start_ml_trading.py"]
    
    # Add assets
    cmd.extend(["--assets"] + args.assets)
    
    # Add options
    if args.live:
        cmd.append("--live")
    
    if args.reset:
        cmd.append("--reset-portfolio")
    
    if args.optimize:
        cmd.append("--optimize-first")
    
    # Add parameters
    cmd.extend(["--initial-capital", str(args.capital)])
    cmd.extend(["--interval", str(args.interval)])
    
    if args.iterations:
        cmd.extend(["--max-iterations", str(args.iterations)])
    
    # Always use extreme leverage and ML position sizing
    cmd.append("--extreme-leverage")
    cmd.append("--ml-position-sizing")
    
    return cmd

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print(f"OPTIMIZED ML TRADING BOT - {'LIVE' if args.live else 'SANDBOX'} MODE")
    print("=" * 80)
    print(f"Trading assets: {args.assets}")
    print(f"Initial capital: ${args.capital:.2f}")
    print(f"Trading interval: {args.interval} seconds")
    print(f"Reset portfolio: {args.reset}")
    print(f"Run optimization: {args.optimize}")
    if args.iterations:
        print(f"Maximum iterations: {args.iterations}")
    print("=" * 80)
    
    # Check for live mode and warn user
    if args.live:
        confirm = input("WARNING: You are about to start LIVE trading with EXTREME leverage. "
                       "Type 'I UNDERSTAND THE RISKS' to proceed: ")
        if confirm != "I UNDERSTAND THE RISKS":
            print("Live trading not confirmed. Exiting.")
            return
    
    # Build command
    cmd = build_command(args)
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        process.wait()
        
        # Check return code
        if process.returncode == 0:
            logger.info("Trading completed successfully")
        else:
            logger.error(f"Trading failed with return code {process.returncode}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        # Try to terminate the process gracefully
        process.terminate()
        
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")

if __name__ == "__main__":
    main()