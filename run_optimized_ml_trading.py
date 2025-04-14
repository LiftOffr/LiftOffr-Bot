#!/usr/bin/env python3
"""
Run Optimized ML Trading

This script starts the ML-optimized trading bot with all available
ML enhancements and optimizations.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimized_ml_trading.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]
DEFAULT_CAPITAL = 20000.0
CONFIG_PATH = "ml_config.json"
ML_INTEGRATION_SCRIPT = "ml_live_trading_integration.py"
BOT_MANAGER_SCRIPT = "bot_manager_integration.py"

def load_config() -> Dict[str, Any]:
    """Load the ML configuration from file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration from {CONFIG_PATH}")
            return config
        else:
            logger.warning(f"Configuration file {CONFIG_PATH} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def get_capital_allocation(config: Dict[str, Any], assets: List[str]) -> Dict[str, float]:
    """Get capital allocation for the specified assets"""
    try:
        if not config or "global_settings" not in config:
            # Default allocation: equal distribution
            equal_share = 1.0 / len(assets)
            return {asset: equal_share for asset in assets}
        
        if "default_capital_allocation" not in config["global_settings"]:
            # Default allocation: equal distribution
            equal_share = 1.0 / len(assets)
            return {asset: equal_share for asset in assets}
        
        # Get allocation from config
        allocation = config["global_settings"]["default_capital_allocation"]
        
        # Filter to include only the specified assets
        filtered_allocation = {asset: allocation.get(asset, 0.0) for asset in assets if asset in allocation}
        
        # If no allocations were found, use equal distribution
        if not filtered_allocation or sum(filtered_allocation.values()) == 0:
            equal_share = 1.0 / len(assets)
            return {asset: equal_share for asset in assets}
        
        # Normalize allocations to sum to 1.0
        total = sum(filtered_allocation.values())
        return {asset: value / total for asset, value in filtered_allocation.items()}
    
    except Exception as e:
        logger.error(f"Error getting capital allocation: {e}")
        # Default allocation: equal distribution
        equal_share = 1.0 / len(assets)
        return {asset: equal_share for asset in assets}

def start_trading_bot(
    assets: List[str], 
    live: bool = False, 
    reset: bool = False, 
    optimize: bool = True,
    capital: float = DEFAULT_CAPITAL,
    capital_allocation: Optional[Dict[str, float]] = None,
    max_iterations: Optional[int] = None,
    interval_seconds: int = 10
) -> subprocess.Popen:
    """Start the trading bot with the specified assets"""
    try:
        logger.info(f"Starting trading bot for assets: {assets}")
        
        # Determine trading mode
        mode = "live" if live else "sandbox"
        logger.info(f"Trading mode: {mode}")
        
        # Build command
        command = [
            "python", 
            "main.py", 
            "--pair", assets[0],  # Primary pair
            "--multi-strategy", "ml_enhanced_integrated"
        ]
        
        # Add flags
        if not live:
            command.append("--sandbox")
        
        if capital:
            command.extend(["--capital", str(capital)])
        
        # Start the bot
        logger.info(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command)
        
        logger.info(f"Trading bot started with PID {process.pid}")
        
        # If max_iterations is specified, limit the runtime
        if max_iterations is not None:
            logger.info(f"Bot will run for {max_iterations} iterations")
            iteration = 0
            while process.poll() is None and (max_iterations is None or iteration < max_iterations):
                time.sleep(interval_seconds)
                iteration += 1
                logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Terminate the process if it's still running
            if process.poll() is None:
                logger.info(f"Reached {max_iterations} iterations, terminating bot")
                process.terminate()
                process.wait()
        
        return process
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        return None

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run Optimized ML Trading")
    
    # Asset selection
    parser.add_argument("--assets", type=str, nargs="+", default=DEFAULT_ASSETS, 
                      help="Assets to trade (comma-separated or space-separated list)")
    
    # Mode selection
    parser.add_argument("--live", action="store_true", 
                      help="Run in live trading mode (default: sandbox mode)")
    parser.add_argument("--sandbox", action="store_true", 
                      help="Run in sandbox mode (default is already sandbox, this is just for compatibility)")
    
    # Reset option
    parser.add_argument("--reset", action="store_true", 
                      help="Reset trading bot state")
    
    # Optimization option
    parser.add_argument("--optimize", action="store_true", 
                      help="Run with all optimizations enabled")
    
    # Capital option
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, 
                      help=f"Starting capital (default: {DEFAULT_CAPITAL})")
    
    # Interval option
    parser.add_argument("--interval", type=int, default=10, 
                      help="Interval between iterations in seconds (default: 10)")
    
    # Max iterations option
    parser.add_argument("--iterations", type=int, 
                      help="Maximum number of iterations to run (default: unlimited)")
    
    args = parser.parse_args()
    
    # Process assets argument
    if len(args.assets) == 1 and "," in args.assets[0]:
        # Handle comma-separated list
        args.assets = [asset.strip() for asset in args.assets[0].split(",")]
    
    return args

def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config()
        
        # Get capital allocation
        capital_allocation = get_capital_allocation(config, args.assets)
        logger.info(f"Capital allocation: {capital_allocation}")
        
        # Validate assets
        if not args.assets:
            logger.error("No assets specified")
            sys.exit(1)
        
        # Start trading bot
        process = start_trading_bot(
            assets=args.assets,
            live=args.live,
            reset=args.reset,
            optimize=args.optimize,
            capital=args.capital,
            capital_allocation=capital_allocation,
            max_iterations=args.iterations,
            interval_seconds=args.interval
        )
        
        if not process:
            logger.error("Failed to start trading bot")
            sys.exit(1)
        
        # Wait for the process to finish
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Exiting")

if __name__ == "__main__":
    main()