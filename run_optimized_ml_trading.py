#!/usr/bin/env python3
"""
Run Optimized ML Trading

This script runs the ML-enhanced trading bot with optimized models for live trading.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from datetime import datetime

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run optimized ML trading bot')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading assets to trade')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Run in sandbox mode (default)')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Trading interval in seconds')
    
    parser.add_argument('--optimize-first', action='store_true',
                      help='Optimize models before starting trading')
    
    parser.add_argument('--capital', type=float, default=20000.0,
                      help='Initial trading capital')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                      help='Maximum number of trading iterations')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    return args

def optimize_models(assets):
    """
    Optimize models for live trading
    
    Args:
        assets: List of trading assets
        
    Returns:
        bool: Whether optimization was successful
    """
    try:
        logger.info(f"Optimizing ML models for live trading: {assets}")
        
        # Run the optimizer
        cmd = [
            "python", "ml_live_training_optimizer.py",
            "--assets"
        ] + assets + [
            "--extreme-leverage",
            "--force-retrain"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            logger.info("Model optimization completed successfully")
            return True
        else:
            logger.error(f"Model optimization failed: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error optimizing models: {e}")
        return False

def start_trading_bot(args):
    """
    Start the trading bot with optimized models
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Whether startup was successful
    """
    try:
        logger.info(f"Starting ML trading bot with optimized models: {args.assets}")
        
        # Run the trading bot
        cmd = [
            "python", "run_ml_live_bot.py",
            "--pairs"
        ] + args.assets
        
        if args.sandbox:
            cmd.append("--sandbox")
        else:
            cmd.append("--live")
        
        # Add other arguments
        cmd.extend([
            "--extreme-leverage",
            "--ml-position-sizing",
            "--interval", str(args.interval),
            "--initial-capital", str(args.capital)
        ])
        
        # Add max iterations if specified
        if args.max_iterations is not None:
            cmd.extend(["--max-iterations", str(args.max_iterations)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Start the trading bot
        process = subprocess.Popen(cmd)
        
        # Wait for the process to finish or be terminated
        process.wait()
        
        if process.returncode == 0:
            logger.info("Trading bot completed successfully")
            return True
        else:
            logger.error(f"Trading bot exited with code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        return False

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print("=" * 80)
        print("OPTIMIZED ML TRADING BOT")
        print("=" * 80)
        print(f"Trading assets: {args.assets}")
        print(f"Mode: {'LIVE' if args.live else 'SANDBOX'}")
        print(f"Trading interval: {args.interval} seconds")
        print(f"Initial capital: ${args.capital:.2f}")
        print(f"Optimize first: {args.optimize_first}")
        if args.max_iterations:
            print(f"Max iterations: {args.max_iterations}")
        print("=" * 80)
        
        # Optimize models if requested
        if args.optimize_first:
            success = optimize_models(args.assets)
            if not success:
                print("Model optimization failed. Exiting.")
                return
        
        # Start the trading bot
        success = start_trading_bot(args)
        if not success:
            print("Trading bot failed to start or exited with an error.")
            return
        
    except KeyboardInterrupt:
        print("\nTrading bot interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()