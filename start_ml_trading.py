#!/usr/bin/env python3
"""
Start ML Trading Bot

This script provides an easy way to start the ML-enhanced trading bot.
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_startup.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Start ML-enhanced trading bot')
    
    parser.add_argument('--pairs', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading pairs to trade')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (not sandbox)')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Use sandbox mode (no real trades)')
    
    parser.add_argument('--extreme-leverage', action='store_true', default=True,
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--ml-position-sizing', action='store_true', default=True,
                      help='Use ML-enhanced position sizing')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--retrain', action='store_true',
                      help='Retrain ML models before starting')
    
    parser.add_argument('--capital', type=float, default=20000.0,
                      help='Initial capital')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    return args

def verify_models(pairs: List[str], force_retrain: bool = False) -> bool:
    """
    Verify ML models exist for all pairs
    
    Args:
        pairs: List of trading pairs
        force_retrain: Whether to force retraining
        
    Returns:
        bool: Whether models exist
    """
    models_exist = True
    
    for pair in pairs:
        pair_filename = pair.replace("/", "")
        
        # Check for ensemble model
        ensemble_path = f"models/ensemble/{pair_filename}_ensemble.json"
        position_sizing_path = f"models/ensemble/{pair_filename}_position_sizing.json"
        
        if not os.path.exists(ensemble_path) or not os.path.exists(position_sizing_path) or force_retrain:
            models_exist = False
            break
    
    return models_exist

def train_models(pairs: List[str], extreme_leverage: bool = False) -> bool:
    """
    Train ML models for all pairs
    
    Args:
        pairs: List of trading pairs
        extreme_leverage: Whether to use extreme leverage settings
        
    Returns:
        bool: Whether training was successful
    """
    try:
        # Build command
        cmd = ["python", "train_ml_live_integration.py"]
        
        # Add arguments
        cmd.append("--assets")
        cmd.extend(pairs)
        
        if extreme_leverage:
            cmd.append("--extreme-leverage")
        
        cmd.append("--force-retrain")
        
        # Run training
        logger.info(f"Training ML models with command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check result
        if process.returncode == 0:
            logger.info("Model training completed successfully")
            return True
        else:
            logger.error(f"Model training failed: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return False

def start_trading_bot(args) -> None:
    """
    Start the ML trading bot
    
    Args:
        args: Command line arguments
    """
    try:
        # Build command
        cmd = ["python", "run_ml_live_bot.py"]
        
        # Add arguments
        cmd.append("--pairs")
        cmd.extend(args.pairs)
        
        if args.live:
            cmd.append("--live")
        else:
            cmd.append("--sandbox")
        
        if args.extreme_leverage:
            cmd.append("--extreme-leverage")
        
        if args.ml_position_sizing:
            cmd.append("--ml-position-sizing")
        
        cmd.append("--interval")
        cmd.append(str(args.interval))
        
        cmd.append("--initial-capital")
        cmd.append(str(args.capital))
        
        # Run bot
        logger.info(f"Starting ML trading bot with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait for process to finish
        process.wait()
        
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print startup message
        print("=" * 80)
        print("ML-ENHANCED TRADING BOT STARTUP")
        print("=" * 80)
        print(f"Trading pairs: {args.pairs}")
        print(f"Mode: {'LIVE' if args.live else 'SANDBOX'}")
        print(f"Extreme leverage: {args.extreme_leverage}")
        print(f"ML position sizing: {args.ml_position_sizing}")
        print(f"Initial capital: ${args.capital:.2f}")
        print(f"Interval: {args.interval} seconds")
        print("=" * 80)
        
        # Verify models exist
        models_exist = verify_models(args.pairs, args.retrain)
        
        # Train models if needed
        if not models_exist:
            print("\nML models not found or retraining requested. Training models...")
            success = train_models(args.pairs, args.extreme_leverage)
            
            if not success:
                print("Error training models. Exiting.")
                return
        
        # Start trading bot
        print("\nStarting ML trading bot...")
        start_trading_bot(args)
        
    except KeyboardInterrupt:
        print("\nStartup interrupted")
    except Exception as e:
        logger.error(f"Error in startup: {e}")

if __name__ == "__main__":
    main()