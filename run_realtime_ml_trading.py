#!/usr/bin/env python3
"""
Run Realtime ML Trading System

This script starts the realtime ML trading system in sandbox mode,
integrating the advanced ensemble ML models with the Kraken trading bot.
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime
import threading

# Import our modules
from realtime_ml_trader import RealtimeMLTrader
import advanced_ml_integration as ami
from trade_optimizer import TradeOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'realtime_ml_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Realtime ML Trading System")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to trade")
    parser.add_argument("--interval", type=int, default=60,
                        help="Trading loop interval in seconds")
    parser.add_argument("--training", action="store_true",
                        help="Run full training before starting")
    parser.add_argument("--optimize", action="store_true",
                        help="Run optimization before starting")
    return parser.parse_args()

def run_training(pairs):
    """Run ML model training for all pairs"""
    logger.info("Starting ML model training")
    
    # Create Advanced ML Trader
    trader = ami.AdvancedMLTrader(pairs)
    
    # Create Continuous Learning Manager
    learning_manager = ami.ContinuousLearningManager(trader)
    
    # Run full training
    learning_manager.run_full_training()
    
    logger.info("ML model training complete")

def run_optimization(pairs):
    """Run trading optimization for all pairs"""
    logger.info("Starting trade optimization")
    
    # Create Trade Optimizer
    optimizer = TradeOptimizer(pairs)
    
    # Run optimization
    from integrate_trade_optimizer import optimize_trading
    optimize_trading(pairs)
    
    logger.info("Trade optimization complete")

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Run training if requested
    if args.training:
        run_training(args.pairs)
    
    # Run optimization if requested
    if args.optimize:
        run_optimization(args.pairs)
    
    # Create realtime ML trader
    trader = RealtimeMLTrader(args.pairs, args.sandbox)
    
    # Start data collection
    trader.start_data_collection()
    
    # Wait for initial data
    logger.info("Waiting for initial data (10 seconds)...")
    time.sleep(10)
    
    # Run trading loop
    trader.run_trading_loop(args.interval)

if __name__ == "__main__":
    try:
        logger.info("Starting Realtime ML Trading System")
        main()
    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.error(f"Error running trading system: {e}")
        import traceback
        logger.error(traceback.format_exc())