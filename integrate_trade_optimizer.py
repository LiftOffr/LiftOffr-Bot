#!/usr/bin/env python3
"""
Integrate Trade Optimizer With ML Trading System

This script connects the TradeOptimizer with the RealtimeMLManager
to optimize trade entries/exits and maximize profitability.
"""
import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any

# Import our modules
from trade_optimizer import TradeOptimizer
from realtime_ml_manager import RealtimeMLManager
import kraken_api_client as kraken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
OPTIMIZATION_INTERVAL = 3600  # 1 hour

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrate Trade Optimizer with ML Trading System")
    
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[
            "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
            "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
        ],
        help="Trading pairs to optimize"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (no sandbox)"
    )
    
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run optimization only without trading"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=OPTIMIZATION_INTERVAL,
        help="Optimization interval in seconds"
    )
    
    return parser.parse_args()

def get_current_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Get current prices for all pairs
    
    Args:
        pairs: List of trading pairs
    
    Returns:
        Dictionary of current prices by pair
    """
    current_prices = {}
    
    for pair in pairs:
        try:
            # Get ticker data from Kraken API
            ticker_data = kraken.get_ticker(pair)
            
            if not ticker_data:
                logger.warning(f"No ticker data for {pair}")
                continue
            
            # Extract last price
            current_price = float(ticker_data.get('c', [0])[0])
            
            if current_price > 0:
                current_prices[pair] = current_price
                logger.debug(f"Current price for {pair}: ${current_price}")
        
        except Exception as e:
            logger.error(f"Error getting price for {pair}: {e}")
    
    return current_prices

def create_ml_manager_callback(optimizer: TradeOptimizer):
    """
    Create callback function for ML manager
    
    Args:
        optimizer: TradeOptimizer instance
    
    Returns:
        Callback function
    """
    def optimizer_callback(pair: str, prediction: Dict[str, Any], market_data: Dict[str, Any]):
        """
        Callback function for ML predictions
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            market_data: Market data
        """
        try:
            # Get current price
            current_price = prediction.get('price', 0)
            if not current_price and market_data and 'c' in market_data:
                current_price = float(market_data['c'][0])
            
            if not current_price:
                return
            
            # Update market state
            optimizer.update_market_state(pair, current_price)
            
            # Check if time is optimal for entry/exit
            is_good_entry_time = optimizer.get_best_entry_time(pair)
            is_good_exit_time = optimizer.get_best_exit_time(pair)
            
            logger.info(f"Optimization data for {pair}: "
                        f"Entry time optimal: {is_good_entry_time}, "
                        f"Exit time optimal: {is_good_exit_time}")
            
            # More logic can be added here to modify the prediction based on optimizer
        
        except Exception as e:
            logger.error(f"Error in optimizer callback: {e}")
    
    return optimizer_callback

def optimize_and_integrate(
    trading_pairs: List[str],
    sandbox: bool = True,
    optimize_only: bool = False,
    interval: int = OPTIMIZATION_INTERVAL
):
    """
    Run optimization and integrate with ML trading system
    
    Args:
        trading_pairs: List of trading pairs to optimize
        sandbox: Whether to run in sandbox mode
        optimize_only: Whether to run optimization only without trading
        interval: Optimization interval in seconds
    """
    # Create TradeOptimizer
    optimizer = TradeOptimizer(trading_pairs, sandbox)
    
    # Get current prices
    current_prices = get_current_prices(trading_pairs)
    if not current_prices:
        logger.error("Failed to get current prices for any pairs")
        return
    
    # Run initial optimization
    optimization_results = optimizer.run_optimization(current_prices)
    logger.info(f"Initial optimization complete: {len(optimization_results.get('allocations', {}))} pairs optimized")
    
    # Exit if optimize_only flag is set
    if optimize_only:
        logger.info("Optimization complete. Exiting as requested.")
        return
    
    # Create RealtimeMLManager if continuing with trading
    ml_manager = RealtimeMLManager(
        trading_pairs=trading_pairs,
        sandbox=sandbox
    )
    
    # Register optimizer callback with ML manager
    optimizer_callback = create_ml_manager_callback(optimizer)
    ml_manager.ws_integration.register_callback(optimizer_callback)
    
    # Start ML manager
    ml_manager.start()
    logger.info("Started Realtime ML Manager with optimizer integration")
    
    try:
        # Main loop
        last_optimization = time.time()
        
        while True:
            # Sleep for a bit
            time.sleep(60)
            
            # Check if it's time to run optimization again
            now = time.time()
            if now - last_optimization >= interval:
                # Get current prices
                current_prices = get_current_prices(trading_pairs)
                
                if current_prices:
                    # Run optimization
                    optimizer.run_optimization(current_prices)
                    logger.info("Periodic optimization complete")
                    
                    # Update last optimization time
                    last_optimization = now
    
    except KeyboardInterrupt:
        logger.info("Stopping due to user interrupt")
    
    finally:
        # Stop ML manager
        if ml_manager:
            ml_manager.stop()
            logger.info("Stopped Realtime ML Manager")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Run optimization and integration
    optimize_and_integrate(
        trading_pairs=args.pairs,
        sandbox=not args.live,
        optimize_only=args.optimize_only,
        interval=args.interval
    )

if __name__ == "__main__":
    main()