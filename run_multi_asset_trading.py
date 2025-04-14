#!/usr/bin/env python3
"""
Multi-Asset Trading Bot Runner

This script runs the multi-asset trading system with the enhanced ML models
for SOL/USD, ETH/USD, and BTC/USD trading pairs.

Features:
1. Real-time trading with extreme leverage settings
2. Enhanced transformer-based ML model integration
3. Adaptive position sizing based on ML confidence
4. Cross-asset correlation analysis for portfolio optimization
5. Automatic model retraining and performance monitoring
"""

import os
import sys
import time
import logging
import argparse
import threading
from typing import Dict, List, Optional

from multi_asset_trading import MultiAssetManager, start_multi_asset_trading, SUPPORTED_ASSETS
from enhanced_transformer_models import train_models_for_all_assets, load_model_with_metadata
from historical_data_fetcher import fetch_historical_data
from bot_manager import BotManager
from config import INITIAL_CAPITAL
import market_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("multi_asset_trading.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-Asset Trading Bot with Enhanced ML')
    
    parser.add_argument('--assets', nargs='+', default=SUPPORTED_ASSETS,
                        help='Trading pairs to trade (default: SOL/USD ETH/USD BTC/USD)')
    
    parser.add_argument('--allocation', nargs='+', type=float, default=None,
                        help='Capital allocation percentages (must sum to 1.0)')
    
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                        help=f'Initial capital (default: ${INITIAL_CAPITAL:.2f})')
    
    parser.add_argument('--retrain', action='store_true',
                        help='Force retraining of ML models before starting')
    
    parser.add_argument('--fetch-data', action='store_true',
                        help='Fetch fresh historical data before starting')
    
    parser.add_argument('--sandbox', action='store_true',
                        help='Run in sandbox mode (no real trades)')
    
    parser.add_argument('--live', action='store_true',
                        help='Run in live trading mode')
    
    parser.add_argument('--monitor-only', action='store_true',
                        help='Monitor markets without trading')
    
    return parser.parse_args()

def fetch_data_for_assets(assets: List[str]) -> Dict[str, object]:
    """
    Fetch historical data for the specified assets
    
    Args:
        assets: List of trading pairs
        
    Returns:
        Dict: Dictionary mapping assets to historical data DataFrames
    """
    data_sources = {}
    
    for asset in assets:
        try:
            logger.info(f"Fetching historical data for {asset}...")
            data = fetch_historical_data(
                symbol=asset,
                timeframe='1h',
                limit=5000  # Get more historical data for training
            )
            
            if data is not None and len(data) > 0:
                logger.info(f"Successfully fetched {len(data)} candles for {asset}")
                data_sources[asset] = data
            else:
                logger.error(f"Failed to fetch data for {asset}")
        
        except Exception as e:
            logger.error(f"Error fetching data for {asset}: {e}")
    
    return data_sources

def train_models(assets: List[str], data_sources: Dict[str, object]) -> Dict[str, object]:
    """
    Train ML models for the specified assets
    
    Args:
        assets: List of trading pairs
        data_sources: Dictionary mapping assets to historical data
        
    Returns:
        Dict: Dictionary mapping assets to trained models
    """
    logger.info(f"Training enhanced ML models for {', '.join(assets)}...")
    
    try:
        model_results = train_models_for_all_assets(assets, data_sources)
        return model_results
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return {}

def create_allocation(assets: List[str], allocation_percentages: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Create allocation dictionary from assets and percentages
    
    Args:
        assets: List of trading pairs
        allocation_percentages: List of allocation percentages (must sum to 1.0)
        
    Returns:
        Dict: Dictionary mapping assets to allocation percentages
    """
    if allocation_percentages is None:
        # Default to equal allocation
        allocation = {asset: 1.0 / len(assets) for asset in assets}
    else:
        # Check if number of percentages matches number of assets
        if len(allocation_percentages) != len(assets):
            logger.warning(f"Number of allocation percentages ({len(allocation_percentages)}) " 
                          f"doesn't match number of assets ({len(assets)}). Using equal allocation.")
            allocation = {asset: 1.0 / len(assets) for asset in assets}
        else:
            # Check if percentages sum to 1.0
            total = sum(allocation_percentages)
            if abs(total - 1.0) > 0.001:
                logger.warning(f"Allocation percentages sum to {total}, normalizing to 1.0")
                allocation_percentages = [p / total for p in allocation_percentages]
            
            allocation = {asset: percentage for asset, percentage in zip(assets, allocation_percentages)}
    
    logger.info(f"Capital allocation: {allocation}")
    return allocation

def main():
    """Main function to run multi-asset trading"""
    args = parse_arguments()
    
    # Use specified assets or default
    assets = args.assets
    
    # Override environment variables for sandbox/live mode
    if args.sandbox:
        os.environ['USE_SANDBOX'] = 'True'
        logger.info("Running in SANDBOX mode (no real trades)")
    elif args.live:
        os.environ['USE_SANDBOX'] = 'False'
        logger.info("Running in LIVE trading mode")
    
    # Fetch fresh historical data if requested
    data_sources = {}
    if args.fetch_data:
        logger.info("Fetching fresh historical data...")
        data_sources = fetch_data_for_assets(assets)
    
    # Train or load models
    models = {}
    if args.retrain:
        logger.info("Retraining ML models...")
        models = train_models(assets, data_sources)
    
    # Create capital allocation
    allocation = create_allocation(assets, args.allocation)
    
    # Start multi-asset trading
    if args.monitor_only:
        logger.info("Starting in monitor-only mode (no trading)")
        # TODO: Implement monitor-only mode
    else:
        logger.info("Starting multi-asset trading...")
        manager = start_multi_asset_trading(allocation)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()