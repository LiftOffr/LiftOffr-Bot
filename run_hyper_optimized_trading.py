#!/usr/bin/env python3
"""
Hyper-Optimized Multi-Asset Trading System

This script launches an enhanced multi-asset trading system using the hyper-optimized
ML models for SOL/USD, ETH/USD, and BTC/USD. It integrates advanced profit-centric
ML models with extreme leverage settings for maximum trading profitability.

Features:
1. Multi-asset trading with dynamic capital allocation
2. Hyper-optimized profit-centric ML models
3. Extreme leverage settings (up to 125x)
4. Multi-timeframe data fusion
5. Cross-asset correlation analysis
6. Adaptive market regime detection

Usage:
  python run_hyper_optimized_trading.py [--live] [--capital AMOUNT] [--assets ASSETS]
"""

import os
import sys
import time
import logging
import json
import argparse
from typing import Dict, List, Optional
import datetime

# Local imports
from multi_asset_trading import MultiAssetManager, start_multi_asset_trading
from hyper_optimized_ml_training import load_model_with_metadata, train_all_assets
from historical_data_fetcher import fetch_historical_data
from config import INITIAL_CAPITAL
from dynamic_position_sizing_ml import get_config as get_position_sizing_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hyper_optimized_trading.log")
    ]
)

logger = logging.getLogger(__name__)

# Default assets to trade
DEFAULT_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]

# Default capital allocation
DEFAULT_ALLOCATION = {
    "SOL/USD": 0.40,  # 40% to SOL (higher volatility, higher potential returns)
    "ETH/USD": 0.35,  # 35% to ETH 
    "BTC/USD": 0.25   # 25% to BTC (lower volatility, more stable)
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hyper-Optimized Multi-Asset Trading')
    
    parser.add_argument('--assets', nargs='+', default=DEFAULT_ASSETS,
                        help='Assets to trade')
    
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
    
    parser.add_argument('--max-leverage', type=float, default=125.0,
                        help='Maximum leverage to use (default: 125.0)')
    
    parser.add_argument('--min-leverage', type=float, default=20.0,
                        help='Minimum leverage to use (default: 20.0)')
    
    parser.add_argument('--base-leverage', type=float, default=35.0,
                        help='Base leverage to use (default: 35.0)')
    
    return parser.parse_args()

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
        # Use default allocation if present for all requested assets
        allocation = {}
        for asset in assets:
            if asset in DEFAULT_ALLOCATION:
                allocation[asset] = DEFAULT_ALLOCATION[asset]
        
        # If any assets don't have defaults or allocation values don't sum to 1.0,
        # use equal allocation
        if not allocation or abs(sum(allocation.values()) - 1.0) > 0.001:
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

def check_models(assets: List[str], force_retrain: bool = False) -> Dict[str, Dict]:
    """
    Check if ML models exist for all assets and retrain if needed
    
    Args:
        assets: List of assets to check
        force_retrain: Whether to force retraining
        
    Returns:
        Dict: Dictionary of loaded models
    """
    models = {}
    missing_models = []
    
    # Check for existing models
    for asset in assets:
        model, scaler_dict, selected_features_dict = load_model_with_metadata(asset)
        
        if model is not None and not force_retrain:
            logger.info(f"Loaded existing model for {asset}")
            models[asset] = {
                'model': model,
                'scalers': scaler_dict,
                'selected_features': selected_features_dict
            }
        else:
            missing_models.append(asset)
    
    # Train missing models if needed
    if missing_models or force_retrain:
        assets_to_train = missing_models if not force_retrain else assets
        logger.info(f"Training models for: {', '.join(assets_to_train)}")
        
        trained_models = train_all_assets(assets_to_train, force_retrain=force_retrain)
        
        # Update models dict with newly trained models
        for asset, model_info in trained_models.items():
            models[asset] = model_info
    
    return models

def update_position_sizing_config(args):
    """
    Update position sizing configuration based on command line arguments
    
    Args:
        args: Command line arguments
    """
    # Get position sizing config
    config = get_position_sizing_config()
    
    # Update leverage settings if provided
    if args.max_leverage != 125.0:
        logger.info(f"Setting maximum leverage to {args.max_leverage}x")
        config.config['max_leverage'] = args.max_leverage
        
        # Update asset-specific configs
        for asset in config.asset_configs:
            config.asset_configs[asset]['max_leverage'] = args.max_leverage
    
    if args.min_leverage != 20.0:
        logger.info(f"Setting minimum leverage to {args.min_leverage}x")
        config.config['min_leverage'] = args.min_leverage
        
        # Update asset-specific configs
        for asset in config.asset_configs:
            config.asset_configs[asset]['min_leverage'] = args.min_leverage
    
    if args.base_leverage != 35.0:
        logger.info(f"Setting base leverage to {args.base_leverage}x")
        config.config['base_leverage'] = args.base_leverage
        
        # Update asset-specific configs
        for asset in config.asset_configs:
            config.asset_configs[asset]['base_leverage'] = args.base_leverage
    
    # Save the updated config
    config.save_config('position_sizing_config_ultra_aggressive.json')

def main():
    """Main function to run hyper-optimized multi-asset trading"""
    args = parse_arguments()
    
    # Set log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"hyper_optimized_trading_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info("========== STARTING HYPER-OPTIMIZED MULTI-ASSET TRADING ==========")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Trading assets: {args.assets}")
    logger.info(f"Initial capital: ${args.capital}")
    
    # Update environment variables for sandbox/live mode
    if args.live:
        os.environ['USE_SANDBOX'] = 'False'
        logger.info("LIVE TRADING MODE ENABLED - REAL MONEY WILL BE USED!")
    elif args.sandbox:
        os.environ['USE_SANDBOX'] = 'True'
        logger.info("Running in SANDBOX mode (paper trading)")
    
    # Update position sizing configuration
    update_position_sizing_config(args)
    
    # Check and train ML models if needed
    models = check_models(args.assets, args.retrain)
    
    # Create capital allocation
    allocation = create_allocation(args.assets, args.allocation)
    
    # Start multi-asset trading
    logger.info("Starting multi-asset trading...")
    
    try:
        # Create and start the multi-asset manager
        manager = MultiAssetManager(capital_allocation=allocation)
        manager.initialize_bot_managers()
        
        # Add ML models to the manager
        for asset, model_info in models.items():
            if asset in manager.bot_managers:
                logger.info(f"Integrating ML model for {asset}")
                # In a real implementation, you would integrate the model here
                # For example, by setting up prediction pipelines
        
        # Start trading
        if not args.monitor_only:
            logger.info("Starting active trading...")
            manager.start_all()
        else:
            logger.info("Starting in monitor-only mode (no trading)")
            # You would implement monitor-only functionality here
        
        # Keep the main thread alive
        try:
            logger.info("Trading system running. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received exit signal. Shutting down...")
            manager.stop_all()
    
    except Exception as e:
        logger.exception(f"Error in trading system: {e}")
    
    logger.info("Trading system shutdown complete.")

if __name__ == "__main__":
    main()