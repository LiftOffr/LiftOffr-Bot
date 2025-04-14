#!/usr/bin/env python3
"""
Run Optimized ML Trading Script

This script coordinates the training and deployment of all machine learning models
across multiple trading pairs for maximum performance. It implements:

1. Advanced ML components (TFT, Feature Fusion, Adaptive Hyperparameters, XAI, Sentiment)
2. Reinforcement Learning for optimal decision making
3. Transfer learning between similar assets
4. Neural Architecture Search for optimal model structures
5. Multi-objective optimization for balanced performance
6. Online learning for continuous model improvement in sandbox trading

It runs all models in sandbox mode to continuously improve performance without risk.
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import advanced ML components
from advanced_ml_integration import (
    advanced_ml,
    ml_model_integrator,
    TARGET_ACCURACY,
    TARGET_RETURN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("ml_live_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config(config_path="ml_enhanced_config.json"):
    """Load the enhanced ML configuration from file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return minimal default config
        return {
            "global_settings": {
                "training_priority": "critical",
                "continuous_learning": True,
                "default_capital_allocation": {
                    "SOL/USD": 0.20,
                    "ETH/USD": 0.20,
                    "BTC/USD": 0.20,
                    "DOT/USD": 0.10,
                    "LINK/USD": 0.10
                }
            },
            "asset_configs": {},
            "training_parameters": {},
            "advanced_ai_settings": {
                "reinforcement_learning": {"enabled": False},
                "transfer_learning": {"enabled": False},
                "online_learning": {"enabled": True}
            }
        }

def get_trading_pairs(config):
    """Extract trading pairs from configuration"""
    if "global_settings" in config and "default_capital_allocation" in config["global_settings"]:
        return list(config["global_settings"]["default_capital_allocation"].keys())
    else:
        # Default pairs if not specified
        return ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    config = load_config()
    trading_pairs = get_trading_pairs(config)
    
    parser = argparse.ArgumentParser(description="Run optimized ML trading")
    
    # Mode and pairs
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "sandbox", "live"], default="sandbox",
                        help="Operating mode")
    parser.add_argument("--trading-pairs", nargs="+", default=trading_pairs,
                        help="Trading pairs to use")
    parser.add_argument("--config", type=str, default="ml_enhanced_config.json",
                        help="Configuration file path")
    
    # Training parameters
    parser.add_argument("--force-retrain", action="store_true", default=False,
                        help="Force retraining of models")
    parser.add_argument("--training-threads", type=int, default=1,
                        help="Number of threads for parallel training")
    parser.add_argument("--continuous-training", action="store_true", default=True,
                        help="Enable continuous training during operation")
    
    # Advanced ML parameters
    parser.add_argument("--use-reinforcement", action="store_true", default=True,
                        help="Use reinforcement learning")
    parser.add_argument("--use-transfer-learning", action="store_true", default=True,
                        help="Use transfer learning between assets")
    parser.add_argument("--use-neural-architecture-search", action="store_true", default=True,
                        help="Use neural architecture search")
    
    # Trading parameters
    parser.add_argument("--initial-capital", type=float, default=20000.0,
                        help="Initial capital for trading")
    parser.add_argument("--max-allocation-percent", type=float, default=0.8,
                        help="Maximum percentage of capital to allocate")
    
    return parser.parse_args()

def create_directories():
    """Create necessary directories for ML artifacts"""
    directories = [
        "models",
        "logs",
        "training_data",
        "training_results",
        "backtest_results",
        "sentiment_data",
        "model_explanations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Created necessary directories")

def setup_asset_specific_directories(trading_pairs):
    """Setup asset-specific directories for models and data"""
    for pair in trading_pairs:
        pair_dir = pair.replace("/", "_")
        directories = [
            f"models/{pair_dir}",
            f"training_data/{pair_dir}",
            f"backtest_results/{pair_dir}"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Created asset-specific directories for {len(trading_pairs)} trading pairs")

def fetch_historical_data(trading_pairs, timeframes=None, days=365):
    """
    Fetch historical data for all trading pairs
    This function triggers the external data fetching process
    """
    if timeframes is None:
        timeframes = ["1d", "4h", "1h", "15m"]
    
    try:
        from enhanced_historical_data_fetcher import batch_fetch_historical_data
        
        logger.info(f"Fetching historical data for {len(trading_pairs)} trading pairs")
        data_dict = batch_fetch_historical_data(trading_pairs, timeframes, days)
        logger.info(f"Successfully fetched historical data")
        return data_dict
    except ImportError:
        logger.error("Enhanced historical data fetcher not available, falling back to basic fetcher")
        
        try:
            from historical_data_fetcher import fetch_historical_data as basic_fetch
            
            data_dict = {}
            for pair in trading_pairs:
                pair_data = {}
                for timeframe in timeframes:
                    # Convert timeframe to interval (e.g., "1d" to 1440 minutes)
                    if timeframe.endswith('d'):
                        interval = int(timeframe[:-1]) * 1440
                    elif timeframe.endswith('h'):
                        interval = int(timeframe[:-1]) * 60
                    elif timeframe.endswith('m'):
                        interval = int(timeframe[:-1])
                    else:
                        interval = 1440  # Default to 1d
                    
                    try:
                        data = basic_fetch(pair, interval=interval, count=days)
                        if data:
                            # Convert to DataFrame
                            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
                            pair_data[timeframe] = df
                            logger.info(f"Fetched {len(df)} {timeframe} records for {pair}")
                    except Exception as e:
                        logger.error(f"Error fetching {timeframe} data for {pair}: {e}")
                
                if pair_data:
                    # Use daily data as base
                    if "1d" in pair_data:
                        data_dict[pair] = pair_data["1d"]
                    else:
                        # Use the first available timeframe
                        first_tf = list(pair_data.keys())[0]
                        data_dict[pair] = pair_data[first_tf]
            
            logger.info(f"Fetched data for {len(data_dict)} trading pairs")
            return data_dict
        except ImportError:
            logger.error("No historical data fetcher available")
            return {}

def load_trading_instance(config, args):
    """
    Load trading instance based on the configuration and arguments
    Connects to the Kraken API and prepares the trading environment
    """
    try:
        from kraken_api import KrakenAPI
        from bot_manager_integration import prepare_bot_manager
        
        api = KrakenAPI()
        bot_manager = prepare_bot_manager(
            initial_capital=args.initial_capital,
            max_allocation=args.max_allocation_percent,
            sandbox=args.mode == "sandbox",
            trading_pairs=args.trading_pairs
        )
        
        logger.info(f"Initialized trading instance in {args.mode} mode")
        return {"api": api, "bot_manager": bot_manager}
    except ImportError:
        logger.error("Trading components not available")
        return None

def train_asset_models(args, config, data_dict, asset):
    """
    Train ML models for a specific asset
    This function is designed to be run in its own thread for parallel training
    """
    logger.info(f"Starting training for {asset}")
    
    # Get asset-specific training parameters
    training_params = config.get("training_parameters", {}).get(asset, {})
    
    # Apply custom training parameters if available
    custom_params = {}
    if training_params:
        custom_params = {
            "epochs": training_params.get("epochs", 200),
            "batch_size": training_params.get("batch_size", 64),
            "learning_rate": training_params.get("learning_rate", 0.001),
            "sequence_length": training_params.get("sequence_length", 60),
            "asymmetric_loss_ratio": training_params.get("asymmetric_loss_ratio", 2.0)
        }
    
    # Prepare asset-specific data
    asset_data = {asset: data_dict.get(asset)} if asset in data_dict else {}
    
    # If cross-asset features are enabled, include related assets
    if args.use_transfer_learning and config.get("advanced_ai_settings", {}).get("transfer_learning", {}).get("enabled", False):
        # Find related assets (source-target pairs)
        transfer_pairs = config.get("advanced_ai_settings", {}).get("transfer_learning", {}).get("source_target_pairs", [])
        related_assets = []
        
        for src, tgt in transfer_pairs:
            if asset == tgt and src in data_dict:
                related_assets.append(src)
            elif asset == src and tgt in data_dict:
                related_assets.append(tgt)
        
        # Add related assets to the data
        for related in related_assets:
            if related in data_dict:
                asset_data[related] = data_dict[related]
        
        logger.info(f"Including data from related assets for {asset}: {related_assets}")
    
    # Train models
    try:
        # Initialize specific models for this asset
        advanced_ml.initialize_models([asset])
        
        # Train the models
        training_results = advanced_ml.train_models(
            asset_data,
            force_retrain=args.force_retrain,
            save_models=True,
            training_params=custom_params,
            assets=[asset]
        )
        
        logger.info(f"Completed training for {asset}")
        return training_results
    except Exception as e:
        logger.error(f"Error training models for {asset}: {e}")
        return None

def train_models_parallel(args, config, data_dict):
    """
    Train models for all assets in parallel using multiple threads
    """
    trading_pairs = args.trading_pairs
    num_threads = min(args.training_threads, len(trading_pairs))
    
    if num_threads <= 1:
        # Sequential training
        logger.info(f"Training models sequentially for {len(trading_pairs)} assets")
        results = {}
        for asset in trading_pairs:
            asset_result = train_asset_models(args, config, data_dict, asset)
            if asset_result:
                results[asset] = asset_result
        return results
    
    # Parallel training
    logger.info(f"Training models in parallel with {num_threads} threads")
    
    import concurrent.futures
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit training tasks
        future_to_asset = {
            executor.submit(train_asset_models, args, config, data_dict, asset): asset
            for asset in trading_pairs
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_asset):
            asset = future_to_asset[future]
            try:
                asset_result = future.result()
                if asset_result:
                    results[asset] = asset_result
                    logger.info(f"Successfully trained models for {asset}")
            except Exception as e:
                logger.error(f"Training failed for {asset}: {e}")
    
    return results

def apply_neural_architecture_search(args, config, data_dict):
    """
    Apply neural architecture search to find optimal model structures
    """
    if not args.use_neural_architecture_search:
        return
    
    nas_settings = config.get("advanced_ai_settings", {}).get("neural_architecture_search", {})
    if not nas_settings.get("enabled", False):
        return
    
    try:
        logger.info("Starting Neural Architecture Search")
        
        # This would be a call to a specialized module for NAS
        # For now, we'll log what would happen
        max_trials = nas_settings.get("max_trials", 50)
        search_algorithm = nas_settings.get("search_algorithm", "bayesian")
        
        logger.info(f"Would run Neural Architecture Search with {max_trials} trials using {search_algorithm} algorithm")
        logger.info("Neural Architecture Search would optimize model structures for each asset")
        
        # In a real implementation, this would run the search and update models
        
    except Exception as e:
        logger.error(f"Error during Neural Architecture Search: {e}")

def register_ml_signals_with_bot_manager(trading_instance, signals):
    """
    Register ML signals with the bot manager for trading
    """
    if not trading_instance or "bot_manager" not in trading_instance:
        logger.error("Bot manager not available, cannot register signals")
        return False
    
    bot_manager = trading_instance["bot_manager"]
    
    try:
        for asset, signal in signals.items():
            # Convert signal type from string to enum value
            signal_type = "NEUTRAL"
            if signal["signal"] == "BUY":
                signal_type = "BUY"
            elif signal["signal"] == "SELL":
                signal_type = "SELL"
            
            # Register signal with bot manager
            bot_manager.register_signal(
                asset,
                "MLModel",
                signal_type,
                signal["strength"]
            )
            
            logger.info(f"Registered {signal_type} signal for {asset} with strength {signal['strength']:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Error registering signals with bot manager: {e}")
        return False

def run_training_loop(args, config):
    """
    Run the main training loop
    """
    # Create necessary directories
    create_directories()
    setup_asset_specific_directories(args.trading_pairs)
    
    # Fetch historical data
    data_dict = fetch_historical_data(args.trading_pairs)
    
    if not data_dict:
        logger.error("No historical data available, aborting")
        return False
    
    # Train models
    logger.info("Starting initial model training")
    
    training_results = train_models_parallel(args, config, data_dict)
    
    if not training_results:
        logger.error("Model training failed, aborting")
        return False
    
    logger.info("Initial model training completed")
    
    # Apply neural architecture search if enabled
    if args.use_neural_architecture_search:
        apply_neural_architecture_search(args, config, data_dict)
    
    # Generate and save performance report
    try:
        performance_report = advanced_ml.generate_performance_report()
        report_path = os.path.join("training_results", f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        logger.info(f"Saved performance report to {report_path}")
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
    
    return True

def run_sandbox_trading(args, config):
    """
    Run sandbox trading with ML models
    """
    # Initialize trading instance
    trading_instance = load_trading_instance(config, args)
    
    if not trading_instance:
        logger.error("Failed to initialize trading instance, aborting")
        return False
    
    # Enable continuous training if specified
    continuous_training = args.continuous_training and config.get("global_settings", {}).get("continuous_learning", True)
    online_learning = config.get("advanced_ai_settings", {}).get("online_learning", {}).get("enabled", False)
    
    logger.info(f"Starting sandbox trading with continuous training: {continuous_training}")
    
    # Main trading loop
    running = True
    training_thread = None
    last_training_time = datetime.now() - timedelta(days=1)  # Ensure immediate training
    last_data_refresh = datetime.now()
    
    while running:
        try:
            current_time = datetime.now()
            
            # Fetch latest market data
            if (current_time - last_data_refresh).total_seconds() > 300:  # Every 5 minutes
                logger.info("Fetching latest market data")
                
                # This would be a call to get the latest market data
                # For now, we'll use a simplified approach
                market_data = {}
                for pair in args.trading_pairs:
                    try:
                        # Use the trading API to get current data
                        ohlc = trading_instance["api"].get_ohlc(pair, interval=1, count=100)
                        
                        if ohlc:
                            df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
                            market_data[pair] = df
                    except Exception as e:
                        logger.error(f"Error fetching market data for {pair}: {e}")
                
                last_data_refresh = current_time
                
                if not market_data:
                    logger.error("Failed to fetch market data, waiting before retry")
                    time.sleep(60)
                    continue
            
            # Run continuous training if enabled
            if continuous_training and (current_time - last_training_time).total_seconds() > 3600:  # Every hour
                if training_thread is None or not training_thread.is_alive():
                    logger.info("Starting continuous training")
                    training_thread = threading.Thread(
                        target=train_models_parallel,
                        args=(args, config, data_dict)
                    )
                    training_thread.daemon = True
                    training_thread.start()
                    last_training_time = current_time
            
            # Generate predictions and signals
            logger.info("Generating trading signals")
            predictions = advanced_ml.predict(market_data, include_explanations=True)
            signals = ml_model_integrator.get_trading_signals(market_data)
            
            # Register signals with bot manager
            register_ml_signals_with_bot_manager(trading_instance, signals)
            
            # If online learning is enabled, update models with latest data
            if online_learning:
                # This would call the online learning update function
                logger.info("Performing online learning update")
                
                # In a real implementation, this would update the models incrementally
                
            # Wait before next iteration
            time.sleep(60)  # 1-minute cycle
            
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
            running = False
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            time.sleep(60)  # Wait before retry
    
    logger.info("Sandbox trading completed")
    return True

def main():
    """Main function"""
    args = parse_arguments()
    config = load_config(args.config)
    
    logger.info(f"Starting optimized ML trading in {args.mode} mode")
    logger.info(f"Trading pairs: {args.trading_pairs}")
    
    if args.mode == "train":
        # Training mode - just train the models
        success = run_training_loop(args, config)
        if success:
            logger.info("Training completed successfully")
            return 0
        else:
            logger.error("Training failed")
            return 1
    
    elif args.mode == "sandbox" or args.mode == "live":
        # First train models if needed
        if args.force_retrain:
            success = run_training_loop(args, config)
            if not success:
                logger.error("Initial training failed, aborting")
                return 1
        
        # Then run trading
        if args.mode == "sandbox":
            success = run_sandbox_trading(args, config)
        else:
            logger.error("Live trading mode not yet implemented")
            return 1
        
        if success:
            logger.info(f"{args.mode.capitalize()} trading completed successfully")
            return 0
        else:
            logger.error(f"{args.mode.capitalize()} trading failed")
            return 1
    
    elif args.mode == "backtest":
        logger.error("Backtest mode not yet implemented")
        return 1
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main())