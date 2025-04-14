#!/usr/bin/env python3
"""
ML Live Training for All Assets

This script incorporates and trains ML models for all supported assets (SOL/USD, ETH/USD, BTC/USD)
while continuing to trade live in the market. It runs in the background and
continuously improves the ML models based on new market data.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_training_all_assets.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "ml_config.json"
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD"]
HISTORICAL_DATA_DIR = "historical_data"
MODELS_DIR = "models"
OPTIMIZATION_RESULTS_DIR = "optimization_results"
ML_LIVE_TRADING_SCRIPT = "run_optimized_ml_trading.py"

# Ensure directories exist
for directory in [HISTORICAL_DATA_DIR, MODELS_DIR, OPTIMIZATION_RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

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
            return {
                "global_settings": {
                    "extreme_leverage_enabled": True,
                    "model_pruning_threshold": 0.4,
                    "model_pruning_min_samples": 10,
                    "model_selection_frequency": 24,
                    "default_capital_allocation": {
                        "SOL/USD": 0.40,
                        "ETH/USD": 0.35,
                        "BTC/USD": 0.25
                    }
                },
                "asset_configs": {},
                "training_parameters": {}
            }
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def save_config(config: Dict[str, Any]) -> None:
    """Save the ML configuration to file"""
    try:
        # Create a backup first
        if os.path.exists(CONFIG_PATH):
            backup_path = f"{CONFIG_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(CONFIG_PATH, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            logger.info(f"Created backup of config at {backup_path}")
        
        # Save the new config
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved ML configuration to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def fetch_historical_data(asset: str, days: int = 30) -> bool:
    """Fetch historical data for an asset"""
    try:
        logger.info(f"Fetching historical data for {asset} (last {days} days)")
        
        # Create a clean asset filename (remove / and other special chars)
        asset_filename = asset.replace("/", "")
        
        # Run the historical data fetcher script
        command = [
            "python", "enhanced_historical_data_fetcher.py", 
            "--symbol", asset, 
            "--days", str(days),
            "--output", f"{HISTORICAL_DATA_DIR}/{asset_filename}_data.csv"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully fetched historical data for {asset}")
            return True
        else:
            logger.error(f"Failed to fetch historical data for {asset}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error fetching historical data for {asset}: {e}")
        return False

def train_models(asset: str, force_retrain: bool = False) -> bool:
    """Train ML models for an asset"""
    try:
        logger.info(f"Training ML models for {asset}")
        
        # Create a clean asset filename (remove / and other special chars)
        asset_filename = asset.replace("/", "")
        
        # Get training parameters from config
        config = load_config()
        training_params = config.get("training_parameters", {}).get(asset, {})
        
        # Build command for training
        command = [
            "python", "hyper_optimized_ml_training.py",
            "--asset", asset,
            "--input", f"{HISTORICAL_DATA_DIR}/{asset_filename}_data.csv",
            "--output", f"{MODELS_DIR}/{asset_filename}",
            "--optimize"
        ]
        
        # Add any training parameters
        if "epochs" in training_params:
            command.extend(["--epochs", str(training_params["epochs"])])
        if "batch_size" in training_params:
            command.extend(["--batch_size", str(training_params["batch_size"])])
        if "learning_rate" in training_params:
            command.extend(["--learning_rate", str(training_params["learning_rate"])])
        if "force_retrain" in training_params or force_retrain:
            command.append("--force_retrain")
        
        # Run the training script
        logger.info(f"Running training command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully trained models for {asset}")
            logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            logger.error(f"Failed to train models for {asset}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error training models for {asset}: {e}")
        return False

def create_ensemble(asset: str) -> bool:
    """Create an ensemble of models for an asset"""
    try:
        logger.info(f"Creating ensemble for {asset}")
        
        # Create a clean asset filename (remove / and other special chars)
        asset_filename = asset.replace("/", "")
        
        # Run the ensemble creation script
        command = [
            "python", "strategy_ensemble_trainer.py",
            "--asset", asset,
            "--models_dir", f"{MODELS_DIR}/{asset_filename}",
            "--output", f"{MODELS_DIR}/ensemble/{asset_filename}_ensemble.json"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully created ensemble for {asset}")
            return True
        else:
            logger.error(f"Failed to create ensemble for {asset}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error creating ensemble for {asset}: {e}")
        return False

def optimize_position_sizing(asset: str) -> bool:
    """Optimize position sizing for an asset"""
    try:
        logger.info(f"Optimizing position sizing for {asset}")
        
        # Create a clean asset filename (remove / and other special chars)
        asset_filename = asset.replace("/", "")
        
        # Run the position sizing optimization script
        command = [
            "python", "dynamic_position_sizing_ml.py",
            "--asset", asset,
            "--input", f"{HISTORICAL_DATA_DIR}/{asset_filename}_data.csv",
            "--output", f"{MODELS_DIR}/ensemble/{asset_filename}_position_sizing.json",
            "--optimize"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully optimized position sizing for {asset}")
            return True
        else:
            logger.error(f"Failed to optimize position sizing for {asset}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error optimizing position sizing for {asset}: {e}")
        return False

def train_asset(asset: str, force_retrain: bool = False) -> bool:
    """Train all models for an asset"""
    try:
        # Make sure ensemble directory exists
        os.makedirs(f"{MODELS_DIR}/ensemble", exist_ok=True)
        
        # Step 1: Fetch historical data
        if not fetch_historical_data(asset, days=60):
            logger.error(f"Failed to fetch historical data for {asset}, skipping training")
            return False
        
        # Step 2: Train models
        if not train_models(asset, force_retrain):
            logger.error(f"Failed to train models for {asset}")
            return False
        
        # Step 3: Create ensemble
        if not create_ensemble(asset):
            logger.error(f"Failed to create ensemble for {asset}")
            return False
        
        # Step 4: Optimize position sizing
        if not optimize_position_sizing(asset):
            logger.error(f"Failed to optimize position sizing for {asset}")
            return False
        
        logger.info(f"Successfully completed training pipeline for {asset}")
        return True
    except Exception as e:
        logger.error(f"Error in training pipeline for {asset}: {e}")
        return False

def start_trading_bot(assets: List[str], sandbox: bool = True, reset: bool = False) -> subprocess.Popen:
    """Start the trading bot with the specified assets"""
    try:
        logger.info(f"Starting trading bot for assets: {assets}")
        
        # Build command
        command = ["python", "main.py", "--pair", assets[0], "--multi-strategy", "ml_enhanced_integrated"]
        
        # Add flags
        if sandbox:
            command.append("--sandbox")
        if reset:
            command.append("--reset")
        
        # Start the bot
        logger.info(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command)
        
        logger.info(f"Trading bot started with PID {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        return None

def continuous_training_loop(assets: List[str], interval_hours: int = 24, force_retrain: bool = False):
    """Run continuous training loop in background"""
    try:
        while True:
            for asset in assets:
                logger.info(f"Starting training cycle for {asset}")
                train_asset(asset, force_retrain)
                logger.info(f"Completed training cycle for {asset}")
            
            # Sleep until next training cycle
            sleep_seconds = interval_hours * 3600
            logger.info(f"Training cycle completed. Sleeping for {interval_hours} hours")
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        logger.info("Training loop interrupted by user")
    except Exception as e:
        logger.error(f"Error in continuous training loop: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ML Live Training for All Assets")
    parser.add_argument("--assets", type=str, default="SOL/USD,ETH/USD,BTC/USD", 
                      help="Comma-separated list of assets to train and trade")
    parser.add_argument("--train_interval", type=int, default=24, 
                      help="Interval between training cycles in hours")
    parser.add_argument("--force_retrain", action="store_true", 
                      help="Force retraining even if models already exist")
    parser.add_argument("--sandbox", action="store_true", default=True,
                      help="Use sandbox mode for trading")
    parser.add_argument("--reset", action="store_true", 
                      help="Reset the trading bot state")
    parser.add_argument("--train_only", action="store_true", 
                      help="Only train models, don't start trading")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Parse assets
    assets = [asset.strip() for asset in args.assets.split(",")]
    
    # Validate assets
    for asset in assets:
        if asset not in SUPPORTED_ASSETS:
            logger.warning(f"Unsupported asset: {asset}. Supported assets are: {SUPPORTED_ASSETS}")
    
    assets = [asset for asset in assets if asset in SUPPORTED_ASSETS]
    
    if not assets:
        logger.error("No valid assets specified")
        sys.exit(1)
    
    try:
        # If train_only, just train the models and exit
        if args.train_only:
            logger.info(f"Training models for assets: {assets}")
            for asset in assets:
                train_asset(asset, args.force_retrain)
            logger.info("Training completed")
            return
        
        # Start trading bot
        trading_process = start_trading_bot(assets, args.sandbox, args.reset)
        if not trading_process:
            logger.error("Failed to start trading bot")
            sys.exit(1)
        
        # Start continuous training in background thread
        logger.info(f"Starting continuous training loop with interval {args.train_interval} hours")
        training_thread = threading.Thread(
            target=continuous_training_loop,
            args=(assets, args.train_interval, args.force_retrain),
            daemon=True
        )
        training_thread.start()
        
        # Wait for the trading process to finish
        trading_process.wait()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Exiting")

if __name__ == "__main__":
    main()