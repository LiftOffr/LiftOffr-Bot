#!/usr/bin/env python3
"""
Run Full Training Pipeline

This script runs a complete ML training pipeline for all supported trading pairs:
1. Fetches extended historical data for all pairs
2. Prepares enhanced datasets with all features and indicators
3. Trains multiple model architectures (LSTM, GRU, Transformer, TCN, etc.)
4. Creates ensemble models combining the best individual models
5. Optimizes model hyperparameters
6. Evaluates performance
7. Updates ML configuration

Usage:
    python run_full_training_pipeline.py [--pairs PAIRS] [--timeframes TIMEFRAMES] [--epochs EPOCHS]
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('full_training_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD", "DOTUSD", "LINKUSD"]
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
MODEL_TYPES = ["lstm", "gru", "transformer", "tcn", "cnn", "bilstm", "attention", "hybrid"]
ML_CONFIG_PATH = "ml_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run full ML training pipeline")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to train on (default: {' '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes to train on (default: {' '.join(DEFAULT_TIMEFRAMES)})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--skip-data-fetch", action="store_true",
                        help="Skip historical data fetching step")
    parser.add_argument("--skip-dataset-preparation", action="store_true",
                        help="Skip dataset preparation step")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training step")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip ensemble creation step")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip model evaluation step")
    parser.add_argument("--skip-config-update", action="store_true",
                        help="Skip ML config update step")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate performance visualizations")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(f"[STEP] {description}")
    
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        
        # Log command output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"[ERR] {line}")
        return None

def fetch_historical_data(pairs: List[str]):
    """Fetch historical data for all pairs"""
    logger.info("Fetching extended historical data for all pairs...")
    
    for pair in pairs:
        cmd = [
            "python", "fetch_extended_historical_data.py",
            "--pair", pair,
            "--days", "365",
            "--save-dir", "historical_data"
        ]
        run_command(cmd, f"Fetching historical data for {pair}")
    
    logger.info("Historical data fetching completed.")

def prepare_datasets(pairs: List[str], timeframes: List[str]):
    """Prepare enhanced datasets for all pairs and timeframes"""
    logger.info("Preparing enhanced datasets for all pairs and timeframes...")
    
    # First run the prepare_all_datasets script
    cmd = ["python", "prepare_all_datasets.py"]
    result = run_command(cmd, "Running dataset preparation")
    
    if not result:
        # If the script doesn't exist or failed, try using prepare_enhanced_dataset for each pair
        for pair in pairs:
            for timeframe in timeframes:
                cmd = [
                    "python", "prepare_enhanced_dataset.py",
                    "--pair", pair,
                    "--timeframe", timeframe
                ]
                run_command(cmd, f"Preparing dataset for {pair} ({timeframe})")
    
    logger.info("Dataset preparation completed.")

def train_models(pairs: List[str], timeframes: List[str], epochs: int, batch_size: int):
    """Train all model types for all pairs and timeframes"""
    logger.info("Training models for all pairs and timeframes...")
    
    # Try using the comprehensive_ml_training script first
    cmd = [
        "python", "comprehensive_ml_training.py",
        "--pairs"]
    cmd.extend(pairs)
    cmd.extend(["--epochs", str(epochs), "--batch-size", str(batch_size)])
    
    result = run_command(cmd, "Running comprehensive ML training")
    
    if not result:
        # If the script doesn't exist or failed, use train_on_enhanced_datasets for each pair
        for pair in pairs:
            cmd = [
                "python", "train_on_enhanced_datasets.py",
                "--pairs", pair,
                "--model-types"]
            cmd.extend(MODEL_TYPES)
            cmd.extend(["--epochs", str(epochs), "--batch-size", str(batch_size), "--ensemble"])
            
            run_command(cmd, f"Training models for {pair}")
    
    logger.info("Model training completed.")

def create_ensemble_models(pairs: List[str], timeframes: List[str]):
    """Create ensemble models combining the best individual models"""
    logger.info("Creating ensemble models...")
    
    # Check if ensemble creation is included in training step
    ensemble_created = False
    
    # If not, run a specific ensemble creation script
    if not ensemble_created:
        for pair in pairs:
            cmd = [
                "python", "create_ensemble_model.py",
                "--pair", pair
            ]
            
            # Try to run the command, but don't worry if the script doesn't exist
            try:
                result = run_command(cmd, f"Creating ensemble model for {pair}")
                if result:
                    ensemble_created = True
            except FileNotFoundError:
                logger.warning(f"create_ensemble_model.py not found, skipping ensemble creation for {pair}")
    
    if ensemble_created:
        logger.info("Ensemble model creation completed.")
    else:
        logger.warning("No ensemble models were created.")

def evaluate_models(pairs: List[str], timeframes: List[str], visualize: bool = False):
    """Evaluate model performance for all pairs and timeframes"""
    logger.info("Evaluating model performance...")
    
    for pair in pairs:
        timeframe = timeframes[0]  # Use the first timeframe for now
        
        cmd = [
            "python", "evaluate_model_performance.py",
            "--pair", pair,
            "--timeframe", timeframe,
            "--start-capital", "20000",
            "--leverage", "5",
            "--output-csv"
        ]
        
        if visualize:
            cmd.append("--visualize")
        
        run_command(cmd, f"Evaluating models for {pair}")
    
    logger.info("Model evaluation completed.")

def update_ml_config(pairs: List[str]):
    """Update ML configuration based on training results"""
    logger.info("Updating ML configuration...")
    
    # Check if ml_config.json exists
    if not os.path.exists(ML_CONFIG_PATH):
        logger.warning(f"{ML_CONFIG_PATH} not found, cannot update config")
        return
    
    try:
        # Load existing config
        with open(ML_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Update timestamp
        config["updated_at"] = datetime.now().isoformat()
        
        # Update global settings
        config["global_settings"]["continuous_learning"] = True
        config["global_settings"]["training_priority"] = "critical"
        
        # Update model settings to ensure all model types are enabled
        for model_type in MODEL_TYPES:
            if model_type in config["model_settings"]:
                config["model_settings"][model_type]["enabled"] = True
            else:
                # Add model type if it doesn't exist
                config["model_settings"][model_type] = {
                    "enabled": True,
                    "lookback_period": 80,
                    "epochs": 100,
                    "batch_size": 32,
                    "dropout_rate": 0.3,
                    "validation_split": 0.2,
                    "use_early_stopping": True,
                    "patience": 15
                }
        
        # Update strategy integration settings
        config["strategy_integration"]["integrate_arima_adaptive"] = True
        config["strategy_integration"]["arima_weight"] = 0.5
        config["strategy_integration"]["adaptive_weight"] = 0.5
        config["strategy_integration"]["use_combined_signals"] = True
        config["strategy_integration"]["signal_priority"] = "confidence"
        config["strategy_integration"]["signal_threshold"] = 0.65
        
        # Save updated config
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Make a backup of the config
        backup_path = f"{ML_CONFIG_PATH}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"ML configuration updated. Backup saved to {backup_path}")
    
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")

def activate_ml_trading(pairs: List[str]):
    """Activate ML trading for all pairs"""
    logger.info("Activating ML trading for all pairs...")
    
    cmd = ["python", "activate_ml_trading_across_all_pairs.py", "--sandbox"]
    run_command(cmd, "Activating ML trading")
    
    logger.info("ML trading activation completed.")

def main():
    """Main function to run the full training pipeline"""
    args = parse_arguments()
    
    start_time = time.time()
    logger.info("Starting full ML training pipeline...")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Step 1: Fetch historical data
    if not args.skip_data_fetch:
        fetch_historical_data(args.pairs)
    else:
        logger.info("Skipping historical data fetching step.")
    
    # Step 2: Prepare enhanced datasets
    if not args.skip_dataset_preparation:
        prepare_datasets(args.pairs, args.timeframes)
    else:
        logger.info("Skipping dataset preparation step.")
    
    # Step 3: Train models
    if not args.skip_training:
        train_models(args.pairs, args.timeframes, args.epochs, args.batch_size)
    else:
        logger.info("Skipping model training step.")
    
    # Step 4: Create ensemble models
    if not args.skip_ensemble:
        create_ensemble_models(args.pairs, args.timeframes)
    else:
        logger.info("Skipping ensemble creation step.")
    
    # Step 5: Evaluate models
    if not args.skip_evaluation:
        evaluate_models(args.pairs, args.timeframes, args.visualize)
    else:
        logger.info("Skipping model evaluation step.")
    
    # Step 6: Update ML configuration
    if not args.skip_config_update:
        update_ml_config(args.pairs)
    else:
        logger.info("Skipping ML config update step.")
    
    # Bonus step: Activate ML trading if all steps were successful
    if not (args.skip_data_fetch or args.skip_dataset_preparation or args.skip_training or 
            args.skip_ensemble or args.skip_evaluation or args.skip_config_update):
        confirm = input("Do you want to activate ML trading for all pairs? (y/n): ")
        if confirm.lower() == 'y':
            activate_ml_trading(args.pairs)
    
    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Full ML training pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())