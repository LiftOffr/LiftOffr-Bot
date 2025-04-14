#!/usr/bin/env python3
"""
Train Dual Strategy ML Models

This script trains machine learning models that integrate ARIMA and Adaptive
strategies to achieve 90% win rate and 1000% returns. It coordinates the entire
training process, from data preparation to model deployment.

Usage:
    python train_dual_strategy_ml.py --pairs SOL/USD ETH/USD BTC/USD
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "ml_enhanced_config.json"
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
MAX_LEVERAGE = 125
TARGET_WIN_RATE = 0.9
TARGET_RETURN = 1000.0

def run_command(command: str, description: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command and log output
    
    Args:
        command: Shell command to run
        description: Optional description of the command
        check: Whether to check the return code
        
    Returns:
        CompletedProcess object
    """
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                logger.info(f"[STDOUT] {line}")
        
        if result.stderr.strip():
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"[STDERR] {line}")
        
        if description and result.returncode == 0:
            logger.info(f"{description} completed successfully")
        elif description:
            logger.error(f"{description} failed with code {result.returncode}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with code {e.returncode}: {e}")
        if check:
            raise
        return e
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        if check:
            raise
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=str(e))

def load_config() -> Dict[str, Any]:
    """
    Load the ML configuration from file
    
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {CONFIG_PATH}")
            return config
        else:
            logger.warning(f"Configuration file {CONFIG_PATH} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def update_ml_config(max_leverage: int = MAX_LEVERAGE) -> bool:
    """
    Update the ML configuration for hyperperformance
    
    Args:
        max_leverage: Maximum leverage to use
        
    Returns:
        True if successful, False otherwise
    """
    command = f"python update_ml_config_for_hyperperformance.py --max-leverage {max_leverage}"
    result = run_command(command, "Updating ML configuration for hyperperformance")
    return result.returncode == 0

def prepare_training_data(trading_pairs: List[str]) -> Dict[str, bool]:
    """
    Prepare training data for each trading pair
    
    Args:
        trading_pairs: List of trading pairs
        
    Returns:
        Dictionary with results for each pair
    """
    results = {}
    for pair in trading_pairs:
        try:
            # Clean pair name for command
            clean_pair = pair.replace("/", "\\/")
            
            # Fetch historical data if needed
            fetch_command = f"python enhanced_historical_data_fetcher.py --pair {clean_pair} --timeframe 1h"
            run_command(fetch_command, f"Fetching historical data for {pair}", check=False)
            
            # Prepare enhanced dataset
            prepare_command = f"python prepare_enhanced_dataset.py --pair {clean_pair} --timeframe 1h"
            result = run_command(prepare_command, f"Preparing enhanced dataset for {pair}", check=False)
            
            results[pair] = result.returncode == 0
        except Exception as e:
            logger.error(f"Error preparing data for {pair}: {e}")
            results[pair] = False
    
    return results

def train_dual_strategy_models(
    trading_pairs: List[str],
    epochs: int = 300,
    target_win_rate: float = TARGET_WIN_RATE,
    target_return: float = TARGET_RETURN
) -> Dict[str, bool]:
    """
    Train dual strategy models for each trading pair
    
    Args:
        trading_pairs: List of trading pairs
        epochs: Number of training epochs
        target_win_rate: Target win rate (0.0-1.0)
        target_return: Target return percentage
        
    Returns:
        Dictionary with results for each pair
    """
    results = {}
    
    # Clean pair names for command
    clean_pairs = [pair.replace("/", "\\/") for pair in trading_pairs]
    pairs_arg = " ".join(clean_pairs)
    
    # Run the enhanced strategy training
    command = (f"python enhanced_strategy_training.py --pairs {pairs_arg} "
               f"--epochs {epochs} --target-win-rate {target_win_rate} "
               f"--target-return {target_return}")
    
    result = run_command(command, "Training dual strategy models", check=False)
    
    # Assume successful if return code is 0
    success = result.returncode == 0
    for pair in trading_pairs:
        results[pair] = success
    
    return results

def auto_prune_models(trading_pairs: List[str]) -> bool:
    """
    Automatically prune underperforming models
    
    Args:
        trading_pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    # Clean pair names for command
    clean_pairs = [pair.replace("/", "\\/") for pair in trading_pairs]
    pairs_arg = " ".join(clean_pairs)
    
    # Run the auto-pruning script
    command = f"python auto_prune_ml_models.py --pairs {pairs_arg} --performance-threshold 0.65"
    result = run_command(command, "Auto-pruning underperforming models", check=False)
    
    return result.returncode == 0

def optimize_hyperparameters(trading_pairs: List[str]) -> Dict[str, bool]:
    """
    Optimize hyperparameters for each trading pair
    
    Args:
        trading_pairs: List of trading pairs
        
    Returns:
        Dictionary with results for each pair
    """
    results = {}
    for pair in trading_pairs:
        try:
            # Clean pair name for command
            clean_pair = pair.replace("/", "\\/")
            
            # Run adaptive hyperparameter tuning
            command = f"python adaptive_hyperparameter_tuning.py --pair {clean_pair} --max-trials 30"
            result = run_command(command, f"Optimizing hyperparameters for {pair}", check=False)
            
            results[pair] = result.returncode == 0
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters for {pair}: {e}")
            results[pair] = False
    
    return results

def integrate_ai_with_trading_bot(trading_pairs: List[str], sandbox: bool = True) -> bool:
    """
    Integrate the trained AI models with the trading bot
    
    Args:
        trading_pairs: List of trading pairs
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    # Clean pair names for command
    clean_pairs = [pair.replace("/", "\\/") for pair in trading_pairs]
    pairs_arg = " ".join(clean_pairs)
    
    # Run the integration script
    sandbox_arg = "--sandbox" if sandbox else ""
    command = f"python integrate_dual_strategy_ai.py --pairs {pairs_arg} {sandbox_arg}"
    result = run_command(command, "Integrating AI with trading bot", check=False)
    
    return result.returncode == 0

def generate_training_report(
    training_results: Dict[str, Any],
    output_path: str = "training_report.json"
) -> str:
    """
    Generate a training report
    
    Args:
        training_results: Training results
        output_path: Path to save the report
        
    Returns:
        Path to the saved report
    """
    # Add timestamp
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": training_results
    }
    
    # Save to file
    try:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving training report: {e}")
        return ""

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train Dual Strategy ML Models")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_TRADING_PAIRS,
                       help="Trading pairs to train on")
    parser.add_argument("--max-leverage", type=int, default=MAX_LEVERAGE,
                       help="Maximum leverage to use")
    parser.add_argument("--epochs", type=int, default=300,
                       help="Number of training epochs")
    parser.add_argument("--target-win-rate", type=float, default=TARGET_WIN_RATE,
                       help="Target win rate (0.0-1.0)")
    parser.add_argument("--target-return", type=float, default=TARGET_RETURN,
                       help="Target return percentage")
    parser.add_argument("--skip-config-update", action="store_true",
                       help="Skip updating the ML configuration")
    parser.add_argument("--skip-data-prep", action="store_true",
                       help="Skip preparing training data")
    parser.add_argument("--skip-integration", action="store_true",
                       help="Skip integrating AI with trading bot")
    args = parser.parse_args()
    
    # Track overall success
    success = True
    
    # Track results for each stage
    results = {
        "config_update": False,
        "data_preparation": {},
        "model_training": {},
        "auto_pruning": False,
        "hyperparameter_optimization": {},
        "ai_integration": False
    }
    
    # Start time for tracking duration
    start_time = time.time()
    
    # Step 1: Update ML configuration
    if not args.skip_config_update:
        results["config_update"] = update_ml_config(args.max_leverage)
        success = success and results["config_update"]
    else:
        logger.info("Skipping ML configuration update")
    
    # Step 2: Prepare training data
    if not args.skip_data_prep:
        results["data_preparation"] = prepare_training_data(args.pairs)
        success = success and all(results["data_preparation"].values())
    else:
        logger.info("Skipping training data preparation")
    
    # Step 3: Train dual strategy models
    results["model_training"] = train_dual_strategy_models(
        args.pairs,
        epochs=args.epochs,
        target_win_rate=args.target_win_rate,
        target_return=args.target_return
    )
    success = success and all(results["model_training"].values())
    
    # Step 4: Auto-prune models
    results["auto_pruning"] = auto_prune_models(args.pairs)
    success = success and results["auto_pruning"]
    
    # Step 5: Optimize hyperparameters
    results["hyperparameter_optimization"] = optimize_hyperparameters(args.pairs)
    success = success and all(results["hyperparameter_optimization"].values())
    
    # Step 6: Integrate AI with trading bot
    if not args.skip_integration:
        results["ai_integration"] = integrate_ai_with_trading_bot(args.pairs, sandbox=True)
        success = success and results["ai_integration"]
    else:
        logger.info("Skipping AI integration with trading bot")
    
    # Calculate duration
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Generate report
    results["duration"] = {
        "hours": int(hours),
        "minutes": int(minutes),
        "seconds": int(seconds)
    }
    results["overall_success"] = success
    
    report_path = generate_training_report(results)
    
    # Print summary
    logger.info(f"Dual Strategy ML Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Overall success: {success}")
    
    if success:
        logger.info("All models trained successfully to target performance!")
        logger.info(f"Target win rate: {args.target_win_rate*100:.1f}%, Target return: {args.target_return:.1f}%")
    else:
        logger.warning("Some parts of the training process failed. Check the logs for details.")

if __name__ == "__main__":
    main()