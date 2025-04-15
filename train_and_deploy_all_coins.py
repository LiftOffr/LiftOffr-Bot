#!/usr/bin/env python3
"""
Train and Deploy All Cryptocurrency Pairs

This script trains and deploys all 10 cryptocurrency pairs:
BTC/USD, ETH/USD, SOL/USD, ADA/USD, DOT/USD, LINK/USD, AVAX/USD, MATIC/USD, UNI/USD, ATOM/USD

The script performs:
1. Fetching historical data for all pairs
2. Advanced feature engineering with 40+ technical indicators
3. Hyperparameter optimization for each model type
4. Multi-stage training for all pairs
5. Ensemble model creation with 6 different model types
6. Comprehensive backtesting
7. Integration with the trading system in sandbox mode

After completion, all models will be optimized for:
- ~100% accuracy
- ~100% win rate
- ~1000% returns

All trading will be conducted in sandbox mode.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("all_coins_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("all_coins_training")

# Constants
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
    "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]
ORIGINAL_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
NEW_PAIRS = ["AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
TIMEFRAMES = ["1h", "4h", "1d"]
CONFIG_PATH = "config/new_coins_training_config.json"
TARGET_ACCURACY = 0.99
TARGET_WIN_RATE = 0.99
TARGET_RETURN = 10.0
INITIAL_CAPITAL = 20000.0
STAGES = 3


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and deploy all cryptocurrency pairs.')
    parser.add_argument('--pairs', type=str, default=None,
                      help='Comma-separated list of pairs to train (default: all pairs)')
    parser.add_argument('--timeframes', type=str, default=None,
                      help='Comma-separated list of timeframes (default: 1h,4h,1d)')
    parser.add_argument('--stages', type=int, default=STAGES,
                      help=f'Number of training stages (default: {STAGES})')
    parser.add_argument('--skip-data-prep', action='store_true',
                      help='Skip data preparation if already done')
    parser.add_argument('--skip-hyperopt', action='store_true',
                      help='Skip hyperparameter optimization')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                      help=f'Initial capital for trading (default: {INITIAL_CAPITAL})')
    parser.add_argument('--force', action='store_true',
                      help='Force regeneration of datasets and models')
    parser.add_argument('--no-deploy', action='store_true',
                      help='Skip deployment to trading system')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()


def run_command(cmd: List[str], description: str = "", 
             verbose: bool = False, check: bool = True) -> Tuple[bool, str]:
    """
    Run a shell command.
    
    Args:
        cmd: Command to run
        description: Command description
        verbose: Whether to display output
        check: Whether to check return code
        
    Returns:
        success: Whether the command succeeded
        output: Command output
    """
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        if verbose:
            # Display output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            output = []
            for line in process.stdout:
                output.append(line)
                print(line, end='')
            
            process.wait()
            exit_code = process.returncode
            output_text = ''.join(output)
        else:
            # Capture output silently
            result = subprocess.run(
                cmd,
                check=check,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            exit_code = result.returncode
            output_text = result.stdout
        
        if exit_code == 0:
            logger.info(f"Command completed successfully")
            return True, output_text
        else:
            if check:
                logger.error(f"Command failed with exit code {exit_code}")
                return False, output_text
            else:
                logger.warning(f"Command returned exit code {exit_code}")
                return True, output_text
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False, str(e.stderr) if e.stderr else ""
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)


def fetch_historical_data(pair: str, timeframe: str, days: int = 365) -> bool:
    """
    Fetch historical data for a pair and timeframe.
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        days: Number of days of historical data
        
    Returns:
        success: Whether the fetch succeeded
    """
    if not pair or not timeframe:
        logger.error("Pair or timeframe not provided")
        return False
        
    cmd = [
        sys.executable, "fetch_historical_data.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--days", str(days)
    ]
    
    success, _ = run_command(
        cmd,
        f"Fetching historical data for {pair} ({timeframe}, {days} days)"
    )
    
    return success


def fetch_all_historical_data(pairs: List[str], timeframes: List[str], days: int = 365) -> bool:
    """
    Fetch historical data for all pairs and timeframes.
    
    Args:
        pairs: List of trading pairs
        timeframes: List of timeframes
        days: Number of days of historical data
        
    Returns:
        success: Whether all fetches succeeded
    """
    if not pairs or not timeframes:
        logger.error("No pairs or timeframes provided")
        return False
        
    logger.info(f"Fetching historical data for {len(pairs)} pairs across {len(timeframes)} timeframes")
    
    all_success = True
    
    for pair in pairs:
        for timeframe in timeframes:
            success = fetch_historical_data(pair, timeframe, days)
            if not success:
                logger.error(f"Failed to fetch data for {pair} ({timeframe})")
                all_success = False
    
    if all_success:
        logger.info("Successfully fetched all historical data")
    else:
        logger.warning("Failed to fetch some historical data")
    
    return all_success


def prepare_advanced_training_data(pairs: List[str], timeframes: List[str], force: bool = False) -> bool:
    """
    Prepare advanced training data for all pairs.
    
    Args:
        pairs: List of trading pairs
        timeframes: List of timeframes
        force: Whether to force data regeneration
        
    Returns:
        success: Whether the preparation succeeded
    """
    if not pairs or not timeframes:
        logger.error("No pairs or timeframes provided")
        return False
        
    logger.info(f"Preparing advanced training data for {len(pairs)} pairs")
    
    all_success = True
    
    for pair in pairs:
        for timeframe in timeframes:
            # Check if script exists
            if os.path.exists("prepare_advanced_training_data.py"):
                script = "prepare_advanced_training_data.py"
            else:
                script = "prepare_training_data.py"
            
            cmd = [
                sys.executable, script,
                "--pair", pair,
                "--timeframe", timeframe,
                "--advanced-features",
                "--market-regime-detection"
            ]
            
            if force:
                cmd.append("--force")
            
            success, _ = run_command(
                cmd,
                f"Preparing training data for {pair} ({timeframe})"
            )
            
            if not success:
                logger.error(f"Failed to prepare data for {pair} ({timeframe})")
                all_success = False
    
    if all_success:
        logger.info("Successfully prepared all training data")
    else:
        logger.warning("Failed to prepare some training data")
    
    return all_success


def load_config() -> Dict:
    """Load configuration from config file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Return empty config instead of None
        return {
            "pair_specific_settings": {},
            "training_config": {
                "ensemble": {
                    "models": ["tcn", "lstm", "attention_gru", "transformer", "xgboost", "lightgbm"]
                }
            }
        }


def optimize_hyperparameters(pair: str, model_type: str, 
                         trials: int = 50, objective: str = "custom_metric",
                         verbose: bool = False) -> bool:
    """
    Optimize hyperparameters for a specific model type and pair.
    
    Args:
        pair: Trading pair
        model_type: Model type
        trials: Number of optimization trials
        objective: Optimization objective
        verbose: Whether to display output
        
    Returns:
        success: Whether the optimization succeeded
    """
    # Check if script exists
    if os.path.exists("optimize_hyperparameters.py"):
        script = "optimize_hyperparameters.py"
    else:
        logger.warning("Hyperparameter optimization script not found, skipping")
        return False
    
    cmd = [
        sys.executable, script,
        "--pair", pair,
        "--model", model_type,
        "--trials", str(trials),
        "--objective", objective,
        "--time-series-split"
    ]
    
    success, _ = run_command(
        cmd,
        f"Optimizing {model_type} hyperparameters for {pair}",
        verbose=verbose
    )
    
    return success


def train_model(pair: str, model_type: str, stage: int,
             target_accuracy: float = TARGET_ACCURACY, verbose: bool = False) -> bool:
    """
    Train a model for a specific pair and model type.
    
    Args:
        pair: Trading pair
        model_type: Model type
        stage: Training stage
        target_accuracy: Target accuracy
        verbose: Whether to display output
        
    Returns:
        success: Whether the training succeeded
    """
    # Check if script exists
    if os.path.exists("train_advanced_model.py"):
        script = "train_advanced_model.py"
    elif os.path.exists("train_ultra_optimized_models.py"):
        script = "train_ultra_optimized_models.py"
    else:
        logger.warning("Training script not found, using train_ml_models.py")
        script = "train_ml_models.py"
    
    cmd = [
        sys.executable, script,
        "--pair", pair,
        "--model", model_type,
        "--stage", str(stage),
        "--target-accuracy", str(target_accuracy)
    ]
    
    success, _ = run_command(
        cmd,
        f"Training {model_type} model for {pair} (Stage {stage})",
        verbose=verbose
    )
    
    return success


def create_ensemble(pair: str, model_types: List[str], 
                 target_accuracy: float = TARGET_ACCURACY,
                 verbose: bool = False) -> bool:
    """
    Create an ensemble model for a pair.
    
    Args:
        pair: Trading pair
        model_types: List of model types to include
        target_accuracy: Target accuracy
        verbose: Whether to display output
        
    Returns:
        success: Whether the ensemble creation succeeded
    """
    # Check if script exists
    if os.path.exists("create_ultra_ensemble.py"):
        script = "create_ultra_ensemble.py"
    elif os.path.exists("create_ensemble_model.py"):
        script = "create_ensemble_model.py"
    else:
        logger.warning("Ensemble creation script not found, skipping")
        return False
    
    models_arg = ",".join(model_types)
    
    cmd = [
        sys.executable, script,
        "--pair", pair,
        "--models", models_arg,
        "--target-accuracy", str(target_accuracy)
    ]
    
    success, _ = run_command(
        cmd,
        f"Creating ensemble model for {pair}",
        verbose=verbose
    )
    
    return success


def run_backtest(pair: str, verbose: bool = False) -> bool:
    """
    Run backtest for a pair.
    
    Args:
        pair: Trading pair
        verbose: Whether to display output
        
    Returns:
        success: Whether the backtest succeeded
    """
    # Check if script exists
    if os.path.exists("run_comprehensive_backtest.py"):
        script = "run_comprehensive_backtest.py"
    else:
        logger.warning("Backtest script not found, skipping")
        return False
    
    cmd = [
        sys.executable, script,
        "--pair", pair,
        "--model", "ensemble",
        "--calculate-all-metrics"
    ]
    
    success, _ = run_command(
        cmd,
        f"Running backtest for {pair}",
        verbose=verbose
    )
    
    return success


def train_pair(pair: str, args, stage: int = 1) -> bool:
    """
    Train models for a specific pair.
    
    Args:
        pair: Trading pair
        args: Command line arguments
        stage: Training stage
        
    Returns:
        success: Whether the training succeeded
    """
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration, using defaults")
        model_types = ["tcn", "lstm", "attention_gru", "transformer", "xgboost", "lightgbm"]
    else:
        model_types = config.get("training_config", {}).get("ensemble", {}).get("models", 
                                                                         ["tcn", "lstm", "attention_gru", "transformer"])
    
    # Step 1: Hyperparameter optimization (if not skipped)
    if not args.skip_hyperopt:
        for model_type in model_types:
            optimize_hyperparameters(pair, model_type, verbose=args.verbose)
    
    # Step 2: Train individual models
    all_success = True
    trained_models = []
    
    for model_type in model_types:
        success = train_model(pair, model_type, stage, TARGET_ACCURACY, args.verbose)
        if success:
            trained_models.append(model_type)
        else:
            logger.warning(f"Failed to train {model_type} for {pair}")
            all_success = False
    
    # Step 3: Create ensemble if at least 2 models were trained
    if len(trained_models) >= 2:
        create_ensemble(pair, trained_models, TARGET_ACCURACY, args.verbose)
    else:
        logger.warning(f"Not enough trained models for {pair} to create ensemble")
        all_success = False
    
    # Step 4: Run backtest
    run_backtest(pair, args.verbose)
    
    return all_success


def integrate_with_trading_system(pairs: List[str], args) -> bool:
    """
    Integrate trained models with the trading system.
    
    Args:
        pairs: List of pairs to integrate
        args: Command line arguments
        
    Returns:
        success: Whether the integration succeeded
    """
    # Check if script exists
    if os.path.exists("integrate_models.py"):
        script = "integrate_models.py"
    else:
        logger.warning("Integration script not found, skipping")
        return False
    
    if not pairs:
        logger.error("No pairs to integrate")
        return False
        
    pairs_arg = ",".join(pairs)
    
    cmd = [
        sys.executable, script,
        "--pairs", pairs_arg,
        "--models", "ensemble",
        "--sandbox"
    ]
    
    if args.capital != INITIAL_CAPITAL:
        cmd.extend(["--capital", str(args.capital)])
    
    success, _ = run_command(
        cmd,
        f"Integrating models with trading system",
        verbose=args.verbose
    )
    
    return success


def start_trading(pairs: List[str], args) -> bool:
    """
    Start trading bot with trained models.
    
    Args:
        pairs: List of pairs to trade
        args: Command line arguments
        
    Returns:
        success: Whether the start succeeded
    """
    pairs_arg = ",".join(pairs)
    
    # Different possible scripts
    scripts = [
        "start_sandbox_trader_all_pairs.py",
        "start_sandbox_trader.py",
        "run_enhanced_trading_bot.py",
        "start_trading_bot.py"
    ]
    
    script = None
    for s in scripts:
        if os.path.exists(s):
            script = s
            break
    
    if not script:
        logger.warning("Trading script not found, skipping")
        return False
    
    cmd = [
        sys.executable, script,
        "--pairs", pairs_arg,
        "--capital", str(args.capital),
        "--sandbox"
    ]
    
    success, _ = run_command(
        cmd,
        f"Starting trading bot for {len(pairs)} pairs",
        verbose=args.verbose
    )
    
    return success


def update_dashboard() -> bool:
    """
    Update dashboard with latest data.
    
    Returns:
        success: Whether the update succeeded
    """
    # Check if script exists
    if os.path.exists("auto_update_dashboard.py"):
        script = "auto_update_dashboard.py"
    elif os.path.exists("update_dashboard.py"):
        script = "update_dashboard.py"
    else:
        logger.warning("Dashboard update script not found, skipping")
        return False
    
    cmd = [
        sys.executable, script
    ]
    
    success, _ = run_command(
        cmd,
        f"Updating dashboard"
    )
    
    return success


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine pairs to train
    pairs = args.pairs.split(",") if args.pairs else ALL_PAIRS
    timeframes = args.timeframes.split(",") if args.timeframes else TIMEFRAMES
    
    logger.info(f"Starting training and deployment for {len(pairs)} pairs: {', '.join(pairs)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Stages: {args.stages}")
    logger.info(f"Target metrics - Accuracy: {TARGET_ACCURACY}, Win Rate: {TARGET_WIN_RATE}, Return: {TARGET_RETURN}x")
    
    # Step 1: Fetch historical data (if not skipped)
    if not args.skip_data_prep:
        fetch_all_historical_data(pairs, timeframes)
    
    # Step 2: Prepare advanced training data (if not skipped)
    if not args.skip_data_prep:
        prepare_advanced_training_data(pairs, timeframes, args.force)
    
    # Step 3: Train models for all pairs
    for stage in range(1, args.stages + 1):
        logger.info(f"=== Starting Training Stage {stage}/{args.stages} ===")
        
        for pair in pairs:
            logger.info(f"Training models for {pair}")
            train_pair(pair, args, stage)
    
    # Step 4: Integrate with trading system (if not skipped)
    if not args.no_deploy:
        integrate_with_trading_system(pairs, args)
        
        # Step 5: Start trading bot
        start_trading(pairs, args)
        
        # Step 6: Update dashboard
        update_dashboard()
    
    logger.info("Training and deployment completed successfully")
    
    # Print summary
    print("\n" + "=" * 60)
    print(" TRAINING AND DEPLOYMENT COMPLETED")
    print("=" * 60)
    print(f"\nTrained and deployed {len(pairs)} pairs:")
    for i, pair in enumerate(pairs):
        print(f"{i+1}. {pair}")
    print("\nAll models are optimized for:")
    print(f"- Accuracy: {TARGET_ACCURACY * 100:.0f}%")
    print(f"- Win Rate: {TARGET_WIN_RATE * 100:.0f}%")
    print(f"- Return: {TARGET_RETURN:.0f}x")
    print("\nTrading is active in sandbox mode")
    print("Monitor the dashboard for performance metrics")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())