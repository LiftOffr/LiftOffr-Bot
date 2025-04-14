#!/usr/bin/env python3
"""
Enhanced Dual Strategy AI Trainer

This script provides comprehensive training for the dual-strategy AI system,
targeting 90% win rate and 1000% returns. It orchestrates the entire process:

1. Preparing enhanced datasets that combine ARIMA and Adaptive strategies
2. Training advanced neural network models with asymmetric loss functions
3. Optimizing hyperparameters for maximum performance
4. Auto-pruning underperforming models
5. Deploying trained models to live trading
6. Tracking and reporting performance metrics

Usage:
    python enhanced_dual_strategy_trainer.py --pairs SOL/USD [BTC/USD ETH/USD ...] [--epochs 500] [--target-win-rate 0.9] [--target-return 1000]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import CompletedProcess
from typing import Dict, List, Any, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ["SOL/USD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_EPOCHS = 500
TARGET_WIN_RATE = 0.9
TARGET_RETURN = 1000.0
ML_CONFIG_FILE = "ml_config.json"
MODELS_DIR = "models"
BACKTEST_RESULTS_DIR = "backtest_results"
OPTIMIZATION_RESULTS_DIR = "optimization_results"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Dual Strategy AI Trainer")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help=f"Trading pairs to prepare data for (default: {DEFAULT_PAIRS})"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help=f"Timeframes to prepare data for (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=TARGET_WIN_RATE,
        help=f"Target win rate (default: {TARGET_WIN_RATE})"
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=TARGET_RETURN,
        help=f"Target return percentage (default: {TARGET_RETURN})"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Deploy in sandbox mode (default: False)"
    )
    
    return parser.parse_args()


def run_command(command: str, description: Optional[str] = None, check: bool = True) -> CompletedProcess:
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
        logger.info(description)
    
    logger.info(f"Running command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Command completed with return code: {result.returncode}")
        
        if result.stdout.strip():
            logger.info(f"STDOUT: {result.stdout.strip()}")
        
        if result.stderr.strip():
            logger.warning(f"STDERR: {result.stderr.strip()}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return e


def load_config() -> Dict[str, Any]:
    """
    Load the ML configuration from file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(ML_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load ML config: {e}")
        return {}


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup of the existing file
        if os.path.exists(ML_CONFIG_FILE):
            backup_file = f"{ML_CONFIG_FILE}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
            os.rename(ML_CONFIG_FILE, backup_file)
            logger.info(f"Created backup of ML config file: {backup_file}")
        
        with open(ML_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved ML config to {ML_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save ML config: {e}")
        return False


def update_ml_config_for_hyperperformance() -> bool:
    """
    Update the ML configuration for hyperperformance
    
    Returns:
        True if successful, False otherwise
    """
    config = load_config()
    
    if not config:
        logger.error("Failed to load ML config for update")
        return False
    
    # Update configuration for hyperperformance
    
    # Model architecture
    if "model_architecture" in config:
        config["model_architecture"].update({
            "use_attention": True,
            "use_residual_connections": True,
            "use_batch_normalization": True,
            "dropout_rate": 0.3,
            "recurrent_dropout_rate": 0.3,
            "use_bidirectional": True,
            "use_tcn": True,
            "tcn_nb_filters": 128,
            "tcn_kernel_size": 3,
            "tcn_nb_stacks": 2,
            "use_transformer": True,
            "transformer_num_heads": 8,
            "transformer_ff_dim": 256
        })
    
    # Training parameters
    if "training_parameters" in config:
        config["training_parameters"].update({
            "batch_size": 64,
            "validation_split": 0.2,
            "early_stopping_patience": 20,
            "reduce_lr_patience": 10,
            "reduce_lr_factor": 0.5,
            "use_class_weights": True,
            "use_focal_loss": True,
            "focal_loss_gamma": 2.0,
            "use_asymmetric_loss": True,
            "asymmetric_loss_gamma_pos": 1.0,
            "asymmetric_loss_gamma_neg": 4.0
        })
    
    # Feature engineering
    if "feature_engineering" in config:
        config["feature_engineering"].update({
            "use_cross_asset_features": True,
            "use_sentiment_features": True,
            "use_on_chain_features": True,
            "use_market_regime_features": True,
            "feature_selection_method": "mutual_info",
            "feature_importance_threshold": 0.05,
            "use_feature_engineering": True
        })
    
    # Trading parameters
    if "trading_parameters" in config:
        config["trading_parameters"].update({
            "min_confidence_threshold": 0.7,
            "max_leverage": 125,
            "min_leverage": 20,
            "base_leverage": 25,
            "dynamic_leverage": True,
            "max_risk_percentage": 5.0,
            "base_risk_percentage": 1.0,
            "use_trailing_stop": True,
            "trailing_stop_activation_percentage": 0.5,
            "trailing_stop_callback_percentage": 0.2,
            "use_dual_limit_orders": True,
            "limit_order_offset_percentage": 0.1
        })
    
    # Logging parameters
    if "logging_parameters" in config:
        config["logging_parameters"].update({
            "log_level": "INFO",
            "log_to_file": True,
            "log_file_path": "ml_trading.log",
            "verbose_logging": True,
            "log_trades": True,
            "log_signals": True,
            "log_portfolio": True
        })
    
    # Save updated configuration
    return save_config(config)


def prepare_training_data(trading_pairs: List[str], timeframes: List[str]) -> Dict[str, bool]:
    """
    Prepare training data for each trading pair and timeframe
    
    Args:
        trading_pairs: List of trading pairs
        timeframes: List of timeframes
        
    Returns:
        Dictionary with results for each pair/timeframe combination
    """
    results = {}
    
    for pair in trading_pairs:
        for timeframe in timeframes:
            pair_key = f"{pair}_{timeframe}"
            
            # Run prepare_enhanced_dataset.py for this pair and timeframe
            command = f"python prepare_enhanced_dataset.py --pair '{pair}' --timeframe '{timeframe}'"
            description = f"Preparing enhanced dataset for {pair} ({timeframe})"
            
            result = run_command(command, description)
            
            results[pair_key] = result.returncode == 0
            
            if not results[pair_key]:
                logger.error(f"Failed to prepare training data for {pair} ({timeframe})")
            
            # Allow some time between requests to avoid resource contention
            time.sleep(1)
    
    return results


def train_dual_strategy_models(
    trading_pairs: List[str],
    timeframes: List[str],
    epochs: int = DEFAULT_EPOCHS,
    target_win_rate: float = TARGET_WIN_RATE,
    target_return: float = TARGET_RETURN
) -> Dict[str, bool]:
    """
    Train dual strategy models for each trading pair and timeframe
    
    Args:
        trading_pairs: List of trading pairs
        timeframes: List of timeframes
        epochs: Number of training epochs
        target_win_rate: Target win rate (0.0-1.0)
        target_return: Target return percentage
        
    Returns:
        Dictionary with results for each pair
    """
    results = {}
    
    for pair in trading_pairs:
        pair_success = True
        
        for timeframe in timeframes:
            pair_key = f"{pair}_{timeframe}"
            
            # Run dual strategy training for this pair and timeframe
            command = (
                f"python dual_strategy_ai_integration.py --pair '{pair}' --timeframe '{timeframe}' "
                f"--epochs {epochs} --target-win-rate {target_win_rate} --target-return {target_return}"
            )
            description = f"Training dual strategy model for {pair} ({timeframe})"
            
            result = run_command(command, description)
            
            if result.returncode != 0:
                logger.error(f"Failed to train dual strategy model for {pair} ({timeframe})")
                pair_success = False
            
            # Allow some time between trainings
            time.sleep(2)
        
        results[pair] = pair_success
    
    return results


def auto_prune_models(trading_pairs: List[str], timeframes: List[str]) -> bool:
    """
    Automatically prune underperforming models
    
    Args:
        trading_pairs: List of trading pairs
        timeframes: List of timeframes
        
    Returns:
        True if successful, False otherwise
    """
    success = True
    
    for pair in trading_pairs:
        for timeframe in timeframes:
            command = f"python auto_prune_ml_models.py --pair '{pair}' --timeframe '{timeframe}'"
            description = f"Auto-pruning models for {pair} ({timeframe})"
            
            result = run_command(command, description, check=False)
            
            if result.returncode != 0:
                logger.warning(f"Auto-pruning models for {pair} ({timeframe}) failed, but continuing")
                success = False
    
    return success


def optimize_hyperparameters(trading_pairs: List[str], timeframes: List[str]) -> Dict[str, bool]:
    """
    Optimize hyperparameters for each trading pair and timeframe
    
    Args:
        trading_pairs: List of trading pairs
        timeframes: List of timeframes
        
    Returns:
        Dictionary with results for each pair/timeframe combination
    """
    results = {}
    
    for pair in trading_pairs:
        for timeframe in timeframes:
            pair_key = f"{pair}_{timeframe}"
            
            command = f"python adaptive_hyperparameter_tuning.py --pair '{pair}' --timeframe '{timeframe}'"
            description = f"Optimizing hyperparameters for {pair} ({timeframe})"
            
            result = run_command(command, description, check=False)
            
            results[pair_key] = result.returncode == 0
            
            if not results[pair_key]:
                logger.warning(f"Hyperparameter optimization for {pair} ({timeframe}) failed, but continuing")
    
    return results


def backtest_models(trading_pairs: List[str], timeframes: List[str]) -> Dict[str, Any]:
    """
    Backtest the trained models
    
    Args:
        trading_pairs: List of trading pairs
        timeframes: List of timeframes
        
    Returns:
        Dictionary with backtest results
    """
    results = {}
    
    for pair in trading_pairs:
        for timeframe in timeframes:
            pair_key = f"{pair}_{timeframe}"
            
            command = f"python enhanced_backtesting.py --pair '{pair}' --timeframe '{timeframe}' --use-ml"
            description = f"Backtesting model for {pair} ({timeframe})"
            
            result = run_command(command, description, check=False)
            
            results[pair_key] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
            
            if not results[pair_key]["success"]:
                logger.warning(f"Backtesting for {pair} ({timeframe}) failed, but continuing")
    
    return results


def deploy_to_live_trading(trading_pairs: List[str], sandbox: bool = True) -> bool:
    """
    Deploy trained models to live trading
    
    Args:
        trading_pairs: List of trading pairs
        sandbox: Whether to use sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    pairs_arg = " ".join([f"'{pair}'" for pair in trading_pairs])
    sandbox_arg = "--sandbox" if sandbox else ""
    
    command = f"python activate_ml_trading_across_all_pairs.py --pairs {pairs_arg} {sandbox_arg}"
    description = f"Deploying trained models to {'sandbox' if sandbox else 'live'} trading"
    
    result = run_command(command, description)
    
    return result.returncode == 0


def calculate_portfolio_metrics(trading_pairs: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate portfolio metrics for each trading pair
    
    Args:
        trading_pairs: List of trading pairs
        
    Returns:
        Dictionary with metrics for each pair
    """
    metrics = {}
    
    for pair in trading_pairs:
        command = f"python get_current_status_new.py --pair '{pair}' --output-json"
        description = f"Calculating metrics for {pair}"
        
        result = run_command(command, description, check=False)
        
        try:
            if result.returncode == 0 and result.stdout.strip():
                metrics[pair] = json.loads(result.stdout.strip())
            else:
                logger.warning(f"Failed to calculate metrics for {pair}")
                metrics[pair] = {
                    "pair": pair,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "pnl_usd": 0.0,
                    "pnl_pct": 0.0
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metrics JSON for {pair}")
            metrics[pair] = {
                "pair": pair,
                "total_trades": 0,
                "win_rate": 0.0,
                "pnl_usd": 0.0,
                "pnl_pct": 0.0
            }
    
    return metrics


def generate_performance_report(
    training_results: Dict[str, Any],
    backtest_results: Dict[str, Any],
    portfolio_metrics: Dict[str, Any],
    output_path: str = "performance_report.json"
) -> str:
    """
    Generate a performance report
    
    Args:
        training_results: Training results
        backtest_results: Backtest results
        portfolio_metrics: Portfolio metrics
        output_path: Path to save the report
        
    Returns:
        Path to the saved report
    """
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_results": training_results,
        "backtest_results": backtest_results,
        "portfolio_metrics": portfolio_metrics,
        "summary": {
            "total_pairs": len(portfolio_metrics),
            "total_trades": sum(m.get("total_trades", 0) for m in portfolio_metrics.values()),
            "average_win_rate": np.mean([m.get("win_rate", 0.0) for m in portfolio_metrics.values()]),
            "total_pnl_usd": sum(m.get("pnl_usd", 0.0) for m in portfolio_metrics.values()),
            "total_pnl_pct": np.mean([m.get("pnl_pct", 0.0) for m in portfolio_metrics.values()])
        }
    }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Performance report saved to {output_path}")
    
    return output_path


def create_performance_charts(portfolio_metrics: Dict[str, Dict[str, Any]], output_dir: str = "performance_charts"):
    """
    Create performance charts
    
    Args:
        portfolio_metrics: Portfolio metrics
        output_dir: Directory to save the charts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    pairs = list(portfolio_metrics.keys())
    win_rates = [portfolio_metrics[pair].get("win_rate", 0.0) * 100 for pair in pairs]
    pnl_pcts = [portfolio_metrics[pair].get("pnl_pct", 0.0) for pair in pairs]
    
    # Create win rate chart
    plt.figure(figsize=(12, 6))
    plt.bar(pairs, win_rates, color='green')
    plt.axhline(y=50, color='r', linestyle='--', label='Break-even')
    plt.title('Win Rate by Trading Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rates.png'))
    
    # Create P&L chart
    plt.figure(figsize=(12, 6))
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_pcts]
    plt.bar(pairs, pnl_pcts, color=colors)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('P&L by Trading Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('P&L (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pnl.png'))
    
    logger.info(f"Performance charts saved to {output_dir}")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
    os.makedirs(OPTIMIZATION_RESULTS_DIR, exist_ok=True)
    
    # 1. Update ML configuration for hyperperformance
    logger.info("Updating ML configuration for hyperperformance")
    if not update_ml_config_for_hyperperformance():
        logger.warning("Failed to update ML configuration, but continuing with existing config")
    
    # 2. Prepare training data
    logger.info("Preparing training data")
    data_results = prepare_training_data(args.pairs, args.timeframes)
    
    # Count successful data preparations
    successful_data_preps = sum(1 for result in data_results.values() if result)
    total_data_preps = len(data_results)
    
    logger.info(f"Successfully prepared {successful_data_preps}/{total_data_preps} datasets")
    
    if successful_data_preps == 0:
        logger.error("No datasets were successfully prepared, aborting")
        sys.exit(1)
    
    # 3. Train dual strategy models
    logger.info("Training dual strategy models")
    training_results = train_dual_strategy_models(
        args.pairs,
        args.timeframes,
        args.epochs,
        args.target_win_rate,
        args.target_return
    )
    
    # Count successful training runs
    successful_trainings = sum(1 for result in training_results.values() if result)
    total_trainings = len(training_results)
    
    logger.info(f"Successfully trained {successful_trainings}/{total_trainings} models")
    
    if successful_trainings == 0:
        logger.error("No models were successfully trained, aborting")
        sys.exit(1)
    
    # 4. Auto-prune models
    logger.info("Auto-pruning models")
    auto_prune_models(args.pairs, args.timeframes)
    
    # 5. Optimize hyperparameters
    logger.info("Optimizing hyperparameters")
    hyperparameter_results = optimize_hyperparameters(args.pairs, args.timeframes)
    
    # 6. Backtest models
    logger.info("Backtesting models")
    backtest_results = backtest_models(args.pairs, args.timeframes)
    
    # 7. Deploy to live trading
    if args.sandbox:
        logger.info("Deploying models to sandbox trading")
        if deploy_to_live_trading(args.pairs, sandbox=True):
            logger.info("Models deployed successfully to sandbox trading")
        else:
            logger.warning("Failed to deploy models to sandbox trading")
    
    # 8. Calculate portfolio metrics
    logger.info("Calculating portfolio metrics")
    portfolio_metrics = calculate_portfolio_metrics(args.pairs)
    
    # 9. Generate performance report
    logger.info("Generating performance report")
    report_path = generate_performance_report(
        training_results,
        backtest_results,
        portfolio_metrics
    )
    
    # 10. Create performance charts
    logger.info("Creating performance charts")
    create_performance_charts(portfolio_metrics)
    
    # Summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Pairs: {', '.join(args.pairs)}")
    logger.info(f"Timeframes: {', '.join(args.timeframes)}")
    logger.info(f"Successful data preparations: {successful_data_preps}/{total_data_preps}")
    logger.info(f"Successful trainings: {successful_trainings}/{total_trainings}")
    
    logger.info("\n=== Portfolio Metrics ===")
    for pair, metrics in portfolio_metrics.items():
        win_rate = metrics.get("win_rate", 0.0) * 100
        pnl_usd = metrics.get("pnl_usd", 0.0)
        pnl_pct = metrics.get("pnl_pct", 0.0)
        
        logger.info(f"{pair}: Win Rate: {win_rate:.2f}%, P&L: ${pnl_usd:.2f} ({pnl_pct:.2f}%)")
    
    logger.info(f"\nPerformance report saved to {report_path}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()