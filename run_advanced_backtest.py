#!/usr/bin/env python3
"""
Advanced Backtesting Runner

This script provides a user-friendly interface for running enhanced backtests
with the Kraken Trading Bot, combining all the advanced features into a single command.

It acts as a unified entry point for:
1. Running comprehensive backtests
2. Optimizing strategy parameters
3. Training and evaluating ML models
4. Auto-pruning underperforming model components
5. Generating detailed performance reports

Usage example:
  python run_advanced_backtest.py --full-optimization --symbol SOLUSD --timeframe 1h
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOLS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
RESULTS_DIR = "backtest_results/advanced"
OPTIMIZATION_DIR = "optimization_results"

# Ensure results directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

def run_script(script_name, args=None, description=None):
    """
    Run a Python script with the given arguments
    
    Args:
        script_name (str): Name of script to run
        args (list, optional): Command line arguments
        description (str, optional): Description for logging
        
    Returns:
        bool: Success status
    """
    if description:
        logger.info(description)
    
    cmd = [sys.executable, script_name]
    
    if args:
        cmd.extend(args)
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        return False

def run_enhanced_backtesting(args):
    """
    Run enhanced backtesting
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    # Prepare script arguments
    script_args = []
    
    # Add symbol and timeframe
    script_args.extend(["--symbol", args.symbol, "--timeframe", args.timeframe])
    
    # Add strategy
    if args.strategy:
        script_args.extend(["--strategy", args.strategy])
    
    # Add optimization flags
    if args.optimize:
        script_args.append("--optimize")
    
    if args.walk_forward:
        script_args.append("--walk-forward")
    
    if args.use_best_params:
        script_args.append("--use-best-params")
    
    # Add multi-strategy flag
    if args.multi_strategy:
        script_args.append("--multi-strategy")
    
    # Add multi-asset flag
    if args.multi_asset:
        script_args.append("--multi-asset")
        
        if args.symbols:
            script_args.extend(["--symbols"] + args.symbols)
    
    # Add cross-timeframe flag
    if args.cross_timeframe:
        script_args.append("--cross-timeframe")
        
        if args.timeframes:
            script_args.extend(["--timeframes"] + args.timeframes)
    
    # Add ML flag
    if args.ml:
        script_args.append("--ml")
    
    # Add plot flag
    if args.plot:
        script_args.append("--plot")
    
    # Add output directory
    if not args.no_save:
        script_args.extend(["--output", os.path.join(RESULTS_DIR, args.symbol, args.timeframe)])
    
    # Run script
    return run_script(
        "run_enhanced_backtesting.py",
        script_args,
        f"Running enhanced backtesting for {args.symbol} on {args.timeframe} timeframe"
    )

def run_parameter_optimization(args):
    """
    Run parameter optimization
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    # Prepare script arguments
    script_args = []
    
    # Add strategy, symbol, and timeframe
    script_args.extend(["--strategy", args.strategy, "--symbol", args.symbol, "--timeframe", args.timeframe])
    
    # Add scoring metric
    if args.scoring:
        script_args.extend(["--scoring", args.scoring])
    
    # Add Bayesian optimization flag
    if args.bayesian:
        script_args.append("--bayesian")
    
    # Add market regimes flag
    if args.market_regimes:
        script_args.append("--market-regimes")
    
    # Add multi-asset flag
    if args.multi_asset:
        script_args.append("--multi-asset")
        
        if args.symbols:
            script_args.extend(["--symbols"] + args.symbols)
    
    # Add multi-timeframe flag
    if args.multi_timeframe:
        script_args.append("--multi-timeframe")
        
        if args.timeframes:
            script_args.extend(["--timeframes"] + args.timeframes)
    
    # Add no-plot flag
    if not args.plot:
        script_args.append("--no-plot")
    
    # Add no-save flag
    if args.no_save:
        script_args.append("--no-save")
    
    # Run script
    return run_script(
        "optimize_strategy_parameters.py",
        script_args,
        f"Optimizing parameters for {args.strategy} strategy on {args.symbol} {args.timeframe}"
    )

def train_ml_model(args):
    """
    Train ML model
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    # Prepare script arguments
    script_args = []
    
    # Add symbol and timeframe
    script_args.extend(["--symbol", args.symbol, "--timeframe", args.timeframe])
    
    # Add model parameters
    if args.sequence_length:
        script_args.extend(["--sequence-length", str(args.sequence_length)])
    
    if args.filters:
        script_args.extend(["--filters", str(args.filters)])
    
    if args.kernel_size:
        script_args.extend(["--kernel-size", str(args.kernel_size)])
    
    if args.dropout_rate:
        script_args.extend(["--dropout-rate", str(args.dropout_rate)])
    
    # Add architecture flags
    if args.no_attention:
        script_args.append("--no-attention")
    
    if args.no_transformer:
        script_args.append("--no-transformer")
    
    # Add multi-timeframe flag
    if args.multi_timeframe:
        script_args.append("--multi-timeframe")
    
    # Add output directory
    script_args.extend(["--output-dir", os.path.join("models", "tcn_enhanced", args.symbol, args.timeframe)])
    
    # Run script
    return run_script(
        "enhanced_tcn_model.py",
        script_args,
        f"Training enhanced TCN model for {args.symbol} on {args.timeframe} timeframe"
    )

def auto_prune_models(args):
    """
    Auto-prune ML models
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    # Prepare script arguments
    script_args = []
    
    # Add symbol and timeframe
    script_args.extend(["--symbol", args.symbol, "--timeframe", args.timeframe])
    
    # Add performance threshold
    if args.performance_threshold:
        script_args.extend(["--performance-threshold", str(args.performance_threshold)])
    
    # Add auto-prune-all flag
    if args.auto_prune_all:
        script_args.append("--auto-prune-all")
    
    # Add multi-asset and multi-timeframe support
    if args.symbols:
        script_args.extend(["--symbols"] + args.symbols)
    
    if args.timeframes:
        script_args.extend(["--timeframes"] + args.timeframes)
    
    # Add output directory
    script_args.extend(["--output-dir", os.path.join("models", "pruned", args.symbol, args.timeframe)])
    
    # Run script
    return run_script(
        "auto_prune_ml_models.py",
        script_args,
        f"Auto-pruning ML models for {args.symbol} on {args.timeframe} timeframe"
    )

def run_full_optimization(args):
    """
    Run full optimization process
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    logger.info(f"Starting full optimization process for {args.symbol} on {args.timeframe} timeframe")
    
    # Create timestamp for this optimization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    result_dir = os.path.join(RESULTS_DIR, args.symbol, args.timeframe, f"full_optimization_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(result_dir, "optimization.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    
    # Step 1: Optimize basic ARIMA strategy
    logger.info("Step 1: Optimizing ARIMA strategy")
    arima_args = argparse.Namespace(
        strategy="arima",
        symbol=args.symbol,
        timeframe=args.timeframe,
        scoring="sharpe_ratio",
        bayesian=True,
        market_regimes=False,
        multi_asset=False,
        multi_timeframe=False,
        plot=True,
        no_save=False,
        symbols=None,
        timeframes=None
    )
    run_parameter_optimization(arima_args)
    
    # Step 2: Optimize integrated strategy
    logger.info("Step 2: Optimizing integrated strategy")
    integrated_args = argparse.Namespace(
        strategy="integrated",
        symbol=args.symbol,
        timeframe=args.timeframe,
        scoring="sharpe_ratio",
        bayesian=True,
        market_regimes=False,
        multi_asset=False,
        multi_timeframe=False,
        plot=True,
        no_save=False,
        symbols=None,
        timeframes=None
    )
    run_parameter_optimization(integrated_args)
    
    # Step 3: Train enhanced TCN model
    logger.info("Step 3: Training enhanced TCN model")
    ml_args = argparse.Namespace(
        symbol=args.symbol,
        timeframe=args.timeframe,
        sequence_length=60,
        filters=64,
        kernel_size=3,
        dropout_rate=0.3,
        no_attention=False,
        no_transformer=False,
        multi_timeframe=True,
        output_dir=os.path.join("models", "tcn_enhanced", args.symbol, args.timeframe)
    )
    train_ml_model(ml_args)
    
    # Step 4: Auto-prune ML models
    logger.info("Step 4: Auto-pruning ML models")
    prune_args = argparse.Namespace(
        symbol=args.symbol,
        timeframe=args.timeframe,
        performance_threshold=0.55,
        auto_prune_all=True,
        symbols=None,
        timeframes=None,
        output_dir=os.path.join("models", "pruned", args.symbol, args.timeframe)
    )
    auto_prune_models(prune_args)
    
    # Step 5: Optimize ML-enhanced strategy
    logger.info("Step 5: Optimizing ML-enhanced strategy")
    ml_enhanced_args = argparse.Namespace(
        strategy="ml_enhanced",
        symbol=args.symbol,
        timeframe=args.timeframe,
        scoring="sharpe_ratio",
        bayesian=True,
        market_regimes=True,
        multi_asset=False,
        multi_timeframe=False,
        plot=True,
        no_save=False,
        symbols=None,
        timeframes=None
    )
    run_parameter_optimization(ml_enhanced_args)
    
    # Step 6: Run final backtest with all optimized strategies
    logger.info("Step 6: Running final backtest with all optimized strategies")
    final_args = argparse.Namespace(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy=None,
        optimize=False,
        walk_forward=False,
        use_best_params=True,
        multi_strategy=True,
        multi_asset=False,
        cross_timeframe=False,
        ml=True,
        plot=True,
        no_save=False,
        symbols=None,
        timeframes=None
    )
    run_enhanced_backtesting(final_args)
    
    logger.info("Full optimization process completed")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return True

def main():
    """Main function for advanced backtest runner"""
    parser = argparse.ArgumentParser(description="Advanced Backtesting Runner")
    
    # Main options
    parser.add_argument("--symbol", type=str, default="SOLUSD",
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                       help="Timeframe (default: 1h)")
    parser.add_argument("--strategy", type=str, default="arima",
                       help="Strategy to use (default: arima)")
    
    # Backtesting options
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize strategy parameters")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Use walk-forward optimization")
    parser.add_argument("--use-best-params", action="store_true",
                       help="Use previously optimized parameters")
    parser.add_argument("--multi-strategy", action="store_true",
                       help="Use multiple strategies")
    parser.add_argument("--multi-asset", action="store_true",
                       help="Run backtest for multiple assets")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="List of symbols to backtest")
    parser.add_argument("--cross-timeframe", action="store_true",
                       help="Run cross-timeframe backtest")
    parser.add_argument("--timeframes", nargs="+", default=None,
                       help="List of timeframes to use")
    parser.add_argument("--ml", action="store_true",
                       help="Include ML-enhanced strategies")
    
    # Parameter optimization options
    parser.add_argument("--parameter-optimization", action="store_true",
                       help="Run parameter optimization only")
    parser.add_argument("--scoring", type=str, default="sharpe_ratio",
                       help="Scoring metric for optimization (default: sharpe_ratio)")
    parser.add_argument("--bayesian", action="store_true",
                       help="Use Bayesian optimization")
    parser.add_argument("--market-regimes", action="store_true",
                       help="Optimize for market regimes")
    
    # ML model options
    parser.add_argument("--train-ml", action="store_true",
                       help="Train ML model only")
    parser.add_argument("--sequence-length", type=int, default=None,
                       help="Sequence length for ML model")
    parser.add_argument("--filters", type=int, default=None,
                       help="Number of filters for ML model")
    parser.add_argument("--kernel-size", type=int, default=None,
                       help="Kernel size for ML model")
    parser.add_argument("--dropout-rate", type=float, default=None,
                       help="Dropout rate for ML model")
    parser.add_argument("--no-attention", action="store_true",
                       help="Disable attention mechanism in ML model")
    parser.add_argument("--no-transformer", action="store_true",
                       help="Disable transformer components in ML model")
    
    # Auto-pruning options
    parser.add_argument("--auto-prune", action="store_true",
                       help="Auto-prune ML models only")
    parser.add_argument("--auto-prune-all", action="store_true",
                       help="Auto-prune all ML models")
    parser.add_argument("--performance-threshold", type=float, default=None,
                       help="Performance threshold for auto-pruning")
    
    # Full optimization
    parser.add_argument("--full-optimization", action="store_true",
                       help="Run full optimization process")
    
    # Output options
    parser.add_argument("--plot", action="store_true",
                       help="Plot results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results")
    
    args = parser.parse_args()
    
    # Run appropriate function based on arguments
    if args.full_optimization:
        run_full_optimization(args)
    elif args.parameter_optimization:
        run_parameter_optimization(args)
    elif args.train_ml:
        train_ml_model(args)
    elif args.auto_prune:
        auto_prune_models(args)
    else:
        run_enhanced_backtesting(args)


if __name__ == "__main__":
    main()