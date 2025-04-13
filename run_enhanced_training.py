#!/usr/bin/env python3
"""
Enhanced Training and Optimization Process for Kraken Trading Bot

This script orchestrates a comprehensive training and optimization
process for the Kraken trading bot, including:

1. Fetching historical data for multiple cryptocurrencies
2. Analyzing correlations and market regimes
3. Training ML models with advanced techniques
4. Optimizing trading strategies through backtesting
5. Generating detailed reports and visualizations
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("enhanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
OUTPUT_DIR = "optimization_results"
DATA_DIR = "historical_data"


def execute_script(script, args=None, description=None):
    """
    Execute a Python script with optional arguments
    
    Args:
        script (str): Script name
        args (list): Command line arguments
        description (str): Description for logging
    
    Returns:
        bool: Success status
    """
    if description:
        logger.info(f"{description}...")
    
    if args is None:
        args = []
    
    command = [sys.executable, script] + args
    logger.info(f"Executing: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log output in real-time
        for line in process.stdout:
            logger.info(f"[{script}] {line.strip()}")
        
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read()
            logger.error(f"Error in {script}: {stderr}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error executing {script}: {e}")
        return False


def fetch_historical_data(pairs, timeframes):
    """
    Fetch historical data for multiple pairs and timeframes
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
    
    Returns:
        bool: Success status
    """
    logger.info("Starting historical data fetching process")
    
    # Ensure data directories exist
    for pair in pairs:
        pair_dir = os.path.join(DATA_DIR, pair.replace("/", ""))
        os.makedirs(pair_dir, exist_ok=True)
    
    success = execute_script(
        "enhanced_historical_data_fetcher.py",
        description="Fetching historical data from Kraken"
    )
    
    if not success:
        logger.error("Failed to fetch historical data")
        return False
    
    return True


def analyze_correlations(pairs, timeframes):
    """
    Analyze correlations between assets
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
    
    Returns:
        bool: Success status
    """
    logger.info("Starting correlation analysis")
    
    success = execute_script(
        "multi_asset_correlation_analyzer.py",
        description="Analyzing correlations between assets"
    )
    
    if not success:
        logger.error("Failed to complete correlation analysis")
        return False
    
    return True


def train_ml_models(pairs):
    """
    Train ML models for each pair
    
    Args:
        pairs (list): List of trading pairs
    
    Returns:
        bool: Success status
    """
    logger.info("Starting ML model training")
    
    all_success = True
    
    for pair in pairs:
        symbol = pair.replace("/", "")
        
        success = execute_script(
            "advanced_ml_training.py",
            args=["--symbol", symbol, "--epochs", "50", "--early-stopping"],
            description=f"Training ML models for {pair}"
        )
        
        if not success:
            logger.error(f"Failed to train ML models for {pair}")
            all_success = False
    
    return all_success


def optimize_strategies(pairs, timeframes, strategies):
    """
    Optimize trading strategies through backtesting
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        strategies (list): List of strategies to optimize
    
    Returns:
        dict: Dictionary with optimization results
    """
    logger.info("Starting strategy optimization through backtesting")
    
    results = {}
    
    for pair in pairs:
        pair_results = {}
        
        for timeframe in timeframes:
            # Create directory for this pair/timeframe combination
            result_dir = os.path.join(OUTPUT_DIR, pair.replace("/", ""), timeframe)
            os.makedirs(result_dir, exist_ok=True)
            
            for strategy in strategies:
                logger.info(f"Optimizing {strategy} strategy for {pair} on {timeframe} timeframe")
                
                output_file = os.path.join(result_dir, f"{strategy}_optimization.json")
                
                success = execute_script(
                    "comprehensive_backtest.py",
                    args=[
                        "--pair", pair,
                        "--timeframe", timeframe,
                        "--strategy", strategy,
                        "--optimize",
                        "--plot",
                        "--output", result_dir
                    ],
                    description=f"Running backtest optimization for {strategy} on {pair} ({timeframe})"
                )
                
                if not success:
                    logger.error(f"Failed to optimize {strategy} for {pair} on {timeframe}")
                    continue
                
                # Check if optimization results were saved
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            strategy_results = json.load(f)
                            
                        if timeframe not in pair_results:
                            pair_results[timeframe] = {}
                            
                        pair_results[timeframe][strategy] = strategy_results
                        
                    except Exception as e:
                        logger.error(f"Error reading optimization results: {e}")
            
        results[pair] = pair_results
    
    return results


def run_multi_strategy_backtest(pairs, timeframes):
    """
    Run backtests with multiple strategies
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
    
    Returns:
        bool: Success status
    """
    logger.info("Running multi-strategy backtests")
    
    all_success = True
    
    for pair in pairs:
        for timeframe in timeframes:
            result_dir = os.path.join(OUTPUT_DIR, pair.replace("/", ""), timeframe)
            
            success = execute_script(
                "comprehensive_backtest.py",
                args=[
                    "--pair", pair,
                    "--timeframe", timeframe,
                    "--multi-strategy",
                    "--ml",
                    "--plot",
                    "--output", result_dir
                ],
                description=f"Running multi-strategy backtest for {pair} on {timeframe}"
            )
            
            if not success:
                logger.error(f"Failed to run multi-strategy backtest for {pair} on {timeframe}")
                all_success = False
    
    return all_success


def generate_summary_report(optimization_results):
    """
    Generate a summary report of optimization results
    
    Args:
        optimization_results (dict): Optimization results
        
    Returns:
        dict: Summary report
    """
    logger.info("Generating summary report")
    
    summary = {
        "pairs": {},
        "strategies": {},
        "best_combinations": [],
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Initialize strategy stats
    strategy_stats = {}
    
    # Process each pair
    for pair, pair_results in optimization_results.items():
        pair_summary = {
            "timeframes": {}
        }
        
        best_pair_combination = None
        best_pair_performance = -float('inf')
        
        # Process each timeframe
        for timeframe, timeframe_results in pair_results.items():
            timeframe_summary = {
                "strategies": {}
            }
            
            best_tf_combination = None
            best_tf_performance = -float('inf')
            
            # Process each strategy
            for strategy, strategy_results in timeframe_results.items():
                # Initialize strategy stats if needed
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        "total_profit": 0.0,
                        "win_rate_sum": 0.0,
                        "count": 0
                    }
                
                # Extract metrics
                if "best_metrics" in strategy_results:
                    metrics = strategy_results["best_metrics"]
                    
                    # Update strategy stats
                    strategy_stats[strategy]["total_profit"] += metrics.get("profit_loss_pct", 0.0)
                    strategy_stats[strategy]["win_rate_sum"] += metrics.get("win_rate", 0.0)
                    strategy_stats[strategy]["count"] += 1
                    
                    # Add to timeframe summary
                    timeframe_summary["strategies"][strategy] = {
                        "profit_loss_pct": metrics.get("profit_loss_pct", 0.0),
                        "win_rate": metrics.get("win_rate", 0.0),
                        "max_drawdown": metrics.get("max_drawdown", 0.0),
                        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0)
                    }
                    
                    # Check if this is the best combination for this timeframe
                    performance = metrics.get("profit_loss_pct", 0.0)
                    if performance > best_tf_performance:
                        best_tf_performance = performance
                        best_tf_combination = {
                            "pair": pair,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "profit_loss_pct": performance,
                            "win_rate": metrics.get("win_rate", 0.0)
                        }
            
            # Add best combination for this timeframe
            if best_tf_combination:
                timeframe_summary["best_combination"] = best_tf_combination
                
                # Check if this is the best combination for this pair
                if best_tf_performance > best_pair_performance:
                    best_pair_performance = best_tf_performance
                    best_pair_combination = best_tf_combination
            
            # Add timeframe summary to pair summary
            pair_summary["timeframes"][timeframe] = timeframe_summary
        
        # Add best combination for this pair
        if best_pair_combination:
            pair_summary["best_combination"] = best_pair_combination
            summary["best_combinations"].append(best_pair_combination)
        
        # Add pair summary to main summary
        summary["pairs"][pair] = pair_summary
    
    # Compute average statistics for each strategy
    for strategy, stats in strategy_stats.items():
        if stats["count"] > 0:
            summary["strategies"][strategy] = {
                "avg_profit_pct": stats["total_profit"] / stats["count"],
                "avg_win_rate": stats["win_rate_sum"] / stats["count"],
                "test_count": stats["count"]
            }
    
    # Save summary report
    summary_file = os.path.join(OUTPUT_DIR, "optimization_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary report to {summary_file}")
    
    return summary


def main():
    """Main function to run the enhanced training process"""
    parser = argparse.ArgumentParser(description="Enhanced training and optimization for Kraken Trading Bot")
    
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                       help=f"Trading pairs to process (default: {DEFAULT_PAIRS})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                       help=f"Timeframes to process (default: {DEFAULT_TIMEFRAMES})")
    parser.add_argument("--strategies", nargs="+", default=["arima", "integrated", "mlenhanced"],
                       help="Strategies to optimize (default: arima integrated mlenhanced)")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip historical data fetching")
    parser.add_argument("--skip-correlation", action="store_true",
                       help="Skip correlation analysis")
    parser.add_argument("--skip-ml", action="store_true",
                       help="Skip ML model training")
    parser.add_argument("--skip-optimization", action="store_true",
                       help="Skip strategy optimization")
    parser.add_argument("--skip-multi", action="store_true",
                       help="Skip multi-strategy backtests")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start time measurement
    start_time = time.time()
    
    logger.info("Starting enhanced training and optimization process")
    logger.info(f"Processing pairs: {args.pairs}")
    logger.info(f"Processing timeframes: {args.timeframes}")
    logger.info(f"Optimizing strategies: {args.strategies}")
    
    # Step 1: Fetch historical data
    if not args.skip_data:
        if not fetch_historical_data(args.pairs, args.timeframes):
            logger.error("Historical data fetching failed, aborting process")
            return
    else:
        logger.info("Skipping historical data fetching")
    
    # Step 2: Analyze correlations
    if not args.skip_correlation:
        if not analyze_correlations(args.pairs, args.timeframes):
            logger.warning("Correlation analysis failed, continuing with process")
    else:
        logger.info("Skipping correlation analysis")
    
    # Step 3: Train ML models
    if not args.skip_ml:
        if not train_ml_models(args.pairs):
            logger.warning("ML model training failed for some pairs, continuing with process")
    else:
        logger.info("Skipping ML model training")
    
    # Step 4: Optimize trading strategies
    optimization_results = {}
    if not args.skip_optimization:
        optimization_results = optimize_strategies(args.pairs, args.timeframes, args.strategies)
        
        # Generate summary report
        generate_summary_report(optimization_results)
    else:
        logger.info("Skipping strategy optimization")
    
    # Step 5: Run multi-strategy backtests
    if not args.skip_multi:
        if not run_multi_strategy_backtest(args.pairs, args.timeframes):
            logger.warning("Multi-strategy backtests failed for some combinations")
    else:
        logger.info("Skipping multi-strategy backtests")
    
    # Calculate total runtime
    runtime = time.time() - start_time
    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Enhanced training and optimization process completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()