#!/usr/bin/env python3
"""
Run Optimization Pipeline

This script provides a unified interface to run the complete optimization pipeline:
1. Optimize ML models to push accuracy close to 100%
2. Optimize position sizing and entry/exit parameters for maximum returns
3. Run comprehensive backtesting to validate performance
4. Generate detailed reports and visualizations

Usage:
    python run_optimization_pipeline.py [--pairs PAIRS] [--mode MODE]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/optimization_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
DEFAULT_TARGET_ACCURACY = 0.99  # 99% target accuracy
OPTIMIZATION_MODES = ["standard", "aggressive", "ultra", "balanced"]
RESULTS_DIR = "pipeline_results"
LOGS_DIR = "logs"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def run_command(cmd, description=None, check=True):
    """Run a shell command and log output"""
    if description:
        logger.info(f"{description}...")
        
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        # Log output
        for line in stdout.splitlines():
            logger.info(line)
            
        for line in stderr.splitlines():
            logger.warning(line)
            
        ret_code = process.returncode
        
        if check and ret_code != 0:
            logger.error(f"Command failed with return code {ret_code}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        return False

def get_optimization_params(mode):
    """
    Get optimization parameters based on mode
    
    Args:
        mode: Optimization mode
        
    Returns:
        dict: Parameters for optimization
    """
    # Default parameters
    params = {
        "target_accuracy": DEFAULT_TARGET_ACCURACY,
        "iterations": 50,
        "parallel": 2,
        "leverage": 20.0,
        "max_leverage": 125.0,
        "risk": 0.20,
        "confidence": 0.65,
        "days": 90
    }
    
    # Adjust based on mode
    if mode == "aggressive":
        params.update({
            "target_accuracy": 0.98,  # Slightly lower accuracy target for speed
            "iterations": 100,
            "parallel": 3,
            "leverage": 50.0,
            "max_leverage": 150.0,
            "risk": 0.30,
            "confidence": 0.60,
            "days": 120
        })
    elif mode == "ultra":
        params.update({
            "target_accuracy": 0.995,  # Higher accuracy target
            "iterations": 200,
            "parallel": 3,
            "leverage": 75.0,
            "max_leverage": 200.0,
            "risk": 0.40,
            "confidence": 0.55,
            "days": 180
        })
    elif mode == "balanced":
        params.update({
            "target_accuracy": 0.97,  # Balanced accuracy target
            "iterations": 75,
            "parallel": 2,
            "leverage": 30.0,
            "max_leverage": 75.0,
            "risk": 0.15,
            "confidence": 0.70,
            "days": 60
        })
        
    return params

def fetch_historical_data(pairs):
    """
    Fetch historical data for all pairs
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Fetching historical data...")
    
    # Run historical data fetcher script
    cmd = [
        "python", "fetch_extended_historical_data.py",
        "--pairs"] + pairs + [
        "--days", "365",
        "--timeframes", "1h", "4h", "1d"
    ]
    
    return run_command(cmd, "Fetching historical data")

def run_ml_optimization(pairs, params):
    """
    Run ML model optimization
    
    Args:
        pairs: List of trading pairs
        params: Optimization parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Running ML model optimization...")
    
    # Run hyper-optimized ML ensemble script
    cmd = [
        "python", "hyper_optimize_ml_ensemble.py",
        "--pairs"] + pairs + [
        "--target-accuracy", str(params["target_accuracy"]),
        "--iterations", str(params["iterations"]),
        "--verbose"
    ]
    
    return run_command(cmd, "Optimizing ML models")

def run_returns_optimization(pairs, params):
    """
    Run returns optimization
    
    Args:
        pairs: List of trading pairs
        params: Optimization parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Running returns optimization...")
    
    # Run maximize returns script for each pair
    success = True
    for pair in pairs:
        logger.info(f"Optimizing returns for {pair}...")
        
        cmd = [
            "python", "maximize_returns_backtest.py",
            "--pairs", pair,
            "--days", str(params["days"]),
            "--leverage", str(params["leverage"]),
            "--max-leverage", str(params["max_leverage"]),
            "--risk", str(params["risk"]),
            "--confidence", str(params["confidence"])
        ]
        
        if not run_command(cmd, f"Optimizing returns for {pair}"):
            logger.error(f"Returns optimization failed for {pair}")
            success = False
            
    return success

def run_comprehensive_backtest(pairs, params):
    """
    Run comprehensive backtest
    
    Args:
        pairs: List of trading pairs
        params: Optimization parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Running comprehensive backtest...")
    
    # Run improved comprehensive backtest script
    cmd = [
        "python", "improved_comprehensive_backtest.py",
        "--pairs"] + pairs + [
        "--days", str(params["days"]),
        "--optimize",
        "--trials", str(params["iterations"]),
        "--target-accuracy", str(params["target_accuracy"]),
        "--leverage", str(params["leverage"]),
        "--max-leverage", str(params["max_leverage"]),
        "--risk", str(params["risk"])
    ]
    
    return run_command(cmd, "Running comprehensive backtest")

def run_optimization_pipeline(pairs, mode):
    """
    Run complete optimization pipeline
    
    Args:
        pairs: List of trading pairs
        mode: Optimization mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting optimization pipeline for {len(pairs)} pairs in {mode} mode...")
    
    # Get optimization parameters
    params = get_optimization_params(mode)
    
    # Start timer
    start_time = time.time()
    
    # Step 1: Fetch historical data
    if not fetch_historical_data(pairs):
        logger.error("Failed to fetch historical data")
        return False
        
    # Step 2: Run ML optimization
    if not run_ml_optimization(pairs, params):
        logger.error("Failed to run ML optimization")
        return False
        
    # Step 3: Run returns optimization
    if not run_returns_optimization(pairs, params):
        logger.warning("Returns optimization had some failures, continuing anyway")
        
    # Step 4: Run comprehensive backtest
    if not run_comprehensive_backtest(pairs, params):
        logger.error("Failed to run comprehensive backtest")
        return False
        
    # Step 5: Generate consolidated results
    if not run_command(["python", "optimize_all_trading_pairs.py", "--pairs"] + pairs):
        logger.warning("Failed to generate consolidated results")
        
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Optimization pipeline complete in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate a summary file
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pairs": pairs,
        "mode": mode,
        "parameters": params,
        "execution_time_seconds": total_time
    }
    
    summary_path = os.path.join(RESULTS_DIR, f"pipeline_summary_{mode}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Pipeline summary saved to {summary_path}")
    
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Optimization Pipeline")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=SUPPORTED_PAIRS,
                        help="Trading pairs to optimize")
    
    parser.add_argument("--mode", type=str, choices=OPTIMIZATION_MODES, default="standard",
                        help="Optimization mode")
    
    parser.add_argument("--historical-only", action="store_true",
                        help="Only fetch historical data")
    
    parser.add_argument("--ml-only", action="store_true",
                        help="Only run ML optimization")
    
    parser.add_argument("--returns-only", action="store_true",
                        help="Only run returns optimization")
    
    parser.add_argument("--backtest-only", action="store_true",
                        help="Only run comprehensive backtest")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Print banner
    print("=" * 80)
    print("OPTIMIZATION PIPELINE")
    print("=" * 80)
    print(f"Trading pairs: {' '.join(args.pairs)}")
    print(f"Mode: {args.mode}")
    
    if args.historical_only:
        print("Running historical data fetch only")
        fetch_historical_data(args.pairs)
    elif args.ml_only:
        print("Running ML optimization only")
        params = get_optimization_params(args.mode)
        run_ml_optimization(args.pairs, params)
    elif args.returns_only:
        print("Running returns optimization only")
        params = get_optimization_params(args.mode)
        run_returns_optimization(args.pairs, params)
    elif args.backtest_only:
        print("Running comprehensive backtest only")
        params = get_optimization_params(args.mode)
        run_comprehensive_backtest(args.pairs, params)
    else:
        print("Running complete optimization pipeline")
        run_optimization_pipeline(args.pairs, args.mode)
        
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    
if __name__ == "__main__":
    main()