#!/usr/bin/env python3
"""
Optimize All Trading Pairs

This script runs optimization for all trading pairs to achieve:
1. Maximum prediction accuracy (targeting near 100%)
2. Maximum returns in backtesting
3. Improved risk-adjusted performance metrics

It coordinates all the optimization components into a single pipeline
for consistent improvements across all cryptocurrency pairs.

Usage:
    python optimize_all_trading_pairs.py [--target-accuracy 0.99] [--iterations 100]
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Dict, List, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/optimize_all_pairs.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_TARGET_ACCURACY = 0.99  # 99% accuracy target
DEFAULT_ITERATIONS = 100
DEFAULT_PARALLEL_JOBS = 2  # Number of parallel optimization jobs
RESULTS_DIR = "optimization_results/combined"

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
        
        stdout_lines = []
        stderr_lines = []
        
        def read_pipe(pipe, lines):
            for line in pipe:
                lines.append(line)
                if description:
                    logger.info(f"  {line.strip()}")
                else:
                    logger.info(line.strip())
                    
        # Read both pipes simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(read_pipe, process.stdout, stdout_lines),
                pool.submit(read_pipe, process.stderr, stderr_lines)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                future.result()
                
        # Wait for process to complete
        ret_code = process.wait()
        
        if check and ret_code != 0:
            logger.error(f"Command failed with return code {ret_code}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        return False

def optimize_ml_models(pair, target_accuracy, iterations):
    """
    Optimize ML models for a specific trading pair
    
    Args:
        pair: Trading pair (e.g., "SOL/USD")
        target_accuracy: Target accuracy (0.0-1.0)
        iterations: Number of optimization iterations
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting ML model optimization for {pair}...")
    
    # Run hyper-optimized ML ensemble script
    cmd = [
        "python", "hyper_optimize_ml_ensemble.py",
        "--pairs", pair,
        "--target-accuracy", str(target_accuracy),
        "--iterations", str(iterations),
        "--verbose"
    ]
    
    return run_command(cmd, f"Optimizing ML models for {pair}")

def optimize_returns(pair, days=90):
    """
    Optimize for maximum returns on a specific trading pair
    
    Args:
        pair: Trading pair (e.g., "SOL/USD")
        days: Number of days for backtesting
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting returns optimization for {pair}...")
    
    # Run maximize returns script
    cmd = [
        "python", "maximize_returns_backtest.py",
        "--pairs", pair,
        "--days", str(days)
    ]
    
    return run_command(cmd, f"Optimizing returns for {pair}")

def run_comprehensive_backtest(pairs, days=90):
    """
    Run comprehensive backtest on multiple pairs
    
    Args:
        pairs: List of trading pairs
        days: Number of days for backtesting
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Running comprehensive backtest...")
    
    # Run improved comprehensive backtest script
    cmd = [
        "python", "improved_comprehensive_backtest.py",
        "--pairs"] + pairs + [
        "--days", str(days),
        "--optimize",
        "--trials", "50",
        "--target-accuracy", "0.99",
        "--target-return", "2000"
    ]
    
    return run_command(cmd, "Running comprehensive backtest")

def generate_combined_report(results):
    """
    Generate a combined report of all optimization results
    
    Args:
        results: Dictionary of results
        
    Returns:
        str: Path to generated report
    """
    logger.info("Generating combined report...")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Combine results
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_summary": {
            "pairs_optimized": len(results),
            "average_accuracy": sum(r.get("ml_accuracy", 0) for r in results.values()) / len(results) if results else 0,
            "average_return": sum(r.get("backtest_return_pct", 0) for r in results.values()) / len(results) if results else 0,
        },
        "pair_results": results
    }
    
    # Save summary to file
    summary_path = os.path.join(RESULTS_DIR, "optimization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Generate report text
    report = ["=" * 80,
              "OPTIMIZATION RESULTS SUMMARY",
              "=" * 80,
              f"Time: {summary['timestamp']}",
              f"Pairs Optimized: {summary['overall_summary']['pairs_optimized']}",
              f"Average Accuracy: {summary['overall_summary']['average_accuracy'] * 100:.2f}%",
              f"Average Return: {summary['overall_summary']['average_return'] * 100:.2f}%",
              "=" * 80,
              "INDIVIDUAL PAIR RESULTS:",
              "=" * 80]
              
    for pair, result in results.items():
        report.append(f"Pair: {pair}")
        report.append(f"  ML Accuracy: {result.get('ml_accuracy', 0) * 100:.2f}%")
        report.append(f"  Backtest Return: {result.get('backtest_return_pct', 0) * 100:.2f}%")
        report.append(f"  Win Rate: {result.get('win_rate', 0) * 100:.2f}%")
        report.append(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
        report.append(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Max Drawdown: {result.get('max_drawdown_pct', 0) * 100:.2f}%")
        report.append("-" * 40)
        
    report.append("=" * 80)
    
    # Save report to file
    report_path = os.path.join(RESULTS_DIR, "optimization_report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
        
    # Plot comparison charts
    plot_comparison_charts(results)
    
    return report_path

def plot_comparison_charts(results):
    """
    Plot comparison charts for optimization results
    
    Args:
        results: Dictionary of results
    """
    pairs = list(results.keys())
    
    if not pairs:
        logger.warning("No results to plot")
        return
        
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    accuracies = [results[pair].get('ml_accuracy', 0) * 100 for pair in pairs]
    plt.bar(pairs, accuracies)
    plt.title('ML Model Accuracy by Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('Accuracy (%)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"))
    
    # Plot return comparison
    plt.figure(figsize=(12, 6))
    returns = [results[pair].get('backtest_return_pct', 0) * 100 for pair in pairs]
    plt.bar(pairs, returns)
    plt.title('Backtest Return by Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('Return (%)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "return_comparison.png"))
    
    # Plot win rate comparison
    plt.figure(figsize=(12, 6))
    win_rates = [results[pair].get('win_rate', 0) * 100 for pair in pairs]
    plt.bar(pairs, win_rates)
    plt.title('Win Rate by Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('Win Rate (%)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "win_rate_comparison.png"))
    
    # Plot profit factor comparison
    plt.figure(figsize=(12, 6))
    profit_factors = [min(results[pair].get('profit_factor', 0), 10) for pair in pairs]  # Cap at 10 for readability
    plt.bar(pairs, profit_factors)
    plt.title('Profit Factor by Pair')
    plt.xlabel('Trading Pair')
    plt.ylabel('Profit Factor')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "profit_factor_comparison.png"))

def load_existing_results():
    """
    Load existing optimization results if available
    
    Returns:
        dict: Existing results or empty dict if none
    """
    results_path = os.path.join(RESULTS_DIR, "optimization_summary.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                return data.get("pair_results", {})
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
    
    return {}

def extract_ml_results(pair):
    """
    Extract ML optimization results from files
    
    Args:
        pair: Trading pair
        
    Returns:
        dict: ML optimization results
    """
    pair_code = pair.replace("/", "")
    results_path = os.path.join("optimization_results", "enhanced", f"{pair_code}_results.json")
    
    if not os.path.exists(results_path):
        logger.warning(f"ML results not found for {pair}: {results_path}")
        return {}
        
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
            return {
                "ml_accuracy": data.get("achieved_accuracy", 0),
                "target_accuracy": data.get("target_accuracy", 0),
                "optimal_weights": data.get("optimal_weights", {})
            }
    except Exception as e:
        logger.error(f"Error extracting ML results for {pair}: {e}")
        return {}

def extract_backtest_results(pair):
    """
    Extract backtest results from files
    
    Args:
        pair: Trading pair
        
    Returns:
        dict: Backtest results
    """
    backtest_path = os.path.join("backtest_results", "maximized", "maximized_returns_backtest.json")
    
    if not os.path.exists(backtest_path):
        logger.warning(f"Backtest results not found: {backtest_path}")
        return {}
        
    try:
        with open(backtest_path, 'r') as f:
            data = json.load(f)
            
            # Extract pair-specific results if available
            pair_results = {}
            if "asset_summary" in data and pair in data["asset_summary"]:
                pair_stats = data["asset_summary"][pair]
                pair_results = {
                    "trades": pair_stats.get("trades", 0),
                    "win_rate": pair_stats.get("win_rate", 0),
                    "profit": pair_stats.get("profit", 0),
                    "profit_pct": pair_stats.get("profit_pct", 0)
                }
            
            return {
                "backtest_return": data.get("total_return", 0),
                "backtest_return_pct": data.get("total_return_pct", 0),
                "win_rate": data.get("win_rate", 0),
                "profit_factor": data.get("profit_factor", 0),
                "sharpe_ratio": data.get("sharpe_ratio", 0),
                "max_drawdown_pct": data.get("max_drawdown", 0),
                "pair_specific": pair_results
            }
    except Exception as e:
        logger.error(f"Error extracting backtest results: {e}")
        return {}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimize All Trading Pairs")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=SUPPORTED_PAIRS,
                        help="Trading pairs to optimize")
    
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY,
                        help="Target accuracy for ML models")
    
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Number of optimization iterations")
    
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL_JOBS,
                        help="Number of pairs to optimize in parallel")
    
    parser.add_argument("--ml-only", action="store_true",
                        help="Only run ML model optimization")
    
    parser.add_argument("--returns-only", action="store_true",
                        help="Only run returns optimization")
    
    parser.add_argument("--backtest-only", action="store_true",
                        help="Only run comprehensive backtest")
    
    parser.add_argument("--resume", action="store_true",
                        help="Resume optimization from last state")
    
    return parser.parse_args()

def optimize_pair(pair, target_accuracy, iterations, ml_only=False, returns_only=False):
    """
    Run complete optimization for a single pair
    
    Args:
        pair: Trading pair
        target_accuracy: Target accuracy for ML models
        iterations: Number of optimization iterations
        ml_only: Only run ML optimization
        returns_only: Only run returns optimization
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Starting optimization for {pair}...")
    
    # Optimize ML models
    if not returns_only:
        if not optimize_ml_models(pair, target_accuracy, iterations):
            logger.error(f"ML model optimization failed for {pair}")
            return {}
            
    # Optimize returns
    if not ml_only:
        if not optimize_returns(pair):
            logger.error(f"Returns optimization failed for {pair}")
            # Continue anyway, we might have partial results
            
    # Extract results
    ml_results = extract_ml_results(pair)
    backtest_results = extract_backtest_results(pair)
    
    # Combine results
    results = {**ml_results, **backtest_results}
    
    logger.info(f"Optimization complete for {pair}")
    logger.info(f"  ML Accuracy: {results.get('ml_accuracy', 0) * 100:.2f}%")
    logger.info(f"  Backtest Return: {results.get('backtest_return_pct', 0) * 100:.2f}%")
    
    return results

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Print banner
    print("=" * 80)
    print("OPTIMIZE ALL TRADING PAIRS")
    print("=" * 80)
    print(f"Trading pairs: {' '.join(args.pairs)}")
    print(f"Target accuracy: {args.target_accuracy * 100:.2f}%")
    print(f"Optimization iterations: {args.iterations}")
    print(f"Parallel jobs: {args.parallel}")
    
    if args.ml_only:
        print("Mode: ML optimization only")
    elif args.returns_only:
        print("Mode: Returns optimization only")
    elif args.backtest_only:
        print("Mode: Comprehensive backtest only")
    else:
        print("Mode: Complete optimization")
        
    if args.resume:
        print("Resuming from previous state")
        
    print("=" * 80)
    
    # Load existing results if resuming
    results = {}
    if args.resume:
        results = load_existing_results()
        logger.info(f"Loaded existing results for {len(results)} pairs")
        
    # Run comprehensive backtest only if requested
    if args.backtest_only:
        run_comprehensive_backtest(args.pairs)
        print("Comprehensive backtest complete")
        return
        
    # Optimize each pair
    start_time = time.time()
    
    # Process pairs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # Create tasks
        tasks = {}
        for pair in args.pairs:
            # Skip if already optimized and resuming
            if args.resume and pair in results:
                logger.info(f"Skipping {pair} (already optimized)")
                continue
                
            # Submit task
            tasks[executor.submit(
                optimize_pair,
                pair,
                args.target_accuracy,
                args.iterations,
                args.ml_only,
                args.returns_only
            )] = pair
            
        # Process results as they complete
        for future in concurrent.futures.as_completed(tasks):
            pair = tasks[future]
            try:
                pair_results = future.result()
                if pair_results:
                    results[pair] = pair_results
                    
                    # Save intermediate results
                    generate_combined_report(results)
                    
            except Exception as e:
                logger.error(f"Error optimizing {pair}: {e}")
                
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total optimization time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Generate final report
    report_path = generate_combined_report(results)
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Optimized {len(results)} pairs")
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Report saved to: {report_path}")
    print("=" * 80)
    
    # Print accuracy summary
    print("ACCURACY SUMMARY:")
    for pair, result in results.items():
        accuracy = result.get('ml_accuracy', 0) * 100
        target = args.target_accuracy * 100
        status = "✓" if accuracy >= target else "✗"
        print(f"  {pair}: {accuracy:.2f}% / {target:.2f}% {status}")
        
    print("=" * 80)
    
if __name__ == "__main__":
    main()