#!/usr/bin/env python3
"""
Train All Pairs and Save Metrics

This script orchestrates the training of specialized models for all 10 cryptocurrency pairs,
saving detailed metrics after each training step to allow for incremental progress.

Usage:
    python train_all_pairs_with_metrics.py [--start-pair PAIR] [--start-timeframe TIMEFRAME]
                                           [--force] [--only-metrics]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_metrics.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'SOL/USD',  # Starting with SOL/USD as requested
    'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MODEL_TYPES = ['entry', 'exit', 'cancel', 'sizing', 'ensemble']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train all pairs and save metrics')
    parser.add_argument('--start-pair', type=str, default=None,
                      help='Pair to start training from (e.g., ETH/USD)')
    parser.add_argument('--start-timeframe', type=str, default=None,
                      help='Timeframe to start training from (e.g., 1h)')
    parser.add_argument('--force', action='store_true',
                      help='Force retraining of existing models')
    parser.add_argument('--only-metrics', action='store_true',
                      help='Only collect metrics without training')
    return parser.parse_args()


def run_command(cmd: List[str], description: str = None, timeout: int = None) -> bool:
    """
    Run a shell command and log output
    
    Args:
        cmd: Command to run
        description: Description of the command
        timeout: Timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if description:
        logging.info(f"{description}")
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to log
        for line in process.stdout:
            line = line.strip()
            if line:
                logging.info(line)
        
        # Wait for process to complete
        return_code = process.wait(timeout=timeout)
        
        if return_code != 0:
            logging.error(f"Command failed with exit code {return_code}")
            return False
        
        return True
    
    except subprocess.TimeoutExpired:
        process.kill()
        logging.warning(f"Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False


def fetch_historical_data(pair: str) -> bool:
    """
    Fetch historical data for a specific pair
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Fetching historical data for {pair}...")
    
    # Fetch standard timeframes
    cmd = ["python", "fetch_kraken_historical_data.py", "--pair", pair]
    if not run_command(cmd, f"Fetching standard timeframes for {pair}", timeout=300):
        logging.error(f"Failed to fetch standard timeframes for {pair}")
        return False
    
    # Fetch small timeframes
    cmd = ["python", "fetch_kraken_small_timeframes.py", "--pairs", pair, "--days", "14"]
    if not run_command(cmd, f"Fetching small timeframes for {pair}", timeout=300):
        logging.error(f"Failed to fetch small timeframes for {pair}")
        return False
    
    return True


def train_models_for_pair(pair: str, force: bool = False) -> bool:
    """
    Train all specialized models for a specific pair
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        force: Whether to force retraining
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Training specialized models for {pair}...")
    
    cmd = ["python", "enhanced_model_training.py", "--pair", pair]
    if force:
        cmd.append("--force")
    
    if not run_command(cmd, f"Training specialized models for {pair}", timeout=1800):  # 30 minutes timeout
        logging.error(f"Failed to train specialized models for {pair}")
        return False
    
    return True


def collect_metrics(pair: str = None) -> Dict:
    """
    Collect metrics for trained models
    
    Args:
        pair: Specific pair to collect metrics for (optional)
        
    Returns:
        Dictionary with metrics
    """
    logging.info(f"Collecting metrics{'for ' + pair if pair else ''}...")
    
    metrics = {}
    metrics_path = "training_metrics.json"
    
    # Load existing metrics if available
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logging.error(f"Error loading existing metrics: {e}")
            metrics = {}
    
    # Determine which pairs to process
    if pair:
        pairs_to_process = [pair]
    else:
        pairs_to_process = SUPPORTED_PAIRS
    
    # Process each pair
    for pair in pairs_to_process:
        pair_metrics = {}
        pair_symbol = pair.replace('/', '_')
        
        # Process each timeframe
        for timeframe in TIMEFRAMES:
            timeframe_metrics = {}
            
            # Process each model type
            for model_type in MODEL_TYPES:
                # Skip ensemble for now (will be calculated from components)
                if model_type == 'ensemble':
                    continue
                
                # Check if model info exists
                model_info_path = f"ml_models/{pair_symbol}_{timeframe}_{model_type}_info.json"
                if not os.path.exists(model_info_path):
                    logging.warning(f"Model info not found: {model_info_path}")
                    continue
                
                # Load model info
                try:
                    with open(model_info_path, 'r') as f:
                        model_info = json.load(f)
                    
                    # Extract metrics
                    if 'metrics' in model_info:
                        timeframe_metrics[model_type] = model_info['metrics']
                        logging.info(f"Collected metrics for {pair} ({timeframe}, {model_type})")
                except Exception as e:
                    logging.error(f"Error loading model info for {pair} ({timeframe}, {model_type}): {e}")
            
            # Check if ensemble model exists
            ensemble_info_path = f"ml_models/{pair_symbol}_{timeframe}_ensemble_info.json"
            if os.path.exists(ensemble_info_path):
                try:
                    with open(ensemble_info_path, 'r') as f:
                        ensemble_info = json.load(f)
                    
                    # Extract metrics
                    if 'metrics' in ensemble_info:
                        timeframe_metrics['ensemble'] = ensemble_info['metrics']
                        logging.info(f"Collected ensemble metrics for {pair} ({timeframe})")
                except Exception as e:
                    logging.error(f"Error loading ensemble info for {pair} ({timeframe}): {e}")
            
            # Add timeframe metrics if not empty
            if timeframe_metrics:
                pair_metrics[timeframe] = timeframe_metrics
        
        # Add pair metrics if not empty
        if pair_metrics:
            metrics[pair] = pair_metrics
    
    # Save metrics
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
    
    return metrics


def generate_metrics_report(metrics: Dict) -> str:
    """
    Generate a human-readable metrics report
    
    Args:
        metrics: Dictionary with metrics
        
    Returns:
        Report string
    """
    report = []
    report.append("# Training Metrics Report")
    report.append("")
    
    total_pairs = len(metrics)
    report.append(f"## Summary: {total_pairs} Trading Pairs")
    report.append("")
    
    # Calculate overall statistics
    win_rates = []
    profit_factors = []
    sharpe_ratios = []
    
    for pair, pair_metrics in metrics.items():
        for timeframe, timeframe_metrics in pair_metrics.items():
            if 'ensemble' in timeframe_metrics:
                ensemble_metrics = timeframe_metrics['ensemble']
                if 'win_rate' in ensemble_metrics:
                    win_rates.append(ensemble_metrics['win_rate'])
                if 'profit_factor' in ensemble_metrics:
                    profit_factors.append(ensemble_metrics['profit_factor'])
                if 'sharpe_ratio' in ensemble_metrics:
                    sharpe_ratios.append(ensemble_metrics['sharpe_ratio'])
    
    # Calculate averages
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
    avg_profit_factor = sum(profit_factors) / len(profit_factors) if profit_factors else 0
    avg_sharpe_ratio = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
    
    report.append(f"Average Win Rate: {avg_win_rate:.2%}")
    report.append(f"Average Profit Factor: {avg_profit_factor:.2f}")
    report.append(f"Average Sharpe Ratio: {avg_sharpe_ratio:.2f}")
    report.append("")
    
    # Add detailed metrics for each pair
    for pair, pair_metrics in metrics.items():
        report.append(f"## {pair}")
        report.append("")
        
        best_timeframe = None
        best_profit_factor = 0
        
        # Find best timeframe based on profit factor
        for timeframe, timeframe_metrics in pair_metrics.items():
            if 'ensemble' in timeframe_metrics:
                ensemble_metrics = timeframe_metrics['ensemble']
                if 'profit_factor' in ensemble_metrics:
                    profit_factor = ensemble_metrics['profit_factor']
                    if profit_factor > best_profit_factor:
                        best_profit_factor = profit_factor
                        best_timeframe = timeframe
        
        # Highlight best timeframe
        if best_timeframe:
            report.append(f"Best Timeframe: {best_timeframe} (Profit Factor: {best_profit_factor:.2f})")
            report.append("")
        
        # Add all timeframes
        for timeframe, timeframe_metrics in pair_metrics.items():
            if 'ensemble' in timeframe_metrics:
                ensemble_metrics = timeframe_metrics['ensemble']
                win_rate = ensemble_metrics.get('win_rate', 0)
                profit_factor = ensemble_metrics.get('profit_factor', 0)
                sharpe_ratio = ensemble_metrics.get('sharpe_ratio', 0)
                
                report.append(f"### {timeframe}")
                report.append("")
                report.append(f"- Win Rate: {win_rate:.2%}")
                report.append(f"- Profit Factor: {profit_factor:.2f}")
                report.append(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
                
                # Add component model metrics
                if 'entry' in timeframe_metrics:
                    entry_metrics = timeframe_metrics['entry']
                    precision = entry_metrics.get('precision', 0)
                    recall = entry_metrics.get('recall', 0)
                    f1_score = entry_metrics.get('f1_score', 0)
                    
                    report.append("")
                    report.append("#### Entry Model")
                    report.append(f"- Precision: {precision:.2f}")
                    report.append(f"- Recall: {recall:.2f}")
                    report.append(f"- F1 Score: {f1_score:.2f}")
                
                report.append("")
    
    return "\n".join(report)


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create required directories
    os.makedirs("ml_models", exist_ok=True)
    os.makedirs("historical_data", exist_ok=True)
    
    # Determine starting point
    start_pair_index = 0
    if args.start_pair:
        try:
            start_pair_index = SUPPORTED_PAIRS.index(args.start_pair)
        except ValueError:
            logging.error(f"Unknown pair: {args.start_pair}")
            return False
    
    # If only collecting metrics, skip training
    if args.only_metrics:
        metrics = collect_metrics()
        report = generate_metrics_report(metrics)
        
        # Save report
        with open("metrics_report.md", "w") as f:
            f.write(report)
        
        logging.info(f"Metrics report saved to metrics_report.md")
        return True
    
    # Process each pair
    for i, pair in enumerate(SUPPORTED_PAIRS[start_pair_index:], start=start_pair_index):
        logging.info(f"=== Processing pair {i+1}/{len(SUPPORTED_PAIRS)}: {pair} ===")
        
        # Fetch historical data
        if not fetch_historical_data(pair):
            logging.error(f"Failed to fetch historical data for {pair}")
            continue
        
        # Train models
        if not train_models_for_pair(pair, args.force):
            logging.error(f"Failed to train models for {pair}")
            continue
        
        # Collect metrics for this pair
        metrics = collect_metrics(pair)
        
        # Generate and save report
        report = generate_metrics_report(metrics)
        with open("metrics_report.md", "w") as f:
            f.write(report)
        
        logging.info(f"Metrics for {pair} saved to metrics_report.md")
        
        # Save progress
        with open("training_progress.txt", "w") as f:
            f.write(f"Last completed pair: {pair}\n")
            f.write(f"Completed {i+1}/{len(SUPPORTED_PAIRS)} pairs\n")
        
        logging.info(f"Completed training for {pair}")
    
    logging.info("=== Training completed for all pairs ===")
    return True


if __name__ == "__main__":
    main()