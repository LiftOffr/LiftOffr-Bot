#!/usr/bin/env python3
"""
Train Hybrid Models for All Cryptocurrency Pairs

This script trains hybrid models for all 10 cryptocurrency pairs:
1. Uses simplified architecture for faster training
2. Applies enhanced risk management with 25% max portfolio risk
3. Saves models and results for each pair

Usage:
    python train_all_pairs.py --epochs 10 --batch_size 32
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_all_pairs.log")
    ]
)
logger = logging.getLogger(__name__)

# Import train_one_crypto_pair module
try:
    from train_one_crypto_pair import train_pair, parse_arguments as parse_single_pair_args
except ImportError:
    logger.error("Failed to import train_one_crypto_pair module")
    sys.exit(1)

# Constants
TRADING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]
CONFIG_DIR = "config"
RESULTS_DIR = "training_results"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"

# Create required directories
for directory in [CONFIG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train hybrid models for all cryptocurrency pairs")
    parser.add_argument("--pairs", type=str, nargs="+", default=TRADING_PAIRS,
                        help=f"Trading pairs to train (default: all 10 pairs)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (default: 1h)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per model (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--update_ml_config", action="store_true", default=True,
                        help="Update ML configuration with new models")
    return parser.parse_args()

def update_ml_config_with_max_risk(max_portfolio_risk):
    """Update ML configuration with maximum portfolio risk"""
    try:
        # Load existing configuration if it exists
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        else:
            config = {"models": {}, "global_settings": {}}
        
        # Update global settings
        if "global_settings" not in config:
            config["global_settings"] = {}
        
        # Set maximum portfolio risk
        config["global_settings"]["max_portfolio_risk"] = max_portfolio_risk
        
        # Save configuration
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated ML configuration with maximum portfolio risk: {max_portfolio_risk:.2%}")
        return True
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False

def create_args_for_pair(pair, args):
    """Create arguments for training a single pair"""
    # Create a namespace object with the same attributes as args
    single_pair_args = parse_single_pair_args()
    
    # Copy attributes from args to single_pair_args
    single_pair_args.pair = pair
    single_pair_args.timeframe = args.timeframe
    single_pair_args.epochs = args.epochs
    single_pair_args.batch_size = args.batch_size
    single_pair_args.sequence_length = 60
    single_pair_args.test_size = 0.2
    single_pair_args.validation_size = 0.2
    single_pair_args.max_portfolio_risk = args.max_portfolio_risk
    
    return single_pair_args

def generate_summary_report(results):
    """Generate a summary report of training results"""
    if not results:
        logger.warning("No training results to report")
        return
    
    report = "\n" + "=" * 80 + "\n"
    report += "TRAINING SUMMARY REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Overall statistics
    total_pairs = len(results)
    successful_pairs = sum(1 for result in results.values() if result["success"])
    failed_pairs = total_pairs - successful_pairs
    
    avg_accuracy = sum(result["metrics"]["accuracy"] for result in results.values() if result["success"]) / max(1, successful_pairs)
    avg_win_rate = sum(result["metrics"]["win_rate"] for result in results.values() if result["success"]) / max(1, successful_pairs)
    
    report += f"Total Pairs: {total_pairs}\n"
    report += f"Successful: {successful_pairs}\n"
    report += f"Failed: {failed_pairs}\n"
    report += f"Average Accuracy: {avg_accuracy:.4f}\n"
    report += f"Average Win Rate: {avg_win_rate:.4f}\n\n"
    
    # Individual pair results
    report += "INDIVIDUAL PAIR RESULTS:\n"
    report += "-" * 80 + "\n"
    
    for pair, result in results.items():
        if result["success"]:
            metrics = result["metrics"]
            report += f"{pair}:\n"
            report += f"  Status: {'Success' if result['success'] else 'Failed'}\n"
            report += f"  Model Path: {result['model_path']}\n"
            report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"  Win Rate: {metrics['win_rate']:.4f}\n"
            report += f"  Signal Distribution: Bearish={metrics['signal_distribution']['bearish']:.2%}, "
            report += f"Neutral={metrics['signal_distribution']['neutral']:.2%}, "
            report += f"Bullish={metrics['signal_distribution']['bullish']:.2%}\n"
        else:
            report += f"{pair}:\n"
            report += f"  Status: Failed\n"
            report += f"  Error: {result.get('error', 'Unknown error')}\n"
        
        report += "-" * 80 + "\n"
    
    # Save report to file
    report_path = f"{RESULTS_DIR}/training_summary_report.txt"
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
    
    # Also log the report
    logger.info(report)

def main():
    """Main function"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING HYBRID MODELS FOR ALL CRYPTOCURRENCY PAIRS")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(args.pairs)}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info("=" * 80 + "\n")
    
    # Update ML configuration with maximum portfolio risk
    if args.update_ml_config:
        update_ml_config_with_max_risk(args.max_portfolio_risk)
    
    # Train models for all pairs
    results = {}
    for i, pair in enumerate(args.pairs):
        logger.info(f"\nTraining pair {i+1}/{len(args.pairs)}: {pair}")
        
        try:
            # Create arguments for training this pair
            pair_args = create_args_for_pair(pair, args)
            
            # Train model for this pair
            model_path, metrics = train_pair(pair, pair_args)
            
            # Store results
            results[pair] = {
                "success": True,
                "model_path": model_path,
                "metrics": metrics
            }
            
            logger.info(f"Successfully trained model for {pair}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Win Rate: {metrics['win_rate']:.4f}")
        except Exception as e:
            logger.error(f"Error training model for {pair}: {e}")
            import traceback
            traceback.print_exc()
            
            # Store error result
            results[pair] = {
                "success": False,
                "error": str(e)
            }
    
    # Generate summary report
    generate_summary_report(results)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nTotal execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    logger.info("\nTraining complete for all pairs.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training all pairs: {e}")
        import traceback
        traceback.print_exc()