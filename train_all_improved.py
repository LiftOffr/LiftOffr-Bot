#!/usr/bin/env python3
"""
Train Improved Hybrid Models for All Cryptocurrency Pairs

This script initiates training for improved hybrid models across 
all cryptocurrency pairs, leveraging the enhanced architecture with:
1. More sophisticated model architecture
2. Advanced feature engineering (40+ indicators)
3. Multi-class prediction for fine-grained signals
4. Data augmentation techniques
5. Dynamic learning rate and batch size adjustments

Usage:
    python train_all_improved.py [--epochs 20] [--batch_size 32]
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
        logging.FileHandler("training_all_improved.log")
    ]
)
logger = logging.getLogger(__name__)

# Import train_improved_model module
try:
    from train_improved_model import main as train_single_pair
    from train_improved_model import parse_arguments as parse_improved_args
except ImportError:
    logger.error("Failed to import train_improved_model module")
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
    parser = argparse.ArgumentParser(description="Train improved hybrid models for all cryptocurrency pairs")
    parser.add_argument("--pairs", type=str, nargs="+", default=TRADING_PAIRS,
                        help=f"Trading pairs to train (default: all 10 pairs)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe to use for training (default: 1h)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs per model (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--update_ml_config", action="store_true", default=True,
                        help="Update ML configuration with new models")
    parser.add_argument("--data_augmentation", action="store_true", default=True,
                        help="Apply data augmentation techniques")
    return parser.parse_args()

def generate_summary_report(results):
    """Generate a summary report of training results"""
    if not results:
        logger.warning("No training results to report")
        return
    
    report = "\n" + "=" * 80 + "\n"
    report += "IMPROVED TRAINING SUMMARY REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Overall statistics
    total_pairs = len(results)
    successful_pairs = sum(1 for result in results.values() if result["success"])
    failed_pairs = total_pairs - successful_pairs
    
    avg_accuracy = sum(result["metrics"]["accuracy"] for result in results.values() if result["success"]) / max(1, successful_pairs)
    avg_win_rate = sum(result["metrics"]["win_rate"] for result in results.values() if result["success"]) / max(1, successful_pairs)
    avg_direction = sum(result["metrics"]["direction_accuracy"] for result in results.values() if result["success"]) / max(1, successful_pairs)
    
    report += f"Total Pairs: {total_pairs}\n"
    report += f"Successful: {successful_pairs}\n"
    report += f"Failed: {failed_pairs}\n"
    report += f"Average Accuracy: {avg_accuracy:.4f}\n"
    report += f"Average Win Rate: {avg_win_rate:.4f}\n"
    report += f"Average Direction Accuracy: {avg_direction:.4f}\n\n"
    
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
            report += f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}\n"
            
            # Add class-wise accuracy
            report += f"  Class-wise Accuracy:\n"
            for class_name, acc in metrics['class_accuracy'].items():
                report += f"    {class_name}: {acc:.4f}\n"
            
            # Add signal distribution
            report += f"  Signal Distribution:\n"
            for class_name, dist in metrics['signal_distribution'].items():
                report += f"    {class_name}: {dist:.2%}\n"
        else:
            report += f"{pair}:\n"
            report += f"  Status: Failed\n"
            report += f"  Error: {result.get('error', 'Unknown error')}\n"
        
        report += "-" * 80 + "\n"
    
    # Save report to file
    report_path = f"{RESULTS_DIR}/improved_training_summary_report.txt"
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
    
    # Also log the report
    logger.info(report)

def train_single_improved_model(pair, args):
    """Train an improved model for a single pair"""
    # Create args for training
    train_args = parse_improved_args()
    train_args.pair = pair
    train_args.timeframe = args.timeframe
    train_args.epochs = args.epochs
    train_args.batch_size = args.batch_size
    train_args.sequence_length = 120
    train_args.test_size = 0.2
    train_args.validation_size = 0.2
    train_args.max_portfolio_risk = args.max_portfolio_risk
    train_args.optimizer_learning_rate = 0.0005
    train_args.data_augmentation = args.data_augmentation
    
    # Set environment variables to disable TF/CUDA warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    logger.info(f"\n{'='*80}\nTraining improved model for {pair}\n{'='*80}")
    
    # Run training
    try:
        model_path, metrics = train_single_pair()
        
        if model_path and metrics:
            return {
                "success": True,
                "model_path": model_path,
                "metrics": metrics
            }
        else:
            return {
                "success": False,
                "error": "Training failed (no model path or metrics returned)"
            }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        logger.error(f"Error training model for {pair}: {e}")
        logger.error(error_msg)
        
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING IMPROVED HYBRID MODELS FOR ALL CRYPTOCURRENCY PAIRS")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(args.pairs)}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Data Augmentation: {args.data_augmentation}")
    logger.info("=" * 80 + "\n")
    
    # Train models for all pairs
    results = {}
    for i, pair in enumerate(args.pairs):
        logger.info(f"\nTraining pair {i+1}/{len(args.pairs)}: {pair}")
        
        # Train model
        result = train_single_improved_model(pair, args)
        results[pair] = result
        
        # Log result
        if result["success"]:
            logger.info(f"Successfully trained improved model for {pair}")
            logger.info(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            logger.info(f"Win Rate: {result['metrics']['win_rate']:.4f}")
            logger.info(f"Direction Accuracy: {result['metrics']['direction_accuracy']:.4f}")
        else:
            logger.error(f"Failed to train improved model for {pair}: {result.get('error', 'Unknown error')}")
    
    # Generate summary report
    generate_summary_report(results)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"\nTotal execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    logger.info("\nImproved training complete for all pairs.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training all pairs: {e}")
        import traceback
        traceback.print_exc()