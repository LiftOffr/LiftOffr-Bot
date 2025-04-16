#!/usr/bin/env python3
"""
Train Improved Models for All Pairs

This script efficiently trains improved models for all cryptocurrency pairs:
1. Uses optimized architecture for CPU training
2. Trains one model at a time to conserve memory
3. Provides detailed progress updates
4. Automatically integrates models with the trading system

Usage:
    python train_improved_all_pairs.py [--epochs 3] [--pairs BTC/USD,ETH/USD]
"""

import os
import sys
import json
import logging
import argparse
import time
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_all_improved.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [CONFIG_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train improved models for all pairs")
    parser.add_argument("--pairs", type=str, default="ALL",
                        help="Trading pairs to train, comma-separated (default: ALL)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs per model (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--sequence_length", type=int, default=40,
                        help="Sequence length for time series (default: 40)")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--activate_after_training", action="store_true", default=True,
                        help="Activate models after training (default: True)")
    parser.add_argument("--reset_portfolio", action="store_true", default=False,
                        help="Reset sandbox portfolio after training (default: False)")
    return parser.parse_args()

def run_command(command: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    logger.info(f"Running command: {' '.join(command)}")
    
    try:
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log first part of output (for brevity)
        output_lines = process.stdout.split('\n')
        if len(output_lines) > 20:
            logger.info(f"Command output (truncated):\n{' '.join(output_lines[:20])}\n[truncated]")
        else:
            logger.info(f"Command output:\n{process.stdout}")
        
        if process.stderr:
            logger.warning(f"Command stderr:\n{process.stderr}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        return None

def train_one_pair(pair: str, args) -> Dict[str, Any]:
    """Train a model for a single trading pair"""
    logger.info(f"Training model for {pair}...")
    
    # Build command
    command = [
        "python", "train_with_faster_progress.py",
        "--pair", pair,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--sequence_length", str(args.sequence_length),
        "--max_portfolio_risk", str(args.max_portfolio_risk)
    ]
    
    # Run training
    start_time = time.time()
    result = run_command(command, f"Training {pair} model with {args.epochs} epochs")
    training_time = time.time() - start_time
    
    # Check result
    if result is None:
        logger.error(f"Training failed for {pair}")
        return {
            "pair": pair,
            "success": False,
            "error": "Training command failed",
            "training_time": training_time
        }
    
    # Check for model file
    pair_clean = pair.replace("/", "_").lower()
    model_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_quick_model.h5"
    
    if os.path.exists(model_path):
        logger.info(f"Model file found for {pair}: {model_path}")
        return {
            "pair": pair,
            "success": True,
            "model_path": model_path,
            "training_time": training_time
        }
    else:
        logger.error(f"Model file not found for {pair}: {model_path}")
        return {
            "pair": pair,
            "success": False,
            "error": "Model file not found",
            "training_time": training_time
        }

def check_ml_config():
    """Check if ML configuration exists and is valid"""
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            if not isinstance(config, dict):
                logger.warning(f"ML configuration is not a dictionary: {type(config)}")
                return False
            
            if "models" not in config:
                logger.warning("ML configuration missing 'models' key")
                return False
            
            if "global_settings" not in config:
                logger.warning("ML configuration missing 'global_settings' key")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking ML configuration: {e}")
            return False
    else:
        logger.warning(f"ML configuration not found: {ML_CONFIG_PATH}")
        return False

def activate_improved_models(args):
    """Activate the improved models in the trading system"""
    logger.info("Activating improved models in trading system...")
    
    # Build command
    command = [
        "python", "activate_improved_model_trading.py",
        "--sandbox"
    ]
    
    # Add reset portfolio flag if requested
    if args.reset_portfolio:
        command.extend(["--reset_portfolio", "--starting_capital", "20000.0"])
    
    # Run command
    result = run_command(command, "Activating improved models")
    
    if result is None:
        logger.error("Failed to activate improved models")
        return False
    else:
        logger.info("Successfully activated improved models")
        return True

def integrate_improved_models():
    """Integrate the improved models with the trading system"""
    logger.info("Integrating improved models with trading system...")
    
    # Build command
    command = ["python", "integrate_improved_model.py"]
    
    # Run command
    result = run_command(command, "Integrating improved models")
    
    if result is None:
        logger.error("Failed to integrate improved models")
        return False
    else:
        logger.info("Successfully integrated improved models")
        return True

def generate_summary_report(training_results: Dict[str, Dict]) -> str:
    """Generate a summary report of training results"""
    if not training_results:
        return "No training results to report"
    
    # Calculate statistics
    total_pairs = len(training_results)
    successful_pairs = sum(1 for result in training_results.values() if result["success"])
    failed_pairs = total_pairs - successful_pairs
    
    avg_training_time = sum(result["training_time"] for result in training_results.values()) / total_pairs
    
    # Format training time
    minutes, seconds = divmod(int(avg_training_time), 60)
    hours, minutes = divmod(minutes, 60)
    avg_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Create report
    report = "\n" + "=" * 80 + "\n"
    report += "IMPROVED MODEL TRAINING SUMMARY\n"
    report += "=" * 80 + "\n\n"
    
    report += f"Total Pairs: {total_pairs}\n"
    report += f"Successfully Trained: {successful_pairs}\n"
    report += f"Failed: {failed_pairs}\n"
    report += f"Average Training Time: {avg_time_str}\n\n"
    
    report += "INDIVIDUAL PAIR RESULTS:\n"
    report += "-" * 80 + "\n"
    
    for pair, result in training_results.items():
        report += f"Pair: {pair}\n"
        report += f"  Success: {result['success']}\n"
        
        minutes, seconds = divmod(int(result["training_time"]), 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        report += f"  Training Time: {time_str}\n"
        
        if result["success"]:
            report += f"  Model Path: {result['model_path']}\n"
        else:
            report += f"  Error: {result.get('error', 'Unknown error')}\n"
        
        report += "-" * 80 + "\n"
    
    return report

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Parse pairs
    pairs = ALL_PAIRS if args.pairs == "ALL" else args.pairs.split(",")
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING IMPROVED MODELS FOR ALL PAIRS")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Maximum Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Activate After Training: {args.activate_after_training}")
    logger.info(f"Reset Portfolio: {args.reset_portfolio}")
    logger.info("=" * 80 + "\n")
    
    # Track start time
    start_time = time.time()
    
    # Train models for all pairs
    training_results = {}
    for i, pair in enumerate(pairs):
        logger.info(f"\nTraining pair {i+1}/{len(pairs)}: {pair}")
        
        # Train model
        result = train_one_pair(pair, args)
        training_results[pair] = result
        
        # Log result
        if result["success"]:
            logger.info(f"Successfully trained model for {pair}")
        else:
            logger.error(f"Failed to train model for {pair}: {result.get('error', 'Unknown error')}")
    
    # Generate summary report
    report = generate_summary_report(training_results)
    logger.info(report)
    
    # Save report to file
    report_path = f"{RESULTS_DIR}/training_summary.txt"
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
    
    # Check ML configuration
    ml_config_valid = check_ml_config()
    if not ml_config_valid:
        logger.warning("ML configuration is invalid or missing, trying to integrate models")
        
        # Integrate models
        integrate_success = integrate_improved_models()
        if not integrate_success:
            logger.error("Failed to integrate models, activation may fail")
    
    # Activate models if requested
    if args.activate_after_training:
        logger.info("Activating improved models...")
        activate_success = activate_improved_models(args)
        if activate_success:
            logger.info("Models activated successfully")
        else:
            logger.error("Failed to activate models")
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes, seconds = divmod(int(total_time), 60)
    hours, minutes = divmod(minutes, 60)
    
    logger.info(f"\nTotal execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info("\nImproved model training complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in training all pairs: {e}")
        import traceback
        traceback.print_exc()