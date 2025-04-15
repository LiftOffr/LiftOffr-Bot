#!/usr/bin/env python3

"""
Check Model Status

This script checks the status of the machine learning models for all trading pairs.
It verifies:
1. If models exist for each pair
2. Current accuracy and performance metrics
3. Which pairs need training or optimization

Usage:
    python check_model_status.py [--pairs PAIRS]
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_MODELS_DIR = "ml_models"
ENSEMBLE_DIR = f"{ML_MODELS_DIR}/ensemble"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check Model Status")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--verbose", action="store_true",
                        help="Display detailed information")
    return parser.parse_args()

def load_json_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def check_model_files(pair) -> Dict[str, Any]:
    """
    Check if model files exist for a specific pair
    
    Returns:
        dict: Status of model files
    """
    pair_filename = pair.replace('/', '_')
    result = {
        "ensemble_config": False,
        "model_files": [],
        "missing_models": [],
        "models_found": 0,
        "all_models_found": False
    }
    
    # Check ensemble configuration
    ensemble_file = f"{ENSEMBLE_DIR}/{pair_filename}_weights.json"
    result["ensemble_config"] = os.path.exists(ensemble_file)
    
    if result["ensemble_config"]:
        # Load ensemble configuration
        ensemble_data = load_json_file(ensemble_file, {})
        models = ensemble_data.get("models", [])
        
        # Check each model file
        for model_info in models:
            model_file = model_info.get("file", "")
            model_type = model_info.get("type", "")
            
            if model_file and os.path.exists(model_file):
                result["model_files"].append({
                    "type": model_type,
                    "file": model_file,
                    "exists": True
                })
            else:
                result["missing_models"].append(model_type)
                result["model_files"].append({
                    "type": model_type,
                    "file": model_file,
                    "exists": False
                })
        
        result["models_found"] = len(result["model_files"]) - len(result["missing_models"])
        result["all_models_found"] = len(result["missing_models"]) == 0
    
    return result

def check_ml_config(pair) -> Dict[str, Any]:
    """
    Check if ML configuration exists for a specific pair
    
    Returns:
        dict: ML configuration status
    """
    # Load ML config
    ml_config = load_json_file(ML_CONFIG_FILE, {"pairs": {}})
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    return {
        "exists": bool(pair_config),
        "accuracy": pair_config.get("accuracy", 0.0),
        "backtest_return": pair_config.get("backtest_return", 0.0),
        "win_rate": pair_config.get("win_rate", 0.0),
        "sharpe_ratio": pair_config.get("sharpe_ratio", 0.0),
        "models": pair_config.get("models", [])
    }

def check_all_pairs(pairs, verbose=False) -> Dict[str, Dict[str, Any]]:
    """
    Check status of all pairs
    
    Returns:
        dict: Status for each pair
    """
    results = {}
    
    for pair in pairs:
        model_status = check_model_files(pair)
        config_status = check_ml_config(pair)
        
        accuracy = config_status.get("accuracy", 0.0)
        target_accuracy = 0.999  # 99.9%
        needs_training = accuracy < target_accuracy
        
        results[pair] = {
            "model_status": model_status,
            "config_status": config_status,
            "accuracy": accuracy,
            "needs_training": needs_training,
            "ready_for_trading": model_status.get("all_models_found", False) and config_status.get("exists", False)
        }
        
        if verbose:
            logger.info(f"Checked {pair}:")
            logger.info(f"  Models found: {model_status.get('models_found', 0)}")
            logger.info(f"  Missing models: {model_status.get('missing_models', [])}")
            logger.info(f"  Accuracy: {accuracy * 100:.2f}%")
            logger.info(f"  Needs training: {needs_training}")
            logger.info(f"  Ready for trading: {results[pair]['ready_for_trading']}")
    
    return results

def get_optimized_pairs(pairs, results) -> List[str]:
    """Get a list of pairs that are already optimized and ready for trading"""
    optimized = []
    for pair in pairs:
        status = results.get(pair, {})
        if (status.get("ready_for_trading", False) and 
            status.get("accuracy", 0.0) >= 0.95):  # 95% or higher accuracy
            optimized.append(pair)
    return optimized

def get_needs_training_pairs(pairs, results) -> List[str]:
    """Get a list of pairs that need training"""
    needs_training = []
    for pair in pairs:
        status = results.get(pair, {})
        if status.get("needs_training", True):
            needs_training.append(pair)
    return needs_training

def format_percentage(value, decimals=2):
    """Format a value as a percentage"""
    return f"{value * 100:.{decimals}f}%"

def main():
    """Main function"""
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    logger.info(f"Checking model status for {len(pairs)} pairs...")
    
    # Check status for all pairs
    results = check_all_pairs(pairs, args.verbose)
    
    # Display summary
    print("\nModel Status Summary:")
    print("=" * 80)
    print(f"{'PAIR':<10} | {'ACCURACY':<10} | {'RETURN':<10} | {'MODELS':<10} | {'NEEDS TRAINING':<15} | {'READY TO TRADE':<15}")
    print("-" * 80)
    
    for pair, status in results.items():
        accuracy = status.get("accuracy", 0.0)
        backtest_return = status.get("config_status", {}).get("backtest_return", 0.0)
        models_found = status.get("model_status", {}).get("models_found", 0)
        needs_training = status.get("needs_training", True)
        ready_for_trading = status.get("ready_for_trading", False)
        
        print(f"{pair:<10} | {format_percentage(accuracy):>10} | {format_percentage(backtest_return):>10} | {models_found:>10} | {str(needs_training):>15} | {str(ready_for_trading):>15}")
    
    print("=" * 80)
    
    # Get pairs that need training
    needs_training = get_needs_training_pairs(pairs, results)
    optimized = get_optimized_pairs(pairs, results)
    
    # Show pairs that need training
    if needs_training:
        print(f"\nPairs that need training or optimization ({len(needs_training)}):")
        for pair in needs_training:
            accuracy = results.get(pair, {}).get("accuracy", 0.0)
            print(f"  {pair:<10} - Current accuracy: {format_percentage(accuracy)}")
    else:
        print("\nAll pairs are already optimized.")
    
    # Show pairs ready for trading
    if optimized:
        print(f"\nPairs ready for trading ({len(optimized)}):")
        for pair in optimized:
            accuracy = results.get(pair, {}).get("accuracy", 0.0)
            print(f"  {pair:<10} - Current accuracy: {format_percentage(accuracy)}")
    else:
        print("\nNo pairs are currently optimized and ready for trading.")
    
    # Provide recommendations
    print("\nRecommendations:")
    if needs_training:
        pairs_str = ",".join(needs_training)
        print(f"  Run enhanced training for pairs that need optimization:")
        print(f"  python run_enhanced_training.py --pairs {pairs_str}")
    else:
        print(f"  All pairs are optimized. Start trading with all pairs:")
        print(f"  python run_enhanced_training.py --skip-training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())