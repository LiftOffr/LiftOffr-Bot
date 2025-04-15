#!/usr/bin/env python3

"""
Check ML and Trading Metrics

This script checks and displays the current ML model accuracy and trading metrics
for all trading pairs.
"""

import os
import sys
import json
from datetime import datetime

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return default

def check_ml_metrics(pairs):
    """Check ML metrics for all pairs"""
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    print("\nML Metrics:")
    print("=" * 100)
    print(f"{'PAIR':<10} | {'ACCURACY':<10} | {'RETURN':<10} | {'WIN RATE':<10} | {'SHARPE':<8} | {'MAX DD':<8} | {'BASE LEV':<8} | {'MAX LEV':<8} | {'RISK %':<8}")
    print("-" * 100)
    
    pairs_data = {}
    
    for pair in pairs:
        # Get pair config
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Extract metrics
        accuracy = pair_config.get("accuracy", 0.0)
        backtest_return = pair_config.get("backtest_return", 0.0)
        win_rate = pair_config.get("win_rate", 0.0)
        sharpe_ratio = pair_config.get("sharpe_ratio", 0.0)
        max_drawdown = pair_config.get("max_drawdown", 0.0)
        base_leverage = pair_config.get("base_leverage", 20.0)
        max_leverage = pair_config.get("max_leverage", 125.0)
        risk_percentage = pair_config.get("risk_percentage", 0.2)
        
        # Store data
        pairs_data[pair] = {
            "accuracy": accuracy,
            "backtest_return": backtest_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "base_leverage": base_leverage,
            "max_leverage": max_leverage,
            "risk_percentage": risk_percentage
        }
        
        # Print metrics
        print(f"{pair:<10} | {accuracy*100:>8.2f}% | {backtest_return*100:>8.2f}% | {win_rate*100:>8.2f}% | {sharpe_ratio:>6.2f} | {max_drawdown*100:>6.2f}% | {base_leverage:>6.2f}x | {max_leverage:>6.2f}x | {risk_percentage*100:>6.2f}%")
    
    print("=" * 100)
    
    # Calculate average metrics
    avg_accuracy = sum(data["accuracy"] for data in pairs_data.values()) / len(pairs_data)
    avg_backtest_return = sum(data["backtest_return"] for data in pairs_data.values()) / len(pairs_data)
    avg_win_rate = sum(data["win_rate"] for data in pairs_data.values()) / len(pairs_data)
    avg_sharpe_ratio = sum(data["sharpe_ratio"] for data in pairs_data.values()) / len(pairs_data)
    
    print(f"Average: | {avg_accuracy*100:>8.2f}% | {avg_backtest_return*100:>8.2f}% | {avg_win_rate*100:>8.2f}% | {avg_sharpe_ratio:>6.2f} |")
    
    return pairs_data

def check_model_weights(pairs):
    """Check model weights for all pairs"""
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    print("\nModel Weights:")
    print("=" * 80)
    
    for pair in pairs:
        # Get pair config
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Get models
        models = pair_config.get("models", [])
        
        if not models:
            print(f"{pair}: No models found")
            continue
        
        print(f"{pair} Models:")
        print("-" * 80)
        print(f"{'MODEL TYPE':<15} | {'WEIGHT':<10}")
        print("-" * 80)
        
        for model in models:
            model_type = model.get("type", "Unknown")
            weight = model.get("weight", 0.0)
            
            print(f"{model_type:<15} | {weight*100:>8.2f}%")
        
        print("")
    
    print("=" * 80)

def main():
    """Main function"""
    pairs = DEFAULT_PAIRS
    
    print(f"Checking metrics for {len(pairs)} pairs...")
    
    # Check ML metrics
    ml_metrics = check_ml_metrics(pairs)
    
    # Check model weights
    check_model_weights(pairs)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())