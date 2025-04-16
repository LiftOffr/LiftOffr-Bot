#!/usr/bin/env python3
"""
Integrate Hybrid Model into Trading System

This script integrates the advanced hybrid model architecture into the existing trading system:
1. Updates the ML configuration to use hybrid models
2. Sets up dynamic model selection based on market conditions
3. Activates the enhanced ML trading system

Usage:
    python integrate_hybrid_model.py [--pairs PAIRS] [--sandbox]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from typing import List, Dict, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
MODEL_WEIGHTS_DIR = "model_weights"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "ADA/USD",
    "DOT/USD",
    "LINK/USD",
    "AVAX/USD",
    "MATIC/USD",
    "UNI/USD",
    "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrate hybrid model into trading system")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to integrate hybrid models for (default: {', '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    parser.add_argument("--base_leverage", type=float, default=5.0,
                        help="Base leverage for trading (default: 5.0)")
    parser.add_argument("--max_leverage", type=float, default=75.0,
                        help="Maximum leverage for high-confidence trades (default: 75.0)")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                        help="Confidence threshold for ML trading (default: 0.65)")
    parser.add_argument("--risk_percentage", type=float, default=0.20,
                        help="Risk percentage for each trade (default: 0.20)")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """
    Run a shell command and log output
    
    Args:
        cmd: List of command and arguments
        description: Description of the command
        
    Returns:
        CompletedProcess if successful, None if failed
    """
    try:
        if description:
            logger.info(f"{description}...")
        
        process = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"Command completed successfully: {' '.join(cmd)}")
        if process.stdout.strip():
            logger.debug(f"Output: {process.stdout.strip()}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Error: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return None

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [CONFIG_DIR, MODEL_WEIGHTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_hybrid_model_files(pairs: List[str]) -> bool:
    """
    Check if hybrid model files exist for the specified pairs
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if all required model files exist, False otherwise
    """
    logger.info("Checking hybrid model files...")
    
    missing_models = []
    
    for pair in pairs:
        # Convert pair format for filename (e.g., BTC/USD -> btc_usd)
        pair_filename = pair.replace("/", "_").lower()
        model_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_filename}_model.h5"
        
        if not os.path.exists(model_path):
            missing_models.append(pair)
    
    if missing_models:
        logger.warning(f"Missing hybrid model files for pairs: {', '.join(missing_models)}")
        logger.info("You need to train hybrid models for these pairs first")
        logger.info("Use the train_hybrid_model.py script to train models")
        return False
    
    logger.info("All hybrid model files exist")
    return True

def load_ml_config() -> Dict[str, Any]:
    """
    Load ML configuration from file
    
    Returns:
        ML configuration dictionary
    """
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration with {len(config.get('models', {}))} models")
            return config
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
    
    # Return default configuration if file doesn't exist or loading fails
    logger.warning("Using default ML configuration")
    return {
        "models": {},
        "global_settings": {
            "base_leverage": 5.0,
            "max_leverage": 75.0,
            "confidence_threshold": 0.65,
            "risk_percentage": 0.20,
            "sandbox_mode": True
        }
    }

def update_ml_config(pairs: List[str], args) -> bool:
    """
    Update ML configuration to use hybrid models
    
    Args:
        pairs: List of trading pairs
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Updating ML configuration...")
    
    # Load existing configuration
    config = load_ml_config()
    
    # Update global settings
    config["global_settings"] = {
        "base_leverage": args.base_leverage,
        "max_leverage": args.max_leverage,
        "confidence_threshold": args.confidence_threshold,
        "risk_percentage": args.risk_percentage,
        "sandbox_mode": args.sandbox
    }
    
    # Update model configurations
    for pair in pairs:
        # Convert pair format for filename and configuration
        pair_filename = pair.replace("/", "_").lower()
        
        # Create or update model configuration
        config["models"][pair] = {
            "model_type": "hybrid",
            "model_path": f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_filename}_model.h5",
            "base_leverage": args.base_leverage,
            "max_leverage": args.max_leverage,
            "confidence_threshold": args.confidence_threshold,
            "risk_percentage": args.risk_percentage,
            "active": True
        }
    
    # Save configuration
    try:
        os.makedirs(os.path.dirname(ML_CONFIG_PATH), exist_ok=True)
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"ML configuration saved to {ML_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML configuration: {e}")
        return False

def activate_ml_trading(sandbox: bool = True) -> bool:
    """
    Activate ML trading with hybrid models
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating ML trading with hybrid models (sandbox={sandbox})...")
    
    # Use activate_ml_trading.py script
    cmd = ["python", "activate_ml_trading.py"]
    if sandbox:
        cmd.append("--sandbox")
    
    process = run_command(cmd, "Activating ML trading")
    
    if process:
        logger.info("ML trading activated successfully")
        return True
    else:
        logger.error("Failed to activate ML trading")
        return False

def train_missing_hybrid_models(pairs: List[str], args) -> bool:
    """
    Train hybrid models for pairs with missing model files
    
    Args:
        pairs: List of trading pairs
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Checking for missing hybrid models...")
    
    missing_models = []
    for pair in pairs:
        # Convert pair format for filename (e.g., BTC/USD -> btc_usd)
        pair_filename = pair.replace("/", "_").lower()
        model_path = f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_filename}_model.h5"
        
        if not os.path.exists(model_path):
            missing_models.append(pair)
    
    if not missing_models:
        logger.info("All hybrid models exist, no training needed")
        return True
    
    logger.info(f"Training hybrid models for pairs: {', '.join(missing_models)}")
    
    success = True
    for pair in missing_models:
        logger.info(f"Training hybrid model for {pair}...")
        
        cmd = [
            "python", "train_hybrid_model.py",
            "--pair", pair,
            "--epochs", "50",
            "--batch_size", "32"
        ]
        
        process = run_command(cmd, f"Training hybrid model for {pair}")
        
        if not process:
            logger.error(f"Failed to train hybrid model for {pair}")
            success = False
    
    return success

def monitor_training_status():
    """Print status information during and after training"""
    # Check if model weights directory exists and contains models
    if not os.path.exists(MODEL_WEIGHTS_DIR):
        logger.error(f"Model weights directory not found: {MODEL_WEIGHTS_DIR}")
        return
    
    # List models in directory
    model_files = [f for f in os.listdir(MODEL_WEIGHTS_DIR) if f.endswith('.h5')]
    
    logger.info(f"Found {len(model_files)} model files in {MODEL_WEIGHTS_DIR}")
    for model_file in model_files:
        logger.info(f"  - {model_file}")
    
    # Check training results directory
    results_dir = "training_results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
        
        logger.info(f"Found {len(result_files)} training result files in {results_dir}")
        for result_file in result_files:
            try:
                with open(os.path.join(results_dir, result_file), 'r') as f:
                    contents = f.read()
                    
                    # Extract key metrics
                    if "Win Rate:" in contents:
                        win_rate_line = [line for line in contents.split('\n') if "Win Rate:" in line][0]
                        win_rate = win_rate_line.split(':')[1].strip()
                        logger.info(f"  - {result_file}: Win Rate = {win_rate}")
            except Exception as e:
                logger.error(f"Error reading {result_file}: {e}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Ensure directories exist
    ensure_directories()
    
    # Train missing hybrid models if needed
    train_missing_hybrid_models(args.pairs, args)
    
    # Check if all hybrid model files exist
    if not check_hybrid_model_files(args.pairs):
        logger.error("Not all required hybrid model files exist")
        return
    
    # Update ML configuration
    if not update_ml_config(args.pairs, args):
        logger.error("Failed to update ML configuration")
        return
    
    # Monitor training status
    monitor_training_status()
    
    # Activate ML trading
    if not activate_ml_trading(args.sandbox):
        logger.error("Failed to activate ML trading")
        return
    
    logger.info("Hybrid model integration complete")
    logger.info(f"Trading system is now using hybrid models for {len(args.pairs)} pairs")
    logger.info("Use 'python show_trading_status.py' to check system status")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in hybrid model integration: {e}")
        raise