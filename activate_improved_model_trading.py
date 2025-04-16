#!/usr/bin/env python3
"""
Activate Improved Model Trading

This script activates the improved ML trading system by:
1. Updating ML configuration with improved model settings
2. Setting enhanced risk management parameters
3. Starting the trading bot with improved models

Usage:
    python activate_improved_model_trading.py [--sandbox] [--pairs ALL|BTC/USD,ETH/USD,...]
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("activate_improved_models.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
MODEL_WEIGHTS_DIR = "model_weights"
DATA_DIR = "data"
SANDBOX_PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
SANDBOX_POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
SANDBOX_TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [CONFIG_DIR, MODEL_WEIGHTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Activate improved ML trading")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Use sandbox mode (default: True)")
    parser.add_argument("--pairs", type=str, default="ALL",
                        help="Trading pairs to activate, comma-separated (default: ALL)")
    parser.add_argument("--base_leverage", type=float, default=5.0,
                        help="Base leverage for trading (default: 5.0)")
    parser.add_argument("--max_leverage", type=float, default=75.0,
                        help="Maximum leverage for high-confidence trades (default: 75.0)")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                        help="Confidence threshold for ML trading (default: 0.65)")
    parser.add_argument("--risk_percentage", type=float, default=0.20,
                        help="Risk percentage for each trade (default: 0.20)")
    parser.add_argument("--max_portfolio_risk", type=float, default=0.25,
                        help="Maximum portfolio risk percentage (default: 0.25)")
    parser.add_argument("--reset_portfolio", action="store_true", default=False,
                        help="Reset sandbox portfolio to starting capital (default: False)")
    parser.add_argument("--starting_capital", type=float, default=20000.0,
                        help="Starting capital for reset portfolio (default: 20000.0)")
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
        
        logger.info(f"Command output:\n{process.stdout}")
        
        if process.stderr:
            logger.warning(f"Command stderr:\n{process.stderr}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        return None

def update_ml_config(args) -> bool:
    """Update ML configuration with improved settings"""
    logger.info("Updating ML configuration with improved settings...")
    
    # Parse pairs
    pairs = ALL_PAIRS if args.pairs == "ALL" else args.pairs.split(",")
    logger.info(f"Activating pairs: {', '.join(pairs)}")
    
    # Load existing config if it exists
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded existing ML configuration with {len(config.get('models', {}))} models")
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
            config = {"models": {}, "global_settings": {}}
    else:
        logger.info("Creating new ML configuration")
        config = {"models": {}, "global_settings": {}}
    
    # Update global settings
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    config["global_settings"]["base_leverage"] = args.base_leverage
    config["global_settings"]["max_leverage"] = args.max_leverage
    config["global_settings"]["confidence_threshold"] = args.confidence_threshold
    config["global_settings"]["risk_percentage"] = args.risk_percentage
    config["global_settings"]["max_portfolio_risk"] = args.max_portfolio_risk
    
    # Update model settings
    if "models" not in config:
        config["models"] = {}
    
    # For each pair, update config or set default values
    for pair in pairs:
        if pair in config["models"]:
            # Update existing model config
            config["models"][pair]["base_leverage"] = args.base_leverage
            config["models"][pair]["max_leverage"] = args.max_leverage
            config["models"][pair]["confidence_threshold"] = args.confidence_threshold
            config["models"][pair]["risk_percentage"] = args.risk_percentage
            config["models"][pair]["active"] = True
            
            logger.info(f"Updated configuration for existing model: {pair}")
        else:
            # Create default config for missing model
            model_path = f"model_weights/hybrid_{pair.replace('/', '_').lower()}_improved_model.h5"
            
            config["models"][pair] = {
                "model_type": "hybrid_improved",
                "model_path": model_path,
                "accuracy": 0.65,  # Default value
                "win_rate": 0.6,  # Default value
                "base_leverage": args.base_leverage,
                "max_leverage": args.max_leverage,
                "confidence_threshold": args.confidence_threshold,
                "risk_percentage": args.risk_percentage,
                "active": True
            }
            
            logger.warning(f"Created default configuration for missing model: {pair}")
    
    # Save updated config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML configuration saved to {ML_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML configuration: {e}")
        return False

def reset_sandbox_portfolio(starting_capital=20000.0) -> bool:
    """Reset sandbox portfolio to starting capital"""
    logger.info(f"Resetting sandbox portfolio to ${starting_capital:,.2f}...")
    
    # Create portfolio data
    portfolio_data = {
        "timestamp": datetime.now().timestamp(),
        "total_balance": starting_capital,
        "available_balance": starting_capital,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "margin_used": 0.0,
        "total_position_size": 0.0,
        "portfolio_risk": 0.0
    }
    
    # Create empty positions data
    positions_data = {}
    
    # Create empty trades data
    trades_data = []
    
    # Save data
    try:
        # Save portfolio
        with open(SANDBOX_PORTFOLIO_PATH, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        # Save positions
        with open(SANDBOX_POSITIONS_PATH, 'w') as f:
            json.dump(positions_data, f, indent=2)
        
        # Save trades
        with open(SANDBOX_TRADES_PATH, 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        logger.info("Sandbox portfolio reset successfully")
        return True
    except Exception as e:
        logger.error(f"Error resetting sandbox portfolio: {e}")
        return False

def check_model_files(pairs: List[str]) -> bool:
    """Check if model files exist for all pairs"""
    logger.info("Checking model files for all pairs...")
    
    all_found = True
    missing_models = []
    
    # Check each pair
    for pair in pairs:
        # Convert pair name for filename
        pair_clean = pair.replace("/", "_").lower()
        
        # Look for different model file patterns
        model_files = [
            f"model_weights/hybrid_{pair_clean}_improved_model.h5",
            f"model_weights/hybrid_{pair_clean}_model.h5",
            f"model_weights/hybrid_{pair_clean}_quick_model.h5"
        ]
        
        if not any(os.path.exists(model_file) for model_file in model_files):
            all_found = False
            missing_models.append(pair)
            logger.warning(f"No model files found for {pair}")
        else:
            logger.info(f"Found model file for {pair}")
    
    if not all_found:
        logger.warning(f"Missing model files for the following pairs: {', '.join(missing_models)}")
        logger.warning("Training may be required for these pairs")
    
    return all_found

def restart_trading_bot(sandbox=True) -> bool:
    """Restart the trading bot"""
    logger.info(f"Restarting trading bot in {'sandbox' if sandbox else 'live'} mode...")
    
    # Build command
    command = ["python", "main.py"]
    if sandbox:
        command.append("--sandbox")
    
    # Run the command
    result = run_command(command, "Starting trading bot")
    
    return result is not None

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("ACTIVATING IMPROVED ML TRADING")
    logger.info("=" * 80)
    logger.info(f"Mode: {'Sandbox' if args.sandbox else 'Live'}")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Base Leverage: {args.base_leverage}x")
    logger.info(f"Max Leverage: {args.max_leverage}x")
    logger.info(f"Confidence Threshold: {args.confidence_threshold}")
    logger.info(f"Risk Percentage: {args.risk_percentage:.2%}")
    logger.info(f"Max Portfolio Risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Reset Portfolio: {args.reset_portfolio}")
    logger.info(f"Starting Capital: ${args.starting_capital:,.2f}")
    logger.info("=" * 80 + "\n")
    
    # Parse pairs
    pairs = ALL_PAIRS if args.pairs == "ALL" else args.pairs.split(",")
    
    # Check model files
    check_model_files(pairs)
    
    # Update ML configuration
    if not update_ml_config(args):
        logger.error("Failed to update ML configuration")
        return False
    
    # Reset sandbox portfolio if requested
    if args.reset_portfolio:
        if not reset_sandbox_portfolio(args.starting_capital):
            logger.error("Failed to reset sandbox portfolio")
            return False
    
    # Restart trading bot
    if not restart_trading_bot(args.sandbox):
        logger.error("Failed to restart trading bot")
        return False
    
    logger.info("\nImproved ML trading activated successfully!")
    logger.info(f"Trading bot is now running in {'sandbox' if args.sandbox else 'live'} mode")
    logger.info(f"Active pairs: {', '.join(pairs)}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error activating improved ML trading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)