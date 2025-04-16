#!/usr/bin/env python3
"""
Restart Multi-Pair Trading

This script restarts the trading bot with the updated ML configuration
that includes multiple trading pairs (SOL/USD, BTC/USD, ETH/USD).

Usage:
    python restart_multi_pair_trading.py [--no-reset]
"""

import os
import sys
import json
import time
import glob
import argparse
import subprocess
import logging
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TRIGGER_FILE = ".trading_bot_restart_trigger"
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_FILE = f"{CONFIG_DIR}/sandbox_portfolio.json"
ENABLED_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD"]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Restart trading bot with multi-pair configuration")
    parser.add_argument("--no-reset", action="store_true", help="Don't reset the sandbox portfolio")
    return parser.parse_args()


def run_command(command: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"STDERR: {line}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"STDERR: {line}")
        
        return None


def validate_ml_config() -> bool:
    """Validate ML configuration"""
    try:
        # Load ML config
        with open(ML_CONFIG_FILE, 'r') as f:
            ml_config = json.load(f)
        
        # Check if ML trading is enabled
        if not ml_config.get("enabled", False):
            logger.warning("ML trading is not enabled in configuration")
            return False
        
        # Check if pairs are configured
        for pair in ENABLED_PAIRS:
            if pair not in ml_config.get("models", {}):
                logger.warning(f"Pair {pair} not found in ML configuration")
                return False
            
            pair_config = ml_config["models"][pair]
            if not pair_config.get("enabled", False):
                logger.warning(f"Pair {pair} is disabled in ML configuration")
            
            # Check model files
            model_path = pair_config.get("model_path", "")
            specialized_models = pair_config.get("specialized_models", {})
            
            for model_type, model_config in specialized_models.items():
                if not model_config.get("enabled", False):
                    continue
                
                model_name = model_config.get("model_name", "")
                model_file = os.path.join(model_path, model_name)
                
                if not os.path.exists(model_file):
                    logger.warning(f"Model file {model_file} not found")
                    logger.info(f"Checking if model exists in filesystem...")
                    # Check if model exists at all
                    if len(glob.glob(f"{model_path}/{model_name}")) == 0:
                        logger.error(f"Model {model_name} not found anywhere in {model_path}")
                        return False
                    else:
                        logger.info(f"Model {model_name} exists but not at the expected path")
                        # Continue anyway since the model exists somewhere
        
        logger.info("ML configuration validated successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error validating ML configuration: {e}")
        return False


def reset_sandbox_portfolio(starting_capital=20000.0) -> bool:
    """Reset sandbox portfolio to starting capital"""
    try:
        # Create portfolio with starting capital
        portfolio = {
            "base_currency": "USD",
            "starting_capital": starting_capital,
            "current_capital": starting_capital,
            "equity": starting_capital,
            "positions": {},
            "completed_trades": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Save portfolio
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        logger.info(f"Sandbox portfolio reset to ${starting_capital:.2f}")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting sandbox portfolio: {e}")
        return False


def restart_trading_bot() -> bool:
    """Restart the trading bot"""
    try:
        # Create restart trigger file
        with open(TRIGGER_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
        
        logger.info("Created trading bot restart trigger")
        
        # Give the bot time to restart
        time.sleep(5)
        
        # Check if trigger file was removed (bot restarted)
        if not os.path.exists(TRIGGER_FILE):
            logger.info("Trading bot restarted successfully")
            return True
        else:
            logger.warning("Trading bot restart trigger file not removed")
            return False
    
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting multi-pair trading bot configuration")
    
    # Validate ML configuration
    if not validate_ml_config():
        logger.error("ML configuration validation failed")
        return 1
    
    # Reset sandbox portfolio if requested
    if not args.no_reset:
        if not reset_sandbox_portfolio():
            logger.error("Failed to reset sandbox portfolio")
            return 1
    
    # Restart trading bot
    if not restart_trading_bot():
        logger.error("Failed to restart trading bot")
        return 1
    
    logger.info("Multi-pair trading bot restarted successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())