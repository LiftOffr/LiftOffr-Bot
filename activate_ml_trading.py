#!/usr/bin/env python3
"""
Activate ML Trading

This script activates the ML trading system by:
1. Checking if required models exist
2. Updating configuration files
3. Starting the trading bot in sandbox mode

Usage:
    python activate_ml_trading.py
"""
import os
import sys
import json
import glob
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories and files
MODEL_WEIGHTS_DIR = "model_weights"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Activate ML trading system")
    parser.add_argument("--sandbox", action="store_true", default=True, 
                        help="Run in sandbox mode (default: True)")
    return parser.parse_args()

def run_command(command, description=None):
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            logger.error(f"STDERR: {stderr}")
            return False
        
        logger.info(f"Command completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    try:
        # Load ML configuration
        if not os.path.exists(ML_CONFIG_PATH):
            logger.error(f"ML configuration file not found: {ML_CONFIG_PATH}")
            return False
        
        with open(ML_CONFIG_PATH, 'r') as f:
            ml_config = json.load(f)
        
        # Get configured models
        models = ml_config.get('models', {})
        if not models:
            logger.error("No models configured in ML configuration")
            return False
        
        # Check if model files exist
        missing_models = []
        for pair, model_info in models.items():
            model_path = model_info.get('model_path', '')
            if not os.path.exists(model_path):
                missing_models.append(f"{pair} ({model_path})")
        
        if missing_models:
            logger.error(f"Missing model files: {', '.join(missing_models)}")
            return False
        
        logger.info(f"All model files exist for {len(models)} configured pairs")
        return True
    except Exception as e:
        logger.error(f"Error checking model files: {e}")
        return False

def update_sandbox_portfolio():
    """Update sandbox portfolio to ensure it's properly initialized"""
    try:
        # Ensure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Check if portfolio file exists
        if not os.path.exists(PORTFOLIO_PATH):
            # Create default portfolio
            portfolio = {
                "balance": 20000.0,
                "initial_balance": 20000.0,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save portfolio
            with open(PORTFOLIO_PATH, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            logger.info(f"Created new sandbox portfolio with $20,000")
        else:
            # Load existing portfolio
            with open(PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            
            # Update last updated time
            portfolio["last_updated"] = datetime.now().isoformat()
            
            # Save updated portfolio
            with open(PORTFOLIO_PATH, 'w') as f:
                json.dump(portfolio, f, indent=2)
            
            logger.info(f"Updated existing sandbox portfolio with balance: ${portfolio.get('balance', 0):.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating sandbox portfolio: {e}")
        return False

def restart_trading_bot(sandbox=True):
    """Restart the trading bot"""
    try:
        # Create restart trigger file
        with open('.trading_bot_restart_trigger', 'w') as f:
            f.write(f"restart_requested={datetime.now().isoformat()}")
        
        logger.info(f"Triggered trading bot restart")
        
        # Run the specific ML trading command
        sandbox_flag = "--sandbox" if sandbox else ""
        return run_command(
            f"python run_realtime_ml_trading.py {sandbox_flag} > logs/ml_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log 2>&1 &",
            "Starting realtime ML trading system"
        )
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    logger.info("Activating ML trading system")
    
    # Check model files
    if not check_model_files():
        logger.error("Model file check failed, exiting")
        sys.exit(1)
    
    # Update sandbox portfolio
    if not update_sandbox_portfolio():
        logger.error("Failed to update sandbox portfolio, exiting")
        sys.exit(1)
    
    # Restart trading bot
    if not restart_trading_bot(sandbox=args.sandbox):
        logger.error("Failed to restart trading bot, exiting")
        sys.exit(1)
    
    logger.info("ML trading system activated successfully")
    logger.info("Check the trading status with: python show_trading_status.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error activating ML trading: {e}")
        raise