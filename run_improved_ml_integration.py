#!/usr/bin/env python3
"""
Improved ML Integration Script

This script orchestrates the entire process of improving the machine learning integration
with the trading system:

1. Retrains models with the correct input shape
2. Updates the ML configuration for better performance
3. Creates optimized ensemble models
4. Activates the ML trading with the improved models

Usage:
    python run_improved_ml_integration.py [--pairs PAIRS] [--sandbox]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run improved ML integration with trading system')
    parser.add_argument('--pairs', type=str, default='SOLUSD,BTCUSD,ETHUSD', 
                        help='Comma-separated list of trading pairs')
    parser.add_argument('--sandbox', action='store_true', default=True,
                        help='Run in sandbox mode (default: True)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset ML configuration and models')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model retraining (use existing models)')
    parser.add_argument('--skip-backtest', action='store_true',
                        help='Skip backtesting step')
    
    return parser.parse_args()

def run_command(cmd, description=None, check=True, shell=False):
    """Run a shell command and log output."""
    if description:
        logger.info(f"{description}")
    
    if isinstance(cmd, str) and not shell:
        cmd = cmd.split()
    
    logger.info(f"Running command: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.stdout:
            logger.info(f"Command output: {process.stdout.strip()}")
        
        if process.stderr:
            if process.returncode != 0:
                logger.error(f"Command error: {process.stderr.strip()}")
            else:
                logger.warning(f"Command stderr: {process.stderr.strip()}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with returncode {e.returncode}: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Failed to run command: {str(e)}")
        raise

def check_dependencies():
    """Check for required Python dependencies."""
    logger.info("Checking for required dependencies")
    
    # List of required packages
    packages = [
        "tensorflow",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "statsmodels"
    ]
    
    missing_packages = []
    
    for package in packages:
        try:
            # Use importlib to check if package is available
            cmd = f"python -c 'import {package}'"
            process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if process.returncode != 0:
                missing_packages.append(package)
                logger.warning(f"Package {package} is not available")
            else:
                logger.info(f"Package {package} is available")
        except Exception as e:
            missing_packages.append(package)
            logger.warning(f"Failed to check {package}: {str(e)}")
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.warning("Some functionality may be limited without these packages")
    else:
        logger.info("All required dependencies are available")
    
    return True  # Continue even if some packages are missing

def check_model_files(pairs):
    """Check if model files exist for specified pairs."""
    logger.info(f"Checking model files for {', '.join(pairs)}")
    
    model_types = ['lstm', 'tcn', 'transformer', 'cnn', 'gru', 'bilstm', 'attention', 'hybrid']
    missing_models = {}
    
    for pair in pairs:
        missing_for_pair = []
        
        for model_type in model_types:
            model_path = f"models/{model_type}/{pair}.h5"
            alt_model_path = f"models/{model_type}/{pair}_{model_type}.h5"
            
            if not os.path.exists(model_path) and not os.path.exists(alt_model_path):
                missing_for_pair.append(model_type)
        
        if missing_for_pair:
            missing_models[pair] = missing_for_pair
    
    if missing_models:
        logger.warning(f"Missing models: {json.dumps(missing_models, indent=2)}")
    else:
        logger.info("All model files found")
    
    return missing_models

def retrain_models(pairs, skip_training=False):
    """Retrain models with the correct input shape."""
    if skip_training:
        logger.info("Skipping model retraining as requested")
        return True
    
    logger.info(f"Retraining models for {', '.join(pairs)}")
    
    # First, check for missing models
    missing_models = check_model_files(pairs)
    
    # Prepare model types to train
    model_types_to_train = ['lstm', 'tcn', 'transformer']
    
    # Prepare pairs string for command
    pairs_str = ','.join(pairs)
    models_str = ','.join(model_types_to_train)
    
    try:
        # Run improved model retraining script
        cmd = [
            "python", "improved_model_retraining.py",
            "--pairs", pairs_str,
            "--models", models_str,
            "--epochs", "200",
            "--batch-size", "32",
            "--create-ensemble",
            "--update-config"
        ]
        
        run_command(cmd, "Running improved model retraining")
        logger.info("Model retraining completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to retrain models: {str(e)}")
        return False

def update_ml_config(pairs, reset=False):
    """Update ML configuration for better integration."""
    logger.info(f"Updating ML configuration for {', '.join(pairs)}")
    
    # Prepare pairs string for command
    pairs_str = ','.join(pairs)
    
    try:
        # Run ML config update script
        cmd = [
            "python", "update_ml_config.py",
            "--pairs", pairs_str,
            "--update-ensemble"
        ]
        
        if reset:
            cmd.append("--reset-weights")
        
        run_command(cmd, "Updating ML configuration")
        logger.info("ML configuration updated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update ML configuration: {str(e)}")
        return False

def run_backtest(pairs, skip_backtest=False):
    """Run backtest to validate the improved models."""
    if skip_backtest:
        logger.info("Skipping backtest as requested")
        return True
    
    logger.info(f"Running backtest for {', '.join(pairs)}")
    
    success = True
    
    for pair in pairs:
        # Convert pair format (e.g., SOLUSD to SOL/USD)
        if '/' not in pair:
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
        else:
            formatted_pair = pair
        
        try:
            # Run backtest for the pair
            cmd = [
                "python", "comprehensive_backtest.py",
                "--pair", formatted_pair,
                "--use-ml",
                "--days", "30"
            ]
            
            run_command(cmd, f"Running backtest for {formatted_pair}")
            logger.info(f"Backtest completed successfully for {formatted_pair}")
            
        except Exception as e:
            logger.error(f"Failed to run backtest for {formatted_pair}: {str(e)}")
            success = False
    
    return success

def activate_ml_trading(pairs, sandbox=True):
    """Activate ML trading with the improved models."""
    logger.info(f"Activating ML trading for {', '.join(pairs)}")
    
    # Prepare pairs string for command
    pairs_str = ','.join(pairs)
    
    try:
        # Run ML trading activation script
        cmd = [
            "python", "activate_ml_with_ensembles.py",
            "--pairs", pairs_str
        ]
        
        if sandbox:
            cmd.append("--sandbox")
        
        run_command(cmd, "Activating ML trading")
        logger.info("ML trading activated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to activate ML trading: {str(e)}")
        return False

def reset_sandbox_portfolio():
    """Reset the sandbox portfolio for testing."""
    logger.info("Resetting sandbox portfolio")
    
    try:
        # Run reset portfolio script if it exists
        if os.path.exists("reset_portfolio.py"):
            run_command("python reset_portfolio.py", "Resetting portfolio")
        else:
            logger.warning("reset_portfolio.py not found, skipping portfolio reset")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset portfolio: {str(e)}")
        return False

def monitor_trading_performance():
    """Monitor trading performance for a short period."""
    logger.info("Monitoring trading performance")
    
    try:
        # Run status check every 30 seconds for 5 minutes
        for i in range(10):
            logger.info(f"Checking trading status (check {i+1}/10)")
            run_command("python get_current_status.py", "Getting trading status")
            
            if i < 9:  # Skip wait on the last iteration
                logger.info("Waiting 30 seconds for next check...")
                time.sleep(30)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to monitor trading performance: {str(e)}")
        return False

def main():
    """Main function to run improved ML integration."""
    args = parse_arguments()
    
    # Parse pairs
    pairs = [pair.strip().upper() for pair in args.pairs.split(',')]
    
    # Check dependencies
    check_dependencies()
    
    # Reset sandbox portfolio if needed
    if args.reset and args.sandbox:
        if not reset_sandbox_portfolio():
            logger.warning("Failed to reset sandbox portfolio, continuing anyway")
    
    # Retrain models
    if not retrain_models(pairs, args.skip_training):
        logger.error("Failed to retrain models, exiting")
        return
    
    # Update ML configuration
    if not update_ml_config(pairs, args.reset):
        logger.error("Failed to update ML configuration, exiting")
        return
    
    # Run backtest
    if not run_backtest(pairs, args.skip_backtest):
        logger.warning("Backtest not completely successful, continuing anyway")
    
    # Activate ML trading
    if not activate_ml_trading(pairs, args.sandbox):
        logger.error("Failed to activate ML trading, exiting")
        return
    
    # Monitor trading performance
    if not monitor_trading_performance():
        logger.warning("Failed to monitor trading performance")
    
    logger.info("Improved ML integration completed successfully")
    logger.info("The trading bot is now running with improved ML models")
    logger.info("You can monitor performance with 'python get_current_status.py'")

if __name__ == "__main__":
    main()