#!/usr/bin/env python3
"""
Activate ML Trading Across All Pairs

This script activates the advanced ML trading system for all supported trading pairs
in sandbox mode. It orchestrates the entire process:

1. Installs required dependencies
2. Prepares the trading environment
3. Trains ML models for all trading pairs
4. Analyzes cross-asset correlations
5. Integrates sentiment analysis
6. Activates the trading system in sandbox mode

It serves as the main entry point for the complete ML-enhanced trading system.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ml_trading_activation.log')
    ]
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
ML_CONFIG_PATH = "ml_config.json"
MODELS_DIR = "models"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Activate ML trading across all pairs")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help=f"Trading pairs to activate (default: {DEFAULT_PAIRS})"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help=f"Timeframes to use (default: {DEFAULT_TIMEFRAMES})"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run in sandbox mode (default: False)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (default: False)"
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip correlation analysis (default: False)"
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip sentiment analysis (default: False)"
    )
    
    return parser.parse_args()


def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("historical_data", exist_ok=True)
    os.makedirs("backtest_results", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("optimization_results", exist_ok=True)
    os.makedirs("analysis_results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def run_command(command, description=None, check=True):
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    logger.info(f"Running command: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout_lines = []
    stderr_lines = []
    
    # Safe handling of stdout
    if process.stdout:
        def read_pipe(pipe, lines):
            for line in iter(pipe.readline, ''):
                if line.strip():
                    logger.info(line.strip())
                    lines.append(line)
        
        read_pipe(process.stdout, stdout_lines)
    
    # Safe handling of stderr
    if process.stderr:
        read_pipe(process.stderr, stderr_lines)
    
    returncode = process.wait()
    
    if returncode != 0 and check:
        logger.error(f"Command failed with exit code {returncode}")
        return False
    
    return True


def install_dependencies(args):
    """Install required dependencies"""
    # Check if we need to install dependencies
    if not run_command("python ensure_ml_dependencies.py", "Checking ML dependencies", check=False):
        logger.warning("Dependencies check failed, attempting to install...")
        
        # Install required Python packages
        run_command(
            "python -m pip install --upgrade pip numpy pandas tensorflow scikit-learn matplotlib seaborn statsmodels joblib nltk",
            "Installing basic ML dependencies"
        )
        
        # Install specialized packages
        run_command(
            "python -m pip install keras-tcn transformers trafilatura websocket-client",
            "Installing specialized ML dependencies"
        )
        
        # Download NLTK data
        run_command(
            "python -c \"import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')\"",
            "Downloading NLTK data"
        )
        
        # Check again
        if not run_command("python ensure_ml_dependencies.py", "Verifying ML dependencies", check=False):
            logger.error("Failed to install all dependencies, but continuing...")
    
    return True


def train_models(args):
    """Train ML models for all trading pairs"""
    if args.skip_training:
        logger.info("Skipping model training as requested")
        return True
    
    success = True
    
    # Run the enhanced dual strategy trainer
    pairs_arg = " ".join([f"'{pair}'" for pair in args.pairs])
    timeframes_arg = " ".join(args.timeframes)
    sandbox_arg = "--sandbox" if args.sandbox else ""
    
    command = f"python enhanced_dual_strategy_trainer.py --pairs {pairs_arg} --timeframes {timeframes_arg} {sandbox_arg}"
    
    if not run_command(command, "Training enhanced dual strategy models", check=False):
        logger.warning("Enhanced dual strategy training failed, but continuing...")
        success = False
    
    # Check if model files exist
    if not check_model_files(args):
        logger.warning("Some model files are missing, but continuing...")
        success = False
    
    return success


def analyze_correlations(args):
    """Analyze correlations between assets"""
    if args.skip_correlation:
        logger.info("Skipping correlation analysis as requested")
        return True
    
    success = True
    
    # Run the multi-asset correlation analyzer
    pairs_arg = " ".join([f"'{pair}'" for pair in args.pairs])
    
    command = f"python multi_asset_correlation_analyzer.py --pairs {pairs_arg}"
    
    if not run_command(command, "Analyzing cross-asset correlations", check=False):
        logger.warning("Cross-asset correlation analysis failed, but continuing...")
        success = False
    
    return success


def analyze_sentiment(args):
    """Analyze sentiment for all assets"""
    if args.skip_sentiment:
        logger.info("Skipping sentiment analysis as requested")
        return True
    
    success = True
    
    # Run the sentiment analysis integration
    pairs_arg = " ".join([f"'{pair}'" for pair in args.pairs])
    
    command = f"python sentiment_analysis_integration.py --pairs {pairs_arg}"
    
    if not run_command(command, "Analyzing market sentiment", check=False):
        logger.warning("Sentiment analysis failed, but continuing...")
        success = False
    
    return success


def start_trading(args):
    """Start the trading system in sandbox mode"""
    # Ensure the system is ready for trading
    if not os.path.exists(ML_CONFIG_PATH):
        logger.error(f"ML configuration file '{ML_CONFIG_PATH}' not found")
        return False
    
    if not check_model_files(args):
        logger.warning("Some model files are missing, but continuing...")
    
    # Start the trading bot with ML integration
    pairs_arg = " ".join([f"'{pair}'" for pair in args.pairs])
    sandbox_arg = "--sandbox" if args.sandbox else ""
    
    command = f"python kraken_trading_bot.py --pairs {pairs_arg} --use-ml --multi-strategy 'ARIMAStrategy,AdaptiveStrategy' {sandbox_arg}"
    
    # Start the trading bot (this will run indefinitely)
    logger.info("Starting trading bot with ML integration")
    logger.info(f"Trading pairs: {', '.join(args.pairs)}")
    logger.info(f"Mode: {'Sandbox' if args.sandbox else 'Live'}")
    
    run_command(command, "Starting trading bot", check=False)
    
    return True


def check_model_files(args):
    """Check if model files exist for all trading pairs"""
    all_files_exist = True
    missing_files = []
    
    for pair in args.pairs:
        pair_code = pair.replace("/", "")
        
        for timeframe in args.timeframes:
            # Check for model files
            model_path = os.path.join(MODELS_DIR, f"{pair_code}_{timeframe}_dual_strategy.h5")
            if not os.path.exists(model_path):
                missing_files.append(model_path)
                all_files_exist = False
    
    if not all_files_exist:
        logger.warning(f"Missing model files: {', '.join(missing_files)}")
    
    return all_files_exist


def main():
    """Main function"""
    args = parse_arguments()
    ensure_directories()
    
    # Step 1: Install dependencies
    if not install_dependencies(args):
        logger.error("Failed to install dependencies")
        sys.exit(1)
    
    # Step 2: Train models for all pairs
    if not train_models(args):
        logger.warning("Model training had some issues, but continuing...")
    
    # Step 3: Analyze correlations between assets
    if not analyze_correlations(args):
        logger.warning("Correlation analysis had some issues, but continuing...")
    
    # Step 4: Analyze sentiment
    if not analyze_sentiment(args):
        logger.warning("Sentiment analysis had some issues, but continuing...")
    
    # Step 5: Start trading
    start_trading(args)


if __name__ == "__main__":
    main()