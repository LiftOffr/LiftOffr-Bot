#!/usr/bin/env python3
"""
Start Sandbox Trading for All Pairs

This script initiates sandbox trading for all 10 cryptocurrency pairs:
BTC/USD, ETH/USD, SOL/USD, ADA/USD, DOT/USD, LINK/USD, AVAX/USD, MATIC/USD, UNI/USD, ATOM/USD

The trading system uses optimized ML models with:
- Advanced ensemble predictions
- Dynamic leverage based on confidence
- Robust risk management
- Cross-strategy signal arbitration

All trading is conducted in sandbox mode for safety.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sandbox_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sandbox_trader")

# Constants
ML_CONFIG_PATH = "config/ml_config.json"
RISK_CONFIG_PATH = "config/risk_config.json"
INITIAL_CAPITAL = 20000.0
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
    "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Start sandbox trading for all pairs.')
    parser.add_argument('--pairs', type=str, default=None,
                      help='Comma-separated list of pairs to trade (default: all pairs)')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                      help=f'Initial capital (default: {INITIAL_CAPITAL})')
    parser.add_argument('--no-dynamic-leverage', action='store_true',
                      help='Disable dynamic leverage')
    parser.add_argument('--no-dashboard-update', action='store_true',
                      help='Disable automatic dashboard updates')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()


def load_ml_config() -> Optional[Dict]:
    """Load ML configuration."""
    try:
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration from {ML_CONFIG_PATH}")
            return config
        else:
            logger.error(f"ML configuration file not found: {ML_CONFIG_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading ML configuration: {e}")
        return None


def load_risk_config() -> Optional[Dict]:
    """Load risk configuration."""
    try:
        if os.path.exists(RISK_CONFIG_PATH):
            with open(RISK_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded risk configuration from {RISK_CONFIG_PATH}")
            return config
        else:
            logger.warning(f"Risk configuration file not found: {RISK_CONFIG_PATH}")
            # Create default risk configuration
            config = {
                "max_drawdown_percentage": 25.0,
                "position_sizing": {
                    "default_risk_percentage": 0.20,
                    "max_portfolio_risk": 0.50
                },
                "stop_loss": {
                    "enable_trailing_stop": True,
                    "trailing_stop_activation_percentage": 1.5,
                    "trailing_stop_distance_percentage": 2.0,
                    "fixed_stop_loss_percentage": 4.0
                }
            }
            return config
    except Exception as e:
        logger.error(f"Error loading risk configuration: {e}")
        return None


def get_active_pairs(ml_config: Dict, args) -> List[str]:
    """
    Get list of active trading pairs.
    
    Args:
        ml_config: ML configuration
        args: Command line arguments
        
    Returns:
        active_pairs: List of active pairs
    """
    if args.pairs:
        # Use pairs specified in command line
        requested_pairs = args.pairs.split(",")
        active_pairs = []
        
        for pair in requested_pairs:
            if pair in ml_config.get("pairs", {}):
                active_pairs.append(pair)
            else:
                logger.warning(f"Pair {pair} not found in ML configuration, skipping")
        
        if not active_pairs:
            logger.error("No valid pairs specified")
            return []
        
        return active_pairs
    else:
        # Use all active pairs from configuration
        active_pairs = []
        
        for pair, settings in ml_config.get("pairs", {}).items():
            if settings.get("active", False) and settings.get("trained", False):
                active_pairs.append(pair)
        
        if not active_pairs:
            logger.warning("No active trained pairs found in configuration")
            return ALL_PAIRS  # Try all pairs as fallback
        
        return active_pairs


def run_command(cmd: List[str], description: str = None) -> bool:
    """
    Run a shell command.
    
    Args:
        cmd: Command to run
        description: Command description
        
    Returns:
        success: Whether the command succeeded
    """
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def start_trading(pairs: List[str], args) -> bool:
    """
    Start trading bot for specified pairs.
    
    Args:
        pairs: List of pairs to trade
        args: Command line arguments
        
    Returns:
        success: Whether the command succeeded
    """
    pairs_arg = ",".join(pairs)
    dynamic_leverage = not args.no_dynamic_leverage
    
    # Prepare command
    cmd = [
        sys.executable, "run_enhanced_trading_bot.py",
        "--pairs", pairs_arg,
        "--capital", str(args.capital),
        "--sandbox"
    ]
    
    if not dynamic_leverage:
        cmd.append("--no-dynamic-leverage")
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run command
    return run_command(
        cmd,
        f"Starting sandbox trading for {len(pairs)} pairs: {pairs_arg}"
    )


def start_dashboard_updater(args) -> bool:
    """
    Start automatic dashboard updater.
    
    Args:
        args: Command line arguments
        
    Returns:
        success: Whether the command succeeded
    """
    if args.no_dashboard_update:
        logger.info("Automatic dashboard updates disabled")
        return True
    
    cmd = [
        sys.executable, "auto_update_dashboard.py",
        "--interval", "60"  # Update every 60 seconds
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run command in background
    cmd.append("&")
    
    return run_command(
        cmd,
        "Starting automatic dashboard updater"
    )


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configurations
    ml_config = load_ml_config()
    if ml_config is None:
        logger.error("Failed to load ML configuration, exiting")
        return 1
    
    risk_config = load_risk_config()
    if risk_config is None:
        logger.warning("Failed to load risk configuration, using defaults")
    
    # Get active pairs
    active_pairs = get_active_pairs(ml_config, args)
    if not active_pairs:
        logger.error("No active pairs found, exiting")
        return 1
    
    logger.info(f"Starting sandbox trading with {len(active_pairs)} pairs: {', '.join(active_pairs)}")
    
    # Start trading
    success = start_trading(active_pairs, args)
    if not success:
        logger.error("Failed to start trading, exiting")
        return 1
    
    # Start dashboard updater
    start_dashboard_updater(args)
    
    logger.info("Sandbox trading successfully started")
    logger.info("Monitor the dashboard for trading performance")
    
    print("\n" + "=" * 60)
    print(" SANDBOX TRADING STARTED")
    print("=" * 60)
    print(f"\nTrading {len(active_pairs)} pairs in sandbox mode:")
    for i, pair in enumerate(active_pairs):
        print(f"{i+1}. {pair}")
    print("\nInitial capital: ${:,.2f}".format(args.capital))
    print("\nAll trading is in sandbox mode (no real funds at risk)")
    print("Monitor the dashboard for performance metrics")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())