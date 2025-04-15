#!/usr/bin/env python3
"""
Kraken Trading Bot - Main

This script runs the main trading bot for Kraken exchange using multiple
independent strategies that can be selected at runtime:
- Adaptive
- ARIMA
- Integrated ML

The bot supports running on multiple trading pairs, with various risk
management features including dynamic position sizing, trailing stops,
and cross-strategy coordination. It can run in sandbox mode for testing
or in live mode for real trading.
"""

import os
import argparse
import logging
import json
import time
import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
ML_MODELS_DIR = "ml_models"
DEFAULT_PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_POSITION_FILE = f"{DATA_DIR}/sandbox_positions.json"
DEFAULT_TRADE_HISTORY_FILE = f"{DATA_DIR}/sandbox_trades.json"
DEFAULT_RISK_METRICS_FILE = f"{DATA_DIR}/risk_metrics.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Kraken Trading Bot")
    parser.add_argument("--pair", type=str, default="SOL/USD",
                      help="Trading pair (default: SOL/USD)")
    parser.add_argument("--quantity", type=float, default=0.0,
                      help="Quantity to trade (default: auto-calculated)")
    parser.add_argument("--strategy", type=str, default="integrated_ml",
                      choices=["adaptive", "arima", "integrated_ml"],
                      help="Trading strategy (default: integrated_ml)")
    parser.add_argument("--multi-strategy", type=str, default="true",
                      choices=["true", "false"],
                      help="Run multiple strategies simultaneously")
    parser.add_argument("--sandbox", action="store_true",
                      help="Run in sandbox mode")
    parser.add_argument("--capital", type=float, default=20000.0,
                      help="Starting capital amount")
    parser.add_argument("--leverage", type=float, default=0.0,
                      help="Leverage to use (default: from config)")
    parser.add_argument("--margin", type=float, default=0.0,
                      help="Margin amount (default: auto-calculated)")
    parser.add_argument("--web", action="store_true",
                      help="Enable web interface")
    parser.add_argument("--live", action="store_true",
                      help="Enable live trading")
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config {config_file}: {e}")
        return {}

def initialize_portfolio(capital, sandbox=True):
    """Initialize portfolio with starting capital"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize or load portfolio history
    if os.path.exists(DEFAULT_PORTFOLIO_FILE):
        try:
            with open(DEFAULT_PORTFOLIO_FILE, 'r') as f:
                portfolio_history = json.load(f)
                
            # Check if we need to update the starting capital
            if portfolio_history and portfolio_history[0]["portfolio_value"] != capital:
                # Update with new capital value
                for point in portfolio_history:
                    # Keep any percentage gains/losses but update the absolute values
                    ratio = point["portfolio_value"] / portfolio_history[0]["portfolio_value"]
                    point["portfolio_value"] = capital * ratio
                    point["cash"] = capital * (point["cash"] / portfolio_history[0]["portfolio_value"])
                
                # Save updated portfolio history
                with open(DEFAULT_PORTFOLIO_FILE, 'w') as f:
                    json.dump(portfolio_history, f, indent=2)
                
                logger.info(f"Updated portfolio history with new capital: ${capital:.2f}")
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
            portfolio_history = []
    else:
        portfolio_history = []
    
    # If no history exists, create initial entry
    if not portfolio_history:
        now = datetime.datetime.now().isoformat()
        portfolio_history = [
            {
                "timestamp": now,
                "portfolio_value": capital,
                "cash": capital,
                "positions": 0,
                "drawdown": 0.0
            }
        ]
        
        # Save initial portfolio
        with open(DEFAULT_PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio_history, f, indent=2)
        
        logger.info(f"Portfolio initialized with ${capital:.2f}")
    
    # Initialize positions file if it doesn't exist
    if not os.path.exists(DEFAULT_POSITION_FILE):
        with open(DEFAULT_POSITION_FILE, 'w') as f:
            json.dump([], f, indent=2)
    
    # Initialize trade history file if it doesn't exist
    if not os.path.exists(DEFAULT_TRADE_HISTORY_FILE):
        with open(DEFAULT_TRADE_HISTORY_FILE, 'w') as f:
            json.dump([], f, indent=2)
    
    return True

def initialize_ml(pairs):
    """Initialize ML models for all pairs"""
    # Load ML config
    ml_config_file = f"{CONFIG_DIR}/ml_config.json"
    ml_config = load_config(ml_config_file)
    
    if not ml_config:
        logger.error("Failed to load ML configuration")
        return False
    
    # Check if ML is enabled
    if not ml_config.get("use_ml", False):
        logger.info("ML is disabled in configuration")
        return True
    
    # Check that all pairs are configured
    for pair in pairs:
        if pair not in ml_config.get("pairs", {}):
            logger.warning(f"Pair {pair} not configured in ML config")
            return False
        
        # Check if ensemble weights exist
        weights_file = f"{ML_MODELS_DIR}/ensemble/{pair.replace('/', '_')}_weights.json"
        if not os.path.exists(weights_file):
            logger.warning(f"Ensemble weights not found for {pair}: {weights_file}")
            return False
    
    logger.info(f"ML initialized successfully for {len(pairs)} pairs")
    return True

def load_risk_config():
    """Load risk management configuration"""
    risk_config_file = f"{CONFIG_DIR}/risk_config.json"
    return load_config(risk_config_file)

def load_integrated_risk_config():
    """Load integrated risk management configuration"""
    risk_config_file = f"{CONFIG_DIR}/integrated_risk_config.json"
    return load_config(risk_config_file)

def load_dynamic_params_config():
    """Load dynamic parameters configuration"""
    params_config_file = f"{CONFIG_DIR}/dynamic_params_config.json"
    return load_config(params_config_file)

def update_risk_metrics(risk_profile="aggressive"):
    """Update risk metrics based on current state"""
    # Load current risk metrics if they exist
    if os.path.exists(DEFAULT_RISK_METRICS_FILE):
        with open(DEFAULT_RISK_METRICS_FILE, 'r') as f:
            risk_metrics = json.load(f)
    else:
        risk_metrics = {}
    
    # Update timestamp
    risk_metrics["last_updated"] = datetime.datetime.now().isoformat()
    
    # Adjust risk level based on profile
    if risk_profile == "conservative":
        risk_metrics["current_risk_level"] = "Low"
    elif risk_profile == "balanced":
        risk_metrics["current_risk_level"] = "Medium"
    elif risk_profile == "aggressive":
        risk_metrics["current_risk_level"] = "High"
    elif risk_profile == "ultra":
        risk_metrics["current_risk_level"] = "Very High"
    
    # Update recommendations
    if "risk_recommendations" not in risk_metrics:
        risk_metrics["risk_recommendations"] = {}
    
    if risk_profile == "aggressive" or risk_profile == "ultra":
        risk_metrics["risk_recommendations"]["current_leverage_cap"] = 50.0
        risk_metrics["risk_recommendations"]["current_position_size_cap"] = 0.2
    else:
        risk_metrics["risk_recommendations"]["current_leverage_cap"] = 25.0
        risk_metrics["risk_recommendations"]["current_position_size_cap"] = 0.1
    
    # Save updated metrics
    with open(DEFAULT_RISK_METRICS_FILE, 'w') as f:
        json.dump(risk_metrics, f, indent=2)
    
    return True

def start_trading_bot(args):
    """Start the trading bot with specified arguments"""
    logger.info(f"Starting trading bot with strategy: {args.strategy}")
    logger.info(f"Mode: {'Sandbox' if args.sandbox else 'Live'}")
    
    # In a real implementation, this would start the trading bot
    # For this example, we'll just simulate it
    logger.info("Trading bot started")
    logger.info("Press Ctrl+C to stop")
    
    # Simulated bot loop
    try:
        while True:
            logger.info("Bot running...")
            time.sleep(60)  # Wait for 60 seconds
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    
    return True

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Select primary pair and ensure it's in uppercase
    primary_pair = args.pair.upper()
    
    # Set up additional trading pairs
    # In a real implementation, these would be configurable
    all_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    
    # Initialize portfolio
    if not initialize_portfolio(args.capital, args.sandbox):
        logger.error("Failed to initialize portfolio")
        return
    
    # Initialize ML for all pairs
    if args.strategy == "integrated_ml" and not initialize_ml(all_pairs):
        logger.error("Failed to initialize ML")
        return
    
    # Load risk configuration
    risk_config = load_risk_config()
    if not risk_config:
        logger.error("Failed to load risk configuration")
        return
    
    # Load integrated risk configuration
    integrated_risk_config = load_integrated_risk_config()
    if not integrated_risk_config:
        logger.error("Failed to load integrated risk configuration")
        return
    
    # Load dynamic parameters configuration
    dynamic_params_config = load_dynamic_params_config()
    if not dynamic_params_config:
        logger.error("Failed to load dynamic parameters configuration")
        return
    
    # Update risk metrics
    risk_profile = integrated_risk_config.get("risk_profile", "balanced")
    if not update_risk_metrics(risk_profile):
        logger.error("Failed to update risk metrics")
        return
    
    # Log configuration summary
    logger.info("=" * 80)
    logger.info("KRAKEN TRADING BOT - CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Mode: {'Sandbox' if args.sandbox else 'Live'}")
    logger.info(f"Risk Profile: {risk_profile.capitalize()}")
    logger.info(f"Trading Pairs: {', '.join(all_pairs)}")
    logger.info(f"Starting Capital: ${args.capital:.2f}")
    logger.info(f"Multi-Strategy: {args.multi_strategy}")
    logger.info("=" * 80)
    
    # In a real implementation, this would start the actual trading bot
    # For this example, we'll just simulate it
    if args.strategy == "integrated_ml":
        logger.info("Using ML-enhanced integrated strategy")
        logger.info(f"ML Self-Optimization: {dynamic_params_config.get('dynamic_parameters', {}).get('enabled', False)}")
        
        # Log accuracy targets
        logger.info("ML Accuracy Targets:")
        ml_config = load_config(f"{CONFIG_DIR}/ml_config.json")
        for pair in all_pairs:
            if pair in ml_config.get("pairs", {}):
                accuracy = ml_config["pairs"][pair].get("accuracy", 0.0)
                logger.info(f"  {pair}: {accuracy:.4f}")
    
    # Start trading bot
    if not start_trading_bot(args):
        logger.error("Failed to start trading bot")
        return
    
    logger.info("Trading bot exited")

if __name__ == "__main__":
    main()