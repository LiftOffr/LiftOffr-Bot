#!/usr/bin/env python3
"""
Run Enhanced Trading Bot

This script runs the Kraken trading bot with enhanced risk management in sandbox mode.
It integrates realistic trading conditions including fees, liquidation risks, slippage,
and advanced risk metrics.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import random
import threading
from typing import Dict, List, Tuple, Optional, Union, Any

# Import risk-aware sandbox trader
from risk_aware_sandbox_trader import RiskAwareSandboxTrader

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
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run enhanced Kraken trading bot')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    parser.add_argument('--pairs', type=str, default='SOL/USD,BTC/USD,ETH/USD,ADA/USD,DOT/USD,LINK/USD',
                        help='Comma-separated list of trading pairs')
    parser.add_argument('--strategies', type=str, default='Adaptive,ARIMA',
                        help='Comma-separated list of strategies to use')
    parser.add_argument('--interval', type=int, default=5,
                        help='Trading interval in minutes')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def load_config(filepath: str, default: Optional[Dict] = None) -> Dict:
    """Load configuration from a JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading config {filepath}: {e}")
        return default if default is not None else {}

def simulate_market_price(pair: str, base_price: float) -> float:
    """Simulate market price movement for testing"""
    if pair.endswith('USD'):
        # Simulate small price changes
        change_pct = random.uniform(-0.002, 0.002)  # 0.2% max change
        return base_price * (1 + change_pct)
    return base_price

def get_ml_prediction(
    pair: str,
    strategy: str,
    current_price: float,
    ml_config: Dict
) -> Tuple[str, float, float]:
    """
    Get ML-based prediction for a trading pair
    
    Args:
        pair: Trading pair
        strategy: Trading strategy
        current_price: Current market price
        ml_config: ML configuration
        
    Returns:
        (direction, confidence, target_price)
    """
    # Get pair-specific configuration
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    accuracy = pair_config.get("accuracy", 0.65)
    
    # Simulate prediction based on ML accuracy
    if random.random() < accuracy:
        # Correct prediction (based on accuracy)
        if random.random() < 0.53:  # Slight bullish bias
            direction = "Long"
            confidence = random.uniform(0.65, 0.98)
            price_change = random.uniform(0.02, 0.12)  # 2-12% price target
            target_price = current_price * (1 + price_change)
        else:
            direction = "Short"
            confidence = random.uniform(0.65, 0.95)
            price_change = random.uniform(0.02, 0.1)  # 2-10% price target
            target_price = current_price * (1 - price_change)
    else:
        # Incorrect prediction (based on 1-accuracy)
        if random.random() < 0.5:
            direction = "Long"
            confidence = random.uniform(0.5, 0.75)  # Lower confidence for wrong predictions
            price_change = random.uniform(-0.08, -0.01)  # Wrong direction
            target_price = current_price * (1 + price_change)
        else:
            direction = "Short"
            confidence = random.uniform(0.5, 0.75)
            price_change = random.uniform(-0.08, -0.01)
            target_price = current_price * (1 - price_change)
    
    # Adjust based on strategy
    if strategy == "ARIMA":
        # ARIMA strategy focuses on short-term price movements
        confidence = min(confidence, 0.92)  # Cap confidence
        # Adjust target price to be more conservative
        if direction == "Long":
            target_price = current_price * (1 + price_change * 0.7)
        else:
            target_price = current_price * (1 - price_change * 0.7)
    
    # Apply strategy-specific adjustments from ML config
    strategy_config = ml_config.get("strategies", {}).get(strategy, {})
    confidence_factor = strategy_config.get("confidence_factor", 1.0)
    confidence *= confidence_factor
    
    return direction, confidence, target_price

def calculate_size_and_leverage(
    direction: str,
    confidence: float,
    current_price: float,
    portfolio_value: float,
    dynamic_params: Dict,
    risk_config: Dict
) -> Tuple[float, float]:
    """
    Calculate position size and leverage based on confidence
    
    Args:
        direction: "Long" or "Short"
        confidence: Prediction confidence (0.0-1.0)
        current_price: Current market price
        portfolio_value: Current portfolio value
        dynamic_params: Dynamic parameters configuration
        risk_config: Risk management configuration
        
    Returns:
        (size, leverage)
    """
    # Get base parameters
    base_leverage = risk_config.get("base_leverage", 20.0)
    max_leverage = risk_config.get("max_leverage", 125.0)
    confidence_threshold = risk_config.get("confidence_threshold", 0.65)
    base_risk_percentage = risk_config.get("risk_percentage", 0.2)
    max_risk_percentage = risk_config.get("max_risk_percentage", 0.3)
    
    # Apply dynamic parameters based on confidence
    if confidence < confidence_threshold:
        # Below threshold, reduce risk
        leverage = base_leverage * 0.5
        risk_percentage = base_risk_percentage * 0.5
    else:
        # Scale leverage based on confidence
        confidence_scale = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
        leverage_range = max_leverage - base_leverage
        leverage = base_leverage + (confidence_scale * leverage_range)
        
        # Scale risk percentage based on confidence
        risk_range = max_risk_percentage - base_risk_percentage
        risk_percentage = base_risk_percentage + (confidence_scale * risk_range * 0.5)
    
    # Apply dynamic parameter adjustments
    leverage_adjustment = dynamic_params.get("leverage_adjustment", 1.0)
    risk_adjustment = dynamic_params.get("risk_adjustment", 1.0)
    
    leverage *= leverage_adjustment
    risk_percentage *= risk_adjustment
    
    # Cap leverage and risk
    leverage = min(max_leverage, max(1.0, leverage))
    risk_percentage = min(max_risk_percentage, max(0.05, risk_percentage))
    
    # Calculate position size based on risk percentage
    margin = portfolio_value * risk_percentage
    notional_value = margin * leverage
    size = notional_value / current_price
    
    return size, leverage

def run_trading_bot(args: argparse.Namespace):
    """
    Run the enhanced trading bot
    
    Args:
        args: Command-line arguments
    """
    # Initialize risk-aware sandbox trader
    trader = RiskAwareSandboxTrader()
    
    # Load configurations
    ml_config = load_config(ML_CONFIG_FILE, {"pairs": {}, "strategies": {}})
    risk_config = load_config(RISK_CONFIG_FILE, {
        "base_leverage": 20.0,
        "max_leverage": 125.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.2,
        "max_risk_percentage": 0.3
    })
    dynamic_params = load_config(DYNAMIC_PARAMS_CONFIG_FILE, {"leverage_adjustment": 1.0, "risk_adjustment": 1.0})
    
    # Parse trading pairs and strategies
    pairs = args.pairs.split(',')
    strategies = args.strategies.split(',')
    
    # Store initial prices for simulation
    simulated_prices = {
        "SOL/USD": 160.0,
        "BTC/USD": 70000.0,
        "ETH/USD": 3500.0,
        "ADA/USD": 0.45,
        "DOT/USD": 7.0,
        "LINK/USD": 18.0
    }
    
    # Initialize prices for any missing pairs
    for pair in pairs:
        if pair not in simulated_prices:
            simulated_prices[pair] = 100.0  # Default placeholder
    
    # Initialize last funding fee time
    last_funding_fee_time = datetime.datetime.now()
    
    # Trading loop
    while True:
        try:
            current_time = datetime.datetime.now()
            logger.info(f"Trading cycle at {current_time}")
            
            # Update simulated prices
            for pair in pairs:
                simulated_prices[pair] = simulate_market_price(pair, simulated_prices[pair])
            
            # Update position prices
            trader.update_position_prices(simulated_prices)
            
            # Check if we need to apply funding fees (every 8 hours)
            if (current_time - last_funding_fee_time).total_seconds() >= 28800:  # 8 hours
                trader.apply_funding_fees()
                last_funding_fee_time = current_time
            
            # Get current portfolio value
            portfolio_value = trader.get_current_portfolio_value()
            logger.info(f"Current portfolio value: ${portfolio_value:.2f}")
            
            # Check for trading signals
            for pair in pairs:
                current_price = simulated_prices[pair]
                
                # Check each strategy
                for strategy in strategies:
                    # Skip if we already have a position for this pair and strategy
                    if any(pos["pair"] == pair and pos["strategy"] == strategy for pos in trader.positions):
                        continue
                    
                    # Get ML prediction
                    direction, confidence, target_price = get_ml_prediction(
                        pair, strategy, current_price, ml_config
                    )
                    
                    # Check confidence threshold
                    if confidence < risk_config.get("confidence_threshold", 0.65):
                        logger.info(f"Skipping {pair} with {strategy}: confidence {confidence:.2f} below threshold")
                        continue
                    
                    # Calculate size and leverage
                    size, leverage = calculate_size_and_leverage(
                        direction, confidence, current_price, portfolio_value,
                        dynamic_params, risk_config
                    )
                    
                    # Open position
                    logger.info(f"Opening {direction} position for {pair} with {strategy} strategy")
                    logger.info(f"  Price: ${current_price:.2f}, Size: {size:.6f}, Leverage: {leverage:.1f}x")
                    logger.info(f"  Confidence: {confidence:.2f}, Target: ${target_price:.2f}")
                    
                    success, position = trader.open_position(
                        pair=pair,
                        direction=direction,
                        size=size,
                        entry_price=current_price,
                        leverage=leverage,
                        strategy=strategy,
                        confidence=confidence
                    )
                    
                    if success:
                        logger.info(f"Successfully opened position for {pair}")
                    else:
                        logger.warning(f"Failed to open position for {pair}")
            
            # Log current positions
            if trader.positions:
                logger.info(f"Current positions: {len(trader.positions)}")
                for pos in trader.positions:
                    unrealized_pnl = pos["unrealized_pnl"] * 100
                    logger.info(f"  {pos['pair']} {pos['direction']} {pos['leverage']}x: {unrealized_pnl:.2f}%")
            else:
                logger.info("No open positions")
            
            # Sleep until next cycle
            sleep_time = args.interval * 60
            logger.info(f"Sleeping for {args.interval} minutes")
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting")
            break
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            time.sleep(60)  # Wait a bit before retrying

def main():
    """Main function"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up required directories
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    
    # Start the trading bot
    logger.info(f"Starting enhanced trading bot with pairs: {args.pairs}")
    logger.info(f"Using strategies: {args.strategies}")
    logger.info(f"Sandbox mode: {args.sandbox}")
    
    run_trading_bot(args)

if __name__ == '__main__':
    main()