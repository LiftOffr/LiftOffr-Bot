#!/usr/bin/env python3
"""
Run Enhanced Trading Simulation

This script runs the Kraken trading bot with ultra-realistic trading conditions:
1. Order book simulation with depth and spread
2. Partial fills for large orders
3. Variable slippage based on order size and market liquidity
4. Exchange latency simulation
5. Market impact modeling
6. Tier-based fee structure
7. Flash crash stress testing
8. Dynamic parameter adjustment based on market conditions
"""

import os
import sys
import time
import json
import random
import logging
import datetime
import threading
import argparse
from typing import Dict, List, Tuple, Optional

# Import enhanced simulation
from enhanced_simulation_upgrades import EnhancedSimulation, OrderBook

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
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
SIMULATION_LOG_FILE = f"{DATA_DIR}/simulation_log.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run enhanced trading simulation')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    parser.add_argument('--pairs', type=str, default='SOL/USD,BTC/USD,ETH/USD,ADA/USD,DOT/USD,LINK/USD',
                      help='Comma-separated list of trading pairs')
    parser.add_argument('--strategies', type=str, default='Adaptive,ARIMA',
                      help='Comma-separated list of strategies to use')
    parser.add_argument('--interval', type=int, default=5,
                      help='Trading interval in minutes')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--flash-crash', action='store_true', help='Enable flash crash simulation')
    parser.add_argument('--latency', action='store_true', help='Enable network latency simulation')
    parser.add_argument('--stress-test', action='store_true', help='Enable stress testing')
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

def get_ml_prediction(
    pair: str,
    strategy: str,
    current_price: float,
    ml_config: Dict,
    is_flash_crash: bool = False
) -> Tuple[str, float, float]:
    """
    Get ML-based prediction for a trading pair
    
    Args:
        pair: Trading pair
        strategy: Trading strategy
        current_price: Current market price
        ml_config: ML configuration
        is_flash_crash: Whether we're in a flash crash
        
    Returns:
        (direction, confidence, target_price)
    """
    # Get pair-specific configuration
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    accuracy = pair_config.get("accuracy", 0.65)
    
    # Adjust accuracy during flash crash (reduced accuracy)
    if is_flash_crash:
        accuracy = max(0.5, accuracy * 0.8)  # 20% reduction but min 50%
    
    # Simulate prediction based on ML accuracy
    if random.random() < accuracy:
        # Correct prediction (based on accuracy)
        if is_flash_crash:
            # During flash crash, more likely to predict short
            direction = "Short" if random.random() < 0.7 else "Long"
        else:
            # Normal conditions, slight bullish bias
            direction = "Long" if random.random() < 0.53 else "Short"
        
        confidence = random.uniform(0.65, 0.98)
        
        if direction == "Long":
            price_change = random.uniform(0.02, 0.12)  # 2-12% price target
            target_price = current_price * (1 + price_change)
        else:  # Short
            price_change = random.uniform(0.02, 0.1)  # 2-10% price target
            target_price = current_price * (1 - price_change)
    else:
        # Incorrect prediction (based on 1-accuracy)
        if is_flash_crash:
            # During flash crash, more likely to incorrectly predict long
            direction = "Long" if random.random() < 0.7 else "Short"
        else:
            direction = "Long" if random.random() < 0.5 else "Short"
        
        confidence = random.uniform(0.5, 0.75)  # Lower confidence for wrong predictions
        
        if direction == "Long":
            price_change = random.uniform(-0.08, -0.01)  # Wrong direction
            target_price = current_price * (1 + price_change)
        else:  # Short
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
    risk_config: Dict,
    market_volatility: float = 0.01
) -> Tuple[float, float]:
    """
    Calculate position size and leverage based on confidence and market conditions
    
    Args:
        direction: "Long" or "Short"
        confidence: Prediction confidence (0.0-1.0)
        current_price: Current market price
        portfolio_value: Current portfolio value
        dynamic_params: Dynamic parameters configuration
        risk_config: Risk management configuration
        market_volatility: Current market volatility (0.0-1.0)
        
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
    
    # Apply market volatility adjustments
    # Higher volatility = lower leverage and risk
    volatility_factor = 1.0 - (market_volatility * 5.0)
    volatility_factor = max(0.5, volatility_factor)  # Cap reduction at 50%
    
    leverage *= volatility_factor
    risk_percentage *= volatility_factor
    
    # Apply dynamic parameter adjustments
    leverage_adjustment = dynamic_params.get("leverage_adjustment", 1.0)
    risk_adjustment = dynamic_params.get("risk_adjustment", 1.0)
    
    # Apply pair-specific adjustments
    pair_adjustments = dynamic_params.get("pair_specific_adjustments", {})
    if pair_adjustments:
        # Determine pair from context (would be better passed as parameter)
        # This is a simplification for the example
        pair_leverage_adj = 1.0
        pair_risk_adj = 1.0
        leverage_adjustment *= pair_leverage_adj
        risk_adjustment *= pair_risk_adj
    
    # Apply strategy-specific adjustments
    strategy_adjustments = dynamic_params.get("strategy_specific_adjustments", {})
    if strategy_adjustments:
        # Determine strategy from context (would be better passed as parameter)
        # This is a simplification for the example
        strategy_leverage_adj = 1.0
        strategy_risk_adj = 1.0
        leverage_adjustment *= strategy_leverage_adj
        risk_adjustment *= strategy_risk_adj
    
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

def detect_market_regime(
    price_history: List[float],
    returns: List[float],
    window: int = 20
) -> Tuple[str, float]:
    """
    Detect current market regime and volatility
    
    Args:
        price_history: List of historical prices
        returns: List of historical returns
        window: Window size for calculations
        
    Returns:
        (regime, volatility)
    """
    if len(price_history) < window + 1:
        return "unknown", 0.01
    
    # Calculate volatility
    recent_returns = returns[-window:]
    volatility = sum(r**2 for r in recent_returns) / len(recent_returns)
    volatility = volatility ** 0.5  # Standard deviation
    
    # Calculate trend
    recent_prices = price_history[-window:]
    start_price = sum(recent_prices[:5]) / 5  # Average of first 5 prices
    end_price = sum(recent_prices[-5:]) / 5   # Average of last 5 prices
    trend = (end_price / start_price) - 1
    
    # Calculate range-bound indicator (ratio of max-min to standard deviation)
    price_range = max(recent_prices) - min(recent_prices)
    range_indicator = price_range / (volatility * recent_prices[-1])
    
    # Determine regime
    if volatility > 0.03:  # High volatility
        regime = "volatile"
    elif abs(trend) > 0.05:  # Strong trend
        regime = "trending_up" if trend > 0 else "trending_down"
    elif range_indicator < 2.0:  # Range-bound
        regime = "ranging"
    else:
        regime = "normal"
    
    return regime, volatility

def run_trading_simulation(args: argparse.Namespace):
    """
    Run the enhanced trading simulation
    
    Args:
        args: Command-line arguments
    """
    # Initialize enhanced simulation
    simulation = EnhancedSimulation()
    
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
    
    # Initialize price history for regime detection
    price_history = {pair: [] for pair in pairs}
    returns_history = {pair: [] for pair in pairs}
    
    # Initialize simulation log
    simulation_log = {
        "trades": [],
        "portfolio_history": [],
        "market_regimes": [],
        "flash_crashes": [],
        "latency_events": []
    }
    
    # Trading loop
    cycle_count = 0
    try:
        while True:
            cycle_time = datetime.datetime.now().replace(tzinfo=None)
            logger.info(f"Trading cycle #{cycle_count} at {cycle_time}")
            
            # Update market data
            simulation.update_market_data()
            
            # Get current prices from order books
            current_prices = {}
            for pair in pairs:
                if pair in simulation.order_books:
                    current_prices[pair] = simulation.order_books[pair].get_mid_price()
                else:
                    # Use default price if order book doesn't exist
                    default_price = 100.0
                    if "BTC" in pair:
                        default_price = 70000.0
                    elif "ETH" in pair:
                        default_price = 3500.0
                    elif "SOL" in pair:
                        default_price = 160.0
                    
                    current_prices[pair] = default_price
                    
                    # Create order book if it doesn't exist
                    simulation.order_books[pair] = OrderBook(pair, default_price)
            
            # Update price history for regime detection
            for pair, price in current_prices.items():
                price_history[pair].append(price)
                if len(price_history[pair]) > 1:
                    prev_price = price_history[pair][-2]
                    ret = (price / prev_price) - 1
                    returns_history[pair].append(ret)
                
                # Keep history limited to avoid memory issues
                if len(price_history[pair]) > 100:
                    price_history[pair] = price_history[pair][-100:]
                if len(returns_history[pair]) > 100:
                    returns_history[pair] = returns_history[pair][-100:]
            
            # Get current portfolio value
            portfolio_value = simulation.sandbox_trader.get_current_portfolio_value()
            logger.info(f"Current portfolio value: ${portfolio_value:.2f}")
            
            # Log portfolio value
            simulation_log["portfolio_history"].append({
                "timestamp": cycle_time.isoformat(),
                "portfolio_value": portfolio_value
            })
            
            # Update positions with current prices
            simulation.update_positions()
            
            # Check for trading signals
            for pair in pairs:
                current_price = current_prices[pair]
                
                # Detect market regime
                if len(price_history[pair]) > 20 and len(returns_history[pair]) > 20:
                    regime, volatility = detect_market_regime(
                        price_history[pair],
                        returns_history[pair]
                    )
                else:
                    regime = "unknown"
                    volatility = 0.01
                
                # Log market regime
                simulation_log["market_regimes"].append({
                    "timestamp": cycle_time.isoformat(),
                    "pair": pair,
                    "regime": regime,
                    "volatility": volatility
                })
                
                logger.info(f"{pair} regime: {regime}, volatility: {volatility:.4f}")
                
                # Check if in flash crash
                is_flash_crash = simulation.flash_crashes.is_crashing(pair)
                if is_flash_crash:
                    logger.warning(f"{pair} is experiencing a flash crash!")
                    
                    # Log flash crash
                    simulation_log["flash_crashes"].append({
                        "timestamp": cycle_time.isoformat(),
                        "pair": pair,
                        "price": current_price
                    })
                
                # Check each strategy
                for strategy in strategies:
                    # Skip if we already have a position for this pair and strategy
                    if any(pos["pair"] == pair and pos["strategy"] == strategy 
                          for pos in simulation.sandbox_trader.positions):
                        continue
                    
                    # Get ML prediction
                    direction, confidence, target_price = get_ml_prediction(
                        pair, strategy, current_price, ml_config, is_flash_crash
                    )
                    
                    # Check confidence threshold
                    if confidence < risk_config.get("confidence_threshold", 0.65):
                        if args.verbose:
                            logger.info(f"Skipping {pair} with {strategy}: confidence {confidence:.2f} below threshold")
                        continue
                    
                    # Calculate size and leverage based on market conditions
                    size, leverage = calculate_size_and_leverage(
                        direction, confidence, current_price, portfolio_value,
                        dynamic_params, risk_config, volatility
                    )
                    
                    # Adjust strategy based on market regime
                    if regime == "volatile" and args.stress_test:
                        # Reduce size and leverage in volatile markets
                        leverage *= 0.7
                        size *= 0.8
                        logger.info(f"Reducing leverage and size due to volatile market")
                    elif regime == "trending_up" and direction == "Long":
                        # Increase leverage slightly for long in uptrend
                        leverage *= 1.1
                        logger.info(f"Increasing leverage for long in uptrend")
                    elif regime == "trending_down" and direction == "Short":
                        # Increase leverage slightly for short in downtrend
                        leverage *= 1.1
                        logger.info(f"Increasing leverage for short in downtrend")
                    
                    # Skip trades during flash crash unless it's a short
                    if is_flash_crash and direction == "Long" and args.flash_crash:
                        logger.warning(f"Skipping {direction} trade for {pair} during flash crash")
                        continue
                    
                    # Open position
                    logger.info(f"Opening {direction} position for {pair} with {strategy} strategy")
                    logger.info(f"  Price: ${current_price:.2f}, Size: {size:.6f}, Leverage: {leverage:.1f}x")
                    logger.info(f"  Confidence: {confidence:.2f}, Target: ${target_price:.2f}")
                    
                    # Simulate latency if enabled
                    if args.latency:
                        # Check if we should have latency issues
                        if simulation.latency.should_fail():
                            logger.warning(f"Connection failure while opening position for {pair}")
                            
                            # Log latency event
                            simulation_log["latency_events"].append({
                                "timestamp": cycle_time.isoformat(),
                                "pair": pair,
                                "type": "connection_failure",
                                "direction": direction,
                                "strategy": strategy
                            })
                            
                            continue
                        
                        if simulation.latency.should_timeout():
                            logger.warning(f"Request timeout while opening position for {pair}")
                            
                            # Log latency event
                            simulation_log["latency_events"].append({
                                "timestamp": cycle_time.isoformat(),
                                "pair": pair,
                                "type": "timeout",
                                "direction": direction,
                                "strategy": strategy
                            })
                            
                            continue
                    
                    success, position = simulation.execute_trade(
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
                        
                        # Log trade
                        simulation_log["trades"].append({
                            "timestamp": cycle_time.isoformat(),
                            "pair": pair,
                            "direction": direction,
                            "entry_price": position["entry_price"],
                            "size": position["size"],
                            "leverage": position["leverage"],
                            "strategy": strategy,
                            "confidence": confidence,
                            "type": "open",
                            "market_regime": regime
                        })
                    else:
                        logger.warning(f"Failed to open position for {pair}")
            
            # Log current positions
            if simulation.sandbox_trader.positions:
                logger.info(f"Current positions: {len(simulation.sandbox_trader.positions)}")
                for pos in simulation.sandbox_trader.positions:
                    unrealized_pnl = pos.get("unrealized_pnl", 0) * 100
                    logger.info(f"  {pos['pair']} {pos['direction']} {pos['leverage']}x: {unrealized_pnl:.2f}%")
            else:
                logger.info("No open positions")
            
            # Save simulation log periodically
            if cycle_count % 10 == 0:
                with open(SIMULATION_LOG_FILE, 'w') as f:
                    json.dump(simulation_log, f, indent=2)
            
            # Sleep until next cycle
            cycle_count += 1
            sleep_time = args.interval * 60
            logger.info(f"Sleeping for {args.interval} minutes...")
            
            # Use a more responsive sleep that can be interrupted
            for _ in range(sleep_time):
                time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting...")
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)
    finally:
        # Save final simulation log
        with open(SIMULATION_LOG_FILE, 'w') as f:
            json.dump(simulation_log, f, indent=2)
        
        logger.info("Trading simulation complete, results saved to simulation_log.json")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up required directories
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Start the trading simulation
    logger.info(f"Starting enhanced trading simulation with pairs: {args.pairs}")
    logger.info(f"Using strategies: {args.strategies}")
    logger.info(f"Sandbox mode: {args.sandbox}")
    logger.info(f"Flash crash simulation: {args.flash_crash}")
    logger.info(f"Latency simulation: {args.latency}")
    logger.info(f"Stress testing: {args.stress_test}")
    
    run_trading_simulation(args)

if __name__ == '__main__':
    main()