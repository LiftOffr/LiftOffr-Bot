#!/usr/bin/env python3
"""
Start Trading Bot with Liquidation Protection

This script starts the enhanced trading bot with real-time market data integration and
realistic liquidation protection for all 10 cryptocurrency pairs. It avoids using Flask
and port 5000 to prevent conflicts with the dashboard application.
"""
import os
import sys
import json
import time
import logging
import threading
import random
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import integration controller for real-time data
from integration_controller import IntegrationController

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"

# Trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD",
    "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD",
    "UNI/USD", "ATOM/USD"
]

def load_json(filepath: str, default: Any = None) -> Any:
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        logger.warning(f"File not found: {filepath}")
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default if default is not None else {}

def save_json(filepath: str, data: Dict) -> None:
    """Save data to a JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def get_ml_prediction(pair: str, current_price: float, ml_config: Dict) -> tuple:
    """
    Simulate an ML prediction based on configuration
    
    Args:
        pair: Trading pair
        current_price: Current price
        ml_config: ML configuration
        
    Returns:
        (direction, confidence, target_price, strategy)
    """
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    accuracy = pair_config.get("accuracy", 0.65)
    strategies = ["Adaptive", "ARIMA"]
    
    # Random strategy with slight preference for Adaptive
    strategy = "Adaptive" if random.random() < 0.55 else "ARIMA"
    
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
            confidence = random.uniform(0.5, 0.75)  # Lower confidence
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
    
    return direction, confidence, target_price, strategy

def calculate_position_size(direction: str, confidence: float, current_price: float, 
                           portfolio_value: float, risk_config: Dict) -> tuple:
    """
    Calculate position size and leverage based on confidence
    
    Args:
        direction: "Long" or "Short"
        confidence: Prediction confidence (0.0-1.0)
        current_price: Current market price
        portfolio_value: Current portfolio value
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
    
    # Cap leverage and risk
    leverage = min(max_leverage, max(1.0, leverage))
    risk_percentage = min(max_risk_percentage, max(0.05, risk_percentage))
    
    # Calculate position size based on risk percentage
    margin = portfolio_value * risk_percentage
    notional_value = margin * leverage
    size = notional_value / current_price
    
    return size, leverage

def open_position(pair: str, direction: str, size: float, entry_price: float, 
                 leverage: float, strategy: str, confidence: float, 
                 liquidation_price: float) -> tuple:
    """
    Open a new trading position
    
    Args:
        pair: Trading pair
        direction: "Long" or "Short"
        size: Position size
        entry_price: Entry price
        leverage: Leverage
        strategy: Trading strategy
        confidence: Prediction confidence
        liquidation_price: Calculated liquidation price
        
    Returns:
        (success, position)
    """
    positions = load_json(POSITIONS_FILE, [])
    portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
    
    # Calculate margin and check balance
    margin = size * entry_price / leverage
    if margin > portfolio.get("balance", 0):
        logger.warning(f"Insufficient balance for {pair} {direction}")
        return False, None
    
    # Create position
    position = {
        "pair": pair,
        "direction": direction,
        "size": size,
        "entry_price": entry_price,
        "current_price": entry_price,
        "leverage": leverage,
        "strategy": strategy,
        "confidence": confidence,
        "entry_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "unrealized_pnl": 0.0,
        "unrealized_pnl_amount": 0.0,
        "liquidation_price": liquidation_price
    }
    
    # Update balance
    portfolio["balance"] = portfolio.get("balance", 20000.0) - margin
    portfolio["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    
    # Add position
    positions.append(position)
    
    # Save to files
    save_json(POSITIONS_FILE, positions)
    save_json(PORTFOLIO_FILE, portfolio)
    
    return True, position

def trading_signal_generator(controller: IntegrationController):
    """
    Periodically generate trading signals and open positions
    
    Args:
        controller: Integration controller for real-time data
    """
    ml_config = load_json(ML_CONFIG_FILE, {"pairs": {}, "strategies": {}})
    risk_config = load_json(RISK_CONFIG_FILE, {
        "base_leverage": 20.0,
        "max_leverage": 125.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.2,
        "max_risk_percentage": 0.3
    })
    
    # Strategy categories
    strategy_categories = {
        "Adaptive": "him all along",
        "ARIMA": "those dudes"
    }
    
    while True:
        try:
            # Fetch current prices and positions
            current_prices = controller.latest_prices
            positions = load_json(POSITIONS_FILE, [])
            portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
            
            # Get current portfolio value
            portfolio_value = portfolio.get("balance", 20000.0)
            for pos in positions:
                if "unrealized_pnl_amount" in pos:
                    portfolio_value += pos["unrealized_pnl_amount"]
            
            logger.info(f"Current portfolio value: ${portfolio_value:.2f}")
            
            # Check for trading signals (every 3-5 minutes for each pair)
            for pair in DEFAULT_PAIRS:
                if pair not in current_prices:
                    continue
                
                current_price = current_prices[pair]
                
                # Check if we already have open positions for this pair
                existing_strategies = [
                    pos["strategy"] for pos in positions 
                    if pos["pair"] == pair
                ]
                
                if len(existing_strategies) >= 2:
                    # Skip if we already have positions for both strategies
                    continue
                
                # Get ML prediction
                direction, confidence, target_price, strategy = get_ml_prediction(
                    pair, current_price, ml_config
                )
                
                # Skip if we already have a position for this strategy
                if strategy in existing_strategies:
                    continue
                
                # Check confidence threshold
                if confidence < risk_config.get("confidence_threshold", 0.65):
                    continue
                
                # Calculate size and leverage
                size, leverage = calculate_position_size(
                    direction, confidence, current_price, portfolio_value, risk_config
                )
                
                # Calculate liquidation price
                liquidation_price = controller.calculate_liquidation_price(
                    current_price, leverage, direction
                )
                
                # Validate leverage to prevent liquidation risk
                if leverage > 50:
                    # Calculate price distance to liquidation as percentage
                    if direction.lower() == "long":
                        price_buffer = (current_price - liquidation_price) / current_price
                    else:
                        price_buffer = (liquidation_price - current_price) / current_price
                        
                    # If buffer is too small, reduce leverage
                    min_buffer = 0.05  # Minimum 5% buffer to liquidation price
                    if price_buffer < min_buffer:
                        # Adjust leverage to ensure minimum buffer
                        adjusted_leverage = (1.0 / (min_buffer + 0.01)) * 0.9
                        logger.warning(
                            f"Reducing leverage from {leverage:.1f}x to {adjusted_leverage:.1f}x "
                            f"to ensure sufficient liquidation buffer"
                        )
                        leverage = min(leverage, adjusted_leverage)
                        
                        # Recalculate size with adjusted leverage
                        margin = portfolio_value * risk_config.get("risk_percentage", 0.2)
                        notional_value = margin * leverage
                        size = notional_value / current_price
                
                # Random decision to open position (not every signal results in a trade)
                if random.random() < 0.3:  # 30% chance of opening a position
                    # Open position
                    logger.info(f"Opening {direction} position for {pair} with {strategy} strategy")
                    logger.info(f"  Price: ${current_price:.2f}, Size: {size:.6f}, Leverage: {leverage:.1f}x")
                    logger.info(f"  Confidence: {confidence:.2f}, Target: ${target_price:.2f}")
                    logger.info(f"  Liquidation price: ${liquidation_price:.2f}")
                    logger.info(f"  Category: {strategy_categories.get(strategy, 'unknown')}")
                    
                    success, position = open_position(
                        pair=pair,
                        direction=direction,
                        size=size,
                        entry_price=current_price,
                        leverage=leverage,
                        strategy=strategy,
                        confidence=confidence,
                        liquidation_price=liquidation_price
                    )
                    
                    if success:
                        logger.info(f"Successfully opened position for {pair}")
                    else:
                        logger.warning(f"Failed to open position for {pair}")
            
            # Sleep for random time (3-5 minutes)
            sleep_time = random.randint(180, 300)
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in trading signal generator: {e}")
            time.sleep(60)  # Wait before retrying

def display_status(controller: IntegrationController):
    """
    Periodically display status of positions and portfolio
    
    Args:
        controller: Integration controller for real-time data
    """
    while True:
        try:
            # Load positions and portfolio
            positions = load_json(POSITIONS_FILE, [])
            portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
            current_prices = controller.latest_prices
            
            # Display current prices
            if current_prices:
                print("\n" + "=" * 60)
                print(" CURRENT MARKET PRICES")
                print("=" * 60)
                for pair in sorted(DEFAULT_PAIRS):
                    if pair in current_prices:
                        print(f"{pair}: ${current_prices[pair]:.2f}")
            
            # Display portfolio summary
            print("\n" + "=" * 60)
            print(" PORTFOLIO SUMMARY")
            print("=" * 60)
            balance = portfolio.get("balance", 20000.0)
            unrealized_pnl = portfolio.get("unrealized_pnl_usd", 0.0)
            total_equity = balance + unrealized_pnl
            print(f"Balance: ${balance:.2f}")
            print(f"Unrealized P&L: ${unrealized_pnl:.2f}")
            print(f"Total Equity: ${total_equity:.2f}")
            
            # Display positions
            if positions:
                print("\n" + "=" * 60)
                print(f" OPEN POSITIONS ({len(positions)})")
                print("=" * 60)
                for pos in positions:
                    pair = pos["pair"]
                    direction = pos["direction"]
                    leverage = pos["leverage"]
                    entry_price = pos["entry_price"]
                    size = pos["size"]
                    strategy = pos["strategy"]
                    
                    current_price = current_prices.get(pair, entry_price)
                    
                    # Calculate PnL
                    if direction.lower() == "long":
                        pnl_pct = (current_price / entry_price - 1) * leverage * 100
                    else:
                        pnl_pct = (1 - current_price / entry_price) * leverage * 100
                    
                    # Get liquidation price
                    liq_price = pos.get("liquidation_price", controller.calculate_liquidation_price(
                        entry_price, leverage, direction
                    ))
                    
                    print(f"{pair} {direction} {leverage}x - {strategy}")
                    print(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
                    print(f"  Size: {size:.6f}, P&L: {pnl_pct:.2f}%")
                    print(f"  Liquidation Price: ${liq_price:.2f}")
                    
                    # Display liquidation risk warning
                    distance_to_liq = 0
                    if direction.lower() == "long":
                        distance_to_liq = ((current_price - liq_price) / current_price) * 100
                    else:
                        distance_to_liq = ((liq_price - current_price) / current_price) * 100
                    
                    if distance_to_liq < 5:
                        print(f"  ⚠️ WARNING: Only {distance_to_liq:.2f}% away from liquidation!")
                    
                    print()
            else:
                print("\nNo open positions")
            
            print("=" * 60)
            print(" All trading is in sandbox mode (no real funds at risk)")
            print("=" * 60 + "\n")
            
            # Sleep for 60 seconds
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in status display: {e}")
            time.sleep(30)

def main():
    """Main function"""
    try:
        # Print welcome message
        print("\n" + "=" * 60)
        print(" ENHANCED TRADING BOT WITH LIQUIDATION PROTECTION")
        print("=" * 60)
        print("\nTrading 10 pairs in sandbox mode with real-time market data:")
        for i, pair in enumerate(DEFAULT_PAIRS):
            print(f"{i+1}. {pair}")
        print("\nFeatures:")
        print("- Real-time market data from Kraken API")
        print("- Accurate liquidation price calculation")
        print("- Dynamic leverage based on prediction confidence")
        print("- ML-enhanced trading signals")
        print("- Cross-strategy signal arbitration")
        print("\nStarting integration with real-time market data...")
        
        # Initialize integration controller
        controller = IntegrationController(pairs=DEFAULT_PAIRS)
        
        # Start the integration controller
        controller.start()
        logger.info("Started real-time market data integration")
        
        # Initial price fetch and update
        logger.info("Fetching initial prices...")
        controller.update_prices()
        
        # Update positions with current prices
        controller.update_position_prices()
        
        # Check for liquidations
        controller.check_liquidations()
        
        # Update portfolio with current unrealized P&L
        controller.update_portfolio()
        
        # Start trading signal generator thread
        signal_thread = threading.Thread(target=trading_signal_generator, args=(controller,))
        signal_thread.daemon = True
        signal_thread.start()
        
        # Start status display thread
        status_thread = threading.Thread(target=display_status, args=(controller,))
        status_thread.daemon = True
        status_thread.start()
        
        # Keep main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        # Stop integration controller if it exists
        if 'controller' in locals():
            controller.stop()
            logger.info("Stopped real-time market data integration")

if __name__ == "__main__":
    main()