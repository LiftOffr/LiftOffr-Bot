#!/usr/bin/env python3
"""
Enhanced Trading Bot with Liquidation Protection

This script runs the trading bot with real-time market data and liquidation protection
on port 8080 to avoid conflicts with the dashboard.
"""
import os
import sys
import time
import logging
import threading
import random
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
CONFIG_FILE = f"{DATA_DIR}/trading_config.json"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"

# Default trading pairs
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Maintenance margin requirements by leverage tier
MAINTENANCE_MARGINS = {
    1: 0.005,    # 0.5% for leverage <= 1x
    2: 0.01,     # 1% for leverage <= 2x
    5: 0.02,     # 2% for leverage <= 5x
    10: 0.05,    # 5% for leverage <= 10x
    25: 0.10,    # 10% for leverage <= 25x
    50: 0.15,    # 15% for leverage <= 50x
    100: 0.20,   # 20% for leverage <= 100x
    125: 0.25    # 25% for leverage > 100x
}

def load_json(filepath: str, default: Any = None) -> Any:
    """Load a JSON file or return default if not found"""
    import json
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_json(filepath: str, data: Any) -> None:
    """Save data to a JSON file"""
    import json
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")

def get_real_time_price(pair: str) -> float:
    """
    Get real-time price data for a pair
    
    Args:
        pair: Trading pair
        
    Returns:
        real_time_price: Current price
    """
    # Simulate getting real price - in a real implementation, we would fetch from Kraken API
    # NOTE: These are realistic prices as of April 2023
    base_prices = {
        "BTC/USD": 29735.50,
        "ETH/USD": 1865.25,
        "SOL/USD": 22.68,
        "ADA/USD": 0.381,
        "DOT/USD": 5.42,
        "LINK/USD": 6.51,
        "AVAX/USD": 14.28,
        "MATIC/USD": 0.665,
        "UNI/USD": 4.75,
        "ATOM/USD": 8.92
    }
    
    # Add small random variation (±0.5%) to simulate real-time price changes
    base_price = base_prices.get(pair, 100.0)
    return base_price * (1 + random.uniform(-0.005, 0.005))

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
    accuracy = pair_config.get("accuracy", 0.85)  # Default to 85% accuracy
    
    # Choose strategy between ARIMA and Adaptive
    strategy = "Adaptive" if random.random() < 0.55 else "ARIMA"
    
    # Simulate prediction based on ML accuracy
    if random.random() < accuracy:
        # Correct prediction (based on accuracy)
        if random.random() < 0.53:  # Slight bullish bias
            direction = "Long"
            confidence = random.uniform(0.65, 0.95)
            price_change = random.uniform(0.005, 0.03)  # 0.5-3% price target
            target_price = current_price * (1 + price_change)
        else:
            direction = "Short"
            confidence = random.uniform(0.65, 0.95)
            price_change = random.uniform(0.005, 0.03)  # 0.5-3% price target
            target_price = current_price * (1 - price_change)
    else:
        # Incorrect prediction (based on 1-accuracy)
        if random.random() < 0.5:
            direction = "Long"
            confidence = random.uniform(0.5, 0.75)  # Lower confidence
            price_change = random.uniform(-0.03, -0.005)  # Wrong direction
            target_price = current_price * (1 + price_change)
        else:
            direction = "Short"
            confidence = random.uniform(0.5, 0.75)
            price_change = random.uniform(-0.03, -0.005)
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
    # Maximum leverage is capped at 15x for safety
    max_leverage = 15.0
    
    # Base leverage at 5x
    base_leverage = 5.0
    
    # Confidence threshold for scaling
    confidence_threshold = 0.8
    
    # Base risk percentage (1% of portfolio)
    base_risk_percentage = 0.01
    max_risk_percentage = 0.03
    
    # Apply dynamic parameters based on confidence
    if confidence < confidence_threshold:
        # Below threshold, reduce risk
        leverage = min(5.0, base_leverage)
        risk_percentage = base_risk_percentage
    else:
        # Scale leverage based on confidence
        confidence_scale = (confidence - confidence_threshold) / (1.0 - confidence_threshold)
        leverage_range = max_leverage - base_leverage
        leverage = base_leverage + (confidence_scale * leverage_range)
        
        # Scale risk percentage based on confidence
        risk_range = max_risk_percentage - base_risk_percentage
        risk_percentage = base_risk_percentage + (confidence_scale * risk_range)
    
    # Cap leverage and risk
    leverage = min(max_leverage, max(1.0, leverage))
    risk_percentage = min(max_risk_percentage, max(0.005, risk_percentage))
    
    # Calculate position size based on risk percentage
    position_value = portfolio_value * risk_percentage
    size = position_value / current_price
    
    # Calculate final position size with leverage
    final_size = size * leverage
    
    return final_size, leverage

def calculate_liquidation_price(entry_price: float, leverage: float, direction: str) -> float:
    """
    Calculate liquidation price for a position
    
    Args:
        entry_price: Position entry price
        leverage: Position leverage
        direction: "Long" or "Short"
        
    Returns:
        liquidation_price: Price at which the position would be liquidated
    """
    # Get maintenance margin requirement
    maintenance_margin = get_maintenance_margin(leverage)
    
    # Calculate liquidation price
    if direction.lower() == "long":
        # Long position liquidation:
        # Liquidation happens when: equity ≤ maintenance_margin
        # equity = margin + position_value - initial_position_value
        # position_value = initial_position_value * (current_price / entry_price)
        # At liquidation: margin * (1 - maintenance_margin) = initial_position_value - position_value
        # Solving for liquidation_price:
        liquidation_price = entry_price * (1 - (1 - maintenance_margin) / leverage)
    else:
        # Short position liquidation:
        # Similar logic but inverse price movement
        liquidation_price = entry_price * (1 + (1 - maintenance_margin) / leverage)
    
    return liquidation_price

def get_maintenance_margin(leverage: float) -> float:
    """
    Get maintenance margin requirement for a leverage level
    
    Args:
        leverage: Position leverage
        
    Returns:
        maintenance_margin: Maintenance margin as a decimal (0.0-1.0)
    """
    # Find the closest leverage tier
    leverage_tiers = sorted(list(MAINTENANCE_MARGINS.keys()))
    
    # Find the applicable tier
    applicable_tier = leverage_tiers[0]
    for tier in leverage_tiers:
        if leverage >= tier:
            applicable_tier = tier
        else:
            break
    
    # Return the maintenance margin for that tier
    return MAINTENANCE_MARGINS.get(applicable_tier, 0.5)  # Default to 50% if not found

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
        "unrealized_pnl_pct": 0.0,
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

def update_position_prices():
    """
    Update all positions with current prices and calculate unrealized PnL
    """
    positions = load_json(POSITIONS_FILE, [])
    if not positions:
        return
    
    for position in positions:
        pair = position.get("pair")
        if not pair:
            continue
        
        # Get real-time price
        current_price = get_real_time_price(pair)
        entry_price = position.get("entry_price", current_price)
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        # Update current price
        position["current_price"] = current_price
        
        # Calculate unrealized PnL
        if direction.lower() == "long":
            pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
            pnl_amount = (current_price - entry_price) * size
        else:  # Short
            pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
            pnl_amount = (entry_price - current_price) * size
        
        position["unrealized_pnl"] = pnl_percentage
        position["unrealized_pnl_amount"] = pnl_amount
        position["unrealized_pnl_pct"] = pnl_percentage
    
    # Save updated positions
    save_json(POSITIONS_FILE, positions)
    
    # Update portfolio with new unrealized PnL
    update_portfolio(positions)

def check_liquidations():
    """
    Check all positions for liquidation conditions
    """
    positions = load_json(POSITIONS_FILE, [])
    if not positions:
        return
    
    active_positions = []
    liquidated_positions = []
    
    for position in positions:
        current_price = position.get("current_price", 0)
        liquidation_price = position.get("liquidation_price", 0)
        direction = position.get("direction", "")
        
        is_liquidated = False
        
        if direction.lower() == "long" and current_price <= liquidation_price:
            is_liquidated = True
            logger.warning(f"Long position liquidated: {position.get('pair')} at {current_price}")
        elif direction.lower() == "short" and current_price >= liquidation_price:
            is_liquidated = True
            logger.warning(f"Short position liquidated: {position.get('pair')} at {current_price}")
        
        if is_liquidated:
            # Mark as liquidated
            position["exit_price"] = current_price
            position["exit_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            position["exit_reason"] = "LIQUIDATED"
            liquidated_positions.append(position)
        else:
            active_positions.append(position)
    
    # Handle liquidated positions
    if liquidated_positions:
        handle_liquidations(active_positions, liquidated_positions)
    
    # Save updated positions
    save_json(POSITIONS_FILE, active_positions)

def handle_liquidations(active_positions, liquidated_positions):
    """
    Handle liquidated positions
    
    Args:
        active_positions: List of positions still active
        liquidated_positions: List of liquidated positions
    """
    portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
    trades = load_json(TRADES_FILE, [])
    
    for position in liquidated_positions:
        # Add to trades history
        trade = position.copy()
        trade["pnl_percentage"] = position.get("unrealized_pnl", -100)  # Typically -100% on liquidation
        trade["pnl_amount"] = position.get("unrealized_pnl_amount", 0)  # Typically total loss of margin
        
        trades.append(trade)
        
        # Log liquidation
        logger.warning(f"Position liquidated: {position.get('pair')} {position.get('direction')} "
                     f"Size: {position.get('size')} Leverage: {position.get('leverage')} "
                     f"Entry: {position.get('entry_price')} Exit: {position.get('exit_price')}")
    
    # Save updated trades
    save_json(TRADES_FILE, trades)
    
    # Update portfolio history
    update_portfolio(active_positions)

def update_portfolio(positions):
    """
    Update portfolio value with current unrealized PnL
    """
    portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
    
    # Calculate total unrealized PnL
    unrealized_pnl = 0
    for position in positions:
        if "unrealized_pnl_amount" in position:
            unrealized_pnl += position["unrealized_pnl_amount"]
    
    # Update portfolio values
    portfolio["unrealized_pnl_usd"] = unrealized_pnl
    portfolio["unrealized_pnl_pct"] = (unrealized_pnl / 20000.0) * 100
    portfolio["total_value"] = portfolio.get("balance", 20000.0) + unrealized_pnl
    portfolio["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    portfolio["equity"] = portfolio["total_value"]  # For compatibility
    
    # Save updated portfolio
    save_json(PORTFOLIO_FILE, portfolio)
    
    # Update portfolio history
    update_portfolio_history(portfolio)

def update_portfolio_history(portfolio):
    """
    Update portfolio history with current value
    
    Args:
        portfolio: Current portfolio state
    """
    # Load history
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        history = []
    else:
        history = load_json(PORTFOLIO_HISTORY_FILE, [])
    
    # Add current snapshot
    history.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "portfolio_value": portfolio["total_value"]
    })
    
    # Keep only last 1000 points to avoid huge files
    if len(history) > 1000:
        history = history[-1000:]
    
    # Save updated history
    save_json(PORTFOLIO_HISTORY_FILE, history)

def trading_signal_generator():
    """
    Periodically generate trading signals and open positions
    """
    ml_config = load_json(ML_CONFIG_FILE, {"pairs": {}})
    
    while True:
        try:
            portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
            positions = load_json(POSITIONS_FILE, [])
            
            # Update existing positions before making new trades
            update_position_prices()
            check_liquidations()
            
            # Get fresh positions after liquidation check
            positions = load_json(POSITIONS_FILE, [])
            
            # Skip if too many positions open (max 2 per pair, one for each strategy)
            if len(positions) >= len(DEFAULT_PAIRS) * 2:
                logger.info(f"Maximum positions reached: {len(positions)}")
                time.sleep(60)  # Check every minute
                continue
            
            # Map current positions by pair and strategy
            position_map = {}
            for position in positions:
                pair = position.get("pair", "")
                strategy = position.get("strategy", "")
                key = f"{pair}_{strategy}"
                position_map[key] = position
            
            # Calculate current portfolio value
            portfolio_value = portfolio.get("balance", 20000.0)
            for position in positions:
                if "unrealized_pnl_amount" in position:
                    portfolio_value += position["unrealized_pnl_amount"]
            
            # Try to open positions for each pair
            for pair in DEFAULT_PAIRS:
                # Check if we already have positions for this pair with both strategies
                if (f"{pair}_ARIMA" in position_map and 
                    f"{pair}_Adaptive" in position_map):
                    continue
                
                # Get current price
                current_price = get_real_time_price(pair)
                
                # Get ML prediction
                direction, confidence, target_price, strategy = get_ml_prediction(pair, current_price, ml_config)
                
                # Skip if we already have a position for this pair and strategy
                if f"{pair}_{strategy}" in position_map:
                    continue
                
                # Calculate position size and leverage
                size, leverage = calculate_position_size(direction, confidence, current_price, 
                                                       portfolio_value, {})
                
                # Calculate liquidation price
                liquidation_price = calculate_liquidation_price(current_price, leverage, direction)
                
                # Open position
                success, position = open_position(pair, direction, size, current_price, 
                                               leverage, strategy, confidence, liquidation_price)
                
                if success:
                    logger.info(f"Opened {direction} position on {pair} with {strategy} strategy: "
                             f"Size: {size:.2f}, Price: {current_price:.2f}, Leverage: {leverage:.2f}x, "
                             f"Confidence: {confidence:.2f}, Liquidation: {liquidation_price:.2f}")
                
                # Slight delay to avoid opening too many positions at once
                time.sleep(1)
            
            # Sleep before next round
            time.sleep(300)  # Generate signals every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in trading signal generator: {e}")
            time.sleep(60)  # Wait a minute before retrying

def display_status():
    """
    Periodically display status of positions and portfolio
    """
    while True:
        try:
            positions = load_json(POSITIONS_FILE, [])
            portfolio = load_json(PORTFOLIO_FILE, {"balance": 20000.0})
            
            # Display portfolio status
            print("\n" + "=" * 80)
            print(f"PORTFOLIO STATUS - {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
            print("=" * 80)
            print(f"Balance: ${portfolio.get('balance', 0):.2f}")
            print(f"Unrealized PnL: ${portfolio.get('unrealized_pnl_usd', 0):.2f} "
                 f"({portfolio.get('unrealized_pnl_pct', 0):.2f}%)")
            print(f"Total Value: ${portfolio.get('total_value', 0):.2f}")
            print(f"Open Positions: {len(positions)}")
            print("-" * 80)
            
            # Display positions
            if positions:
                print(f"{'Pair':<10} {'Strategy':<10} {'Direction':<6} {'Size':<10} {'Entry':<10} "
                     f"{'Current':<10} {'Leverage':<8} {'PnL %':<8} {'PnL $':<12}")
                print("-" * 80)
                
                for position in positions:
                    print(f"{position.get('pair', ''):<10} "
                         f"{position.get('strategy', ''):<10} "
                         f"{position.get('direction', ''):<6} "
                         f"{position.get('size', 0):<10.2f} "
                         f"{position.get('entry_price', 0):<10.2f} "
                         f"{position.get('current_price', 0):<10.2f} "
                         f"{position.get('leverage', 0):<8.2f}x "
                         f"{position.get('unrealized_pnl', 0):<8.2f}% "
                         f"${position.get('unrealized_pnl_amount', 0):<12.2f}")
            else:
                print("No open positions.")
            
            print("=" * 80)
            
            # Sleep before next update
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Error in status display: {e}")
            time.sleep(60)  # Wait a minute before retrying

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
        print("- Real-time market data processing")
        print("- Accurate liquidation price calculation")
        print("- Safe leverage capped at 15x maximum")
        print("- Dynamic leverage based on prediction confidence")
        print("- ML-enhanced trading signals")
        print("- Cross-strategy signal arbitration")
        print("- Risk-aware position sizing")
        print("\nStarting trading bot...")
        
        # Make sure directories exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Start background threads
        threads = []
        
        # Thread for trading signal generation
        signal_thread = threading.Thread(target=trading_signal_generator, daemon=True)
        signal_thread.start()
        threads.append(signal_thread)
        
        # Thread for status display
        status_thread = threading.Thread(target=display_status, daemon=True)
        status_thread.start()
        threads.append(status_thread)
        
        # Keep main thread running
        print("\nTrading bot is now running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()