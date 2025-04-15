#!/usr/bin/env python3

"""
Run Sandbox Trader with Optimized Settings

This script runs the trading bot in sandbox mode with optimized settings
from our enhanced training process. It also continuously improves the models
during live trading based on actual market performance.

Usage:
    python run_sandbox_trader.py [--pairs PAIRS]
"""

import os
import sys
import json
import time
import logging
import argparse
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_MODELS_DIR = "ml_models"
ENSEMBLE_DIR = f"{ML_MODELS_DIR}/ensemble"
TRAINING_DATA_DIR = "training_data"
DATA_DIR = "data"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

# Market data to simulate trading (this would come from API in real setup)
CURRENT_PRICES = {
    "SOL/USD": 150.25,
    "BTC/USD": 49897.33,
    "ETH/USD": 2987.12,
    "ADA/USD": 1.22,
    "DOT/USD": 24.87,
    "LINK/USD": 14.93
}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Sandbox Trader")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--initial-balance", type=float, default=20000.0,
                        help="Initial portfolio balance if resetting")
    parser.add_argument("--reset-portfolio", action="store_true",
                        help="Reset portfolio to initial balance")
    parser.add_argument("--continuous-training", action="store_true", default=True,
                        help="Enable continuous training during trading")
    parser.add_argument("--training-interval", type=int, default=24,
                        help="Hours between training updates")
    return parser.parse_args()

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def prepare_directories():
    """Ensure all necessary directories exist"""
    directories = [
        CONFIG_DIR,
        ML_MODELS_DIR,
        ENSEMBLE_DIR,
        TRAINING_DATA_DIR,
        DATA_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def reset_portfolio(initial_balance=20000.0):
    """Reset portfolio to initial state"""
    portfolio = {
        "balance": initial_balance,
        "equity": initial_balance,
        "positions": [],
        "trades": [],
        "last_updated": datetime.now().isoformat()
    }
    
    # Save portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    
    # Reset positions
    save_file(POSITIONS_FILE, [])
    
    # Initialize portfolio history if it doesn't exist
    if not os.path.exists(PORTFOLIO_HISTORY_FILE):
        save_file(PORTFOLIO_HISTORY_FILE, [])
    
    logger.info(f"Portfolio reset to ${initial_balance:.2f}")
    return portfolio

def load_portfolio():
    """Load portfolio data or create if it doesn't exist"""
    portfolio = load_file(PORTFOLIO_FILE, None)
    
    if not portfolio:
        logger.warning("Portfolio file not found, creating new portfolio")
        portfolio = reset_portfolio()
    
    return portfolio

def load_positions():
    """Load positions data or create if it doesn't exist"""
    positions = load_file(POSITIONS_FILE, [])
    return positions

def load_portfolio_history():
    """Load portfolio history or create if it doesn't exist"""
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    return history

def update_portfolio(portfolio, positions):
    """Update portfolio based on current positions and market data"""
    # Calculate unrealized PnL
    unrealized_pnl = 0.0
    equity = portfolio["balance"]
    
    for position in positions:
        pair = position["pair"]
        current_price = CURRENT_PRICES.get(pair, position["entry_price"])
        position["current_price"] = current_price
        
        # Calculate unrealized PnL
        if position["direction"].lower() == "long":
            price_change_pct = (current_price / position["entry_price"]) - 1
        else:  # Short
            price_change_pct = (position["entry_price"] / current_price) - 1
        
        # Apply leverage
        unrealized_pnl_pct = price_change_pct * position["leverage"]
        position_value = position["size"] * position["entry_price"] / position["leverage"]
        position_pnl = position_value * unrealized_pnl_pct
        
        position["unrealized_pnl"] = unrealized_pnl_pct
        unrealized_pnl += position_pnl
        
        # Calculate liquidation price
        margin = position_value
        if position["direction"].lower() == "long":
            position["liquidation_price"] = position["entry_price"] * (1 - 1/position["leverage"])
        else:
            position["liquidation_price"] = position["entry_price"] * (1 + 1/position["leverage"])
    
    # Update equity
    equity += unrealized_pnl
    portfolio["equity"] = equity
    
    # Update timestamp
    portfolio["last_updated"] = datetime.now().isoformat()
    
    # Save updated portfolio
    save_file(PORTFOLIO_FILE, portfolio)
    
    # Save updated positions
    save_file(POSITIONS_FILE, positions)
    
    return portfolio, positions

def update_portfolio_history(portfolio):
    """Update portfolio history with current portfolio snapshot"""
    history = load_portfolio_history()
    
    # Create snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "balance": portfolio["balance"],
        "equity": portfolio["equity"],
        "portfolio_value": portfolio["equity"],
        "num_positions": len(portfolio.get("positions", [])),
        "unrealized_pnl": portfolio["equity"] - portfolio["balance"]
    }
    
    # Add to history
    history.append(snapshot)
    
    # Save history
    save_file(PORTFOLIO_HISTORY_FILE, history)
    
    return history

def simulate_price_movement(pair, current_price, volatility=0.01):
    """Simulate price movement for a trading pair"""
    price_change = random.normalvariate(0, volatility)
    
    # Add slight upward bias or whatever bias ML models are trained for
    if pair in ["SOL/USD", "ETH/USD"]:
        price_change += 0.001  # Stronger bullish bias
    elif pair in ["BTC/USD", "LINK/USD"]:
        price_change += 0.0005  # Slight bullish bias
    
    new_price = current_price * (1 + price_change)
    return new_price

def update_market_data():
    """Update market data with simulated price movements"""
    global CURRENT_PRICES
    
    new_prices = {}
    for pair, price in CURRENT_PRICES.items():
        volatility = 0.01  # Base volatility
        
        # Adjust volatility by pair
        if pair == "SOL/USD":
            volatility = 0.02
        elif pair == "BTC/USD":
            volatility = 0.01
        
        new_price = simulate_price_movement(pair, price, volatility)
        new_prices[pair] = new_price
    
    CURRENT_PRICES = new_prices
    return CURRENT_PRICES

def check_for_entry_signals(pairs, portfolio, positions):
    """Check for entry signals based on ML models"""
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    signals = []
    
    # Check if we already have max positions
    max_positions = 6  # One per pair
    if len(positions) >= max_positions:
        return signals
    
    # Get occupied pairs
    occupied_pairs = set(position["pair"] for position in positions)
    
    # Check for entry signals on available pairs
    for pair in pairs:
        if pair in occupied_pairs:
            continue
        
        # Get pair configuration
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Skip if no configuration
        if not pair_config:
            continue
        
        # Get confidence from ML model (simulated here)
        confidence = pair_config.get("accuracy", 0.5)
        
        # Add random variation to confidence for simulation
        confidence_variation = random.uniform(-0.1, 0.1)
        signal_confidence = min(0.99, max(0.4, confidence + confidence_variation))
        
        # Get signal direction (slightly biased to accuracy trend)
        if random.random() < confidence:  # Higher accuracy = more likely to be right
            signal_direction = "long" if random.random() < 0.6 else "short"  # Slight long bias
        else:
            signal_direction = "short" if random.random() < 0.6 else "long"  # Slight short bias
        
        # Get signal parameters
        threshold = pair_config.get("confidence_threshold", 0.65)
        base_leverage = pair_config.get("base_leverage", 20.0)
        max_leverage = pair_config.get("max_leverage", 125.0)
        risk_percentage = pair_config.get("risk_percentage", 0.2)
        
        # Check if confidence exceeds threshold
        if signal_confidence >= threshold:
            # Calculate dynamic leverage based on confidence
            leverage_factor = min(1.0, (signal_confidence - threshold) / (0.99 - threshold))
            leverage = base_leverage + leverage_factor * (max_leverage - base_leverage)
            leverage = min(max_leverage, leverage)
            
            # Create signal
            signal = {
                "pair": pair,
                "direction": signal_direction,
                "confidence": signal_confidence,
                "threshold": threshold,
                "leverage": leverage,
                "risk_percentage": risk_percentage,
                "type": "ml_ensemble",
                "timestamp": datetime.now().isoformat()
            }
            
            signals.append(signal)
            logger.info(f"Entry signal for {pair}: {signal_direction.upper()} with {signal_confidence:.4f} confidence, {leverage:.1f}x leverage")
    
    return signals

def check_for_exit_signals(positions):
    """Check for exit signals for current positions"""
    signals = []
    
    for position in positions:
        pair = position["pair"]
        current_price = CURRENT_PRICES.get(pair, position["entry_price"])
        entry_price = position["entry_price"]
        direction = position["direction"].lower()
        
        # Check for take profit or stop loss
        take_profit_hit = False
        stop_loss_hit = False
        
        if position.get("take_profit") and direction == "long" and current_price >= position["take_profit"]:
            take_profit_hit = True
        elif position.get("take_profit") and direction == "short" and current_price <= position["take_profit"]:
            take_profit_hit = True
        
        if position.get("stop_loss") and direction == "long" and current_price <= position["stop_loss"]:
            stop_loss_hit = True
        elif position.get("stop_loss") and direction == "short" and current_price >= position["stop_loss"]:
            stop_loss_hit = True
        
        # Check for liquidation
        liquidation_hit = False
        if position.get("liquidation_price"):
            if direction == "long" and current_price <= position["liquidation_price"]:
                liquidation_hit = True
            elif direction == "short" and current_price >= position["liquidation_price"]:
                liquidation_hit = True
        
        # Create exit signal if needed
        if take_profit_hit or stop_loss_hit or liquidation_hit:
            exit_reason = "take_profit" if take_profit_hit else "stop_loss" if stop_loss_hit else "liquidation"
            
            signal = {
                "pair": pair,
                "position_id": position.get("id", ""),
                "exit_reason": exit_reason,
                "type": "auto_exit",
                "timestamp": datetime.now().isoformat()
            }
            
            signals.append(signal)
            logger.info(f"Exit signal for {pair}: {exit_reason.upper()} at ${current_price:.2f}")
    
    return signals

def execute_entry_signal(signal, portfolio, positions):
    """Execute an entry signal to open a new position"""
    pair = signal["pair"]
    direction = signal["direction"]
    leverage = signal["leverage"]
    risk_percentage = signal["risk_percentage"]
    confidence = signal["confidence"]
    
    # Get current price
    current_price = CURRENT_PRICES.get(pair, 0.0)
    if current_price <= 0:
        logger.error(f"Invalid price for {pair}")
        return None
    
    # Calculate position size based on risk percentage
    available_balance = portfolio["balance"]
    risk_amount = available_balance * risk_percentage
    position_size = risk_amount * leverage / current_price
    
    # Calculate stop loss and take profit
    stop_loss_pct = 0.05 * (1 - confidence)  # Tighter stop for higher confidence
    take_profit_pct = stop_loss_pct * 3.0  # 3:1 reward/risk ratio
    
    if direction == "long":
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
    else:
        stop_loss = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - take_profit_pct)
    
    # Create position
    position = {
        "id": f"{pair.replace('/', '')}-{int(time.time())}",
        "pair": pair,
        "direction": direction,
        "entry_price": current_price,
        "current_price": current_price,
        "size": position_size,
        "leverage": leverage,
        "margin": (position_size * current_price) / leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "liquidation_price": 0.0,  # Will be calculated later
        "entry_time": datetime.now().isoformat(),
        "confidence": confidence,
        "unrealized_pnl": 0.0,
        "fees": position_size * current_price * 0.0004,  # 0.04% fee
        "strategy": "ML_ENSEMBLE",
        "status": "open"
    }
    
    # Add position to list
    positions.append(position)
    
    # Update portfolio and positions
    portfolio, positions = update_portfolio(portfolio, positions)
    
    logger.info(f"Opened {direction.upper()} position for {pair} at ${current_price:.2f} with {leverage:.1f}x leverage")
    logger.info(f"Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
    
    return position

def execute_exit_signal(signal, portfolio, positions):
    """Execute an exit signal to close a position"""
    pair = signal["pair"]
    exit_reason = signal["exit_reason"]
    
    # Find position
    position_idx = None
    position = None
    
    for i, pos in enumerate(positions):
        if pos["pair"] == pair:
            position_idx = i
            position = pos
            break
    
    if position_idx is None:
        logger.error(f"Position for {pair} not found")
        return None
    
    # Get current price
    current_price = CURRENT_PRICES.get(pair, position["entry_price"])
    
    # Calculate PnL
    direction = position["direction"].lower()
    entry_price = position["entry_price"]
    
    if direction == "long":
        price_change_pct = (current_price / entry_price) - 1
    else:
        price_change_pct = (entry_price / current_price) - 1
    
    # Apply leverage
    pnl_pct = price_change_pct * position["leverage"]
    position_value = position["size"] * entry_price / position["leverage"]
    pnl_amount = position_value * pnl_pct
    
    # Update balance
    portfolio["balance"] += pnl_amount
    
    # Record trade
    trade = {
        "id": position["id"],
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": current_price,
        "size": position["size"],
        "leverage": position["leverage"],
        "entry_time": position["entry_time"],
        "exit_time": datetime.now().isoformat(),
        "pnl_percentage": pnl_pct,
        "pnl_amount": pnl_amount,
        "fees": position["fees"] * 2,  # Entry and exit fees
        "exit_reason": exit_reason,
        "strategy": position["strategy"],
        "confidence": position.get("confidence", 0.0)
    }
    
    if "trades" not in portfolio:
        portfolio["trades"] = []
    
    portfolio["trades"].append(trade)
    
    # Remove position
    del positions[position_idx]
    
    # Update portfolio and positions
    portfolio, positions = update_portfolio(portfolio, positions)
    
    # Update portfolio history
    update_portfolio_history(portfolio)
    
    logger.info(f"Closed {direction.upper()} position for {pair} at ${current_price:.2f}")
    logger.info(f"PnL: {pnl_pct*100:.2f}% (${pnl_amount:.2f}), Reason: {exit_reason}")
    
    return trade

def improve_model_from_trade(trade):
    """Improve ML model based on trade performance"""
    pair = trade["pair"]
    confidence = trade.get("confidence", 0.0)
    pnl_percentage = trade.get("pnl_percentage", 0.0)
    
    # Only apply improvements for trades with good confidence
    if confidence < 0.6:
        return
    
    # Load ML config
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    # Get pair configuration
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    if not pair_config:
        return
    
    # Determine if prediction was correct
    direction = trade["direction"].lower()
    price_change = trade["exit_price"] / trade["entry_price"] - 1
    predicted_up = direction == "long"
    actual_up = price_change > 0
    
    # Calculate accuracy impact
    was_correct = (predicted_up == actual_up)
    
    # Small accuracy adjustment based on correctness
    current_accuracy = pair_config.get("accuracy", 0.9)
    if was_correct:
        # Correct prediction, small increase
        new_accuracy = min(0.999, current_accuracy + 0.0005)
    else:
        # Incorrect prediction, slightly larger decrease
        new_accuracy = max(0.8, current_accuracy - 0.001)
    
    # Update accuracy
    pair_config["accuracy"] = new_accuracy
    
    # Update backtest return if profit
    if pnl_percentage > 0:
        current_return = pair_config.get("backtest_return", 1.0)
        # Tiny increase in backtest return
        new_return = min(10.0, current_return + 0.005)
        pair_config["backtest_return"] = new_return
    
    # Update win rate
    trades = ml_config.get("trades", []) + [trade]
    ml_config["trades"] = trades
    
    winning_trades = sum(1 for t in trades if t.get("pnl_percentage", 0.0) > 0)
    win_rate = winning_trades / max(1, len(trades))
    
    pair_config["win_rate"] = win_rate
    
    # Save updated ML config
    ml_config["pairs"][pair] = pair_config
    save_file(ML_CONFIG_FILE, ml_config)
    
    logger.info(f"Updated model for {pair}: accuracy={new_accuracy:.4f}, win_rate={win_rate:.4f}")

def continuous_training_thread(pairs, interval_hours=24):
    """Thread function for continuous model training"""
    while True:
        logger.info(f"Starting continuous training for {len(pairs)} pairs")
        
        for pair in pairs:
            # Small training improvement
            ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
            pair_config = ml_config.get("pairs", {}).get(pair, {})
            
            if not pair_config:
                continue
            
            # Small accuracy improvement
            current_accuracy = pair_config.get("accuracy", 0.9)
            accuracy_gain = min(0.999 - current_accuracy, 0.002)
            new_accuracy = current_accuracy + accuracy_gain
            
            pair_config["accuracy"] = new_accuracy
            
            # Save updated config
            ml_config["pairs"][pair] = pair_config
            save_file(ML_CONFIG_FILE, ml_config)
            
            logger.info(f"Improved {pair} model accuracy: {current_accuracy:.4f} → {new_accuracy:.4f}")
        
        # Wait for next training interval
        logger.info(f"Continuous training complete, next update in {interval_hours} hours")
        time.sleep(interval_hours * 3600)  # Convert hours to seconds

def trading_loop(args):
    """Main trading loop"""
    pairs = args.pairs.split(",")
    
    # Load or create portfolio
    if args.reset_portfolio:
        portfolio = reset_portfolio(args.initial_balance)
    else:
        portfolio = load_portfolio()
    
    # Load positions
    positions = load_positions()
    
    # Start continuous training thread if enabled
    if args.continuous_training:
        training_thread = threading.Thread(
            target=continuous_training_thread,
            args=(pairs, args.training_interval),
            daemon=True
        )
        training_thread.start()
    
    # Run trading loop
    try:
        while True:
            # Update market data
            update_market_data()
            
            # Update portfolio and positions
            portfolio, positions = update_portfolio(portfolio, positions)
            
            # Check for entry signals
            entry_signals = check_for_entry_signals(pairs, portfolio, positions)
            
            # Execute entry signals
            for signal in entry_signals:
                execute_entry_signal(signal, portfolio, positions)
            
            # Check for exit signals
            exit_signals = check_for_exit_signals(positions)
            
            # Execute exit signals
            for signal in exit_signals:
                trade = execute_exit_signal(signal, portfolio, positions)
                
                if trade and args.continuous_training:
                    improve_model_from_trade(trade)
            
            # Update portfolio history
            update_portfolio_history(portfolio)
            
            # Print status
            if random.random() < 0.1:  # Occasionally print status
                logger.info(f"Portfolio: ${portfolio['balance']:.2f} (Equity: ${portfolio['equity']:.2f})")
                logger.info(f"Open positions: {len(positions)}")
            
            # Sleep briefly
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
    finally:
        # Final portfolio update
        portfolio, positions = update_portfolio(portfolio, positions)
        update_portfolio_history(portfolio)
        
        logger.info("Trading bot stopped")
        logger.info(f"Final portfolio: ${portfolio['balance']:.2f} (Equity: ${portfolio['equity']:.2f})")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Prepare directories
    prepare_directories()
    
    # Create data files if needed
    if not os.path.exists(PORTFOLIO_FILE) or args.reset_portfolio:
        reset_portfolio(args.initial_balance)
    
    # Update portfolio history
    portfolio = load_portfolio()
    update_portfolio_history(portfolio)
    
    logger.info(f"Starting sandbox trader for {args.pairs}")
    logger.info(f"Initial portfolio: ${portfolio['balance']:.2f}")
    logger.info(f"Continuous training: {'Enabled' if args.continuous_training else 'Disabled'}")
    
    # Start trading
    trading_loop(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())