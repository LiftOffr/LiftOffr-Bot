#!/usr/bin/env python3
"""
Simulate ML Trades

This script simulates ML-based trades for demonstration:
1. Uses the ML configuration to generate trading signals
2. Opens positions based on ML model predictions
3. Simulates price movements for open positions
4. Calculates P/L for the portfolio

Usage:
    python simulate_ml_trades.py [--pairs PAIR1 PAIR2 ...] [--trades NUM_TRADES]
"""
import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import numpy as np
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories and files
DATA_DIR = "data"
MODEL_WEIGHTS_DIR = "model_weights"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"

# Default pairs
DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "ADA/USD",
    "DOT/USD",
    "LINK/USD",
    "AVAX/USD",
    "MATIC/USD",
    "UNI/USD",
    "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simulate ML-based trades")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS[:3],
                        help="Trading pairs to simulate (default: BTC/USD, ETH/USD, SOL/USD)")
    parser.add_argument("--trades", type=int, default=3,
                        help="Number of trades to simulate per pair (default: 3)")
    return parser.parse_args()

def load_ml_config():
    """Load ML configuration"""
    try:
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                ml_config = json.load(f)
            logger.info(f"Loaded ML configuration with {len(ml_config.get('models', {}))} models")
            return ml_config
        else:
            logger.error(f"ML configuration file not found: {ML_CONFIG_PATH}")
            return {"models": {}, "global_settings": {}}
    except Exception as e:
        logger.error(f"Error loading ML configuration: {e}")
        return {"models": {}, "global_settings": {}}

def load_portfolio():
    """Load portfolio data"""
    try:
        if os.path.exists(PORTFOLIO_PATH):
            with open(PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            logger.info(f"Loaded portfolio with balance: ${portfolio.get('balance', 0):.2f}")
            return portfolio
        else:
            logger.error(f"Portfolio file not found: {PORTFOLIO_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return None

def save_portfolio(portfolio):
    """Save portfolio data"""
    try:
        portfolio["last_updated"] = datetime.datetime.now().isoformat()
        with open(PORTFOLIO_PATH, 'w') as f:
            json.dump(portfolio, f, indent=2)
        logger.info(f"Saved portfolio with balance: ${portfolio.get('balance', 0):.2f}")
    except Exception as e:
        logger.error(f"Error saving portfolio: {e}")

def load_positions():
    """Load position data"""
    try:
        if os.path.exists(POSITIONS_PATH):
            with open(POSITIONS_PATH, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions")
            return positions
        else:
            logger.warning(f"Positions file not found: {POSITIONS_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return {}

def save_positions(positions):
    """Save position data"""
    try:
        with open(POSITIONS_PATH, 'w') as f:
            json.dump(positions, f, indent=2)
        logger.info(f"Saved {len(positions)} positions")
    except Exception as e:
        logger.error(f"Error saving positions: {e}")

def load_trades():
    """Load trade data"""
    try:
        if os.path.exists(TRADES_PATH):
            with open(TRADES_PATH, 'r') as f:
                trades = json.load(f)
            logger.info(f"Loaded {len(trades)} trades")
            return trades
        else:
            logger.warning(f"Trades file not found: {TRADES_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return {}

def save_trades(trades):
    """Save trade data"""
    try:
        with open(TRADES_PATH, 'w') as f:
            json.dump(trades, f, indent=2)
        logger.info(f"Saved {len(trades)} trades")
    except Exception as e:
        logger.error(f"Error saving trades: {e}")

def get_current_price(pair):
    """Get current price for a pair (simulated for demo)"""
    # Extract symbol from pair (e.g., "BTC" from "BTC/USD")
    symbol = pair.split('/')[0].lower()
    
    # For simulation, generate a somewhat realistic price based on the pair
    if pair == "BTC/USD":
        return 57000 + (np.random.randn() * 500)
    elif pair == "ETH/USD":
        return 3500 + (np.random.randn() * 50)
    elif pair == "SOL/USD":
        return 150 + (np.random.randn() * 5)
    elif pair == "ADA/USD":
        return 0.45 + (np.random.randn() * 0.01)
    elif pair == "DOT/USD":
        return 7.5 + (np.random.randn() * 0.2)
    elif pair == "LINK/USD":
        return 16 + (np.random.randn() * 0.5)
    elif pair == "AVAX/USD":
        return 35 + (np.random.randn() * 1)
    elif pair == "MATIC/USD":
        return 0.75 + (np.random.randn() * 0.02)
    elif pair == "UNI/USD":
        return 10 + (np.random.randn() * 0.3)
    elif pair == "ATOM/USD":
        return 9 + (np.random.randn() * 0.25)
    else:
        # Default case for unknown pairs
        return 100 + (np.random.randn() * 2)

def generate_prediction(pair, ml_config):
    """Generate a trading prediction for a pair"""
    # Get model configuration for the pair
    model_config = ml_config.get('models', {}).get(pair, {})
    if not model_config:
        logger.warning(f"No model configuration found for {pair}")
        return None
    
    # Get confidence threshold from config
    confidence_threshold = model_config.get(
        'confidence_threshold',
        ml_config.get('global_settings', {}).get('confidence_threshold', 0.65)
    )
    
    # Get leverage range from config
    min_leverage = model_config.get('min_leverage', 5.0)
    max_leverage = model_config.get('max_leverage', 75.0)
    
    # Randomly generate a prediction (simulating model output)
    # In a real system, this would come from the ML model
    signal_value = np.random.uniform(-1, 1)
    
    # Determine signal type and confidence
    if signal_value > 0:
        signal = "buy"
        confidence = min(abs(signal_value), 1.0)
    elif signal_value < 0:
        signal = "sell"
        confidence = min(abs(signal_value), 1.0)
    else:
        signal = "neutral"
        confidence = 0.0
    
    # Determine if signal is strong enough
    execute_signal = confidence >= confidence_threshold
    
    # Calculate dynamic leverage based on confidence
    if confidence > 0:
        # Scale leverage based on confidence
        leverage = min_leverage + ((max_leverage - min_leverage) * confidence)
    else:
        leverage = min_leverage
    
    # Cap leverage at max
    leverage = min(leverage, max_leverage)
    
    # Format the prediction result
    prediction = {
        'pair': pair,
        'signal': signal,
        'confidence': confidence,
        'execute_signal': execute_signal,
        'leverage': leverage,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    return prediction

def calculate_position_size(pair, leverage, portfolio, ml_config):
    """Calculate position size based on account balance and risk"""
    # Get balance
    balance = float(portfolio.get('balance', 0))
    
    # Get risk percentage from config or use default
    model_config = ml_config.get('models', {}).get(pair, {})
    risk_percentage = model_config.get(
        'risk_percentage', 
        ml_config.get('global_settings', {}).get('risk_percentage', 0.2)
    )
    
    # Calculate amount to risk (balance * risk_percentage)
    risk_amount = balance * risk_percentage
    
    # Get current price
    current_price = get_current_price(pair)
    
    # Calculate position size (amount / price)
    # For leveraged trading, we actually use less margin but control more
    position_size = risk_amount / current_price * leverage
    
    return position_size, current_price, risk_amount

def open_position(pair, signal, confidence, leverage, portfolio, positions, ml_config):
    """Open a new position"""
    try:
        # Calculate position size and get current price
        size, entry_price, risk_amount = calculate_position_size(
            pair, leverage, portfolio, ml_config
        )
        
        # Determine if long or short
        long = signal == "buy"
        
        # Calculate liquidation price (simple approximation)
        # For longs: liquidation when price falls by (100/leverage)%
        # For shorts: liquidation when price rises by (100/leverage)%
        liquidation_threshold = 1.0 / leverage
        
        if long:
            liquidation_price = entry_price * (1.0 - liquidation_threshold)
        else:
            liquidation_price = entry_price * (1.0 + liquidation_threshold)
        
        # Generate position ID
        position_id = f"{pair.replace('/', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate margin required (controlled value / leverage)
        margin = (size * entry_price) / leverage
        
        # Create position object
        position = {
            'symbol': pair,
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': size,
            'margin': margin,
            'long': long,
            'leverage': leverage,
            'liquidation_price': liquidation_price,
            'entry_time': datetime.datetime.now().isoformat(),
            'last_updated': datetime.datetime.now().isoformat(),
            'unrealized_pnl': 0.0,
            'unrealized_pnl_percentage': 0.0
        }
        
        # Update portfolio balance
        portfolio['balance'] -= margin
        
        # Add position to positions dictionary
        positions[position_id] = position
        
        direction = "LONG" if long else "SHORT"
        logger.info(f"Opened {direction} position {position_id} for {pair} at ${entry_price:.2f}")
        logger.info(f"Size: {size:.6f}, Leverage: {leverage:.1f}x, Margin: ${margin:.2f}")
        logger.info(f"Liquidation price: ${liquidation_price:.2f}")
        
        return position_id, position
    except Exception as e:
        logger.error(f"Error opening position: {e}")
        return None, None

def close_position(position_id, positions, trades, portfolio):
    """Close a position and update portfolio"""
    try:
        # Get position
        position = positions.get(position_id)
        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None
        
        # Get current price
        current_price = get_current_price(position['symbol'])
        
        # Update position with current price
        position['current_price'] = current_price
        
        # Calculate P/L
        if position['long']:
            # For long positions, profit when current_price > entry_price
            pnl_percentage = (current_price - position['entry_price']) / position['entry_price']
        else:
            # For short positions, profit when current_price < entry_price
            pnl_percentage = (position['entry_price'] - current_price) / position['entry_price']
        
        # Apply leverage to P/L
        pnl_percentage = pnl_percentage * position['leverage']
        
        # Calculate absolute P/L
        pnl = position['margin'] * pnl_percentage
        
        # Check for liquidation
        liquidated = False
        if position['long'] and current_price <= position['liquidation_price']:
            liquidated = True
            pnl = -position['margin']  # Lose full margin on liquidation
            pnl_percentage = -1.0
        elif not position['long'] and current_price >= position['liquidation_price']:
            liquidated = True
            pnl = -position['margin']  # Lose full margin on liquidation
            pnl_percentage = -1.0
        
        # Update portfolio balance
        portfolio['balance'] += position['margin'] + pnl
        
        # Create trade record
        direction = "LONG" if position['long'] else "SHORT"
        status = "LIQUIDATED" if liquidated else "CLOSED"
        
        trade = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'direction': direction,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'leverage': position['leverage'],
            'margin': position['margin'],
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'status': status,
            'entry_time': position['entry_time'],
            'exit_time': datetime.datetime.now().isoformat()
        }
        
        # Add trade to trades dictionary
        trade_id = f"trade_{position_id}"
        trades[trade_id] = trade
        
        # Remove position from positions dictionary
        del positions[position_id]
        
        logger.info(f"{status} {direction} position {position_id} for {position['symbol']}")
        logger.info(f"Entry: ${position['entry_price']:.2f}, Exit: ${current_price:.2f}")
        logger.info(f"P/L: ${pnl:.2f} ({pnl_percentage:.2%})")
        logger.info(f"New portfolio balance: ${portfolio['balance']:.2f}")
        
        return trade_id, trade
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return None, None

def update_positions(positions, portfolio):
    """Update all open positions with current prices and calculate P/L"""
    if not positions:
        return
    
    total_unrealized_pnl = 0.0
    
    logger.info(f"Updating {len(positions)} positions")
    
    for position_id, position in list(positions.items()):
        try:
            # Get current price
            current_price = get_current_price(position['symbol'])
            
            # Update position with current price
            position['current_price'] = current_price
            position['last_updated'] = datetime.datetime.now().isoformat()
            
            # Calculate unrealized P/L
            if position['long']:
                # For long positions, profit when current_price > entry_price
                pnl_percentage = (current_price - position['entry_price']) / position['entry_price']
            else:
                # For short positions, profit when current_price < entry_price
                pnl_percentage = (position['entry_price'] - current_price) / position['entry_price']
            
            # Apply leverage to P/L
            pnl_percentage = pnl_percentage * position['leverage']
            
            # Calculate absolute P/L
            pnl = position['margin'] * pnl_percentage
            
            # Update position with P/L
            position['unrealized_pnl'] = pnl
            position['unrealized_pnl_percentage'] = pnl_percentage
            
            direction = "LONG" if position['long'] else "SHORT"
            logger.info(f"Updated {direction} position {position_id} for {position['symbol']}: "
                        f"${position['entry_price']:.2f} -> ${current_price:.2f}, "
                        f"P/L: ${pnl:.2f} ({pnl_percentage:.2%})")
            
            # Check for liquidation
            if position['long'] and current_price <= position['liquidation_price']:
                logger.warning(f"Position {position_id} would be liquidated at current price ${current_price:.2f}")
                # In a real system, we would close the position here
            elif not position['long'] and current_price >= position['liquidation_price']:
                logger.warning(f"Position {position_id} would be liquidated at current price ${current_price:.2f}")
                # In a real system, we would close the position here
            
            total_unrealized_pnl += pnl
        except Exception as e:
            logger.error(f"Error updating position {position_id}: {e}")
    
    logger.info(f"Total unrealized P/L: ${total_unrealized_pnl:.2f}")
    
    return total_unrealized_pnl

def simulate_trades(pairs, num_trades_per_pair):
    """Simulate trades for the specified pairs"""
    try:
        # Load configurations and data
        ml_config = load_ml_config()
        portfolio = load_portfolio()
        positions = load_positions()
        trades = load_trades()
        
        if not portfolio:
            logger.error("Failed to load portfolio")
            return
        
        initial_balance = portfolio.get('balance', 0)
        logger.info(f"Starting simulation with balance: ${initial_balance:.2f}")
        
        # Track simulation metrics
        num_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        
        # Simulate trades for each pair
        for pair in pairs:
            logger.info(f"\n=== Simulating trades for {pair} ===")
            
            for i in range(num_trades_per_pair):
                # Generate prediction
                prediction = generate_prediction(pair, ml_config)
                if not prediction or not prediction['execute_signal']:
                    logger.info(f"Skip trade for {pair} - no actionable prediction")
                    continue
                
                # Open position based on prediction
                position_id, position = open_position(
                    pair,
                    prediction['signal'],
                    prediction['confidence'],
                    prediction['leverage'],
                    portfolio,
                    positions,
                    ml_config
                )
                
                if not position_id:
                    logger.warning(f"Failed to open position for {pair}")
                    continue
                
                # Save positions and portfolio
                save_positions(positions)
                save_portfolio(portfolio)
                
                # Simulate price movement (wait a bit and update)
                time.sleep(1)
                update_positions(positions, portfolio)
                
                # Simulate more price movement (wait a bit more and update again)
                time.sleep(1)
                update_positions(positions, portfolio)
                
                # Close the position
                trade_id, trade = close_position(position_id, positions, trades, portfolio)
                
                if not trade_id:
                    logger.warning(f"Failed to close position {position_id}")
                    continue
                
                # Update simulation metrics
                num_trades += 1
                if trade['pnl'] > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
                total_pnl += trade['pnl']
                
                # Save positions, trades, and portfolio
                save_positions(positions)
                save_trades(trades)
                save_portfolio(portfolio)
                
                # Simulate a short break between trades
                time.sleep(1)
        
        # Calculate simulation results
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        final_balance = portfolio.get('balance', 0)
        roi = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
        
        logger.info(f"\n=== Simulation Results ===")
        logger.info(f"Completed trades: {num_trades}")
        logger.info(f"Winning trades: {winning_trades} ({win_rate:.2%})")
        logger.info(f"Losing trades: {losing_trades} ({1-win_rate:.2%})")
        logger.info(f"Total P/L: ${total_pnl:.2f}")
        logger.info(f"Initial balance: ${initial_balance:.2f}")
        logger.info(f"Final balance: ${final_balance:.2f}")
        logger.info(f"ROI: {roi:.2%}")
        
        return True
    except Exception as e:
        logger.error(f"Error in trade simulation: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    pairs = args.pairs
    num_trades = args.trades
    
    logger.info(f"Starting ML trade simulation")
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Trades per pair: {num_trades}")
    
    # Simulate trades
    result = simulate_trades(pairs, num_trades)
    
    if result:
        logger.info(f"ML trade simulation completed successfully")
    else:
        logger.error(f"ML trade simulation failed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in ML trade simulation: {e}")
        raise