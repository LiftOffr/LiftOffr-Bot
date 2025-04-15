#!/usr/bin/env python3
"""
Risk Management Module for Trading Bot

This module provides risk management functions for the trading bot:
1. Liquidation calculation and handling
2. Risk-based leverage calculation
3. Position sizing based on risk parameters
"""
import math
import logging
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_LEVERAGE = 125.0
MIN_LEVERAGE = 5.0
DEFAULT_RISK_PER_TRADE = 0.02  # 2% of portfolio
MAX_RISK_PER_TRADE = 0.05  # 5% of portfolio
STOP_LOSS_PCT = 4.0  # Fixed at 4%
MAX_POSITION_PCT = 0.20  # Maximum 20% of portfolio per position

def calculate_liquidation_price(entry_price, direction, leverage, safety_margin=0.05):
    """
    Calculate liquidation price for a position.
    
    Args:
        entry_price (float): Entry price of the position
        direction (str): Direction of the position ('long' or 'short')
        leverage (float): Leverage used for the position
        safety_margin (float): Additional safety margin (default 5%)
        
    Returns:
        float: Liquidation price
    """
    # Calculate liquidation threshold (percentage move that would cause liquidation)
    # For example, with 10x leverage, a 10% move against position causes liquidation
    liquidation_threshold = (1 / leverage) * (1 - safety_margin)
    
    if direction.lower() == 'long':
        # For long positions, liquidation price is below entry price
        liquidation_price = entry_price * (1 - liquidation_threshold)
    else:  # short
        # For short positions, liquidation price is above entry price
        liquidation_price = entry_price * (1 + liquidation_threshold)
    
    return liquidation_price

def check_liquidation(position, current_price):
    """
    Check if a position should be liquidated based on current price.
    
    Args:
        position (dict): Position information
        current_price (float): Current price
        
    Returns:
        tuple: (is_liquidated, liquidation_price, liquidation_pnl)
    """
    entry_price = position.get('entry_price', 0)
    direction = position.get('direction', 'long').lower()
    leverage = position.get('leverage', 1.0)
    position_size = position.get('position_size', 0)
    
    # Calculate liquidation price
    liquidation_price = calculate_liquidation_price(entry_price, direction, leverage)
    
    # Check if price has hit liquidation level
    is_liquidated = False
    if direction == 'long' and current_price <= liquidation_price:
        is_liquidated = True
    elif direction == 'short' and current_price >= liquidation_price:
        is_liquidated = True
    
    # Calculate PnL at liquidation
    if is_liquidated:
        # At liquidation, trader loses almost entire position
        liquidation_pnl = -position_size * 0.95  # 95% loss (some small collateral might remain)
        logger.warning(f"LIQUIDATION: {position.get('pair')} {direction} position liquidated at ${current_price}")
        logger.warning(f"Entry: ${entry_price}, Liquidation price: ${liquidation_price}, PnL: ${liquidation_pnl}")
    else:
        liquidation_pnl = 0
    
    return is_liquidated, liquidation_price, liquidation_pnl

def calculate_risk_adjusted_leverage(confidence, volatility=None, market_regime="neutral"):
    """
    Calculate risk-adjusted leverage based on confidence score and market conditions.
    
    Args:
        confidence (float): Model confidence score (0.0-1.0)
        volatility (float): Market volatility score (0.0-1.0)
        market_regime (str): Market regime ('trending', 'ranging', 'volatile', 'neutral')
        
    Returns:
        float: Recommended leverage
    """
    # Base leverage calculation from confidence
    if confidence >= 0.90:
        # High confidence trades get higher leverage
        base_leverage = MIN_LEVERAGE + (confidence * (MAX_LEVERAGE - MIN_LEVERAGE))
    elif confidence >= 0.75:
        # Medium-high confidence
        base_leverage = MIN_LEVERAGE + (confidence * 0.8 * (MAX_LEVERAGE - MIN_LEVERAGE))
    elif confidence >= 0.65:
        # Medium confidence
        base_leverage = MIN_LEVERAGE + (confidence * 0.6 * (MAX_LEVERAGE - MIN_LEVERAGE))
    else:
        # Low confidence, stay close to minimum
        base_leverage = MIN_LEVERAGE + (confidence * 0.3 * (MAX_LEVERAGE - MIN_LEVERAGE))
    
    # Adjust for volatility if provided
    if volatility is not None:
        # Higher volatility should reduce leverage
        volatility_factor = 1 - (volatility * 0.5)  # 0.5-1.0 scaling factor
        base_leverage *= volatility_factor
    
    # Adjust for market regime
    regime_factor = 1.0
    if market_regime == "volatile":
        regime_factor = 0.7  # Reduce leverage in volatile markets
    elif market_regime == "ranging":
        regime_factor = 0.85  # Slightly reduce leverage in ranging markets
    elif market_regime == "trending":
        regime_factor = 1.1  # Slight increase in trending markets
    
    # Apply regime factor
    adjusted_leverage = base_leverage * regime_factor
    
    # Ensure leverage is within allowed range
    final_leverage = max(MIN_LEVERAGE, min(adjusted_leverage, MAX_LEVERAGE))
    
    # Round to 1 decimal place
    final_leverage = round(final_leverage, 1)
    
    return final_leverage

def calculate_position_size(portfolio_balance, leverage, risk_percentage=None, stop_loss_pct=STOP_LOSS_PCT):
    """
    Calculate position size based on risk percentage and leverage.
    
    Args:
        portfolio_balance (float): Total portfolio balance
        leverage (float): Leverage to use
        risk_percentage (float): Percentage of portfolio to risk (decimal form)
        stop_loss_pct (float): Stop loss percentage
        
    Returns:
        float: Position size in quote currency
    """
    if risk_percentage is None:
        risk_percentage = DEFAULT_RISK_PER_TRADE
    
    # Limit risk percentage to maximum
    risk_percentage = min(risk_percentage, MAX_RISK_PER_TRADE)
    
    # Calculate maximum amount we're willing to lose on this trade
    max_loss_amount = portfolio_balance * risk_percentage
    
    # Calculate position size based on stop loss and leverage
    # Formula: Position Size = Max Loss / (Stop Loss Percentage / Leverage)
    position_size = max_loss_amount / (stop_loss_pct / 100 / leverage)
    
    # Limit position size to maximum percentage of portfolio
    max_position_size = portfolio_balance * MAX_POSITION_PCT
    position_size = min(position_size, max_position_size)
    
    return position_size

def analyze_market_regime(prices=None, coin_pair=None):
    """
    Analyze market regime based on price data.
    
    Args:
        prices (list): List of historical prices (optional)
        coin_pair (str): Cryptocurrency pair (optional)
        
    Returns:
        str: Market regime ('trending', 'ranging', 'volatile', 'neutral')
    """
    # If we don't have price data, return a random regime for demonstration
    # In a real implementation, this would use actual price data for analysis
    regimes = ["trending", "ranging", "volatile", "neutral"]
    
    if coin_pair:
        # Seed random based on pair name for consistency across calls
        random.seed(hash(coin_pair) % 1000)
        
    return random.choice(regimes)

def estimate_volatility(prices=None, coin_pair=None):
    """
    Estimate market volatility based on price data.
    
    Args:
        prices (list): List of historical prices (optional)
        coin_pair (str): Cryptocurrency pair (optional)
        
    Returns:
        float: Volatility score (0.0-1.0)
    """
    # If we don't have price data, return a random volatility for demonstration
    # In a real implementation, this would use actual price data for analysis
    if coin_pair:
        # Seed random based on pair name for consistency across calls
        random.seed(hash(coin_pair) % 1000)
    
    return random.uniform(0.1, 0.7)

def prepare_trade_entry(portfolio_balance, pair, confidence, entry_price):
    """
    Prepare trade entry with risk management.
    
    Args:
        portfolio_balance (float): Total portfolio balance
        pair (str): Trading pair
        confidence (float): Model confidence score (0.0-1.0)
        entry_price (float): Entry price
        
    Returns:
        dict: Trade entry parameters
    """
    # Analyze market conditions
    market_regime = analyze_market_regime(coin_pair=pair)
    volatility = estimate_volatility(coin_pair=pair)
    
    # Calculate risk-adjusted leverage
    leverage = calculate_risk_adjusted_leverage(confidence, volatility, market_regime)
    
    # Determine risk percentage based on confidence
    if confidence >= 0.85:
        risk_percentage = 0.03  # Higher risk for high confidence
    elif confidence >= 0.75:
        risk_percentage = 0.025  # Medium-high risk
    else:
        risk_percentage = 0.02  # Standard risk
    
    # Calculate position size
    position_size = calculate_position_size(
        portfolio_balance=portfolio_balance,
        leverage=leverage,
        risk_percentage=risk_percentage
    )
    
    # Calculate take profit based on stop loss (reward/risk ratio)
    reward_risk_ratio = 2.5 + (confidence * 2)  # 2.5-4.5x
    take_profit_pct = STOP_LOSS_PCT * reward_risk_ratio
    
    # Determine direction (for demo, alternate based on pair hash)
    random.seed(hash(pair) % 1000)
    direction = "long" if random.random() > 0.5 else "short"
    
    # Calculate liquidation price
    liquidation_price = calculate_liquidation_price(entry_price, direction, leverage)
    
    return {
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "position_size": position_size,
        "leverage": leverage,
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": round(take_profit_pct, 1),
        "liquidation_price": liquidation_price,
        "confidence": confidence,
        "market_regime": market_regime,
        "volatility": volatility,
        "risk_percentage": risk_percentage,
        "entry_time": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Example usage
    test_confidence = 0.85
    test_portfolio = 20000.0
    test_entry_price = 50000.0
    test_pair = "BTC/USD"
    
    trade_params = prepare_trade_entry(test_portfolio, test_pair, test_confidence, test_entry_price)
    print(f"Trade Parameters: {trade_params}")
    
    is_liq, liq_price, liq_pnl = check_liquidation({
        "entry_price": test_entry_price,
        "direction": "long",
        "leverage": trade_params["leverage"],
        "position_size": trade_params["position_size"]
    }, test_entry_price * 0.85)  # Simulated 15% price drop
    
    print(f"Liquidation Check: {is_liq}, Price: {liq_price}, PnL: {liq_pnl}")