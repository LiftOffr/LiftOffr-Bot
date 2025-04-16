#!/usr/bin/env python3
"""
Enhanced Risk Management for Crypto Trading Bot

This module implements dynamic risk management for the trading bot with:
1. Maximum portfolio risk of 25%
2. Dynamic risk allocation per trade
3. Position sizing based on volatility
4. Risk budget adjustment based on open positions

Usage:
    python enhanced_risk_management.py --update_config
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"

# Default settings
DEFAULT_MAX_PORTFOLIO_RISK = 0.25  # 25% maximum portfolio risk
DEFAULT_BASE_LEVERAGE = 5.0
DEFAULT_MAX_LEVERAGE = 75.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_RISK_PERCENTAGE = 0.20  # 20% risk per trade

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced risk management for trading bot")
    parser.add_argument("--update_config", action="store_true", default=True,
                        help="Update ML configuration with new risk settings")
    parser.add_argument("--max_portfolio_risk", type=float, default=DEFAULT_MAX_PORTFOLIO_RISK,
                        help=f"Maximum portfolio risk percentage (default: {DEFAULT_MAX_PORTFOLIO_RISK})")
    parser.add_argument("--base_leverage", type=float, default=DEFAULT_BASE_LEVERAGE,
                        help=f"Base leverage for trading (default: {DEFAULT_BASE_LEVERAGE})")
    parser.add_argument("--max_leverage", type=float, default=DEFAULT_MAX_LEVERAGE,
                        help=f"Maximum leverage for high-confidence trades (default: {DEFAULT_MAX_LEVERAGE})")
    return parser.parse_args()

def load_ml_config() -> Dict[str, Any]:
    """Load ML configuration from file"""
    try:
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration with {len(config.get('models', {}))} models")
            return config
        else:
            logger.warning(f"ML configuration file not found: {ML_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error loading ML configuration: {e}")
    
    # Return default configuration if file doesn't exist or loading fails
    return {
        "models": {},
        "global_settings": {
            "base_leverage": DEFAULT_BASE_LEVERAGE,
            "max_leverage": DEFAULT_MAX_LEVERAGE,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "risk_percentage": DEFAULT_RISK_PERCENTAGE,
            "max_portfolio_risk": DEFAULT_MAX_PORTFOLIO_RISK
        }
    }

def save_ml_config(config: Dict[str, Any]) -> bool:
    """Save ML configuration to file"""
    try:
        os.makedirs(os.path.dirname(ML_CONFIG_PATH), exist_ok=True)
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved ML configuration to {ML_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML configuration: {e}")
        return False

def load_portfolio() -> Dict[str, Any]:
    """Load portfolio data"""
    try:
        if os.path.exists(PORTFOLIO_PATH):
            with open(PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            logger.info(f"Loaded portfolio with balance: ${portfolio.get('balance', 0):.2f}")
            return portfolio
        else:
            logger.warning(f"Portfolio file not found: {PORTFOLIO_PATH}")
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
    
    # Return default portfolio if file doesn't exist or loading fails
    return {
        "balance": 20000.0,
        "initial_balance": 20000.0,
        "last_updated": datetime.now().isoformat()
    }

def load_positions() -> Dict[str, Any]:
    """Load position data"""
    try:
        if os.path.exists(POSITIONS_PATH):
            with open(POSITIONS_PATH, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions")
            return positions
        else:
            logger.warning(f"Positions file not found: {POSITIONS_PATH}")
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
    
    # Return empty positions dict if file doesn't exist or loading fails
    return {}

def calculate_current_portfolio_risk(portfolio: Dict[str, Any], positions: Dict[str, Any]) -> float:
    """
    Calculate current portfolio risk based on open positions
    
    Args:
        portfolio: Portfolio data
        positions: Open positions data
        
    Returns:
        Current portfolio risk as percentage of portfolio balance
    """
    if not positions:
        return 0.0
    
    portfolio_balance = portfolio.get('balance', 0)
    if portfolio_balance <= 0:
        return 1.0  # 100% risk if balance is zero or negative
    
    # Calculate total margin used
    total_margin = sum(position.get('margin', 0) for position in positions.values())
    
    # Calculate risk as percentage of portfolio balance
    portfolio_risk = total_margin / portfolio_balance
    
    logger.info(f"Current portfolio risk: {portfolio_risk:.2%}")
    logger.info(f"Total margin used: ${total_margin:.2f} of ${portfolio_balance:.2f}")
    
    return portfolio_risk

def calculate_available_risk(
    portfolio: Dict[str, Any], 
    positions: Dict[str, Any], 
    max_portfolio_risk: float
) -> float:
    """
    Calculate available risk for new trades
    
    Args:
        portfolio: Portfolio data
        positions: Open positions data
        max_portfolio_risk: Maximum allowed portfolio risk
        
    Returns:
        Available risk as percentage of portfolio balance
    """
    current_risk = calculate_current_portfolio_risk(portfolio, positions)
    available_risk = max(0, max_portfolio_risk - current_risk)
    
    logger.info(f"Available risk: {available_risk:.2%}")
    
    return available_risk

def calculate_max_position_size(
    pair: str,
    portfolio: Dict[str, Any],
    positions: Dict[str, Any],
    config: Dict[str, Any],
    leverage: float,
    current_price: float
) -> Tuple[float, float]:
    """
    Calculate maximum position size based on available risk
    
    Args:
        pair: Trading pair
        portfolio: Portfolio data
        positions: Open positions data
        config: ML configuration
        leverage: Leverage for the trade
        current_price: Current price of the asset
        
    Returns:
        Tuple of (max_position_size, risk_amount)
    """
    # Get portfolio balance
    portfolio_balance = portfolio.get('balance', 0)
    if portfolio_balance <= 0:
        return 0.0, 0.0
    
    # Get maximum portfolio risk
    max_portfolio_risk = config.get('global_settings', {}).get('max_portfolio_risk', DEFAULT_MAX_PORTFOLIO_RISK)
    
    # Calculate available risk
    available_risk = calculate_available_risk(portfolio, positions, max_portfolio_risk)
    
    # Get risk percentage per trade
    risk_percentage = config.get('models', {}).get(pair, {}).get(
        'risk_percentage',
        config.get('global_settings', {}).get('risk_percentage', DEFAULT_RISK_PERCENTAGE)
    )
    
    # Adjust risk percentage if it would exceed available risk
    adjusted_risk_percentage = min(risk_percentage, available_risk)
    
    # Calculate risk amount
    risk_amount = portfolio_balance * adjusted_risk_percentage
    
    # Calculate maximum position size
    max_position_size = risk_amount / current_price * leverage
    
    logger.info(f"Max position size for {pair}: {max_position_size:.6f}")
    logger.info(f"Risk amount: ${risk_amount:.2f} ({adjusted_risk_percentage:.2%} of ${portfolio_balance:.2f})")
    
    return max_position_size, risk_amount

def update_risk_settings(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Update risk settings in ML configuration
    
    Args:
        config: ML configuration
        args: Command line arguments
        
    Returns:
        Updated ML configuration
    """
    # Update global settings
    if 'global_settings' not in config:
        config['global_settings'] = {}
    
    config['global_settings']['max_portfolio_risk'] = args.max_portfolio_risk
    config['global_settings']['base_leverage'] = args.base_leverage
    config['global_settings']['max_leverage'] = args.max_leverage
    
    # Log changes
    logger.info("Updated risk settings:")
    logger.info(f"Maximum portfolio risk: {args.max_portfolio_risk:.2%}")
    logger.info(f"Base leverage: {args.base_leverage}x")
    logger.info(f"Maximum leverage: {args.max_leverage}x")
    
    return config

class DynamicRiskManager:
    """Dynamic risk manager class for trading bot"""
    
    def __init__(self, max_portfolio_risk=DEFAULT_MAX_PORTFOLIO_RISK):
        """Initialize the risk manager"""
        self.max_portfolio_risk = max_portfolio_risk
        self.config = load_ml_config()
        self.portfolio = load_portfolio()
        self.positions = load_positions()
    
    def calculate_trade_risk(self, pair: str, confidence: float) -> Tuple[float, float]:
        """
        Calculate risk and leverage for a trade based on confidence
        
        Args:
            pair: Trading pair
            confidence: Prediction confidence (0.0-1.0)
            
        Returns:
            Tuple of (risk_percentage, leverage)
        """
        # Get pair-specific or global settings
        pair_config = self.config.get('models', {}).get(pair, {})
        global_settings = self.config.get('global_settings', {})
        
        base_leverage = pair_config.get('base_leverage', global_settings.get('base_leverage', DEFAULT_BASE_LEVERAGE))
        max_leverage = pair_config.get('max_leverage', global_settings.get('max_leverage', DEFAULT_MAX_LEVERAGE))
        base_risk = pair_config.get('risk_percentage', global_settings.get('risk_percentage', DEFAULT_RISK_PERCENTAGE))
        
        # Calculate dynamic leverage based on confidence
        leverage = base_leverage + (max_leverage - base_leverage) * confidence
        leverage = min(leverage, max_leverage)  # Cap at max leverage
        
        # Calculate available risk
        available_risk = calculate_available_risk(self.portfolio, self.positions, self.max_portfolio_risk)
        
        # Adjust risk percentage based on available risk
        risk_percentage = min(base_risk, available_risk)
        
        return risk_percentage, leverage
    
    def can_open_position(self, pair: str, confidence: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a position can be opened based on risk management rules
        
        Args:
            pair: Trading pair
            confidence: Prediction confidence (0.0-1.0)
            
        Returns:
            Tuple of (can_open, risk_info)
        """
        # Get current portfolio risk
        current_risk = calculate_current_portfolio_risk(self.portfolio, self.positions)
        
        # Check if maximum portfolio risk would be exceeded
        if current_risk >= self.max_portfolio_risk:
            return False, {
                "reason": "Maximum portfolio risk exceeded",
                "current_risk": current_risk,
                "max_risk": self.max_portfolio_risk
            }
        
        # Calculate risk percentage and leverage
        risk_percentage, leverage = self.calculate_trade_risk(pair, confidence)
        
        # Check if risk percentage is too small
        min_risk = 0.01  # 1% minimum risk per trade
        if risk_percentage < min_risk:
            return False, {
                "reason": "Available risk too small",
                "available_risk": risk_percentage,
                "min_risk": min_risk
            }
        
        # Return success
        return True, {
            "risk_percentage": risk_percentage,
            "leverage": leverage,
            "current_portfolio_risk": current_risk,
            "available_risk": self.max_portfolio_risk - current_risk
        }

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load ML configuration
    config = load_ml_config()
    
    # Update risk settings
    if args.update_config:
        config = update_risk_settings(config, args)
        save_ml_config(config)
    
    # Load portfolio and positions
    portfolio = load_portfolio()
    positions = load_positions()
    
    # Create risk manager
    risk_manager = DynamicRiskManager(args.max_portfolio_risk)
    
    # Display current risk status
    current_risk = calculate_current_portfolio_risk(portfolio, positions)
    available_risk = calculate_available_risk(portfolio, positions, args.max_portfolio_risk)
    
    print("\n" + "=" * 60)
    print("ENHANCED RISK MANAGEMENT")
    print("=" * 60)
    print(f"Maximum portfolio risk: {args.max_portfolio_risk:.2%}")
    print(f"Current portfolio risk: {current_risk:.2%}")
    print(f"Available risk: {available_risk:.2%}")
    print(f"Portfolio balance: ${portfolio.get('balance', 0):.2f}")
    print(f"Open positions: {len(positions)}")
    
    # Test risk calculations for different pairs and confidence levels
    print("\nRISK CALCULATION EXAMPLES:")
    test_pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
    confidence_levels = [0.6, 0.75, 0.9]
    
    for pair in test_pairs:
        for confidence in confidence_levels:
            can_open, risk_info = risk_manager.can_open_position(pair, confidence)
            status = "ALLOWED" if can_open else "DENIED"
            
            if can_open:
                print(f"\n{pair} @ {confidence:.2f} confidence: {status}")
                print(f"  Risk: {risk_info['risk_percentage']:.2%}")
                print(f"  Leverage: {risk_info['leverage']:.1f}x")
            else:
                print(f"\n{pair} @ {confidence:.2f} confidence: {status}")
                print(f"  Reason: {risk_info['reason']}")
    
    print("\n" + "=" * 60)
    print("ENHANCED RISK MANAGEMENT SUMMARY")
    print("=" * 60)
    print("1. Maximum portfolio risk: 25%")
    print("2. Dynamic risk allocation per trade")
    print("3. Leverage adjustment based on confidence")
    print("4. Position sizing based on available risk")
    print("5. Risk budget protection")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in risk management: {e}")
        import traceback
        traceback.print_exc()