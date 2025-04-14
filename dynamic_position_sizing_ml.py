"""
Dynamic Position Sizing with ML Integration

This module enhances the dynamic position sizing capabilities by using
ML model predictions and confidence scores to determine leverage and
margin allocation for extreme leverage trading.

Features:
1. ML-driven leverage calculation (up to 125x)
2. Confidence-based margin percentage determination
3. Market regime adjustment factors
4. Asset-specific volatility considerations
5. Risk management constraints
"""

import os
import logging
import json
import time
from typing import Dict, Tuple, Optional
import numpy as np

from config import LEVERAGE, MARGIN_PERCENT

logger = logging.getLogger(__name__)

# Default leverage limits
BASE_LEVERAGE = 35.0  # Default base leverage
MAX_LEVERAGE = 125.0  # Maximum leverage (with high confidence)
MIN_LEVERAGE = 20.0   # Minimum leverage floor

# Market regime adjustment factors
MARKET_REGIME_FACTORS = {
    'volatile_trending_up': 1.2,    # Previously 0.8 (now we increase in volatile uptrends)
    'volatile_trending_down': 0.9,  # Previously 0.7 (less reduction in volatile downtrends)
    'normal_trending_up': 1.8,      # Previously 1.4 (much more aggressive in normal uptrends)
    'normal_trending_down': 1.2,    # Previously 0.9 (now we increase even in downtrends)
    'neutral': 1.4                  # Previously 1.1 (much more aggressive in sideways markets)
}

# Default position sizing configuration
DEFAULT_CONFIG = {
    'base_leverage': BASE_LEVERAGE,
    'max_leverage': MAX_LEVERAGE,
    'min_leverage': MIN_LEVERAGE,
    'base_margin_percent': 0.22,    # Enhanced from 0.20
    'max_margin_cap': 0.65,         # Enhanced from 0.50
    'min_margin_floor': 0.15,
    'ml_influence': 0.75,           # Enhanced from 0.50
    'confidence_threshold': 0.80,
    'risk_factor': 1.0,
    'market_regime_factors': MARKET_REGIME_FACTORS,
    'signal_threshold': 0.08,       # Reduced from 0.20 for more signals
    'stop_loss_atr_multiplier': 1.5,
    'take_profit_atr_multiplier': 3.0,
    'limit_order_factor': 0.70,     # Enhanced from 0.65
    'volume_factor': 1.0
}

class PositionSizingConfig:
    """Configuration class for dynamic position sizing parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with either default configuration or from a file
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info(f"Loaded position sizing configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # Asset-specific configurations
        self.asset_configs = {
            'SOL/USD': {
                'base_leverage': 35.0,
                'max_leverage': 125.0,
                'min_leverage': 20.0,
                'volatility_adjustment': 1.1  # Higher volatility, slightly higher adjustment
            },
            'ETH/USD': {
                'base_leverage': 30.0,
                'max_leverage': 100.0,
                'min_leverage': 15.0,
                'volatility_adjustment': 1.0  # Medium volatility, neutral adjustment
            },
            'BTC/USD': {
                'base_leverage': 25.0,
                'max_leverage': 85.0,
                'min_leverage': 12.0,
                'volatility_adjustment': 0.9  # Lower volatility, slightly lower adjustment
            }
        }
    
    def get_asset_config(self, asset: str) -> Dict:
        """
        Get asset-specific configuration
        
        Args:
            asset: Trading pair symbol
            
        Returns:
            Dict: Configuration dictionary for the asset
        """
        # Start with base configuration
        asset_config = self.config.copy()
        
        # Override with asset-specific values if available
        if asset in self.asset_configs:
            asset_config.update(self.asset_configs[asset])
        
        return asset_config
    
    def get_value(self, key: str, asset: Optional[str] = None) -> any:
        """
        Get a configuration value, using asset-specific value if available
        
        Args:
            key: Configuration key
            asset: Optional trading pair symbol
            
        Returns:
            Value for the configuration key
        """
        if asset and asset in self.asset_configs and key in self.asset_configs[asset]:
            return self.asset_configs[asset][key]
        return self.config.get(key, DEFAULT_CONFIG.get(key))
    
    def save_config(self, config_path: str):
        """
        Save current configuration to a file
        
        Args:
            config_path: Path to save the configuration
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved position sizing configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")

# Global configuration instance
_config_instance = None

def get_config() -> PositionSizingConfig:
    """
    Get the global configuration instance
    
    Returns:
        PositionSizingConfig: Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        # Check for asset-specific config file
        config_path = os.environ.get('POSITION_SIZING_CONFIG', 'position_sizing_config_ultra_aggressive.json')
        _config_instance = PositionSizingConfig(config_path)
    return _config_instance

def calculate_dynamic_leverage(
    base_price: float,
    signal_strength: float,
    ml_confidence: float,
    market_regime: str,
    asset: str = 'SOL/USD'
) -> float:
    """
    Calculate dynamic leverage based on signal strength, ML confidence and market regime
    
    Args:
        base_price: Current price
        signal_strength: Strength of the trading signal (0.0-1.0)
        ml_confidence: Confidence score from ML model (0.0-1.0)
        market_regime: Current market regime
        asset: Trading pair symbol
        
    Returns:
        float: Calculated leverage
    """
    config = get_config()
    asset_config = config.get_asset_config(asset)
    
    # Get base leverage and limits
    base_leverage = asset_config['base_leverage']
    max_leverage = asset_config['max_leverage']
    min_leverage = asset_config['min_leverage']
    
    # Combine signal strength and ML confidence
    ml_influence = asset_config['ml_influence']
    combined_confidence = (signal_strength * (1 - ml_influence)) + (ml_confidence * ml_influence)
    
    # Market regime factor
    regime_factor = asset_config['market_regime_factors'].get(market_regime, 1.0)
    
    # Asset-specific volatility adjustment
    volatility_adjustment = asset_config.get('volatility_adjustment', 1.0)
    
    # Calculate leverage
    leverage_multiplier = combined_confidence * regime_factor * volatility_adjustment
    dynamic_leverage = base_leverage * leverage_multiplier
    
    # Apply limits
    dynamic_leverage = max(min_leverage, min(max_leverage, dynamic_leverage))
    
    logger.debug(f"Dynamic leverage calculation for {asset}:")
    logger.debug(f"Base leverage: {base_leverage}, Signal strength: {signal_strength}, ML confidence: {ml_confidence}")
    logger.debug(f"Market regime: {market_regime} (factor: {regime_factor})")
    logger.debug(f"Volatility adjustment: {volatility_adjustment}")
    logger.debug(f"Combined confidence: {combined_confidence}, Final leverage: {dynamic_leverage}")
    
    return dynamic_leverage

def calculate_dynamic_margin_percent(
    portfolio_value: float,
    leverage: float,
    signal_strength: float,
    ml_confidence: float,
    asset: str = 'SOL/USD'
) -> float:
    """
    Calculate dynamic margin percentage based on portfolio value, leverage,
    signal strength and ML confidence
    
    Args:
        portfolio_value: Current portfolio value
        leverage: Calculated leverage
        signal_strength: Strength of the trading signal (0.0-1.0)
        ml_confidence: Confidence score from ML model (0.0-1.0)
        asset: Trading pair symbol
        
    Returns:
        float: Calculated margin percentage
    """
    config = get_config()
    asset_config = config.get_asset_config(asset)
    
    # Get base margin percent and limits
    base_margin = asset_config['base_margin_percent']
    max_margin = asset_config['max_margin_cap']
    min_margin = asset_config['min_margin_floor']
    
    # Combine signal strength and ML confidence
    ml_influence = asset_config['ml_influence']
    combined_confidence = (signal_strength * (1 - ml_influence)) + (ml_confidence * ml_influence)
    
    # Adjust margin percentage based on confidence
    confidence_factor = 1.0 + (combined_confidence - 0.5) * 2.0
    
    # Adjust margin based on leverage (higher leverage = slightly lower margin)
    leverage_factor = 1.0
    if leverage > asset_config['base_leverage']:
        # Gradually reduce margin as leverage increases
        leverage_diff = (leverage - asset_config['base_leverage']) / (asset_config['max_leverage'] - asset_config['base_leverage'])
        leverage_factor = max(0.8, 1.0 - (leverage_diff * 0.2))
    
    # Calculate margin percentage
    margin_percent = base_margin * confidence_factor * leverage_factor
    
    # Apply limits
    margin_percent = max(min_margin, min(max_margin, margin_percent))
    
    logger.debug(f"Dynamic margin calculation for {asset}:")
    logger.debug(f"Base margin: {base_margin}, Signal strength: {signal_strength}, ML confidence: {ml_confidence}")
    logger.debug(f"Combined confidence: {combined_confidence}, Confidence factor: {confidence_factor}")
    logger.debug(f"Leverage factor: {leverage_factor}, Final margin percent: {margin_percent}")
    
    return margin_percent

def adjust_limit_order_price(
    base_price: float,
    direction: str,
    atr_value: float,
    ml_confidence: float,
    asset: str = 'SOL/USD'
) -> float:
    """
    Adjust limit order price based on ATR and ML confidence
    
    Args:
        base_price: Current price
        direction: Trade direction ('buy' or 'sell')
        atr_value: Current ATR value
        ml_confidence: Confidence score from ML model (0.0-1.0)
        asset: Trading pair symbol
        
    Returns:
        float: Adjusted limit order price
    """
    config = get_config()
    asset_config = config.get_asset_config(asset)
    
    # Base factor for limit order price adjustment
    limit_factor = asset_config['limit_order_factor']
    
    # Adjust factor based on ML confidence
    confidence_adjustment = 1.0 + (ml_confidence - 0.5) * 0.5
    adjusted_factor = limit_factor * confidence_adjustment
    
    # Calculate price adjustment
    price_adjustment = atr_value * adjusted_factor
    
    # Apply adjustment based on direction
    if direction.lower() == 'buy':
        limit_price = base_price - price_adjustment
    else:  # sell
        limit_price = base_price + price_adjustment
    
    logger.debug(f"Limit order price adjustment for {asset} ({direction}):")
    logger.debug(f"Base price: {base_price}, ATR: {atr_value}, ML confidence: {ml_confidence}")
    logger.debug(f"Base factor: {limit_factor}, Adjusted factor: {adjusted_factor}")
    logger.debug(f"Price adjustment: {price_adjustment}, Final limit price: {limit_price}")
    
    return limit_price

def calculate_stop_loss_price(
    entry_price: float,
    direction: str,
    atr_value: float,
    ml_confidence: float,
    asset: str = 'SOL/USD'
) -> float:
    """
    Calculate stop loss price based on ATR and ML confidence
    
    Args:
        entry_price: Trade entry price
        direction: Trade direction ('long' or 'short')
        atr_value: Current ATR value
        ml_confidence: Confidence score from ML model (0.0-1.0)
        asset: Trading pair symbol
        
    Returns:
        float: Calculated stop loss price
    """
    config = get_config()
    asset_config = config.get_asset_config(asset)
    
    # Base multiplier for stop loss
    stop_multiplier = asset_config['stop_loss_atr_multiplier']
    
    # Adjust multiplier based on ML confidence (higher confidence = wider stop)
    confidence_adjustment = 1.0 + (ml_confidence - 0.5) * 0.6
    adjusted_multiplier = stop_multiplier * confidence_adjustment
    
    # Calculate price adjustment
    price_adjustment = atr_value * adjusted_multiplier
    
    # Apply adjustment based on direction
    if direction.lower() == 'long':
        stop_price = entry_price - price_adjustment
    else:  # short
        stop_price = entry_price + price_adjustment
    
    logger.debug(f"Stop loss calculation for {asset} ({direction}):")
    logger.debug(f"Entry price: {entry_price}, ATR: {atr_value}, ML confidence: {ml_confidence}")
    logger.debug(f"Base multiplier: {stop_multiplier}, Adjusted multiplier: {adjusted_multiplier}")
    logger.debug(f"Price adjustment: {price_adjustment}, Final stop price: {stop_price}")
    
    return stop_price

def calculate_take_profit_price(
    entry_price: float,
    direction: str,
    atr_value: float,
    ml_confidence: float,
    asset: str = 'SOL/USD'
) -> float:
    """
    Calculate take profit price based on ATR and ML confidence
    
    Args:
        entry_price: Trade entry price
        direction: Trade direction ('long' or 'short')
        atr_value: Current ATR value
        ml_confidence: Confidence score from ML model (0.0-1.0)
        asset: Trading pair symbol
        
    Returns:
        float: Calculated take profit price
    """
    config = get_config()
    asset_config = config.get_asset_config(asset)
    
    # Base multiplier for take profit
    tp_multiplier = asset_config['take_profit_atr_multiplier']
    
    # Adjust multiplier based on ML confidence (higher confidence = wider target)
    confidence_adjustment = 1.0 + (ml_confidence - 0.5) * 1.0
    adjusted_multiplier = tp_multiplier * confidence_adjustment
    
    # Calculate price adjustment
    price_adjustment = atr_value * adjusted_multiplier
    
    # Apply adjustment based on direction
    if direction.lower() == 'long':
        tp_price = entry_price + price_adjustment
    else:  # short
        tp_price = entry_price - price_adjustment
    
    logger.debug(f"Take profit calculation for {asset} ({direction}):")
    logger.debug(f"Entry price: {entry_price}, ATR: {atr_value}, ML confidence: {ml_confidence}")
    logger.debug(f"Base multiplier: {tp_multiplier}, Adjusted multiplier: {adjusted_multiplier}")
    logger.debug(f"Price adjustment: {price_adjustment}, Final take profit price: {tp_price}")
    
    return tp_price

def get_position_sizing_summary(
    portfolio_value: float,
    asset: str,
    price: float,
    signal_strength: float,
    ml_confidence: float,
    market_regime: str,
    atr_value: float
) -> Dict:
    """
    Get a complete position sizing summary
    
    Args:
        portfolio_value: Current portfolio value
        asset: Trading pair symbol
        price: Current asset price
        signal_strength: Strength of the trading signal (0.0-1.0)
        ml_confidence: Confidence score from ML model (0.0-1.0)
        market_regime: Current market regime
        atr_value: Current ATR value
        
    Returns:
        Dict: Complete position sizing information
    """
    # Calculate dynamic leverage
    leverage = calculate_dynamic_leverage(
        base_price=price,
        signal_strength=signal_strength,
        ml_confidence=ml_confidence,
        market_regime=market_regime,
        asset=asset
    )
    
    # Calculate dynamic margin percentage
    margin_percent = calculate_dynamic_margin_percent(
        portfolio_value=portfolio_value,
        leverage=leverage,
        signal_strength=signal_strength,
        ml_confidence=ml_confidence,
        asset=asset
    )
    
    # Calculate margin amount
    margin_amount = portfolio_value * margin_percent
    
    # Calculate position size
    position_size = margin_amount * leverage
    
    # Calculate quantity at current price
    quantity = position_size / price if price > 0 else 0
    
    # Calculate limit order prices for both directions
    buy_limit_price = adjust_limit_order_price(
        base_price=price,
        direction='buy',
        atr_value=atr_value,
        ml_confidence=ml_confidence,
        asset=asset
    )
    
    sell_limit_price = adjust_limit_order_price(
        base_price=price,
        direction='sell',
        atr_value=atr_value,
        ml_confidence=ml_confidence,
        asset=asset
    )
    
    # Create summary
    summary = {
        'asset': asset,
        'price': price,
        'portfolio_value': portfolio_value,
        'signal_strength': signal_strength,
        'ml_confidence': ml_confidence,
        'market_regime': market_regime,
        'atr': atr_value,
        'leverage': leverage,
        'margin_percent': margin_percent,
        'margin_amount': margin_amount,
        'position_size': position_size,
        'quantity': quantity,
        'buy_limit_price': buy_limit_price,
        'sell_limit_price': sell_limit_price,
        'timestamp': time.time()
    }
    
    logger.info(f"Position sizing summary for {asset}:")
    logger.info(f"Signal strength: {signal_strength:.2f}, ML confidence: {ml_confidence:.2f}")
    logger.info(f"Market regime: {market_regime}")
    logger.info(f"Leverage: {leverage:.2f}x, Margin: {margin_percent:.2%} (${margin_amount:.2f})")
    logger.info(f"Position size: ${position_size:.2f}, Quantity: {quantity:.6f}")
    
    return summary

if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.DEBUG)
    
    # Create test configurations
    test_configs = [
        {
            'portfolio_value': 20000,
            'asset': 'SOL/USD',
            'price': 125.50,
            'signal_strength': 0.85,
            'ml_confidence': 0.92,
            'market_regime': 'normal_trending_up',
            'atr_value': 0.18
        },
        {
            'portfolio_value': 20000,
            'asset': 'ETH/USD',
            'price': 3450.75,
            'signal_strength': 0.70,
            'ml_confidence': 0.85,
            'market_regime': 'volatile_trending_up',
            'atr_value': 15.25
        },
        {
            'portfolio_value': 20000,
            'asset': 'BTC/USD',
            'price': 62150.25,
            'signal_strength': 0.60,
            'ml_confidence': 0.78,
            'market_regime': 'neutral',
            'atr_value': 225.50
        }
    ]
    
    # Run test calculations for each configuration
    for cfg in test_configs:
        summary = get_position_sizing_summary(**cfg)
        
        print(f"\n{'='*80}\n{cfg['asset']} Position Sizing\n{'='*80}")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")