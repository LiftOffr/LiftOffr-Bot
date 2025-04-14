#!/usr/bin/env python3
"""
Dynamic Position Sizing Module with ML Integration

This module provides advanced position sizing and leverage control based on:
1. Signal strength from trading strategies
2. ML model confidence
3. Market regime analysis
4. Asset-specific risk profiles
5. Current volatility levels

The module optimizes risk:reward by dynamically adjusting:
- Leverage (from 20x to 125x)
- Position size (percentage of available capital)
- Stop loss and take profit levels
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from market_context import (
    analyze_market_context, 
    get_optimal_trade_parameters,
    get_regime_leverage_factor,
    MarketRegime
)

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "min_leverage": 20.0,       # Minimum leverage to use
    "base_leverage": 35.0,      # Base leverage (will be multiplied by factors)
    "max_leverage": 125.0,      # Maximum leverage allowed
    "min_margin_floor": 0.15,   # Minimum margin as % of available capital
    "base_margin_percent": 0.22, # Base margin as % of available capital
    "max_margin_cap": 0.40,     # Maximum margin as % of available capital
    "signal_strength_multiplier": 1.5,  # Multiplier for signal strength
    "ml_confidence_multiplier": 2.0,    # Multiplier for ML confidence
    "stop_loss_atr_multiplier": 1.0,    # Multiplier for ATR-based stop loss
    "take_profit_atr_multiplier": 3.0,  # Multiplier for ATR-based take profit
    "market_regime_factors": {  # Leverage multipliers for market regimes
        "volatile_trending_up": 1.2,    # High volatility uptrend
        "volatile_trending_down": 0.9,  # High volatility downtrend
        "normal_trending_up": 1.8,      # Normal volatility uptrend
        "normal_trending_down": 1.2,    # Normal volatility downtrend
        "neutral": 1.4                  # Neutral/ranging market
    }
}

# Asset-specific configurations (overrides global config)
DEFAULT_ASSET_CONFIGS = {
    "SOL/USD": {
        "min_leverage": 20.0,  # More volatile, but we want exposure
        "base_leverage": 35.0,
        "max_leverage": 125.0,
        "signal_strength_multiplier": 1.8  # Respond more to SOL signals
    },
    "ETH/USD": {
        "min_leverage": 15.0,  # Moderate volatility
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "signal_strength_multiplier": 1.6  # Moderate response to signals
    },
    "BTC/USD": {
        "min_leverage": 12.0,  # Lower volatility
        "base_leverage": 25.0,
        "max_leverage": 85.0,
        "signal_strength_multiplier": 1.4  # More conservative with BTC
    }
}

@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing"""
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    asset_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: DEFAULT_ASSET_CONFIGS.copy())
    
    def get_asset_config(self, asset: str) -> Dict[str, Any]:
        """Get configuration for a specific asset, falling back to global config"""
        if asset in self.asset_configs:
            # Start with global config
            result = self.config.copy()
            # Override with asset-specific config
            result.update(self.asset_configs[asset])
            return result
        else:
            return self.config.copy()
    
    def save_config(self, filename: str = "position_sizing_config.json") -> None:
        """Save configuration to file"""
        with open(filename, 'w') as f:
            json.dump({
                "global": self.config,
                "assets": self.asset_configs
            }, f, indent=4)
    
    def load_config(self, filename: str = "position_sizing_config.json") -> bool:
        """Load configuration from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if "global" in data:
                    self.config.update(data["global"])
                if "assets" in data:
                    for asset, config in data["assets"].items():
                        if asset in self.asset_configs:
                            self.asset_configs[asset].update(config)
                        else:
                            self.asset_configs[asset] = config
                return True
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load position sizing config: {e}")
            return False

# Singleton config instance
_config_instance = None

def get_config() -> PositionSizingConfig:
    """Get the global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = PositionSizingConfig()
        
        # Try to load from the ultra-aggressive config first
        if not _config_instance.load_config("position_sizing_config_ultra_aggressive.json"):
            # Fall back to the regular config
            _config_instance.load_config("position_sizing_config.json")
    
    return _config_instance

def calculate_dynamic_position_size(
    available_capital: float,
    signal_strength: float,
    ml_confidence: float,
    market_regime: str,
    asset: str = "SOL/USD"
) -> float:
    """
    Calculate the optimal position size as a percentage of available capital
    
    Args:
        available_capital: Available capital for this trade
        signal_strength: Signal strength from strategy (0.0-1.0)
        ml_confidence: Confidence from ML model (0.0-1.0)
        market_regime: Current market regime
        asset: Asset being traded
        
    Returns:
        float: Position size as a percentage of available capital (0.0-1.0)
    """
    # Get config for this asset
    config = get_config().get_asset_config(asset)
    
    # Base margin percentage
    base_margin = config["base_margin_percent"]
    
    # Adjust based on signal strength
    signal_adjustment = signal_strength * config["signal_strength_multiplier"]
    
    # Adjust based on ML confidence
    ml_adjustment = ml_confidence * config["ml_confidence_multiplier"]
    
    # Adjust based on market regime
    regime_adjustment = config["market_regime_factors"].get(market_regime, 1.0)
    
    # Calculate adjusted margin
    adjusted_margin = base_margin * (1.0 + signal_adjustment + ml_adjustment) * regime_adjustment
    
    # Apply limits
    min_margin = config["min_margin_floor"]
    max_margin = config["max_margin_cap"]
    
    position_size = max(min_margin, min(adjusted_margin, max_margin))
    
    logger.debug(f"Position size for {asset}: {position_size:.2%} of ${available_capital:.2f}")
    logger.debug(f"Adjustments - Signal: {signal_adjustment:.2f}, ML: {ml_adjustment:.2f}, "
                f"Regime: {regime_adjustment:.2f}")
    
    return position_size

def calculate_dynamic_leverage(
    base_price: float,
    signal_strength: float,
    ml_confidence: float,
    market_regime: str,
    asset: str = "SOL/USD"
) -> float:
    """
    Calculate the optimal leverage based on current market conditions
    
    Args:
        base_price: Current asset price
        signal_strength: Signal strength from strategy (0.0-1.0)
        ml_confidence: Confidence from ML model (0.0-1.0)
        market_regime: Current market regime
        asset: Asset being traded
        
    Returns:
        float: Leverage amount
    """
    # Get config for this asset
    config = get_config().get_asset_config(asset)
    
    # Base leverage
    base_leverage = config["base_leverage"]
    
    # Adjust based on signal strength (higher signal = more leverage)
    signal_adjustment = signal_strength * config["signal_strength_multiplier"]
    
    # Adjust based on ML confidence (higher confidence = more leverage)
    ml_adjustment = ml_confidence * config["ml_confidence_multiplier"]
    
    # Adjust based on market regime
    regime_adjustment = config["market_regime_factors"].get(market_regime, 1.0)
    
    # Calculate adjusted leverage
    adjusted_leverage = base_leverage * (1.0 + signal_adjustment + ml_adjustment) * regime_adjustment
    
    # Apply limits
    min_leverage = config["min_leverage"]
    max_leverage = config["max_leverage"]
    
    leverage = max(min_leverage, min(adjusted_leverage, max_leverage))
    
    logger.debug(f"Leverage for {asset}: {leverage:.2f}x at ${base_price:.2f}")
    logger.debug(f"Adjustments - Signal: {signal_adjustment:.2f}, ML: {ml_adjustment:.2f}, "
                f"Regime: {regime_adjustment:.2f}")
    
    return leverage

def calculate_stop_loss(
    price: float,
    direction: str,
    atr: float,
    signal_strength: float,
    market_regime: str,
    asset: str = "SOL/USD"
) -> float:
    """
    Calculate the optimal stop loss price
    
    Args:
        price: Entry price
        direction: 'long' or 'short'
        atr: Average True Range value
        signal_strength: Signal strength (0.0-1.0)
        market_regime: Current market regime
        asset: Asset being traded
        
    Returns:
        float: Stop loss price
    """
    # Get config for this asset
    config = get_config().get_asset_config(asset)
    
    # Calculate ATR-based stop distance
    atr_multiplier = config["stop_loss_atr_multiplier"]
    
    # Adjust based on market regime
    regime_factor = config["market_regime_factors"].get(market_regime, 1.0)
    
    # Adjust based on signal strength (stronger signal = tighter stop)
    signal_factor = 1.0 - (signal_strength * 0.3)  # 0.7-1.0 range
    
    # Calculate stop distance
    stop_distance = atr * atr_multiplier * signal_factor * regime_factor
    
    # Calculate stop price
    if direction.lower() == 'long':
        stop_price = price - stop_distance
    else:  # short
        stop_price = price + stop_distance
    
    logger.debug(f"{direction.capitalize()} stop loss for {asset}: ${stop_price:.2f} "
                f"(${stop_distance:.2f} from entry)")
    
    return stop_price

def calculate_take_profit(
    price: float,
    direction: str,
    atr: float,
    signal_strength: float,
    ml_confidence: float,
    market_regime: str,
    asset: str = "SOL/USD"
) -> float:
    """
    Calculate the optimal take profit price
    
    Args:
        price: Entry price
        direction: 'long' or 'short'
        atr: Average True Range value
        signal_strength: Signal strength (0.0-1.0)
        ml_confidence: ML model confidence (0.0-1.0)
        market_regime: Current market regime
        asset: Asset being traded
        
    Returns:
        float: Take profit price
    """
    # Get config for this asset
    config = get_config().get_asset_config(asset)
    
    # Calculate ATR-based take profit distance
    atr_multiplier = config["take_profit_atr_multiplier"]
    
    # Adjust based on market regime
    regime_factor = config["market_regime_factors"].get(market_regime, 1.0)
    
    # Adjust based on signal strength and ML confidence
    signal_factor = 1.0 + (signal_strength * 0.5)  # 1.0-1.5 range
    ml_factor = 1.0 + (ml_confidence * 0.5)  # 1.0-1.5 range
    
    # Calculate take profit distance
    tp_distance = atr * atr_multiplier * signal_factor * ml_factor * regime_factor
    
    # Calculate take profit price
    if direction.lower() == 'long':
        tp_price = price + tp_distance
    else:  # short
        tp_price = price - tp_distance
    
    logger.debug(f"{direction.capitalize()} take profit for {asset}: ${tp_price:.2f} "
                f"(${tp_distance:.2f} from entry)")
    
    return tp_price

def get_optimal_trade_parameters_ml(
    price_data: Dict[str, Any],
    direction: str,
    signal_strength: float,
    ml_confidence: float,
    asset: str = "SOL/USD"
) -> Dict[str, float]:
    """
    Calculate optimal trade parameters using ML and market context
    
    Args:
        price_data: Dictionary with price data (OHLCV)
        direction: Trade direction ('long' or 'short')
        signal_strength: Signal strength from strategy (0.0-1.0)
        ml_confidence: Confidence from ML model (0.0-1.0)
        asset: Asset being traded
        
    Returns:
        Dict: Trade parameters (leverage, margin_pct, stop_loss_pct, take_profit_pct)
    """
    # Analyze market context
    market_context = analyze_market_context(price_data)
    market_regime = market_context['regime']
    
    # Get config for this asset
    config = get_config().get_asset_config(asset)
    
    # Calculate leverage
    leverage = calculate_dynamic_leverage(
        base_price=market_context['price'],
        signal_strength=signal_strength,
        ml_confidence=ml_confidence,
        market_regime=market_regime,
        asset=asset
    )
    
    # Calculate position size
    margin_pct = calculate_dynamic_position_size(
        available_capital=100.0,  # Placeholder, will be scaled by actual capital
        signal_strength=signal_strength,
        ml_confidence=ml_confidence,
        market_regime=market_regime,
        asset=asset
    )
    
    # Get optimal trade parameters for current market conditions
    optimal_params = market_context['long_params'] if direction == 'long' else market_context['short_params']
    
    # Calculate risk percentages
    stop_loss_pct = optimal_params['stop_loss_pct']
    take_profit_pct = optimal_params['take_profit_pct']
    
    return {
        'leverage': leverage,
        'margin_pct': margin_pct,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'market_regime': market_regime,
        'atr': optimal_params['atr'],
        'confidence': max(signal_strength, ml_confidence)
    }

def save_ultra_aggressive_config():
    """Create and save an ultra-aggressive position sizing configuration"""
    config = PositionSizingConfig()
    
    # Update global parameters for ultra-aggressive trading
    config.config.update({
        "min_leverage": 20.0,       # Minimum leverage to use
        "base_leverage": 45.0,      # Base leverage (will be multiplied by factors)
        "max_leverage": 125.0,      # Maximum leverage allowed
        "min_margin_floor": 0.18,   # Minimum margin as % of available capital
        "base_margin_percent": 0.28, # Base margin as % of available capital
        "max_margin_cap": 0.50,     # Maximum margin as % of available capital
        "signal_strength_multiplier": 2.2,  # Higher multiplier for signal strength
        "ml_confidence_multiplier": 2.8,    # Higher multiplier for ML confidence
        "stop_loss_atr_multiplier": 1.1,    # Slightly wider stops
        "take_profit_atr_multiplier": 3.5,  # More ambitious take profits
        "market_regime_factors": {  # More aggressive leverage multipliers
            "volatile_trending_up": 1.4,    # Higher leverage in volatile uptrends
            "volatile_trending_down": 1.0,   # Same leverage in volatile downtrends
            "normal_trending_up": 2.2,      # Much higher leverage in normal uptrends
            "normal_trending_down": 1.5,    # Higher leverage in normal downtrends
            "neutral": 1.8                  # Higher leverage in neutral markets
        }
    })
    
    # Update asset-specific configs
    config.asset_configs["SOL/USD"].update({
        "min_leverage": 20.0,
        "base_leverage": 50.0,    # Significantly higher base leverage for SOL
        "max_leverage": 125.0,    # Maximum leverage for SOL
        "signal_strength_multiplier": 2.5  # More responsive to SOL signals
    })
    
    config.asset_configs["ETH/USD"].update({
        "min_leverage": 15.0,
        "base_leverage": 40.0,    # Higher base leverage for ETH
        "max_leverage": 100.0,    # High leverage for ETH
        "signal_strength_multiplier": 2.2  # More responsive to ETH signals
    })
    
    config.asset_configs["BTC/USD"].update({
        "min_leverage": 12.0,
        "base_leverage": 35.0,    # Higher base leverage for BTC
        "max_leverage": 85.0,     # Moderate leverage for BTC (still high)
        "signal_strength_multiplier": 2.0  # More responsive to BTC signals
    })
    
    # Save the ultra-aggressive config
    config.save_config("position_sizing_config_ultra_aggressive.json")
    logger.info("Created ultra-aggressive position sizing configuration")

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Generate ultra-aggressive config
    save_ultra_aggressive_config()
    
    # Test the module
    config = get_config()
    
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"Base Leverage: {config.config['base_leverage']}x")
    print(f"Max Leverage: {config.config['max_leverage']}x")
    print("\nAsset-specific settings:")
    for asset, asset_config in config.asset_configs.items():
        print(f"\n{asset}:")
        print(f"  Base Leverage: {asset_config.get('base_leverage', config.config['base_leverage'])}x")
        print(f"  Max Leverage: {asset_config.get('max_leverage', config.config['max_leverage'])}x")
    
    # Test dynamic leverage calculation
    print("\nTesting dynamic leverage calculation:")
    for asset in config.asset_configs.keys():
        for signal in [0.3, 0.6, 0.9]:
            for conf in [0.4, 0.7, 0.95]:
                for regime in MarketRegime:
                    leverage = calculate_dynamic_leverage(
                        base_price=100.0,
                        signal_strength=signal,
                        ml_confidence=conf,
                        market_regime=regime.value,
                        asset=asset
                    )
                    print(f"{asset} - Signal: {signal:.1f}, Conf: {conf:.1f}, "
                          f"Regime: {regime.value} -> Leverage: {leverage:.1f}x")