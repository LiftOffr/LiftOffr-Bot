#!/usr/bin/env python3
"""
Update ML Configuration Script

This script updates the ML configuration for better model integration with the trading system.
It can:
1. Update global ML settings (confidence thresholds, leverage settings)
2. Update model-specific settings (feature scaling, signal weights)
3. Update asset-specific settings (position sizing, risk management)
4. Update ensemble configurations for better model integration
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_config_updates.log')
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_BASE_LEVERAGE = 20.0
DEFAULT_MAX_LEVERAGE = 125.0
DEFAULT_MAX_RISK_PER_TRADE = 0.20  # 20% of available capital
DEFAULT_CONFIG_PATH = "ml_config.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update ML configuration')
    
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to ML configuration file (default: {DEFAULT_CONFIG_PATH})'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for ML signals (default: {DEFAULT_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--base-leverage',
        type=float,
        default=DEFAULT_BASE_LEVERAGE,
        help=f'Base leverage for trades (default: {DEFAULT_BASE_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-leverage',
        type=float,
        default=DEFAULT_MAX_LEVERAGE,
        help=f'Maximum leverage for high-confidence trades (default: {DEFAULT_MAX_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-risk',
        type=float,
        default=DEFAULT_MAX_RISK_PER_TRADE,
        help=f'Maximum risk per trade as fraction of capital (default: {DEFAULT_MAX_RISK_PER_TRADE})'
    )
    
    parser.add_argument(
        '--update-strategy',
        action='store_true',
        help='Update strategy integration parameters'
    )
    
    parser.add_argument(
        '--update-ensemble',
        action='store_true',
        help='Update ensemble configurations for specified pairs'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset configuration to defaults'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load ML configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration or empty dict if not found/valid
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Configuration file {config_path} not found, creating new configuration")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save ML configuration to file.
    
    Args:
        config: Dictionary with configuration
        config_path: Path to save configuration
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return False

def update_global_settings(
    config: Dict[str, Any],
    confidence_threshold: float,
    base_leverage: float,
    max_leverage: float,
    max_risk_per_trade: float,
    reset: bool = False
) -> Dict[str, Any]:
    """
    Update global ML settings.
    
    Args:
        config: Dictionary with configuration
        confidence_threshold: Confidence threshold for ML signals
        base_leverage: Base leverage for trades
        max_leverage: Maximum leverage for high-confidence trades
        max_risk_per_trade: Maximum risk per trade as fraction of capital
        reset: Whether to reset to defaults
        
    Returns:
        Updated configuration
    """
    if reset:
        # Reset to defaults
        config['confidence_threshold'] = DEFAULT_CONFIDENCE_THRESHOLD
        config['base_leverage'] = DEFAULT_BASE_LEVERAGE
        config['max_leverage'] = DEFAULT_MAX_LEVERAGE
        config['max_risk_per_trade'] = DEFAULT_MAX_RISK_PER_TRADE
    else:
        # Update with provided values
        config['confidence_threshold'] = confidence_threshold
        config['base_leverage'] = base_leverage
        config['max_leverage'] = max_leverage
        config['max_risk_per_trade'] = max_risk_per_trade
    
    # Add metadata
    config['last_updated'] = datetime.now().isoformat()
    # Handle version as an integer
    current_version = config.get('version', 0)
    if isinstance(current_version, str):
        try:
            current_version = int(current_version)
        except ValueError:
            current_version = 0
    config['version'] = current_version + 1
    
    logger.info(f"Updated global settings: confidence={config['confidence_threshold']}, "
                f"base_leverage={config['base_leverage']}, max_leverage={config['max_leverage']}, "
                f"max_risk={config['max_risk_per_trade']}")
    
    return config

def update_strategy_integration(config: Dict[str, Any], reset: bool = False) -> Dict[str, Any]:
    """
    Update strategy integration parameters.
    
    Args:
        config: Dictionary with configuration
        reset: Whether to reset to defaults
        
    Returns:
        Updated configuration
    """
    # Ensure strategy_integration section exists
    if 'strategy_integration' not in config or reset:
        config['strategy_integration'] = {}
    
    # Set default integration parameters
    strategy_integration = {
        'use_ml_signals': True,
        'ml_signal_weight': 0.7,  # ML signal has 70% weight in final decision
        'strategy_signal_weight': 0.3,  # Traditional strategy has 30% weight
        'minimum_strategy_agreement': 0.5,  # At least 50% of traditional strategies must agree
        'override_threshold': 0.85,  # ML can override traditional strategies at 85% confidence
        'signal_conflation_method': 'weighted_average',  # How to combine signals
        'use_dynamic_weights': True,  # Adjust weights based on performance
        'safety_features': {
            'market_volatility_filter': True,  # Filter signals during high volatility
            'stop_loss_override': True,  # Allow ML to suggest stop loss adjustments
            'position_sizing_control': True,  # Allow ML to control position sizing
            'maximum_drawdown_protection': 0.1,  # Maximum allowed drawdown (10%)
            'maximum_consecutive_losses': 5  # Maximum allowed consecutive losses
        },
        'last_updated': datetime.now().isoformat()
    }
    
    # Update strategy integration parameters
    for key, value in strategy_integration.items():
        config['strategy_integration'][key] = value
    
    logger.info("Updated strategy integration parameters")
    
    return config

def update_ensemble_configuration(
    pair: str,
    base_dir: str = 'models/ensemble',
    reset: bool = False
) -> bool:
    """
    Update ensemble configuration for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        base_dir: Base directory for ensemble configurations
        reset: Whether to reset to defaults
        
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        # Format pair name (e.g., 'SOL/USD' -> 'SOLUSD')
        pair_formatted = pair.replace('/', '')
        
        # Ensure directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Set paths
        ensemble_path = os.path.join(base_dir, f"{pair_formatted}_ensemble.json")
        weights_path = os.path.join(base_dir, f"{pair_formatted}_weights.json")
        position_sizing_path = os.path.join(base_dir, f"{pair_formatted}_position_sizing.json")
        
        # Define ensemble models
        models = [
            {
                "type": "lstm",
                "enabled": True,
                "confidence_scaling": 1.2
            },
            {
                "type": "tcn",
                "enabled": True,
                "confidence_scaling": 1.1
            },
            {
                "type": "transformer",
                "enabled": True,
                "confidence_scaling": 1.15
            },
            {
                "type": "cnn",
                "enabled": True,
                "confidence_scaling": 1.0
            },
            {
                "type": "gru",
                "enabled": True,
                "confidence_scaling": 1.05
            },
            {
                "type": "bilstm",
                "enabled": True,
                "confidence_scaling": 1.1
            },
            {
                "type": "attention",
                "enabled": True,
                "confidence_scaling": 1.15
            },
            {
                "type": "hybrid",
                "enabled": True,
                "confidence_scaling": 1.2
            }
        ]
        
        # Define ensemble parameters
        parameters = {
            "method": "weighted_voting",
            "confidence_threshold": 0.65,
            "use_confidence_scaling": True,
            "use_dynamic_weights": True,
            "update_frequency": 24,  # Update weights every 24 hours
            "last_updated": datetime.now().isoformat()
        }
        
        # Create ensemble configuration
        ensemble_config = {
            "models": models,
            "parameters": parameters
        }
        
        # Save ensemble configuration
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        # Define model weights
        weights = {
            "lstm": 0.2,
            "tcn": 0.2,
            "transformer": 0.15,
            "cnn": 0.1,
            "gru": 0.1,
            "bilstm": 0.1,
            "attention": 0.1,
            "hybrid": 0.05
        }
        
        # Save model weights
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        
        # Define position sizing rules
        position_sizing = {
            "base_risk_percentage": 0.02,  # 2% risk per trade
            "max_risk_percentage": 0.05,  # Maximum 5% risk
            "confidence_scaling": True,  # Scale position size based on confidence
            "volatility_scaling": True,  # Scale position size based on volatility
            "min_position_size": 0.001,  # Minimum position size
            "max_position_size": 0.5,  # Maximum position size (fraction of capital)
            "last_updated": datetime.now().isoformat()
        }
        
        # Save position sizing rules
        with open(position_sizing_path, 'w') as f:
            json.dump(position_sizing, f, indent=2)
        
        logger.info(f"Updated ensemble configuration for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error updating ensemble configuration for {pair}: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Load configuration
    config = load_config(args.config)
    
    # Update global settings
    config = update_global_settings(
        config,
        args.confidence,
        args.base_leverage,
        args.max_leverage,
        args.max_risk,
        args.reset
    )
    
    # Update strategy integration
    if args.update_strategy:
        config = update_strategy_integration(config, args.reset)
    
    # Save configuration
    if not save_config(config, args.config):
        logger.error("Failed to save configuration")
        return 1
    
    # Update ensemble configurations
    if args.update_ensemble:
        for pair in pairs:
            if not update_ensemble_configuration(pair, reset=args.reset):
                logger.error(f"Failed to update ensemble configuration for {pair}")
                return 1
    
    logger.info("Configuration updated successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())