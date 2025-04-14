#!/usr/bin/env python3
"""
ML Parameter Update Tool

This script allows for easy updating of ML parameters in ml_config.json
and optionally restarting the trading bot with the new parameters.
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_parameter_updates.log')
    ]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "ml_config.json"
TRADING_BOT_SCRIPT = "run_optimized_ml_trading.py"

def load_config() -> Dict[str, Any]:
    """Load the ML configuration from file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration from {CONFIG_PATH}")
            return config
        else:
            logger.error(f"Configuration file {CONFIG_PATH} not found")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def save_config(config: Dict[str, Any]) -> None:
    """Save the ML configuration to file"""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved ML configuration to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        sys.exit(1)

def update_leverage_settings(
    config: Dict[str, Any], 
    asset: str,
    min_leverage: float = None,
    default_leverage: float = None,
    max_leverage: float = None,
    confidence_threshold: float = None
) -> Dict[str, Any]:
    """Update leverage settings for an asset"""
    try:
        if 'asset_configs' not in config or asset not in config['asset_configs']:
            logger.error(f"Asset {asset} not found in configuration")
            return config
        
        asset_config = config['asset_configs'][asset]
        if 'leverage_settings' not in asset_config:
            asset_config['leverage_settings'] = {}
        
        leverage_settings = asset_config['leverage_settings']
        
        if min_leverage is not None:
            leverage_settings['min'] = min_leverage
            logger.info(f"Updated min_leverage for {asset} to {min_leverage}")
        
        if default_leverage is not None:
            leverage_settings['default'] = default_leverage
            logger.info(f"Updated default_leverage for {asset} to {default_leverage}")
        
        if max_leverage is not None:
            leverage_settings['max'] = max_leverage
            logger.info(f"Updated max_leverage for {asset} to {max_leverage}")
        
        if confidence_threshold is not None:
            leverage_settings['confidence_threshold'] = confidence_threshold
            logger.info(f"Updated confidence_threshold for {asset} to {confidence_threshold}")
        
        return config
    except Exception as e:
        logger.error(f"Error updating leverage settings: {e}")
        return config

def update_position_sizing(
    config: Dict[str, Any], 
    asset: str,
    confidence_thresholds: List[float] = None,
    size_multipliers: List[float] = None
) -> Dict[str, Any]:
    """Update position sizing settings for an asset"""
    try:
        if 'asset_configs' not in config or asset not in config['asset_configs']:
            logger.error(f"Asset {asset} not found in configuration")
            return config
        
        asset_config = config['asset_configs'][asset]
        if 'position_sizing' not in asset_config:
            asset_config['position_sizing'] = {}
        
        position_sizing = asset_config['position_sizing']
        
        if confidence_thresholds is not None:
            position_sizing['confidence_thresholds'] = confidence_thresholds
            logger.info(f"Updated confidence_thresholds for {asset} to {confidence_thresholds}")
        
        if size_multipliers is not None:
            position_sizing['size_multipliers'] = size_multipliers
            logger.info(f"Updated size_multipliers for {asset} to {size_multipliers}")
        
        return config
    except Exception as e:
        logger.error(f"Error updating position sizing: {e}")
        return config

def update_risk_management(
    config: Dict[str, Any], 
    asset: str,
    max_open_positions: int = None,
    max_drawdown_percent: float = None,
    take_profit_multiplier: float = None,
    stop_loss_multiplier: float = None
) -> Dict[str, Any]:
    """Update risk management settings for an asset"""
    try:
        if 'asset_configs' not in config or asset not in config['asset_configs']:
            logger.error(f"Asset {asset} not found in configuration")
            return config
        
        asset_config = config['asset_configs'][asset]
        if 'risk_management' not in asset_config:
            asset_config['risk_management'] = {}
        
        risk_management = asset_config['risk_management']
        
        if max_open_positions is not None:
            risk_management['max_open_positions'] = max_open_positions
            logger.info(f"Updated max_open_positions for {asset} to {max_open_positions}")
        
        if max_drawdown_percent is not None:
            risk_management['max_drawdown_percent'] = max_drawdown_percent
            logger.info(f"Updated max_drawdown_percent for {asset} to {max_drawdown_percent}")
        
        if take_profit_multiplier is not None:
            risk_management['take_profit_multiplier'] = take_profit_multiplier
            logger.info(f"Updated take_profit_multiplier for {asset} to {take_profit_multiplier}")
        
        if stop_loss_multiplier is not None:
            risk_management['stop_loss_multiplier'] = stop_loss_multiplier
            logger.info(f"Updated stop_loss_multiplier for {asset} to {stop_loss_multiplier}")
        
        return config
    except Exception as e:
        logger.error(f"Error updating risk management: {e}")
        return config

def update_global_settings(
    config: Dict[str, Any],
    extreme_leverage_enabled: bool = None,
    model_pruning_threshold: float = None,
    model_pruning_min_samples: int = None,
    model_selection_frequency: int = None
) -> Dict[str, Any]:
    """Update global settings"""
    try:
        if 'global_settings' not in config:
            config['global_settings'] = {}
        
        global_settings = config['global_settings']
        
        if extreme_leverage_enabled is not None:
            global_settings['extreme_leverage_enabled'] = extreme_leverage_enabled
            logger.info(f"Updated extreme_leverage_enabled to {extreme_leverage_enabled}")
        
        if model_pruning_threshold is not None:
            global_settings['model_pruning_threshold'] = model_pruning_threshold
            logger.info(f"Updated model_pruning_threshold to {model_pruning_threshold}")
        
        if model_pruning_min_samples is not None:
            global_settings['model_pruning_min_samples'] = model_pruning_min_samples
            logger.info(f"Updated model_pruning_min_samples to {model_pruning_min_samples}")
        
        if model_selection_frequency is not None:
            global_settings['model_selection_frequency'] = model_selection_frequency
            logger.info(f"Updated model_selection_frequency to {model_selection_frequency}")
        
        return config
    except Exception as e:
        logger.error(f"Error updating global settings: {e}")
        return config

def update_capital_allocation(
    config: Dict[str, Any],
    sol_alloc: float = None,
    eth_alloc: float = None,
    btc_alloc: float = None
) -> Dict[str, Any]:
    """Update capital allocation settings"""
    try:
        if 'global_settings' not in config:
            config['global_settings'] = {}
        
        if 'default_capital_allocation' not in config['global_settings']:
            config['global_settings']['default_capital_allocation'] = {
                "SOL/USD": 0.40,
                "ETH/USD": 0.35,
                "BTC/USD": 0.25
            }
        
        capital_allocation = config['global_settings']['default_capital_allocation']
        
        if sol_alloc is not None:
            capital_allocation["SOL/USD"] = sol_alloc
            logger.info(f"Updated SOL/USD capital allocation to {sol_alloc}")
        
        if eth_alloc is not None:
            capital_allocation["ETH/USD"] = eth_alloc
            logger.info(f"Updated ETH/USD capital allocation to {eth_alloc}")
        
        if btc_alloc is not None:
            capital_allocation["BTC/USD"] = btc_alloc
            logger.info(f"Updated BTC/USD capital allocation to {btc_alloc}")
        
        # Normalize allocations
        total = sum(capital_allocation.values())
        if total > 0:
            for key in capital_allocation:
                capital_allocation[key] = capital_allocation[key] / total
            logger.info(f"Normalized capital allocations to sum to 1.0")
        
        return config
    except Exception as e:
        logger.error(f"Error updating capital allocation: {e}")
        return config

def restart_trading_bot() -> bool:
    """Restart the trading bot"""
    try:
        logger.info("Restarting trading bot...")
        subprocess.Popen(["python", TRADING_BOT_SCRIPT, "--reset", "--sandbox"])
        logger.info("Trading bot restarted successfully")
        return True
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ML Parameter Update Tool")
    
    # Asset selection
    parser.add_argument("--asset", type=str, choices=["SOL/USD", "ETH/USD", "BTC/USD"], 
                      help="Asset to update parameters for")
    
    # Global settings
    parser.add_argument("--extreme-leverage", type=lambda x: (str(x).lower() == 'true'), 
                      help="Enable extreme leverage (true/false)")
    
    # Leverage settings
    parser.add_argument("--min-leverage", type=float, help="Minimum leverage")
    parser.add_argument("--default-leverage", type=float, help="Default leverage")
    parser.add_argument("--max-leverage", type=float, help="Maximum leverage")
    parser.add_argument("--confidence-threshold", type=float, help="Confidence threshold for leverage")
    
    # Position sizing
    parser.add_argument("--confidence-thresholds", type=str, 
                      help="Confidence thresholds for position sizing (comma-separated floats)")
    parser.add_argument("--size-multipliers", type=str, 
                      help="Size multipliers for position sizing (comma-separated floats)")
    
    # Risk management
    parser.add_argument("--max-drawdown", type=float, help="Maximum drawdown percentage")
    parser.add_argument("--take-profit-multiplier", type=float, help="Take profit multiplier")
    parser.add_argument("--stop-loss-multiplier", type=float, help="Stop loss multiplier")
    
    # Capital allocation
    parser.add_argument("--sol-allocation", type=float, help="Capital allocation for SOL/USD")
    parser.add_argument("--eth-allocation", type=float, help="Capital allocation for ETH/USD")
    parser.add_argument("--btc-allocation", type=float, help="Capital allocation for BTC/USD")
    
    # Restart option
    parser.add_argument("--restart", action="store_true", help="Restart trading bot after updating parameters")
    
    args = parser.parse_args()
    
    # Load the current configuration
    config = load_config()
    
    # Update the configuration based on the provided arguments
    if args.asset:
        # Update leverage settings
        config = update_leverage_settings(
            config, 
            args.asset,
            min_leverage=args.min_leverage,
            default_leverage=args.default_leverage,
            max_leverage=args.max_leverage,
            confidence_threshold=args.confidence_threshold
        )
        
        # Update position sizing
        if args.confidence_thresholds or args.size_multipliers:
            confidence_thresholds = None
            if args.confidence_thresholds:
                confidence_thresholds = [float(x) for x in args.confidence_thresholds.split(',')]
            
            size_multipliers = None
            if args.size_multipliers:
                size_multipliers = [float(x) for x in args.size_multipliers.split(',')]
            
            config = update_position_sizing(
                config, 
                args.asset,
                confidence_thresholds=confidence_thresholds,
                size_multipliers=size_multipliers
            )
        
        # Update risk management
        config = update_risk_management(
            config, 
            args.asset,
            max_drawdown_percent=args.max_drawdown,
            take_profit_multiplier=args.take_profit_multiplier,
            stop_loss_multiplier=args.stop_loss_multiplier
        )
    
    # Update global settings
    config = update_global_settings(
        config,
        extreme_leverage_enabled=args.extreme_leverage
    )
    
    # Update capital allocation
    config = update_capital_allocation(
        config,
        sol_alloc=args.sol_allocation,
        eth_alloc=args.eth_allocation,
        btc_alloc=args.btc_allocation
    )
    
    # Save the updated configuration
    save_config(config)
    
    # Restart the trading bot if requested
    if args.restart:
        restart_trading_bot()

if __name__ == "__main__":
    main()