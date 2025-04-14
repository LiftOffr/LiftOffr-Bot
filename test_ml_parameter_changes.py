#!/usr/bin/env python3
"""
Test ML Parameter Changes

This script demonstrates how to modify ML parameters and observe the effects
on trading behavior. It makes a series of parameter changes with pauses in between
to allow for observation of the effects.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_parameter_test.log')
    ]
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "ml_config.json"

def load_config():
    """Load the ML configuration from file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            return config
        else:
            logger.error(f"Configuration file {CONFIG_PATH} not found")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def save_config(config):
    """Save the ML configuration to file"""
    try:
        # Create a backup first
        if os.path.exists(CONFIG_PATH):
            backup_path = f"{CONFIG_PATH}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(CONFIG_PATH, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            logger.info(f"Created backup of config at {backup_path}")
        
        # Save the new config
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved ML configuration to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        sys.exit(1)

def restart_trading_bot():
    """Restart the trading bot"""
    try:
        logger.info("Restarting trading bot...")
        subprocess.Popen(["python", "run_optimized_ml_trading.py", "--reset", "--sandbox"])
        logger.info("Trading bot restarted successfully")
        return True
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False

def apply_change(change_description, config_modifier_func, pause_seconds=30):
    """Apply a change, restart the bot, and pause to observe effects"""
    logger.info(f"Applying change: {change_description}")
    
    # Load current config
    config = load_config()
    
    # Apply the change
    modified_config = config_modifier_func(config)
    
    # Save the modified config
    save_config(modified_config)
    
    # Restart the trading bot
    restart_trading_bot()
    
    # Log the change for observation
    logger.info(f"Applied change: {change_description}")
    logger.info(f"Pausing for {pause_seconds} seconds to observe effects...")
    time.sleep(pause_seconds)

def test_sol_leverage_increase(config):
    """Increase SOL leverage settings"""
    if 'asset_configs' in config and 'SOL/USD' in config['asset_configs']:
        asset_config = config['asset_configs']['SOL/USD']
        if 'leverage_settings' not in asset_config:
            asset_config['leverage_settings'] = {}
        
        # Increase max leverage to 150
        asset_config['leverage_settings']['max'] = 150.0
        # Increase default leverage to 75
        asset_config['leverage_settings']['default'] = 75.0
        # Lower confidence threshold to 0.6
        asset_config['leverage_settings']['confidence_threshold'] = 0.6
    
    return config

def test_sol_position_sizing(config):
    """Modify SOL position sizing for more aggressive trading"""
    if 'asset_configs' in config and 'SOL/USD' in config['asset_configs']:
        asset_config = config['asset_configs']['SOL/USD']
        if 'position_sizing' not in asset_config:
            asset_config['position_sizing'] = {}
        
        # Make position sizing more aggressive
        asset_config['position_sizing']['confidence_thresholds'] = [0.6, 0.7, 0.8, 0.9]
        asset_config['position_sizing']['size_multipliers'] = [0.5, 0.75, 0.9, 1.0]
    
    return config

def test_sol_capital_allocation(config):
    """Increase SOL capital allocation"""
    if 'global_settings' not in config:
        config['global_settings'] = {}
    
    if 'default_capital_allocation' not in config['global_settings']:
        config['global_settings']['default_capital_allocation'] = {}
    
    # Allocate more capital to SOL
    allocations = config['global_settings']['default_capital_allocation']
    allocations['SOL/USD'] = 0.6  # 60%
    allocations['ETH/USD'] = 0.25  # 25% 
    allocations['BTC/USD'] = 0.15  # 15%
    
    return config

def test_disable_extreme_leverage(config):
    """Disable extreme leverage"""
    if 'global_settings' not in config:
        config['global_settings'] = {}
    
    # Disable extreme leverage
    config['global_settings']['extreme_leverage_enabled'] = False
    
    return config

def test_enable_extreme_leverage(config):
    """Re-enable extreme leverage"""
    if 'global_settings' not in config:
        config['global_settings'] = {}
    
    # Enable extreme leverage
    config['global_settings']['extreme_leverage_enabled'] = True
    
    return config

def test_conservative_settings(config):
    """Set more conservative settings"""
    # Reset to conservative leverage settings for all assets
    for asset in ['SOL/USD', 'ETH/USD', 'BTC/USD']:
        if 'asset_configs' in config and asset in config['asset_configs']:
            asset_config = config['asset_configs'][asset]
            
            # Set conservative leverage
            if 'leverage_settings' not in asset_config:
                asset_config['leverage_settings'] = {}
                
            asset_config['leverage_settings']['min'] = 5.0
            asset_config['leverage_settings']['default'] = 10.0
            asset_config['leverage_settings']['max'] = 20.0
            asset_config['leverage_settings']['confidence_threshold'] = 0.75
            
            # Set conservative position sizing
            if 'position_sizing' not in asset_config:
                asset_config['position_sizing'] = {}
                
            asset_config['position_sizing']['confidence_thresholds'] = [0.75, 0.85, 0.9, 0.95]
            asset_config['position_sizing']['size_multipliers'] = [0.25, 0.5, 0.75, 1.0]
    
    # Distribute capital evenly
    if 'global_settings' not in config:
        config['global_settings'] = {}
        
    if 'default_capital_allocation' not in config['global_settings']:
        config['global_settings']['default_capital_allocation'] = {}
        
    allocations = config['global_settings']['default_capital_allocation']
    allocations['SOL/USD'] = 0.33  # 33%
    allocations['ETH/USD'] = 0.33  # 33%
    allocations['BTC/USD'] = 0.34  # 34%
    
    # Disable extreme leverage
    config['global_settings']['extreme_leverage_enabled'] = False
    
    return config

def restore_original_config():
    """Restore the original configuration from the oldest backup"""
    try:
        # Find the oldest backup
        backups = [f for f in os.listdir('.') if f.startswith(f"{CONFIG_PATH}.backup_")]
        if not backups:
            logger.error("No backup files found")
            return False
        
        oldest_backup = min(backups)
        
        # Restore from the backup
        with open(oldest_backup, "r") as src, open(CONFIG_PATH, "w") as dst:
            dst.write(src.read())
        
        logger.info(f"Restored original configuration from {oldest_backup}")
        return True
    except Exception as e:
        logger.error(f"Error restoring original configuration: {e}")
        return False

def run_test_sequence(pause_seconds=60):
    """Run a sequence of test changes"""
    logger.info("Starting ML parameter test sequence")
    
    # Make sure we have a backup of the original config
    original_config = load_config()
    save_config(original_config)  # This will create a backup
    
    try:
        # Test 1: Increase SOL leverage
        apply_change(
            "Increasing SOL leverage (max=150, default=75, threshold=0.6)", 
            test_sol_leverage_increase,
            pause_seconds
        )
        
        # Test 2: Make SOL position sizing more aggressive
        apply_change(
            "Making SOL position sizing more aggressive", 
            test_sol_position_sizing,
            pause_seconds
        )
        
        # Test 3: Increase SOL capital allocation
        apply_change(
            "Increasing SOL capital allocation to 60%", 
            test_sol_capital_allocation,
            pause_seconds
        )
        
        # Test 4: Disable extreme leverage
        apply_change(
            "Disabling extreme leverage", 
            test_disable_extreme_leverage,
            pause_seconds
        )
        
        # Test 5: Re-enable extreme leverage
        apply_change(
            "Re-enabling extreme leverage", 
            test_enable_extreme_leverage,
            pause_seconds
        )
        
        # Test 6: Set conservative settings
        apply_change(
            "Setting conservative settings for all assets", 
            test_conservative_settings,
            pause_seconds
        )
        
        # Finally, restore original config
        logger.info("Test sequence complete, restoring original configuration")
        restore_original_config()
        restart_trading_bot()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        restore_original_config()
        restart_trading_bot()
    except Exception as e:
        logger.error(f"Error during test sequence: {e}")
        restore_original_config()
        restart_trading_bot()

def main():
    parser = argparse.ArgumentParser(description="Test ML Parameter Changes")
    parser.add_argument("--pause", type=int, default=60, 
                      help="Pause time in seconds between changes (default: 60)")
    args = parser.parse_args()
    
    run_test_sequence(args.pause)

if __name__ == "__main__":
    main()