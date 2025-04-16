#!/usr/bin/env python3
"""
Check Active Timeframes

This script analyzes the configured ML models to determine which timeframes
are being used for active trading decisions.
"""

import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
ML_CONFIG_FILE = "config/ml_config.json"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]


def load_ml_config():
    """Load ML configuration"""
    try:
        if os.path.exists(ML_CONFIG_FILE):
            with open(ML_CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.error(f"ML config file not found: {ML_CONFIG_FILE}")
            return None
    except Exception as e:
        logger.error(f"Error loading ML config: {e}")
        return None


def check_model_files():
    """Check which timeframe model files exist"""
    timeframe_counts = {tf: 0 for tf in TIMEFRAMES}
    
    for root, dirs, files in os.walk("ml_models"):
        for file in files:
            if file.endswith("_model.h5"):
                # Extract timeframe from filename
                for tf in TIMEFRAMES:
                    if f"_{tf}_" in file:
                        timeframe_counts[tf] += 1
                        break
    
    return timeframe_counts


def check_configured_timeframes(config):
    """Check which timeframes are configured for trading"""
    if not config or "models" not in config:
        return {}
    
    timeframe_usage = {tf: [] for tf in TIMEFRAMES}
    
    for pair, pair_config in config["models"].items():
        if pair_config.get("enabled", False):
            timeframe = pair_config.get("timeframe")
            if timeframe in TIMEFRAMES:
                timeframe_usage[timeframe].append(pair)
    
    return timeframe_usage


def check_ensemble_usage(config):
    """Check if ensembles are being used"""
    if not config or "models" not in config:
        return False
    
    ensembles_used = []
    
    for pair, pair_config in config["models"].items():
        if pair_config.get("enabled", False) and pair_config.get("use_ensemble", False):
            ensembles_used.append(pair)
    
    return ensembles_used


def main():
    """Main function"""
    # Load ML config
    config = load_ml_config()
    if not config:
        return 1
    
    # Check if ML trading is enabled
    logger.info(f"ML trading enabled: {config.get('enabled', False)}")
    logger.info(f"Sandbox mode: {config.get('sandbox', True)}")
    
    # Check which timeframe model files exist
    timeframe_counts = check_model_files()
    logger.info("\nModel files by timeframe:")
    for tf, count in timeframe_counts.items():
        logger.info(f"  {tf}: {count} models")
    
    # Check which timeframes are configured for trading
    timeframe_usage = check_configured_timeframes(config)
    logger.info("\nTimeframes configured for trading:")
    for tf, pairs in timeframe_usage.items():
        if pairs:
            logger.info(f"  {tf}: {len(pairs)} pairs - {', '.join(pairs)}")
    
    # Check ensemble usage
    ensembles_used = check_ensemble_usage(config)
    logger.info(f"\nEnsemble models used for {len(ensembles_used)} pairs: {', '.join(ensembles_used) if ensembles_used else 'None'}")
    
    # Check specialized models (entry, exit, etc.)
    logger.info("\nSpecialized model types:")
    specialized_types = set()
    for pair, pair_config in config.get("models", {}).items():
        if not pair_config.get("enabled", False):
            continue
        
        if "specialized_models" in pair_config:
            for model_type, model_config in pair_config["specialized_models"].items():
                if model_config.get("enabled", False):
                    specialized_types.add(model_type)
    
    for model_type in specialized_types:
        logger.info(f"  {model_type}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())