#!/usr/bin/env python3
"""
Update ML Config

This script updates the ML configuration file to use the newly trained models.
It looks for trained models in the models directory and updates the ml_config.json file.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_config_update.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
ML_CONFIG_PATH = "ml_config.json"
MODEL_TYPES = ["tcn", "lstm", "gru", "transformer", "cnn", "bilstm", "attention", "hybrid"]
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]

def backup_config():
    """Create a backup of the current ML config"""
    if os.path.exists(ML_CONFIG_PATH):
        backup_path = f"{ML_CONFIG_PATH}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        try:
            with open(ML_CONFIG_PATH, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    else:
        logger.warning(f"ML config not found: {ML_CONFIG_PATH}")
        return False

def find_trained_models():
    """Find all trained models in the models directories"""
    trained_models = {}
    
    for model_type in MODEL_TYPES:
        model_dir = os.path.join(MODELS_DIR, model_type)
        if not os.path.exists(model_dir):
            continue
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.h5'):
                pair = filename.split('_')[0]
                if pair not in trained_models:
                    trained_models[pair] = []
                trained_models[pair].append(model_type)
    
    return trained_models

def find_ensemble_configs():
    """Find all ensemble configurations"""
    ensemble_configs = {}
    ensemble_dir = os.path.join(MODELS_DIR, "ensemble")
    
    if os.path.exists(ensemble_dir):
        for filename in os.listdir(ensemble_dir):
            if filename.endswith('_ensemble.json'):
                pair = filename.split('_')[0]
                
                try:
                    with open(os.path.join(ensemble_dir, filename), 'r') as f:
                        ensemble_configs[pair] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load ensemble config {filename}: {e}")
    
    return ensemble_configs

def load_current_config():
    """Load the current ML config"""
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ML config: {e}")
            return {}
    else:
        logger.warning(f"ML config not found: {ML_CONFIG_PATH}")
        return {}

def update_config(config, trained_models, ensemble_configs):
    """Update the ML config with trained models"""
    if not config:
        # Create a new config if none exists
        config = {
            "version": "2.0",
            "global_settings": {
                "use_ml": True,
                "confidence_threshold": 0.65,
                "signal_strength_threshold": 0.7,
                "max_position_size_pct": 25,
                "base_leverage": 20,
                "max_leverage": 125,
                "auto_prune_models": True,
                "auto_optimize_ensembles": True
            },
            "model_settings": {}
        }
    
    # Update model settings for each pair
    for pair in trained_models:
        if pair not in config["model_settings"]:
            config["model_settings"][pair] = {}
        
        pair_config = config["model_settings"][pair]
        
        # Update model types
        pair_config["model_types"] = trained_models[pair]
        
        # Update ensemble config if available
        if pair in ensemble_configs:
            pair_config["ensemble"] = ensemble_configs[pair]
        
        # Default settings if not already present
        if "confidence_threshold" not in pair_config:
            pair_config["confidence_threshold"] = 0.65
        
        if "signal_strength_threshold" not in pair_config:
            pair_config["signal_strength_threshold"] = 0.7
        
        if "position_sizing" not in pair_config:
            pair_config["position_sizing"] = {
                "base_leverage": 20,
                "max_leverage": 125,
                "confidence_threshold": 0.65,
                "high_confidence_threshold": 0.85,
                "scaling_factor": 1.5
            }
    
    return config

def save_config(config):
    """Save the updated ML config"""
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"ML config saved to {ML_CONFIG_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to save ML config: {e}")
        return False

def main():
    """Main function"""
    # Backup existing config
    backup_config()
    
    # Find trained models and ensemble configs
    trained_models = find_trained_models()
    ensemble_configs = find_ensemble_configs()
    
    if not trained_models:
        logger.error("No trained models found")
        return False
    
    logger.info(f"Found trained models for pairs: {', '.join(trained_models.keys())}")
    
    # Load current config
    config = load_current_config()
    
    # Update config
    updated_config = update_config(config, trained_models, ensemble_configs)
    
    # Save updated config
    save_config(updated_config)
    
    return True

if __name__ == "__main__":
    main()