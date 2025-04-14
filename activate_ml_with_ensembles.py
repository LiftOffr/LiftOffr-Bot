#!/usr/bin/env python3
"""
Activate ML Trading with Ensemble Models

This script activates the ML trading system by:
1. Validating the model files
2. Setting up ensemble model configurations
3. Updating the ML config
4. Starting the trading system in sandbox mode

Usage:
    python activate_ml_with_ensembles.py [--pairs PAIRS] [--sandbox]
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_activation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
MODEL_TYPES = ["lstm", "gru", "transformer", "tcn", "cnn", "bilstm", "attention", "hybrid"]
MODEL_DIRS = {model_type: f"models/{model_type}" for model_type in MODEL_TYPES}
ENSEMBLE_DIR = "models/ensemble"
ML_CONFIG_PATH = "ml_config.json"
CONFIDENCE_THRESHOLD = 0.7

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Activate ML trading with ensemble models")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to activate (default: {' '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    parser.add_argument("--create-ensembles", action="store_true",
                        help="Create ensemble models if they don't exist")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(f"[STEP] {description}")
    
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        
        # Log command output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"[ERR] {line}")
        return None

def check_model_files(pairs: List[str]) -> bool:
    """Check if model files exist for all trading pairs"""
    all_models_exist = True
    
    for pair in pairs:
        missing_models = []
        for model_type in MODEL_TYPES:
            model_dir = MODEL_DIRS[model_type]
            model_path = os.path.join(model_dir, f"{pair}.h5")
            if not os.path.exists(model_path):
                # Check alternative path with model type in filename
                alt_model_path = os.path.join(model_dir, f"{pair}_{model_type}.h5")
                if not os.path.exists(alt_model_path):
                    missing_models.append(model_type)
        
        if missing_models:
            logger.warning(f"Missing models for {pair}: {', '.join(missing_models)}")
            all_models_exist = False
        else:
            logger.info(f"All model types available for {pair}")
    
    return all_models_exist

def create_ensemble_configs(pairs: List[str]) -> bool:
    """Create ensemble configurations for all pairs"""
    success = True
    
    for pair in pairs:
        ensemble_config_path = os.path.join(ENSEMBLE_DIR, f"{pair}_ensemble.json")
        position_sizing_path = os.path.join(ENSEMBLE_DIR, f"{pair}_position_sizing.json")
        
        # Check if ensemble config already exists
        if os.path.exists(ensemble_config_path) and os.path.exists(position_sizing_path):
            logger.info(f"Ensemble configuration already exists for {pair}")
            continue
        
        # Create ensemble config with equal weights for all available models
        available_models = []
        for model_type in MODEL_TYPES:
            model_dir = MODEL_DIRS[model_type]
            model_path = os.path.join(model_dir, f"{pair}.h5")
            alt_model_path = os.path.join(model_dir, f"{pair}_{model_type}.h5")
            
            if os.path.exists(model_path) or os.path.exists(alt_model_path):
                available_models.append(model_type)
        
        if not available_models:
            logger.error(f"No models available for {pair}")
            success = False
            continue
        
        # Create equal weights for all available models
        weights = {model_type: 1.0 / len(available_models) for model_type in available_models}
        
        # Create simple performances dict
        performances = {model_type: {"accuracy": 0.8, "f1_score": 0.8} for model_type in available_models}
        
        # Create ensemble config
        ensemble_config = {
            "pair": pair,
            "weights": weights,
            "performances": performances,
            "model_types": available_models,
            "created_at": datetime.now().isoformat()
        }
        
        # Create position sizing config
        position_sizing = {
            "pair": pair,
            "base_confidence": 0.75,
            "model_confidences": {model_type: 0.8 for model_type in available_models},
            "leverage_scaling": {
                "min_leverage": 1,
                "max_leverage": 5,
                "confidence_threshold": 0.8,
                "scaling_factor": 1.5
            },
            "position_sizing": {
                "base_size": 0.2,
                "min_size": 0.05,
                "max_size": 0.5,
                "confidence_scaling": True
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Ensure ensemble directory exists
        os.makedirs(ENSEMBLE_DIR, exist_ok=True)
        
        # Save ensemble config
        try:
            with open(ensemble_config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=4)
            
            logger.info(f"Created ensemble configuration for {pair}")
            
            # Save position sizing config
            with open(position_sizing_path, 'w') as f:
                json.dump(position_sizing, f, indent=4)
            
            logger.info(f"Created position sizing configuration for {pair}")
        
        except Exception as e:
            logger.error(f"Error creating ensemble configurations for {pair}: {e}")
            success = False
    
    return success

def update_ml_config(pairs: List[str]) -> bool:
    """Update ML configuration for ensemble models"""
    try:
        # Check if ML config exists
        if not os.path.exists(ML_CONFIG_PATH):
            logger.info(f"ML config not found, creating new one")
            ml_config = {
                "global_settings": {
                    "prediction_horizon": "8h",
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "use_ensemble_predictions": True,
                    "real_time_inference": True,
                    "continuous_learning": True,
                    "training_priority": "critical"
                },
                "trading_pairs": {},
                "model_settings": {},
                "ensemble_settings": {
                    "use_weighted_ensemble": True,
                    "dynamic_weights": True,
                    "weight_update_interval": "1d",
                    "weight_adaptation_rate": 0.2
                },
                "strategy_integration": {
                    "integrate_arima_adaptive": True,
                    "arima_weight": 0.5,
                    "adaptive_weight": 0.5,
                    "use_combined_signals": True,
                    "signal_priority": "confidence",
                    "signal_threshold": 0.65
                },
                "position_sizing": {
                    "use_dynamic_sizing": True,
                    "max_position_size": 0.5,
                    "min_position_size": 0.1,
                    "base_position_size": 0.2,
                    "risk_factor": 1.0
                },
                "risk_management": {
                    "max_drawdown": 0.2,
                    "stop_loss_multiplier": 1.5,
                    "take_profit_multiplier": 2.0,
                    "trailing_stop_activation": 0.01,
                    "trailing_stop_callback": 0.005
                },
                "updated_at": datetime.now().isoformat()
            }
        else:
            # Load existing ML config
            with open(ML_CONFIG_PATH, 'r') as f:
                ml_config = json.load(f)
            
            # Update timestamp
            ml_config["updated_at"] = datetime.now().isoformat()
            
            # Ensure ensemble settings exist
            if "ensemble_settings" not in ml_config:
                ml_config["ensemble_settings"] = {
                    "use_weighted_ensemble": True,
                    "dynamic_weights": True,
                    "weight_update_interval": "1d",
                    "weight_adaptation_rate": 0.2
                }
            else:
                ml_config["ensemble_settings"]["use_weighted_ensemble"] = True
            
            # Ensure global settings include ensemble configuration
            if "global_settings" not in ml_config:
                ml_config["global_settings"] = {
                    "prediction_horizon": "8h",
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "use_ensemble_predictions": True,
                    "real_time_inference": True,
                    "continuous_learning": True
                }
            else:
                ml_config["global_settings"]["use_ensemble_predictions"] = True
        
        # Configure trading pairs
        if "trading_pairs" not in ml_config:
            ml_config["trading_pairs"] = {}
        
        for pair in pairs:
            if pair not in ml_config["trading_pairs"]:
                ml_config["trading_pairs"][pair] = {
                    "enabled": True,
                    "model_type": "ensemble",
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "timeframe": "1h",
                    "max_leverage": 5,
                    "enable_short_positions": True
                }
            else:
                ml_config["trading_pairs"][pair]["enabled"] = True
                ml_config["trading_pairs"][pair]["model_type"] = "ensemble"
        
        # Configure model settings for all model types
        if "model_settings" not in ml_config:
            ml_config["model_settings"] = {}
        
        for model_type in MODEL_TYPES:
            if model_type not in ml_config["model_settings"]:
                ml_config["model_settings"][model_type] = {
                    "enabled": True,
                    "lookback_period": 80,
                    "epochs": 100,
                    "batch_size": 32,
                    "dropout_rate": 0.3,
                    "validation_split": 0.2,
                    "use_early_stopping": True,
                    "patience": 15
                }
            else:
                ml_config["model_settings"][model_type]["enabled"] = True
        
        # Save updated ML config
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(ml_config, f, indent=4)
        
        # Create backup of ML config
        backup_path = f"{ML_CONFIG_PATH}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        with open(backup_path, 'w') as f:
            json.dump(ml_config, f, indent=4)
        
        logger.info(f"Updated ML configuration. Backup saved to {backup_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False

def activate_ml_trading(pairs: List[str], sandbox: bool = True) -> bool:
    """Activate ML trading for all pairs"""
    cmd = ["python", "activate_ml_trading_across_all_pairs.py"]
    
    if sandbox:
        cmd.append("--sandbox")
    
    if len(pairs) > 0:
        cmd.append("--pairs")
        cmd.extend(pairs)
    
    result = run_command(cmd, "Activating ML trading")
    
    if result:
        logger.info("ML trading activation completed successfully")
        return True
    else:
        logger.error("Failed to activate ML trading")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting ML activation with ensemble models...")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Sandbox mode: {args.sandbox}")
    
    # Step 1: Check model files
    models_exist = check_model_files(args.pairs)
    if not models_exist:
        if args.create_ensembles:
            logger.warning("Some models are missing, but proceeding with available models")
        else:
            logger.error("Missing model files. Please train models first")
            return 1
    
    # Step 2: Create ensemble configurations
    if not create_ensemble_configs(args.pairs):
        logger.error("Failed to create ensemble configurations")
        return 1
    
    # Step 3: Update ML configuration
    if not update_ml_config(args.pairs):
        logger.error("Failed to update ML configuration")
        return 1
    
    # Step 4: Activate ML trading
    if not activate_ml_trading(args.pairs, args.sandbox):
        logger.error("Failed to activate ML trading")
        return 1
    
    logger.info("ML activation with ensemble models completed successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())