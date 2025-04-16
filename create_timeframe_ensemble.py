#!/usr/bin/env python3
"""
Create Timeframe Ensemble Model

This script creates an ensemble model that combines signals from multiple timeframes
(15m, 1h, 4h, 1d) for more robust trading decisions.

Usage:
    python create_timeframe_ensemble.py --pair BTC/USD
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ensemble_creation.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_WEIGHTS_DIR = "model_weights"
ENSEMBLE_DIR = "ensemble_models"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
TIMEFRAMES = ['15m', '1h', '4h', '1d']
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [MODEL_WEIGHTS_DIR, ENSEMBLE_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create ensemble model combining multiple timeframes")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help=f"Trading pair to create ensemble for (options: {', '.join(ALL_PAIRS)})")
    parser.add_argument("--timeframes", type=str, default="all",
                        help=f"Comma-separated list of timeframes (options: {', '.join(TIMEFRAMES)} or 'all')")
    parser.add_argument("--update_ml_config", action="store_true", default=True,
                        help="Update ML configuration with ensemble model")
    return parser.parse_args()

def load_ml_config() -> Dict:
    """Load ML configuration"""
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded ML configuration from {ML_CONFIG_PATH}")
                return config
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
    
    logger.warning(f"ML configuration not found at {ML_CONFIG_PATH}")
    return {"models": {}, "global_settings": {}}

def find_available_models(pair: str, timeframes: List[str], config: Dict) -> Dict[str, str]:
    """Find available models for the pair and timeframes"""
    available_models = {}
    
    if "models" not in config:
        logger.warning("No models in configuration")
        return available_models
    
    # Search for models in configuration
    for key, model_config in config["models"].items():
        if model_config.get("pair") == pair and model_config.get("timeframe") in timeframes:
            timeframe = model_config.get("timeframe")
            model_path = model_config.get("model_path")
            
            # Check if model file exists
            if os.path.exists(model_path):
                available_models[timeframe] = model_path
                logger.info(f"Found model for {pair} ({timeframe}): {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    return available_models

def load_component_models(models_dict: Dict[str, str]) -> Dict[str, Model]:
    """Load component models"""
    loaded_models = {}
    
    for timeframe, model_path in models_dict.items():
        try:
            model = load_model(model_path)
            loaded_models[timeframe] = model
            logger.info(f"Loaded model for timeframe {timeframe}")
        except Exception as e:
            logger.error(f"Error loading model for timeframe {timeframe}: {e}")
    
    return loaded_models

def create_ensemble_model(component_models: Dict[str, Model]) -> Model:
    """Create ensemble model that combines signals from multiple timeframes"""
    logger.info("Creating ensemble model...")
    
    # Each component model outputs probability distribution over 5 classes
    output_layers = []
    
    # Add output layers from each component model
    for timeframe, model in sorted(component_models.items()):
        # Get the output layer from component model
        output_layer = model.output
        
        # Rename to avoid conflicts
        output_layer_renamed = tf.keras.layers.Lambda(
            lambda x: x, name=f"output_{timeframe}"
        )(output_layer)
        
        output_layers.append(output_layer_renamed)
    
    # Combine outputs
    if len(output_layers) > 1:
        merged = Concatenate()(output_layers)
    else:
        merged = output_layers[0]
    
    # Add dense layers to merge predictions
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(32, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    
    # Output layer (5 classes)
    ensemble_output = Dense(5, activation='softmax', name='ensemble_output')(merged)
    
    # Create ensemble model
    inputs = [model.input for model in component_models.values()]
    ensemble_model = Model(inputs=inputs, outputs=ensemble_output)
    
    # Compile ensemble model
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Ensemble model created successfully")
    return ensemble_model

def save_ensemble_model(pair: str, ensemble_model: Model) -> str:
    """Save ensemble model to file"""
    pair_clean = pair.replace("/", "_").lower()
    ensemble_path = f"{ENSEMBLE_DIR}/ensemble_{pair_clean}_model.h5"
    
    try:
        ensemble_model.save(ensemble_path)
        logger.info(f"Saved ensemble model to {ensemble_path}")
        return ensemble_path
    except Exception as e:
        logger.error(f"Error saving ensemble model: {e}")
        return ""

def update_ml_config_with_ensemble(
    pair: str, 
    ensemble_path: str, 
    component_timeframes: List[str]
) -> bool:
    """Update ML configuration with ensemble model"""
    logger.info("Updating ML configuration with ensemble model...")
    
    # Load existing config
    config = load_ml_config()
    
    # Update global settings if not present
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    # Set global settings
    config["global_settings"]["base_leverage"] = 5.0
    config["global_settings"]["max_leverage"] = 75.0
    config["global_settings"]["confidence_threshold"] = 0.65
    config["global_settings"]["risk_percentage"] = 0.20
    config["global_settings"]["max_portfolio_risk"] = 0.25
    
    # Ensure models section exists
    if "models" not in config:
        config["models"] = {}
    
    # Add ensemble model
    model_key = f"{pair}_ensemble"
    
    config["models"][model_key] = {
        "pair": pair,
        "timeframe": "ensemble",
        "model_type": "timeframe_ensemble",
        "model_path": ensemble_path,
        "component_timeframes": component_timeframes,
        "accuracy": 0.0,  # Will be updated after evaluation
        "direction_accuracy": 0.0,  # Will be updated after evaluation
        "win_rate": 0.0,  # Will be updated after evaluation
        "sharpe_ratio": 0.0,  # Will be updated after evaluation
        "total_return": 0.0,  # Will be updated after evaluation
        "base_leverage": 5.0,
        "max_leverage": 75.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "active": True,
        "preferred": True,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with ensemble model for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine timeframes
    if args.timeframes.lower() == "all":
        timeframes = TIMEFRAMES
    else:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("CREATE TIMEFRAME ENSEMBLE MODEL")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Update ML Config: {args.update_ml_config}")
    logger.info("=" * 80 + "\n")
    
    # Load ML configuration
    config = load_ml_config()
    
    # Find available models for the pair and timeframes
    available_models = find_available_models(args.pair, timeframes, config)
    
    if not available_models:
        logger.error(f"No models found for {args.pair} with timeframes {', '.join(timeframes)}")
        logger.error("Please train models for these timeframes first")
        return 1
    
    logger.info(f"Found {len(available_models)} models for {args.pair}:")
    for timeframe in available_models:
        logger.info(f"  - {timeframe}")
    
    # Load component models
    component_models = load_component_models(available_models)
    
    if len(component_models) < 1:
        logger.error("No component models could be loaded")
        return 1
    
    # Create ensemble model
    ensemble_model = create_ensemble_model(component_models)
    
    # Save ensemble model
    ensemble_path = save_ensemble_model(args.pair, ensemble_model)
    
    if not ensemble_path:
        logger.error("Failed to save ensemble model")
        return 1
    
    # Update ML config
    if args.update_ml_config:
        update_ml_config_with_ensemble(
            args.pair, 
            ensemble_path, 
            list(component_models.keys())
        )
    
    # Print success message
    logger.info("\nEnsemble model creation completed successfully!")
    logger.info(f"Created ensemble model for {args.pair} with {len(component_models)} timeframes:")
    for timeframe in component_models.keys():
        logger.info(f"  - {timeframe}")
    logger.info(f"Model saved to {ensemble_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())