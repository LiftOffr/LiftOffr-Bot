#!/usr/bin/env python3
"""
Apply Optimized Settings

This script applies the optimized settings from the risk-aware optimization process.
It updates all configuration files with the optimized parameters.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apply optimized settings")
    parser.add_argument("--pairs", type=str, default="SOL/USD,BTC/USD,ETH/USD",
                     help="Comma-separated list of pairs")
    parser.add_argument("--risk-level", type=str, default="balanced",
                     choices=["conservative", "balanced", "aggressive", "ultra"],
                     help="Risk level")
    return parser.parse_args()

def load_optimized_parameters(pair: str, risk_level: str) -> Dict[str, Any]:
    """Load optimized parameters from file"""
    filename = f"optimization_results/{pair.replace('/', '_')}_{risk_level}_optimized.json"
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    else:
        logger.warning(f"Optimization results not found for {pair}: {filename}")
        return {}

def update_risk_config(pairs: List[str], risk_level: str):
    """Update risk configuration with optimized parameters"""
    logger.info("Updating risk configuration...")
    
    if os.path.exists(RISK_CONFIG_FILE):
        with open(RISK_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"Risk configuration file not found: {RISK_CONFIG_FILE}")
        return
    
    # Update pair-specific settings
    for pair in pairs:
        optimized = load_optimized_parameters(pair, risk_level)
        
        if not optimized:
            continue
        
        # Update pair-specific configuration
        if pair in config["pair_specific"]:
            if "optimized_parameters" in optimized:
                config["pair_specific"][pair]["max_leverage"] = optimized["optimized_parameters"]["leverage"]
                
                # Add volatility settings
                if "performance" in optimized:
                    win_rate = optimized["performance"].get("win_rate", 0.5)
                    volatility_adjustment = 1.0 + (win_rate - 0.5) * 2  # Scale between 0-2
                    config["pair_specific"][pair]["volatility_adjustment"] = volatility_adjustment
    
    # Save updated configuration
    with open(RISK_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Risk configuration updated")

def update_ml_config(pairs: List[str], risk_level: str):
    """Update ML configuration with optimized parameters"""
    logger.info("Updating ML configuration...")
    
    if os.path.exists(ML_CONFIG_FILE):
        with open(ML_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"ML configuration file not found: {ML_CONFIG_FILE}")
        return
    
    # Update pair-specific settings
    for pair in pairs:
        optimized = load_optimized_parameters(pair, risk_level)
        
        if not optimized or "ml_parameters" not in optimized:
            continue
        
        # Update pair-specific configuration
        if pair in config["pairs"]:
            ml_params = optimized["ml_parameters"]
            
            # Update parameters for all model types
            for model_type in ["tcn", "lstm", "gru", "transformer"]:
                if model_type in config["pairs"][pair]["models"]:
                    config["pairs"][pair]["models"][model_type]["learning_rate"] = ml_params.get("learning_rate", 0.001)
                    config["pairs"][pair]["models"][model_type]["batch_size"] = ml_params.get("batch_size", 64)
                    
                    # Add dropout if available
                    if "dropout" in ml_params:
                        config["pairs"][pair]["models"][model_type]["dropout"] = ml_params["dropout"]
            
            # Update performance metrics if available
            if "performance" in optimized:
                perf = optimized["performance"]
                accuracy = perf.get("accuracy", 0.8)
                
                # Adjust confidence threshold based on model accuracy
                conf_threshold = max(0.55, min(0.85, accuracy - 0.2))
                config["pairs"][pair]["confidence_threshold"] = conf_threshold
    
    # Save updated configuration
    with open(ML_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("ML configuration updated")

def update_dynamic_params_config(risk_level: str):
    """Update dynamic parameters configuration"""
    logger.info("Updating dynamic parameters configuration...")
    
    if os.path.exists(DYNAMIC_PARAMS_CONFIG_FILE):
        with open(DYNAMIC_PARAMS_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"Dynamic parameters configuration file not found: {DYNAMIC_PARAMS_CONFIG_FILE}")
        return
    
    # Adjust learning rate based on risk level
    if risk_level == "conservative":
        config["dynamic_parameters"]["learning_rate"] = 0.05
    elif risk_level == "balanced":
        config["dynamic_parameters"]["learning_rate"] = 0.1
    elif risk_level == "aggressive":
        config["dynamic_parameters"]["learning_rate"] = 0.15
    elif risk_level == "ultra":
        config["dynamic_parameters"]["learning_rate"] = 0.2
    
    # Save updated configuration
    with open(DYNAMIC_PARAMS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Dynamic parameters configuration updated")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse pairs
    pairs = args.pairs.split(",")
    
    # Update configurations
    update_risk_config(pairs, args.risk_level)
    update_ml_config(pairs, args.risk_level)
    update_dynamic_params_config(args.risk_level)
    
    logger.info("All configurations updated with optimized settings")

if __name__ == "__main__":
    main()