#!/usr/bin/env python3
"""
Apply Dynamic Parameters to Live Trading Environment

This script applies the optimized dynamic parameters to the live trading environment:
1. Loads optimized parameters from the configuration
2. Updates the .env file with new parameter values
3. Notifies strategies to use dynamic parameter adjustments

Usage:
    python apply_dynamic_parameters.py [--pair PAIR]
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import the dynamic parameter optimizer
from dynamic_parameter_optimizer import DynamicParameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dynamic_parameters.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_optimized_parameters(pair: Optional[str] = None) -> Dict[str, Any]:
    """
    Load optimized parameters from configuration.
    
    Args:
        pair: Specific trading pair to load (None = all pairs)
        
    Returns:
        Dictionary of optimized parameters
    """
    params_path = "config/optimized_params.json"
    
    if not os.path.exists(params_path):
        logger.error(f"Optimized parameters file not found: {params_path}")
        return {}
    
    try:
        with open(params_path, 'r') as f:
            all_params = json.load(f)
        
        if pair:
            # Return parameters for specific pair only
            if pair in all_params:
                logger.info(f"Loaded optimized parameters for {pair}")
                return {pair: all_params[pair]}
            else:
                logger.error(f"No optimized parameters found for {pair}")
                return {}
        else:
            # Return parameters for all pairs
            logger.info(f"Loaded optimized parameters for {len(all_params)} pairs")
            return all_params
    except Exception as e:
        logger.error(f"Error loading optimized parameters: {e}")
        return {}

def update_env_file(optimized_params: Dict[str, Any]) -> bool:
    """
    Update .env file with optimized parameters.
    
    Args:
        optimized_params: Dictionary of optimized parameters by pair
        
    Returns:
        True if successful, False otherwise
    """
    env_path = ".env"
    env_lines = []
    
    # Read existing .env file
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            return False
    
    # Process existing lines and remove old optimized parameters
    processed_lines = []
    skip_patterns = ["OPTIMIZED_", "RISK_PERCENTAGE_", "LEVERAGE_", "CONFIDENCE_THRESHOLD_"]
    
    for line in env_lines:
        skip = False
        for pattern in skip_patterns:
            if pattern in line and "=" in line:
                skip = True
                break
        
        if not skip:
            processed_lines.append(line)
    
    # Add header for optimized parameters
    processed_lines.append("\n# Dynamic optimized parameters - generated on {}\n".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # Add optimization date
    processed_lines.append("OPTIMIZED_DATE=\"{}\"\n".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # Add parameters for each pair
    for pair, params in optimized_params.items():
        pair_key = pair.replace("/", "")
        
        processed_lines.append(f"# Optimized parameters for {pair}\n")
        processed_lines.append(f"RISK_PERCENTAGE_{pair_key}={params.get('risk_percentage', 0.2)}\n")
        processed_lines.append(f"BASE_LEVERAGE_{pair_key}={params.get('base_leverage', 20.0)}\n")
        processed_lines.append(f"MAX_LEVERAGE_{pair_key}={params.get('max_leverage', 125.0)}\n")
        processed_lines.append(f"CONFIDENCE_THRESHOLD_{pair_key}={params.get('confidence_threshold', 0.65)}\n")
        processed_lines.append(f"SIGNAL_STRENGTH_THRESHOLD_{pair_key}={params.get('signal_strength_threshold', 0.6)}\n")
        
        # Add strategy weights if available
        if "strategy_weights" in params:
            weights = params["strategy_weights"]
            processed_lines.append(f"ARIMA_WEIGHT_{pair_key}={weights.get('arima', 0.3)}\n")
            processed_lines.append(f"ADAPTIVE_WEIGHT_{pair_key}={weights.get('adaptive', 0.7)}\n")
        
        # Add trailing stop and exit multipliers
        processed_lines.append(f"TRAILING_STOP_ATR_MULTIPLIER_{pair_key}={params.get('trailing_stop_atr_multiplier', 3.0)}\n")
        processed_lines.append(f"EXIT_MULTIPLIER_{pair_key}={params.get('exit_multiplier', 1.5)}\n")
        
        processed_lines.append("\n")
    
    # Add the dynamic parameter flag
    processed_lines.append("# Enable dynamic parameter adjustment based on confidence\n")
    processed_lines.append("USE_DYNAMIC_PARAMETERS=true\n\n")
    
    # Write updated .env file
    try:
        with open(env_path, 'w') as f:
            f.writelines(processed_lines)
        
        logger.info(f"Updated .env file with dynamic parameters for {len(optimized_params)} pairs")
        return True
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        return False

def create_dynamic_parameter_examples() -> None:
    """Create example files showing dynamic parameter calculations"""
    optimizer = DynamicParameterOptimizer()
    
    optimized_params = load_optimized_parameters()
    if not optimized_params:
        logger.error("No optimized parameters available")
        return
    
    example_dir = "dynamic_parameter_examples"
    os.makedirs(example_dir, exist_ok=True)
    
    for pair, params in optimized_params.items():
        # Generate examples with different confidence levels and signal strengths
        confidence_levels = [0.6, 0.7, 0.8, 0.9, 0.95]
        signal_strengths = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        examples = []
        
        for confidence in confidence_levels:
            for signal in signal_strengths:
                # Use the DynamicParameterOptimizer to calculate parameters
                dynamic_params = optimizer.get_dynamic_parameters(
                    pair=pair,
                    confidence=confidence,
                    signal_strength=signal,
                    volatility=0.03  # Example volatility
                )
                
                examples.append({
                    "confidence": confidence,
                    "signal_strength": signal,
                    "dynamic_parameters": dynamic_params
                })
        
        # Save examples to file
        pair_safe = pair.replace("/", "_")
        example_file = os.path.join(example_dir, f"{pair_safe}_examples.json")
        
        with open(example_file, 'w') as f:
            json.dump({
                "pair": pair,
                "base_parameters": params,
                "examples": examples
            }, f, indent=2)
        
        logger.info(f"Created dynamic parameter examples for {pair} at {example_file}")
        
        # Generate a human-readable summary
        summary_file = os.path.join(example_dir, f"{pair_safe}_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Dynamic Parameter Examples for {pair}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Base Parameters:\n")
            f.write(f"  Risk Percentage: {params.get('risk_percentage', 0.2):.2f}\n")
            f.write(f"  Base Leverage: {params.get('base_leverage', 20.0):.1f}\n")
            f.write(f"  Max Leverage: {params.get('max_leverage', 125.0):.1f}\n")
            f.write(f"  Confidence Threshold: {params.get('confidence_threshold', 0.65):.2f}\n")
            f.write(f"  Signal Strength Threshold: {params.get('signal_strength_threshold', 0.6):.2f}\n")
            f.write("\n")
            
            f.write("Dynamic Parameter Examples:\n")
            f.write("-" * 40 + "\n")
            f.write("Conf | Signal | Risk % | Leverage | Trail Stop\n")
            f.write("-" * 40 + "\n")
            
            for example in examples:
                conf = example["confidence"]
                signal = example["signal_strength"]
                params = example["dynamic_parameters"]
                
                f.write(f"{conf:.2f} | {signal:.2f}  | {params['risk_percentage']:.2f}   | "
                       f"{params['leverage']:.1f}      | {params['trailing_stop_atr_multiplier']:.2f}\n")
        
        logger.info(f"Created dynamic parameter summary for {pair} at {summary_file}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apply dynamic parameters to trading environment")
    parser.add_argument("--pair", type=str, help="Specific trading pair to apply parameters for")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    logger.info("Applying dynamic parameters to trading environment")
    
    # Load optimized parameters
    optimized_params = load_optimized_parameters(args.pair)
    
    if not optimized_params:
        logger.error("No optimized parameters to apply")
        return 1
    
    # Update .env file
    if not update_env_file(optimized_params):
        return 1
    
    # Create example files
    create_dynamic_parameter_examples()
    
    logger.info("Dynamic parameters successfully applied")
    print("\n" + "=" * 80)
    print("Dynamic parameters successfully applied to trading environment")
    print(f"Applied to {len(optimized_params)} pairs: {', '.join(optimized_params.keys())}")
    print("=" * 80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())