#!/usr/bin/env python3

"""
Update Trading Configuration

This script updates the trading configuration files with the optimized settings
from our enhanced training process. It configures risk parameters, strategy settings,
and dynamic parameter adjustment based on the trained ML models.

Usage:
    python update_trading_config.py [--pairs PAIRS]
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Update Trading Configuration")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use more aggressive risk settings")
    return parser.parse_args()

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def update_risk_config(pairs, ml_config, aggressive=False):
    """
    Update risk configuration based on ML model metrics
    
    Args:
        pairs (List[str]): List of trading pairs
        ml_config (Dict[str, Any]): ML configuration
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        Dict[str, Any]: Updated risk configuration
    """
    # Load existing risk config or create new one
    risk_config = load_file(RISK_CONFIG_FILE, {"pairs": {}})
    
    # Calculate average metrics across all pairs
    accuracy_sum = 0.0
    win_rate_sum = 0.0
    sharpe_ratio_sum = 0.0
    drawdown_sum = 0.0
    
    for pair in pairs:
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        accuracy_sum += pair_config.get("accuracy", 0.0)
        win_rate_sum += pair_config.get("win_rate", 0.0)
        sharpe_ratio_sum += pair_config.get("sharpe_ratio", 0.0)
        drawdown_sum += pair_config.get("max_drawdown", 0.0)
    
    avg_accuracy = accuracy_sum / len(pairs) if pairs else 0.0
    avg_win_rate = win_rate_sum / len(pairs) if pairs else 0.0
    avg_sharpe_ratio = sharpe_ratio_sum / len(pairs) if pairs else 0.0
    avg_drawdown = drawdown_sum / len(pairs) if pairs else 0.0
    
    # Set global risk parameters
    risk_factor = 1.0
    if aggressive:
        risk_factor = 1.2
    
    # Higher accuracy = tighter risk controls
    max_acceptable_drawdown = min(0.15, avg_drawdown * 3.0)
    daily_var_95 = avg_drawdown / 2.0
    
    # Update global risk parameters
    risk_config["global"] = {
        "max_acceptable_drawdown": max_acceptable_drawdown * risk_factor,
        "daily_var_95": daily_var_95,
        "position_correlation_limit": 0.7,
        "max_open_positions": 12,  # Up to 2 positions per pair
        "max_leverage_total": 100.0 * risk_factor,
        "margin_reserve_percentage": 0.2 / risk_factor,  # Lower reserve for aggressive
        "last_updated": datetime.now().isoformat()
    }
    
    # Add parameters for flash crash protection
    risk_config["flash_crash"] = {
        "enabled": True,
        "trigger_threshold": -0.05,
        "max_acceptable_loss": -0.15,
        "recovery_threshold": 0.03,
        "cooldown_period": 6,  # 6 hours
        "protection_levels": [
            {"threshold": -0.05, "action": "reduce_leverage", "factor": 0.5},
            {"threshold": -0.1, "action": "hedge", "factor": 0.25},
            {"threshold": -0.15, "action": "close_all", "factor": 1.0}
        ]
    }
    
    # Update pair-specific risk parameters
    for pair in pairs:
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Skip if no ML configuration
        if not pair_config:
            continue
        
        # Get metrics
        accuracy = pair_config.get("accuracy", 0.0)
        win_rate = pair_config.get("win_rate", 0.0)
        sharpe_ratio = pair_config.get("sharpe_ratio", 0.0)
        max_drawdown = pair_config.get("max_drawdown", 0.0)
        
        # Determine risk level
        if accuracy >= 0.97 and sharpe_ratio >= 3.0:
            risk_level = "Low"
        elif accuracy >= 0.94 and sharpe_ratio >= 2.0:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        if aggressive and risk_level != "High":
            risk_level = "Low" if risk_level == "Low" else "Medium"
        
        # Set risk parameters based on risk level and metrics
        risk_params = {
            "risk_level": risk_level,
            "max_drawdown_historical": max_drawdown,
            "max_leverage": 125.0 if risk_level == "Low" else (100.0 if risk_level == "Medium" else 75.0),
            "position_size_limit": 0.3 if risk_level == "Low" else (0.25 if risk_level == "Medium" else 0.2),
            "stop_loss_multiplier": 1.5 if risk_level == "Low" else (1.2 if risk_level == "Medium" else 1.0),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add to risk config
        if "pairs" not in risk_config:
            risk_config["pairs"] = {}
        
        risk_config["pairs"][pair] = risk_params
    
    # Save risk config
    save_file(RISK_CONFIG_FILE, risk_config)
    
    logger.info(f"Updated risk configuration for {len(pairs)} pairs")
    
    return risk_config

def update_dynamic_params_config(pairs, ml_config, aggressive=False):
    """
    Update dynamic parameters configuration based on ML model metrics
    
    Args:
        pairs (List[str]): List of trading pairs
        ml_config (Dict[str, Any]): ML configuration
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        Dict[str, Any]: Updated dynamic parameters configuration
    """
    # Load existing dynamic params config or create new one
    dynamic_params_config = load_file(DYNAMIC_PARAMS_CONFIG_FILE, {})
    
    # Set risk factor based on aggressiveness
    risk_factor = 1.0
    if aggressive:
        risk_factor = 1.2
    
    # Set global dynamic parameters
    base_leverage = 20.0 * risk_factor
    risk_percentage = 0.2 * risk_factor
    
    dynamic_params_config["global"] = {
        "base_leverage": base_leverage,
        "max_leverage": 125.0,
        "risk_percentage": risk_percentage,
        "confidence_threshold": 0.65 / risk_factor,  # Lower threshold for aggressive
        "dynamic_leverage_scaling": True,
        "dynamic_position_sizing": True,
        "dynamic_stop_loss": True,
        "dynamic_take_profit": True,
        "last_updated": datetime.now().isoformat()
    }
    
    # Set leverage scaling parameters
    dynamic_params_config["leverage_scaling"] = {
        "min_confidence": 0.65 / risk_factor,
        "max_confidence": 0.95,
        "min_leverage": base_leverage,
        "max_leverage": 125.0,
        "scaling_formula": "linear",  # or "quadratic", "exponential"
        "confidence_weight": 0.7,
        "volatility_weight": 0.2,
        "trend_weight": 0.1
    }
    
    # Set position sizing parameters
    dynamic_params_config["position_sizing"] = {
        "base_risk_percentage": risk_percentage,
        "max_risk_percentage": min(0.3, risk_percentage * 1.5),
        "min_risk_percentage": 0.05,
        "kelly_criterion_weight": 0.5,
        "max_position_size_percentage": 0.3 * risk_factor,
        "adjustment_factors": {
            "high_confidence": 1.2 * risk_factor,
            "medium_confidence": 1.0 * risk_factor,
            "low_confidence": 0.8,
            "high_volatility": 0.8,
            "medium_volatility": 1.0,
            "low_volatility": 1.2 * risk_factor
        }
    }
    
    # Update pair-specific parameters
    dynamic_params_config["pairs"] = {}
    
    for pair in pairs:
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        # Skip if no ML configuration
        if not pair_config:
            continue
        
        # Get metrics
        accuracy = pair_config.get("accuracy", 0.0)
        win_rate = pair_config.get("win_rate", 0.0)
        max_drawdown = pair_config.get("max_drawdown", 0.0)
        
        # Calculate Kelly criterion
        # f* = (bp - q) / b where p is win rate, q is 1-p, and b is avg win/loss ratio
        avg_win_loss_ratio = 2.0  # Assuming 2:1 reward-to-risk ratio
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly = max(0.05, min(0.3, kelly))  # Constrain between 5% and 30%
        
        if aggressive:
            kelly = min(0.3, kelly * 1.2)
        
        # Calculate optimal risk percentage (half Kelly for safety)
        optimal_risk = kelly / 2
        
        # Calculate risk-adjusted leverage
        risk_adjusted_leverage = min(125.0, 100.0 * (1 - max_drawdown))
        
        if aggressive:
            risk_adjusted_leverage = min(125.0, risk_adjusted_leverage * 1.25)
        
        # Set pair-specific parameters
        dynamic_params_config["pairs"][pair] = {
            "kelly_criterion": kelly,
            "optimal_risk_percentage": optimal_risk,
            "risk_adjusted_leverage": risk_adjusted_leverage,
            "adjustment_factor": 1.0 * risk_factor,
            "last_updated": datetime.now().isoformat()
        }
    
    # Save dynamic parameters config
    save_file(DYNAMIC_PARAMS_CONFIG_FILE, dynamic_params_config)
    
    logger.info(f"Updated dynamic parameters configuration for {len(pairs)} pairs")
    
    return dynamic_params_config

def update_integrated_risk_config(pairs, ml_config, risk_config, aggressive=False):
    """
    Update integrated risk configuration based on ML model metrics and risk config
    
    Args:
        pairs (List[str]): List of trading pairs
        ml_config (Dict[str, Any]): ML configuration
        risk_config (Dict[str, Any]): Risk configuration
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        Dict[str, Any]: Updated integrated risk configuration
    """
    # Load existing integrated risk config or create new one
    integrated_risk_config = load_file(INTEGRATED_RISK_CONFIG_FILE, {})
    
    # Build correlation matrix
    correlation_matrix = {}
    for pair1 in pairs:
        correlation_matrix[pair1] = {}
        for pair2 in pairs:
            # Set default correlation
            if pair1 == pair2:
                correlation = 1.0
            elif "BTC" in pair1 and "BTC" in pair2:
                correlation = 0.9
            elif "ETH" in pair1 and "ETH" in pair2:
                correlation = 0.85
            elif (("SOL" in pair1 or "ADA" in pair1 or "DOT" in pair1) and 
                  ("SOL" in pair2 or "ADA" in pair2 or "DOT" in pair2)):
                correlation = 0.75
            else:
                correlation = 0.5
            
            correlation_matrix[pair1][pair2] = correlation
    
    # Calculate aggregate risk score for each pair
    pair_risks = {}
    for pair in pairs:
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        max_drawdown = pair_config.get("max_drawdown", 0.2)
        sharpe_ratio = pair_config.get("sharpe_ratio", 1.0)
        win_rate = pair_config.get("win_rate", 0.5)
        
        # Calculate risk score (lower is better)
        risk_score = max_drawdown * 5 - sharpe_ratio - win_rate
        pair_risks[pair] = risk_score
    
    # Set risk factor based on aggressiveness
    risk_factor = 1.0
    if aggressive:
        risk_factor = 1.2
    
    # Create portfolio diversification rules
    diversification_rules = {
        "max_exposure_per_pair": 0.3 * risk_factor,
        "max_correlation_weight": 0.6 * risk_factor,
        "min_uncorrelated_allocation": 0.2 / risk_factor,
        "correlation_threshold": 0.7
    }
    
    # Build stress testing scenarios
    stress_scenarios = [
        {
            "name": "Market Crash",
            "description": "Sudden market-wide crash of 15-25%",
            "impacts": {pair: {"price_change": -0.2, "volatility_multiplier": 3.0} for pair in pairs},
            "duration": 48,  # 48 hours
            "recovery_pattern": "V-shaped",
            "probability": 0.05
        },
        {
            "name": "Prolonged Bear Market",
            "description": "Extended downtrend with 30-50% decline over weeks",
            "impacts": {pair: {"price_change": -0.4, "volatility_multiplier": 1.5} for pair in pairs},
            "duration": 720,  # 30 days (in hours)
            "recovery_pattern": "U-shaped",
            "probability": 0.15
        },
        {
            "name": "Liquidity Crisis",
            "description": "Sudden drop in liquidity causing slippage and execution issues",
            "impacts": {pair: {"price_change": -0.1, "slippage_multiplier": 5.0, "execution_delay": 10} for pair in pairs},
            "duration": 24,  # 24 hours
            "recovery_pattern": "Volatile",
            "probability": 0.1
        }
    ]
    
    # Get global risk parameters
    global_risk = risk_config.get("global", {})
    max_drawdown_trigger = global_risk.get("max_acceptable_drawdown", 0.15)
    
    # Build risk policies
    risk_policies = {
        "portfolio_protection": {
            "max_drawdown_trigger": max_drawdown_trigger,
            "actions": [
                {"trigger": max_drawdown_trigger, "action": "reduce_leverage", "target": 0.5},
                {"trigger": max_drawdown_trigger + 0.05, "action": "reduce_positions", "target": 0.5},
                {"trigger": max_drawdown_trigger + 0.10, "action": "close_all", "target": 1.0}
            ]
        },
        "volatility_adjustment": {
            "low_volatility": {"leverage_multiplier": 1.2 * risk_factor, "position_size_multiplier": 1.1 * risk_factor},
            "medium_volatility": {"leverage_multiplier": 1.0, "position_size_multiplier": 1.0},
            "high_volatility": {"leverage_multiplier": 0.8, "position_size_multiplier": 0.8}
        },
        "trend_following": {
            "uptrend": {"long_bias": 0.7 * risk_factor, "short_bias": 0.3 / risk_factor},
            "downtrend": {"long_bias": 0.3 / risk_factor, "short_bias": 0.7 * risk_factor},
            "sideways": {"long_bias": 0.5, "short_bias": 0.5}
        }
    }
    
    # Create integrated risk config
    integrated_risk_config = {
        "correlation_matrix": correlation_matrix,
        "pair_risks": pair_risks,
        "diversification_rules": diversification_rules,
        "stress_scenarios": stress_scenarios,
        "risk_policies": risk_policies,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save integrated risk config
    save_file(INTEGRATED_RISK_CONFIG_FILE, integrated_risk_config)
    
    logger.info(f"Updated integrated risk configuration for {len(pairs)} pairs")
    
    return integrated_risk_config

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    logger.info(f"Updating trading configuration for {len(pairs)} pairs")
    
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Load ML config
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    # Update risk configuration
    risk_config = update_risk_config(pairs, ml_config, args.aggressive)
    
    # Update dynamic parameters configuration
    dynamic_params_config = update_dynamic_params_config(pairs, ml_config, args.aggressive)
    
    # Update integrated risk configuration
    integrated_risk_config = update_integrated_risk_config(pairs, ml_config, risk_config, args.aggressive)
    
    logger.info("Trading configuration updated successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())