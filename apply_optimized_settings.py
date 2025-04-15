#!/usr/bin/env python3

"""
Apply Optimized Settings

This script applies the optimized settings from the enhanced training process
to the actual trading configurations. It updates:

1. ML configuration with optimized model weights and parameters
2. Risk configuration with pair-specific risk parameters
3. Dynamic parameters configuration for leverage and position sizing
4. Integrated risk configuration for portfolio management

The script ensures that all trading settings are aligned with the most
accurate models and highest return strategies while maintaining proper risk management.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directories and file paths
CONFIG_DIR = "config"
ML_MODELS_DIR = "ml_models"
ENSEMBLE_DIR = f"{ML_MODELS_DIR}/ensemble"
OPTIMIZATION_RESULTS_DIR = "optimization_results"
BACKTEST_RESULTS_DIR = "backtest_results"

ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
SANDBOX_PORTFOLIO_FILE = "data/sandbox_portfolio.json"

# Default trading pairs
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Apply Optimized Settings")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--reset-portfolio", action="store_true",
                        help="Reset sandbox portfolio to initial balance")
    parser.add_argument("--initial-balance", type=float, default=20000.0,
                        help="Initial portfolio balance if resetting")
    parser.add_argument("--aggressive", action="store_true",
                        help="Use more aggressive settings")
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

def update_ml_config(pairs, aggressive=False):
    """
    Update ML configuration with optimized settings
    
    Args:
        pairs (list): List of trading pairs
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        bool: Success/failure
    """
    # Load existing ML config
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    # Update for each pair
    updated_pairs = 0
    
    for pair in pairs:
        # Load ensemble weights
        pair_filename = pair.replace('/', '_')
        ensemble_file = f"{ENSEMBLE_DIR}/{pair_filename}_weights.json"
        ensemble_data = load_file(ensemble_file)
        
        # Load backtest results
        backtest_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_backtest.json"
        backtest_results = load_file(backtest_file)
        
        # Load trading params
        params_file = f"{OPTIMIZATION_RESULTS_DIR}/{pair_filename}_trading_params.json"
        trading_params = load_file(params_file)
        
        # Skip if missing data
        if not ensemble_data or not backtest_results or not trading_params:
            logger.warning(f"Missing data for {pair}, skipping")
            continue
        
        # Extract key metrics
        accuracy = ensemble_data.get("accuracy", 0.0)
        total_return = backtest_results.get("total_return_pct", 0.0) / 100.0
        max_drawdown = backtest_results.get("max_drawdown", 0.0)
        win_rate = backtest_results.get("win_rate", 0.0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
        
        # Extract trading parameters
        confidence_threshold = trading_params.get("confidence_threshold", 0.65)
        base_leverage = trading_params.get("base_leverage", 20.0)
        risk_percentage = trading_params.get("risk_percentage", 0.2)
        profit_factor = trading_params.get("profit_factor", 2.0)
        stop_loss_pct = trading_params.get("stop_loss_pct", 0.05)
        
        # Apply aggressive adjustments if enabled
        if aggressive:
            base_leverage = min(125.0, base_leverage * 1.25)
            risk_percentage = min(0.3, risk_percentage * 1.2)
            confidence_threshold = max(0.55, confidence_threshold * 0.9)
        
        # Update pair configuration
        pair_config = {
            "accuracy": accuracy,
            "backtest_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "confidence_threshold": confidence_threshold,
            "base_leverage": base_leverage,
            "max_leverage": min(125.0, base_leverage * 2),
            "risk_percentage": risk_percentage,
            "profit_factor": profit_factor,
            "stop_loss_pct": stop_loss_pct,
            "models": [
                {"type": model["type"], "weight": model["weight"], "file": model["file"]}
                for model in ensemble_data.get("models", [])
            ],
            "last_updated": datetime.now().isoformat()
        }
        
        # Add to ML config
        ml_config["pairs"][pair] = pair_config
        updated_pairs += 1
    
    # Update global settings
    ml_config["global"] = {
        "default_confidence_threshold": 0.65 if not aggressive else 0.6,
        "default_base_leverage": 20.0 if not aggressive else 25.0,
        "default_max_leverage": 125.0,
        "default_risk_percentage": 0.2 if not aggressive else 0.25,
        "enabled_pairs": pairs,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save updated config
    success = save_file(ML_CONFIG_FILE, ml_config)
    
    if success:
        logger.info(f"Updated ML configuration for {updated_pairs} pairs")
    else:
        logger.error("Failed to update ML configuration")
    
    return success

def update_risk_configuration(pairs, aggressive=False):
    """
    Update risk configuration with optimized settings
    
    Args:
        pairs (list): List of trading pairs
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        bool: Success/failure
    """
    # Load existing risk config
    risk_config = load_file(RISK_CONFIG_FILE, {"pairs": {}})
    
    # Update for each pair
    updated_pairs = 0
    all_drawdowns = []
    
    for pair in pairs:
        # Load backtest results
        pair_filename = pair.replace('/', '_')
        backtest_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_backtest.json"
        backtest_results = load_file(backtest_file)
        
        # Skip if missing data
        if not backtest_results:
            logger.warning(f"Missing backtest results for {pair}, skipping")
            continue
        
        # Extract key metrics
        max_drawdown = backtest_results.get("max_drawdown", 0.0)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
        win_rate = backtest_results.get("win_rate", 0.0)
        all_drawdowns.append(max_drawdown)
        
        # Determine risk level
        if max_drawdown <= 0.1 and sharpe_ratio >= 2.0 and win_rate >= 0.6:
            risk_level = "Low"
        elif max_drawdown <= 0.2 and sharpe_ratio >= 1.0 and win_rate >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Adjust for aggressive mode
        if aggressive and risk_level != "High":
            risk_level = "Low" if risk_level == "Low" else "Medium"
        
        # Set risk parameters based on risk level
        risk_params = {
            "risk_level": risk_level,
            "max_drawdown_historical": max_drawdown,
            "max_leverage": 125.0 if risk_level == "Low" else (100.0 if risk_level == "Medium" else 75.0),
            "position_size_limit": 0.3 if risk_level == "Low" else (0.25 if risk_level == "Medium" else 0.2),
            "stop_loss_multiplier": 1.5 if risk_level == "Low" else (1.2 if risk_level == "Medium" else 1.0),
            "last_updated": datetime.now().isoformat()
        }
        
        # Add to risk config
        risk_config["pairs"][pair] = risk_params
        updated_pairs += 1
    
    # Update global parameters
    avg_drawdown = sum(all_drawdowns) / len(all_drawdowns) if all_drawdowns else 0.2
    max_acceptable_drawdown = min(0.15, avg_drawdown * 1.5)
    
    if aggressive:
        max_acceptable_drawdown = min(0.20, avg_drawdown * 2.0)
    
    # Update global risk parameters
    risk_config["global"] = {
        "max_acceptable_drawdown": max_acceptable_drawdown,
        "daily_var_95": avg_drawdown / 2.0,  # Approximate VaR
        "position_correlation_limit": 0.7,  # Limit on correlated positions
        "max_open_positions": 12,  # Limit total open positions
        "max_leverage_total": 100.0,  # Maximum total leverage
        "margin_reserve_percentage": 0.2,  # Reserve 20% of margin
        "last_updated": datetime.now().isoformat()
    }
    
    # Add parameters for flash crash protection
    risk_config["flash_crash"] = {
        "enabled": True,
        "trigger_threshold": -0.05,  # 5% sudden drop
        "max_acceptable_loss": -0.15,  # 15% maximum acceptable loss
        "recovery_threshold": 0.03,  # 3% recovery before re-entry
        "cooldown_period": 6,  # 6 hours cooldown after flash crash
        "protection_levels": [
            {"threshold": -0.05, "action": "reduce_leverage", "factor": 0.5},
            {"threshold": -0.1, "action": "hedge", "factor": 0.25},
            {"threshold": -0.15, "action": "close_all", "factor": 1.0}
        ]
    }
    
    # Save updated config
    success = save_file(RISK_CONFIG_FILE, risk_config)
    
    if success:
        logger.info(f"Updated risk configuration for {updated_pairs} pairs")
    else:
        logger.error("Failed to update risk configuration")
    
    return success

def update_dynamic_parameters(pairs, aggressive=False):
    """
    Update dynamic parameters configuration
    
    Args:
        pairs (list): List of trading pairs
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        bool: Success/failure
    """
    # Load existing dynamic parameters config
    dynamic_params_config = load_file(DYNAMIC_PARAMS_CONFIG_FILE, {})
    
    # Global dynamic parameters
    base_leverage = 20.0 if not aggressive else 25.0
    risk_percentage = 0.2 if not aggressive else 0.25
    
    # Set global dynamic parameters
    dynamic_params_config["global"] = {
        "base_leverage": base_leverage,
        "max_leverage": 125.0,
        "risk_percentage": risk_percentage,
        "confidence_threshold": 0.65 if not aggressive else 0.6,
        "dynamic_leverage_scaling": True,
        "dynamic_position_sizing": True,
        "dynamic_stop_loss": True,
        "dynamic_take_profit": True,
        "last_updated": datetime.now().isoformat()
    }
    
    # Set leverage scaling parameters
    dynamic_params_config["leverage_scaling"] = {
        "min_confidence": 0.65 if not aggressive else 0.6,
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
        "max_position_size_percentage": 0.3,  # Maximum size as percentage of portfolio
        "adjustment_factors": {
            "high_confidence": 1.2 if not aggressive else 1.3,
            "medium_confidence": 1.0,
            "low_confidence": 0.8,
            "high_volatility": 0.8 if not aggressive else 0.9,
            "medium_volatility": 1.0,
            "low_volatility": 1.2 if not aggressive else 1.1
        }
    }
    
    # Update pair-specific parameters
    updated_pairs = 0
    dynamic_params_config["pairs"] = {}
    
    for pair in pairs:
        # Load backtest results
        pair_filename = pair.replace('/', '_')
        backtest_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_backtest.json"
        backtest_results = load_file(backtest_file)
        
        # Skip if missing data
        if not backtest_results:
            logger.warning(f"Missing backtest results for {pair}, skipping")
            continue
        
        # Extract key metrics
        win_rate = backtest_results.get("win_rate", 0.5)
        max_drawdown = backtest_results.get("max_drawdown", 0.2)
        
        # Calculate Kelly criterion f* = (bp - q) / b
        # where p is win rate, q is loss rate (1-p), and b is average win/loss ratio
        avg_win_loss_ratio = 2.0  # Assuming 2:1 reward-to-risk ratio
        kelly = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly = max(0.05, min(0.3, kelly))  # Constrain between 5% and 30%
        
        # Adjust for aggressive mode
        if aggressive:
            kelly = min(0.3, kelly * 1.2)
        
        # Calculate optimal risk percentage
        optimal_risk = kelly / 2  # Half-Kelly for safety
        
        # Calculate risk-adjusted leverage
        risk_adjusted_leverage = min(125.0, 100.0 * (1 - max_drawdown))
        
        # Adjust for aggressive mode
        if aggressive:
            risk_adjusted_leverage = min(125.0, risk_adjusted_leverage * 1.25)
        
        # Set pair-specific parameters
        dynamic_params_config["pairs"][pair] = {
            "kelly_criterion": kelly,
            "optimal_risk_percentage": optimal_risk,
            "risk_adjusted_leverage": risk_adjusted_leverage,
            "adjustment_factor": 1.0 if not aggressive else 1.2,
            "last_updated": datetime.now().isoformat()
        }
        
        updated_pairs += 1
    
    # Save updated config
    success = save_file(DYNAMIC_PARAMS_CONFIG_FILE, dynamic_params_config)
    
    if success:
        logger.info(f"Updated dynamic parameters for {updated_pairs} pairs")
    else:
        logger.error("Failed to update dynamic parameters")
    
    return success

def update_integrated_risk_config(pairs, aggressive=False):
    """
    Update integrated risk configuration
    
    Args:
        pairs (list): List of trading pairs
        aggressive (bool): Whether to use aggressive settings
        
    Returns:
        bool: Success/failure
    """
    # Load existing integrated risk config
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
        # Load backtest results
        pair_filename = pair.replace('/', '_')
        backtest_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_backtest.json"
        backtest_results = load_file(backtest_file, {})
        
        max_drawdown = backtest_results.get("max_drawdown", 0.2)
        sharpe_ratio = backtest_results.get("sharpe_ratio", 1.0)
        win_rate = backtest_results.get("win_rate", 0.5)
        
        # Calculate risk score (lower is better)
        risk_score = max_drawdown * 5 - sharpe_ratio - win_rate
        pair_risks[pair] = risk_score
    
    # Create portfolio diversification rules
    diversification_rules = {
        "max_exposure_per_pair": 0.3 if not aggressive else 0.35,
        "max_correlation_weight": 0.6 if not aggressive else 0.7,
        "min_uncorrelated_allocation": 0.2 if not aggressive else 0.15,
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
    
    # Build risk policies
    max_drawdown_trigger = 0.15 if not aggressive else 0.20
    
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
            "low_volatility": {"leverage_multiplier": 1.2, "position_size_multiplier": 1.1},
            "medium_volatility": {"leverage_multiplier": 1.0, "position_size_multiplier": 1.0},
            "high_volatility": {"leverage_multiplier": 0.8, "position_size_multiplier": 0.8}
        },
        "trend_following": {
            "uptrend": {"long_bias": 0.7, "short_bias": 0.3},
            "downtrend": {"long_bias": 0.3, "short_bias": 0.7},
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
    
    # Save updated config
    success = save_file(INTEGRATED_RISK_CONFIG_FILE, integrated_risk_config)
    
    if success:
        logger.info(f"Updated integrated risk configuration for {len(pairs)} pairs")
    else:
        logger.error("Failed to update integrated risk configuration")
    
    return success

def reset_sandbox_portfolio(initial_balance=20000.0):
    """
    Reset sandbox portfolio to initial balance
    
    Args:
        initial_balance (float): Initial portfolio balance
        
    Returns:
        bool: Success/failure
    """
    # Create empty portfolio
    portfolio = {
        "balance": initial_balance,
        "positions": [],
        "last_updated": datetime.now().isoformat()
    }
    
    # Save portfolio
    success = save_file(SANDBOX_PORTFOLIO_FILE, portfolio)
    
    if success:
        logger.info(f"Reset sandbox portfolio to ${initial_balance:.2f}")
    else:
        logger.error("Failed to reset sandbox portfolio")
    
    return success

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    logger.info(f"Applying optimized settings for {len(pairs)} pairs")
    
    # Update configurations
    success = True
    success &= update_ml_config(pairs, args.aggressive)
    success &= update_risk_configuration(pairs, args.aggressive)
    success &= update_dynamic_parameters(pairs, args.aggressive)
    success &= update_integrated_risk_config(pairs, args.aggressive)
    
    # Reset portfolio if requested
    if args.reset_portfolio:
        success &= reset_sandbox_portfolio(args.initial_balance)
    
    if success:
        logger.info("Successfully applied all optimized settings")
    else:
        logger.warning("Some settings were not applied successfully")

if __name__ == "__main__":
    main()