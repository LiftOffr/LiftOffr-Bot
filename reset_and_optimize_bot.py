#!/usr/bin/env python3
"""
Reset and Optimize Trading Bot

This script:
1. Resets the sandbox portfolio to $20,000
2. Enables all risk management optimizations
3. Configures the bot for self-optimization and continuous learning
4. Activates all trading pairs with dynamic parameter adjustment
"""

import os
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("reset_optimize.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
CONFIG_DIR = "config"
ML_MODELS_DIR = "ml_models"
TRADING_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
INITIAL_PORTFOLIO_VALUE = 20000.0
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Reset and optimize the trading bot")
    parser.add_argument("--sandbox", action="store_true", help="Run in sandbox mode")
    parser.add_argument("--pairs", type=str, default=",".join(TRADING_PAIRS),
                     help="Comma-separated list of trading pairs")
    parser.add_argument("--capital", type=float, default=INITIAL_PORTFOLIO_VALUE,
                     help="Initial capital")
    parser.add_argument("--optimize", action="store_true", default=True,
                     help="Run optimization before starting")
    parser.add_argument("--risk-level", type=str, default="balanced",
                     choices=["conservative", "balanced", "aggressive", "ultra"],
                     help="Risk level for trading")
    return parser.parse_args()

def run_command(cmd, description=None):
    """Run a command with proper logging"""
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if description:
            logger.info(f"{description} - Completed")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"STDERR: {e.stderr}")
        return None

def reset_portfolio_data():
    """Reset portfolio data files"""
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Reset positions data
    positions_data = []
    with open(f"{DATA_DIR}/sandbox_positions.json", 'w') as f:
        json.dump(positions_data, f, indent=2)
    
    # Reset trade history
    trades_data = []
    with open(f"{DATA_DIR}/sandbox_trades.json", 'w') as f:
        json.dump(trades_data, f, indent=2)
    
    # Reset portfolio history
    timestamp = datetime.now().isoformat()
    portfolio_history = [
        {
            "timestamp": timestamp,
            "portfolio_value": INITIAL_PORTFOLIO_VALUE,
            "cash": INITIAL_PORTFOLIO_VALUE,
            "positions": 0,
            "drawdown": 0.0
        }
    ]
    with open(f"{DATA_DIR}/sandbox_portfolio_history.json", 'w') as f:
        json.dump(portfolio_history, f, indent=2)
    
    logger.info(f"Portfolio data reset: Starting capital set to ${INITIAL_PORTFOLIO_VALUE:.2f}")

def create_risk_config(risk_level):
    """Create optimized risk configuration"""
    # Risk configurations by level
    risk_configs = {
        "conservative": {
            "max_risk_per_trade": 0.05,
            "max_portfolio_risk": 0.15,
            "max_leverage": 10.0,
            "volatility_scaling_factor": 0.7,
            "kelly_fraction": 0.4,
            "max_drawdown_threshold": 0.1,
            "ratchet_takeprofit_enabled": True,
            "trailing_stop_enabled": True
        },
        "balanced": {
            "max_risk_per_trade": 0.1,
            "max_portfolio_risk": 0.25,
            "max_leverage": 20.0,
            "volatility_scaling_factor": 1.0,
            "kelly_fraction": 0.6,
            "max_drawdown_threshold": 0.15,
            "ratchet_takeprofit_enabled": True,
            "trailing_stop_enabled": True
        },
        "aggressive": {
            "max_risk_per_trade": 0.15,
            "max_portfolio_risk": 0.4,
            "max_leverage": 50.0,
            "volatility_scaling_factor": 1.3,
            "kelly_fraction": 0.8,
            "max_drawdown_threshold": 0.25,
            "ratchet_takeprofit_enabled": True,
            "trailing_stop_enabled": True
        },
        "ultra": {
            "max_risk_per_trade": 0.2,
            "max_portfolio_risk": 0.5,
            "max_leverage": 100.0,
            "volatility_scaling_factor": 1.5,
            "kelly_fraction": 1.0,
            "max_drawdown_threshold": 0.35,
            "ratchet_takeprofit_enabled": True,
            "trailing_stop_enabled": True
        }
    }
    
    # Base risk configuration
    config = {
        "version": "2.0",
        "risk_management": {
            "enabled": True,
            "risk_level": risk_level,
            "max_risk_per_trade": risk_configs[risk_level]["max_risk_per_trade"],
            "max_portfolio_risk": risk_configs[risk_level]["max_portfolio_risk"],
            "max_leverage": risk_configs[risk_level]["max_leverage"],
            "volatility_scaling": {
                "enabled": True,
                "scaling_factor": risk_configs[risk_level]["volatility_scaling_factor"],
                "volatility_window": 20,
                "volatility_threshold": 0.03
            },
            "kelly_criterion": {
                "enabled": True,
                "fraction": risk_configs[risk_level]["kelly_fraction"]
            },
            "drawdown_protection": {
                "enabled": True,
                "max_drawdown_threshold": risk_configs[risk_level]["max_drawdown_threshold"],
                "reduce_exposure_at": 0.75,
                "pause_trading_at": 0.9
            },
            "position_sizing": {
                "method": "risk_based",
                "adaptive": True,
                "confidence_scaling": True
            },
            "stop_loss": {
                "method": "volatility_based",
                "atr_multiplier": 1.5,
                "fixed_percentage": 0.04,
                "max_loss_percentage": 0.04,
                "trailing": risk_configs[risk_level]["trailing_stop_enabled"]
            },
            "take_profit": {
                "method": "adaptive",
                "target_reward_risk_ratio": 1.5,
                "ratcheting": risk_configs[risk_level]["ratchet_takeprofit_enabled"]
            }
        },
        "market_regime_detection": {
            "enabled": True,
            "methods": ["volatility", "trend_strength", "momentum"],
            "lookback_periods": 50,
            "update_frequency": "candle",
            "regime_specific_parameters": True
        },
        "pair_specific": {
            "SOL/USD": {
                "max_leverage": min(125.0, risk_configs[risk_level]["max_leverage"] * 1.25),
                "volatility_adjustment": 1.2
            },
            "BTC/USD": {
                "max_leverage": min(100.0, risk_configs[risk_level]["max_leverage"] * 1.0),
                "volatility_adjustment": 1.0
            },
            "ETH/USD": {
                "max_leverage": min(100.0, risk_configs[risk_level]["max_leverage"] * 1.0),
                "volatility_adjustment": 1.0
            },
            "ADA/USD": {
                "max_leverage": min(50.0, risk_configs[risk_level]["max_leverage"] * 0.8),
                "volatility_adjustment": 1.5
            },
            "DOT/USD": {
                "max_leverage": min(50.0, risk_configs[risk_level]["max_leverage"] * 0.8),
                "volatility_adjustment": 1.5
            },
            "LINK/USD": {
                "max_leverage": min(50.0, risk_configs[risk_level]["max_leverage"] * 0.8),
                "volatility_adjustment": 1.5
            }
        }
    }
    
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Save risk configuration
    with open(RISK_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Risk configuration created with {risk_level} risk profile")
    return config

def create_integrated_risk_config(risk_level):
    """Create integrated risk configuration for cross-strategy coordination"""
    # Base configuration
    config = {
        "version": "1.0",
        "integrated_risk_management": {
            "enabled": True,
            "max_concurrent_trades": 6,
            "max_trades_per_pair": 2,
            "max_portfolio_allocation": 0.75,
            "strategy_ranking": {
                "enabled": True,
                "method": "weighted_performance",
                "metrics": ["win_rate", "profit_factor", "sharpe_ratio", "consistency"],
                "lookback_periods": [30, 90, 180]
            },
            "signal_arbitration": {
                "required_confirmation": 0,
                "signal_strength_priority": True,
                "strategy_strength_scaling": True,
                "minimum_signal_strength": 0.5,
                "contradictory_signals": "stronger_wins"
            },
            "risk_budgeting": {
                "enabled": True,
                "allocation_method": "equal_risk",
                "adjust_by_performance": True,
                "higher_timeframe_bias": True
            }
        },
        "strategy_categories": {
            "those_dudes": {
                "strategies": ["ARIMAStrategy", "LSTMStrategy", "GRUStrategy"],
                "max_allocation": 0.5,
                "priority": 0.6,
                "risk_adjustment": 1.0,
                "integration_rules": {
                    "min_agreement": 0,
                    "signal_amplification": True
                }
            },
            "him_all_along": {
                "strategies": ["AdaptiveStrategy", "TCNStrategy", "EnsembleStrategy"],
                "max_allocation": 0.5,
                "priority": 0.4,
                "risk_adjustment": 1.0,
                "integration_rules": {
                    "min_agreement": 0,
                    "signal_amplification": True
                }
            }
        },
        "cross_strategy_exits": {
            "enabled": True,
            "exit_agreement_threshold": 0.0,
            "prioritize_stop_loss": True,
            "profit_protection": {
                "enabled": True,
                "unrealized_profit_threshold": 0.03,
                "lock_in_percentage": 0.5
            }
        }
    }
    
    # Adjust settings based on risk level
    if risk_level == "conservative":
        config["integrated_risk_management"]["max_concurrent_trades"] = 4
        config["integrated_risk_management"]["max_portfolio_allocation"] = 0.5
        config["integrated_risk_management"]["signal_arbitration"]["minimum_signal_strength"] = 0.7
    elif risk_level == "aggressive":
        config["integrated_risk_management"]["max_concurrent_trades"] = 8
        config["integrated_risk_management"]["max_portfolio_allocation"] = 0.85
        config["integrated_risk_management"]["signal_arbitration"]["minimum_signal_strength"] = 0.4
    elif risk_level == "ultra":
        config["integrated_risk_management"]["max_concurrent_trades"] = 10
        config["integrated_risk_management"]["max_portfolio_allocation"] = 0.95
        config["integrated_risk_management"]["signal_arbitration"]["minimum_signal_strength"] = 0.3
    
    # Save integrated risk configuration
    with open(INTEGRATED_RISK_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Integrated risk configuration created with {risk_level} risk profile")
    return config

def create_ml_config(pairs, risk_level):
    """Create ML configuration with optimal settings"""
    # Base ML configuration
    config = {
        "version": "2.0",
        "ml_trading": {
            "enabled": True,
            "auto_training": {
                "enabled": True,
                "frequency": "daily",
                "min_data_points": 5000,
                "validation_split": 0.2,
                "early_stopping": True
            },
            "prediction": {
                "horizon": 24,
                "confidence_threshold": 0.65,
                "minimum_accuracy": 0.70,
                "ensemble_method": "weighted_voting"
            },
            "feature_engineering": {
                "technical_indicators": True,
                "price_patterns": True,
                "volatility_features": True,
                "volume_analysis": True,
                "sentiment_features": True
            },
            "model_selection": {
                "auto_selection": True,
                "ensemble_models": True,
                "model_types": ["tcn", "lstm", "gru", "transformer"]
            },
            "hyperparameter_tuning": {
                "enabled": True,
                "method": "bayesian",
                "max_trials": 25,
                "parallel_trials": 3
            },
            "adaptive_parameters": {
                "enabled": True,
                "market_regime_specific": True,
                "adjust_by_volatility": True,
                "adjust_by_confidence": True
            }
        },
        "pairs": {}
    }
    
    # Adjust settings based on risk level
    if risk_level == "conservative":
        config["ml_trading"]["prediction"]["confidence_threshold"] = 0.75
        config["ml_trading"]["prediction"]["minimum_accuracy"] = 0.80
    elif risk_level == "aggressive":
        config["ml_trading"]["prediction"]["confidence_threshold"] = 0.60
        config["ml_trading"]["prediction"]["minimum_accuracy"] = 0.65
    elif risk_level == "ultra":
        config["ml_trading"]["prediction"]["confidence_threshold"] = 0.55
        config["ml_trading"]["prediction"]["minimum_accuracy"] = 0.60
    
    # Configure for each trading pair
    for pair in pairs:
        # Base leverage settings
        if pair == "SOL/USD":
            base_leverage = 20.0 if risk_level == "balanced" else (
                10.0 if risk_level == "conservative" else (
                50.0 if risk_level == "aggressive" else 100.0
            ))
            max_leverage = min(125.0, base_leverage * 2)
        else:
            base_leverage = 10.0 if risk_level == "balanced" else (
                5.0 if risk_level == "conservative" else (
                25.0 if risk_level == "aggressive" else 50.0
            ))
            max_leverage = min(100.0, base_leverage * 2)
        
        # Risk percentage
        risk_percentage = 0.1 if risk_level == "balanced" else (
            0.05 if risk_level == "conservative" else (
            0.15 if risk_level == "aggressive" else 0.2
        ))
        
        # Add pair-specific settings
        pair_config = {
            "enabled": True,
            "base_leverage": base_leverage,
            "max_leverage": max_leverage,
            "risk_percentage": risk_percentage,
            "confidence_threshold": config["ml_trading"]["prediction"]["confidence_threshold"],
            "models": {
                "tcn": {
                    "enabled": True,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "batch_size": 64
                },
                "lstm": {
                    "enabled": True,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "batch_size": 64
                },
                "gru": {
                    "enabled": True,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "batch_size": 64
                },
                "transformer": {
                    "enabled": True,
                    "optimizer": "adam",
                    "learning_rate": 0.0005,
                    "batch_size": 32
                }
            },
            "ensemble": {
                "enabled": True,
                "voting_weights": {
                    "tcn": 0.35,
                    "lstm": 0.25,
                    "gru": 0.20,
                    "transformer": 0.20
                }
            }
        }
        
        # Add to config
        config["pairs"][pair] = pair_config
    
    # Save ML configuration
    with open(ML_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ML configuration created with {risk_level} risk profile for {len(pairs)} pairs")
    return config

def create_dynamic_params_config(risk_level):
    """Create dynamic parameter configuration"""
    # Base configuration
    config = {
        "version": "1.0",
        "dynamic_parameters": {
            "enabled": True,
            "update_frequency": "candle",
            "market_regime_detection": {
                "enabled": True,
                "regime_types": ["trending_up", "trending_down", "ranging", "volatile"],
                "detection_methods": ["price_action", "volatility", "momentum"]
            },
            "parameter_adjustment": {
                "volatility_based": True,
                "confidence_based": True,
                "win_rate_based": True,
                "regime_based": True
            },
            "learning_rate": 0.1,
            "min_adjustment": 0.5,
            "max_adjustment": 2.0
        },
        "adjustable_parameters": {
            "leverage": {
                "enabled": True,
                "base_value": 20.0,
                "min_value": 1.0,
                "max_value": 125.0,
                "step_size": 1.0,
                "volatility_scaling": True,
                "confidence_scaling": True
            },
            "risk_percentage": {
                "enabled": True,
                "base_value": 0.1,
                "min_value": 0.01,
                "max_value": 0.2,
                "step_size": 0.01,
                "volatility_scaling": True,
                "confidence_scaling": True
            },
            "take_profit": {
                "enabled": True,
                "base_value": 0.1,
                "min_value": 0.02,
                "max_value": 0.5,
                "step_size": 0.01,
                "volatility_scaling": True,
                "regime_scaling": True
            },
            "stop_loss": {
                "enabled": True,
                "base_value": 0.05,
                "min_value": 0.01,
                "max_value": 0.1,
                "step_size": 0.005,
                "volatility_scaling": True,
                "confidence_scaling": True
            },
            "trailing_stop": {
                "enabled": True,
                "base_value": 0.02,
                "min_value": 0.005,
                "max_value": 0.05,
                "step_size": 0.005,
                "profit_scaling": True
            }
        },
        "regime_specific_values": {
            "trending_up": {
                "leverage_multiplier": 1.2,
                "risk_percentage_multiplier": 1.2,
                "take_profit_multiplier": 1.0,
                "stop_loss_multiplier": 0.8
            },
            "trending_down": {
                "leverage_multiplier": 1.0,
                "risk_percentage_multiplier": 1.0,
                "take_profit_multiplier": 0.8,
                "stop_loss_multiplier": 1.0
            },
            "ranging": {
                "leverage_multiplier": 0.8,
                "risk_percentage_multiplier": 0.8,
                "take_profit_multiplier": 0.7,
                "stop_loss_multiplier": 1.2
            },
            "volatile": {
                "leverage_multiplier": 0.6,
                "risk_percentage_multiplier": 0.6,
                "take_profit_multiplier": 1.2,
                "stop_loss_multiplier": 1.4
            }
        }
    }
    
    # Adjust settings based on risk level
    if risk_level == "conservative":
        config["dynamic_parameters"]["learning_rate"] = 0.05
        config["dynamic_parameters"]["max_adjustment"] = 1.5
        config["adjustable_parameters"]["leverage"]["base_value"] = 10.0
        config["adjustable_parameters"]["risk_percentage"]["base_value"] = 0.05
    elif risk_level == "aggressive":
        config["dynamic_parameters"]["learning_rate"] = 0.15
        config["dynamic_parameters"]["max_adjustment"] = 2.5
        config["adjustable_parameters"]["leverage"]["base_value"] = 50.0
        config["adjustable_parameters"]["risk_percentage"]["base_value"] = 0.15
    elif risk_level == "ultra":
        config["dynamic_parameters"]["learning_rate"] = 0.2
        config["dynamic_parameters"]["max_adjustment"] = 3.0
        config["adjustable_parameters"]["leverage"]["base_value"] = 100.0
        config["adjustable_parameters"]["risk_percentage"]["base_value"] = 0.2
    
    # Save configuration
    with open(DYNAMIC_PARAMS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Dynamic parameter configuration created with {risk_level} risk profile")
    return config

def run_optimization(pairs, risk_level):
    """Run optimization for all trading pairs"""
    logger.info("Running optimization for all trading pairs...")
    
    # Create optimization directory if it doesn't exist
    os.makedirs("optimization_results", exist_ok=True)
    
    # Run risk-aware optimization
    optimize_cmd = [
        "python", "run_risk_aware_optimization.py",
        "--pairs", ",".join(pairs),
        "--risk-level", risk_level,
        "--epochs", "50",
        "--trials", "20"
    ]
    run_command(optimize_cmd, f"Optimizing with {risk_level} risk level")
    
    # Apply optimized settings
    apply_cmd = [
        "python", "apply_optimized_settings.py",
        "--pairs", ",".join(pairs),
        "--risk-level", risk_level
    ]
    run_command(apply_cmd, "Applying optimized settings")
    
    logger.info("Optimization completed")

def start_bot(pairs, capital, sandbox=True):
    """Start the trading bot with optimized settings"""
    # Create command to start the bot
    bot_cmd = [
        "python", "main.py",
        "--pairs", ",".join(pairs),
        "--capital", str(capital),
        "--strategy", "integrated_ml",
        "--multi-strategy", "true"
    ]
    
    if sandbox:
        bot_cmd.append("--sandbox")
    
    # Start the bot
    logger.info(f"Starting trading bot with {len(pairs)} pairs and ${capital:.2f} capital...")
    run_command(bot_cmd, "Starting trading bot")

def create_all_configs(pairs, risk_level):
    """Create all configuration files"""
    create_risk_config(risk_level)
    create_integrated_risk_config(risk_level)
    create_ml_config(pairs, risk_level)
    create_dynamic_params_config(risk_level)
    
    logger.info("All configuration files created successfully")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse trading pairs
    pairs = args.pairs.split(",") if args.pairs else TRADING_PAIRS
    
    # Reset portfolio data
    reset_portfolio_data()
    
    # Create configuration files
    create_all_configs(pairs, args.risk_level)
    
    # Run optimization if enabled
    if args.optimize:
        run_optimization(pairs, args.risk_level)
    
    # Start the bot
    start_bot(pairs, args.capital, args.sandbox)
    
    logger.info("Reset and optimization completed")
    print(f"""
================================================================================
TRADING BOT RESET AND OPTIMIZED
================================================================================

Portfolio has been reset to ${INITIAL_PORTFOLIO_VALUE:.2f}

CONFIGURATION SUMMARY:
- Risk Level: {args.risk_level.upper()}
- Trading Pairs: {', '.join(pairs)}
- Dynamic Parameter Optimization: ENABLED
- ML Self-optimization: ENABLED
- Integrated Risk Management: ENABLED
- Cross-Strategy Coordination: ENABLED

The bot is now running in {'SANDBOX' if args.sandbox else 'LIVE'} mode.
Check the portfolio status with:
  python check_sandbox_portfolio.py

================================================================================
""")

if __name__ == "__main__":
    main()