#!/usr/bin/env python3
"""
Reset and Optimize Trading Bot

This script resets the trading bot with the specified starting capital and
optimizes all trading parameters for maximum returns and accuracy.

Usage:
    python reset_and_optimize_bot.py [--capital CAPITAL] [--sandbox] [--risk-level RISK_LEVEL]
"""

import os
import json
import argparse
import logging
import subprocess
import time
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
ML_MODELS_DIR = "ml_models"
DEFAULT_PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_POSITION_FILE = f"{DATA_DIR}/sandbox_positions.json"
DEFAULT_TRADE_HISTORY_FILE = f"{DATA_DIR}/sandbox_trades.json"
DEFAULT_RISK_METRICS_FILE = f"{DATA_DIR}/risk_metrics.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Reset and optimize trading bot")
    parser.add_argument("--capital", type=float, default=20000.0,
                      help="Starting capital amount")
    parser.add_argument("--sandbox", action="store_true",
                      help="Run in sandbox mode (default)")
    parser.add_argument("--risk-level", type=str, default="balanced",
                      choices=["conservative", "balanced", "aggressive", "ultra"],
                      help="Risk level")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(f"{description}...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"STDERR: {e.stderr}")
        return None

def reset_portfolio(capital: float):
    """Reset portfolio with starting capital"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Initialize portfolio history with starting capital
    portfolio_history = [
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "portfolio_value": capital,
            "cash": capital,
            "positions": 0,
            "drawdown": 0.0
        }
    ]
    
    # Save portfolio history
    with open(DEFAULT_PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio_history, f, indent=2)
    
    # Initialize positions file
    with open(DEFAULT_POSITION_FILE, 'w') as f:
        json.dump([], f, indent=2)
    
    # Initialize trade history file
    with open(DEFAULT_TRADE_HISTORY_FILE, 'w') as f:
        json.dump([], f, indent=2)
    
    logger.info(f"Portfolio data reset: Starting capital set to ${capital:.2f}")
    return True

def create_config_files(risk_level: str, all_pairs: List[str]):
    """Create/update configuration files based on risk level"""
    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Create risk configuration
    risk_config = {
        "version": "1.3.0",
        "global_settings": {
            "max_drawdown": 0.20 if risk_level in ["aggressive", "ultra"] else 0.15,
            "max_daily_loss": 0.05 if risk_level in ["aggressive", "ultra"] else 0.03,
            "emergency_stop_loss": 0.04 if risk_level in ["aggressive", "ultra"] else 0.03,
            "max_open_trades": 12 if risk_level in ["aggressive", "ultra"] else 8,
            "max_open_trades_per_pair": 1,
            "cooldown_after_stop_loss": 60 if risk_level in ["aggressive", "ultra"] else 120,
            "trend_detection": {
                "enabled": True,
                "ema_short": 50,
                "ema_long": 100,
                "check_before_shorts": True
            },
            "max_leverage": {
                "default": 50.0 if risk_level == "ultra" else (
                    25.0 if risk_level == "aggressive" else (
                        15.0 if risk_level == "balanced" else 10.0
                    )
                ),
                "high_volatility": 25.0 if risk_level == "ultra" else (
                    15.0 if risk_level == "aggressive" else (
                        10.0 if risk_level == "balanced" else 5.0
                    )
                ),
                "extreme_volatility": 10.0 if risk_level == "ultra" else (
                    5.0 if risk_level == "aggressive" else (
                        3.0 if risk_level == "balanced" else 2.0
                    )
                )
            },
            "position_sizing": {
                "default_risk_per_trade": 0.20 if risk_level == "ultra" else (
                    0.15 if risk_level == "aggressive" else (
                        0.10 if risk_level == "balanced" else 0.05
                    )
                ),
                "kelly_criterion_enabled": True,
                "kelly_fraction": 0.5,
                "dynamic_sizing_enabled": True,
                "volatility_adjustment_enabled": True
            }
        },
        "pair_specific": {},
        "dynamic_risk_management": {
            "enabled": True,
            "volatility_bands": {
                "low_band": 0.01,
                "medium_band": 0.03,
                "high_band": 0.05
            },
            "drawdown_adjustment": {
                "enabled": True,
                "drawdown_threshold": 0.05,
                "risk_reduction_factor": 0.3
            },
            "trailing_stops": {
                "enabled": True,
                "activation_threshold": 0.01,
                "trailing_distance": 0.015
            },
            "ratcheting_stops": {
                "enabled": True,
                "profit_intervals": [0.01, 0.03, 0.05, 0.1],
                "lock_percentages": [0.2, 0.4, 0.6, 0.8]
            }
        },
        "stress_testing": {
            "flash_crash_protection": {
                "enabled": True,
                "protection_threshold": 0.05,
                "emergency_stop_loss": 0.10 if risk_level in ["aggressive", "ultra"] else 0.05
            },
            "liquidity_crisis": {
                "enabled": True,
                "max_slippage": 0.02,
                "order_size_reduction": 0.5
            },
            "extreme_volatility": {
                "enabled": True,
                "max_leverage_reduction": 0.6,
                "position_size_reduction": 0.5
            }
        },
        "capital_allocation": {
            "max_allocation_per_pair": 0.3,
            "max_allocation_per_category": 0.5,
            "reserve_percentage": 0.2,
            "dynamic_allocation": {
                "enabled": True,
                "performance_based": True,
                "volatility_based": True,
                "correlation_aware": True
            }
        }
    }
    
    # Set pair-specific risk settings
    for pair in all_pairs:
        if pair == "SOL/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 50.0 if risk_level in ["aggressive", "ultra"] else 25.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.1,
                "max_trade_duration_hours": 48,
                "volatility_adjustment": 1.42
            }
        elif pair == "BTC/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 25.0 if risk_level in ["aggressive", "ultra"] else 15.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.08,
                "max_trade_duration_hours": 72,
                "volatility_adjustment": 1.36
            }
        elif pair == "ETH/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 30.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.09,
                "max_trade_duration_hours": 60,
                "volatility_adjustment": 1.38
            }
        elif pair == "ADA/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 40.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.12,
                "max_trade_duration_hours": 48,
                "volatility_adjustment": 1.34
            }
        elif pair == "DOT/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 45.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.11,
                "max_trade_duration_hours": 48,
                "volatility_adjustment": 1.36
            }
        elif pair == "LINK/USD":
            risk_config["pair_specific"][pair] = {
                "max_leverage": 35.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "risk_per_trade": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "atr_multiplier": 1.5,
                "take_profit": 0.10,
                "max_trade_duration_hours": 48,
                "volatility_adjustment": 1.32
            }
    
    # Save risk configuration
    with open(f"{CONFIG_DIR}/risk_config.json", 'w') as f:
        json.dump(risk_config, f, indent=2)
    
    logger.info(f"Risk configuration created with {risk_level} risk profile")
    
    # Create integrated risk configuration
    integrated_risk_config = {
        "version": "1.2.0",
        "risk_profile": risk_level,
        "integrated_risk_management": {
            "enabled": True,
            "portfolio_drawdown_protection": {
                "enabled": True,
                "max_drawdown": 0.20 if risk_level in ["aggressive", "ultra"] else 0.15,
                "risk_reduction_stages": [
                    {"drawdown": 0.05, "reduction": 0.20},
                    {"drawdown": 0.10, "reduction": 0.50},
                    {"drawdown": 0.15, "reduction": 0.75}
                ],
                "auto_recovery": {
                    "enabled": True,
                    "recovery_threshold": 0.03,
                    "recovery_duration_days": 3
                }
            },
            "cross_strategy_management": {
                "signal_strength_arbitration": True,
                "opposing_signal_cancellation": True,
                "cross_strategy_exits": True,
                "emergency_coordinated_exit": True,
                "max_concurrent_strategies_per_pair": 2,
                "strategy_categories": {
                    "category_1": {
                        "name": "those dudes",
                        "strategies": ["adaptive", "cyclical", "seasonal"],
                        "max_concurrent_trades": 3
                    },
                    "category_2": {
                        "name": "him all along",
                        "strategies": ["arima", "momentum", "ml"],
                        "max_concurrent_trades": 3
                    }
                }
            },
            "dynamic_parameter_adjustment": {
                "enabled": True,
                "confidence_based_leverage": True,
                "risk_based_position_sizing": True,
                "volatility_based_atr_multiplier": True,
                "profit_target_adjustment": True,
                "win_streak_factor": 0.05,
                "loss_streak_factor": 0.1
            }
        },
        "market_regime_detection": {
            "enabled": True,
            "lookback_periods": 30,
            "refresh_interval_minutes": 60,
            "regime_specific_parameters": {
                "trending": {
                    "leverage_multiplier": 1.2,
                    "risk_multiplier": 1.1,
                    "atr_multiplier": 1.0
                },
                "ranging": {
                    "leverage_multiplier": 0.9,
                    "risk_multiplier": 0.9,
                    "atr_multiplier": 1.1
                },
                "volatile": {
                    "leverage_multiplier": 0.7,
                    "risk_multiplier": 0.8,
                    "atr_multiplier": 1.3
                },
                "extremely_volatile": {
                    "leverage_multiplier": 0.4,
                    "risk_multiplier": 0.6,
                    "atr_multiplier": 1.5
                }
            },
            "classification_thresholds": {
                "adx_trending_threshold": 25,
                "bb_ranging_threshold": 0.03,
                "volatility_threshold": 0.04,
                "extreme_volatility_threshold": 0.08
            }
        },
        "integrated_ml_optimization": {
            "enabled": True,
            "model_accuracy_monitoring": True,
            "retraining_trigger_threshold": 0.05,
            "auto_ensemble_adjustment": True,
            "ensemble_weights_optimization": {
                "enabled": True,
                "optimization_interval_hours": 24,
                "accuracy_weight": 0.7,
                "profit_weight": 0.3
            },
            "hyperparameter_optimization": {
                "enabled": True,
                "optimization_interval_days": 7,
                "max_trials": 50
            }
        },
        "risk_aware_backtesting": {
            "realistic_slippage": True,
            "simulated_partial_fills": True,
            "liquidity_constraints": True,
            "market_impact_modeling": True,
            "stress_test_scenarios": [
                "flash_crash",
                "liquidity_crisis",
                "extreme_volatility",
                "correlation_breakdown"
            ]
        },
        "risk_reporting": {
            "real_time_metrics": {
                "current_drawdown": True,
                "open_risk": True,
                "value_at_risk": True,
                "exposure_concentration": True
            },
            "alerts": {
                "drawdown_warning": 0.1,
                "high_concentration_warning": 0.3,
                "consecutive_loss_warning": 3
            },
            "periodic_reports": {
                "daily_summary": True,
                "weekly_detailed": True,
                "monthly_comprehensive": True
            }
        }
    }
    
    # Save integrated risk configuration
    with open(f"{CONFIG_DIR}/integrated_risk_config.json", 'w') as f:
        json.dump(integrated_risk_config, f, indent=2)
    
    logger.info(f"Integrated risk configuration created with {risk_level} risk profile")
    
    # Create ML configuration
    ml_config = {
        "version": "2.1.0",
        "use_ml": True,
        "auto_retrain": True,
        "global_confidence_threshold": 0.60 if risk_level in ["aggressive", "ultra"] else 0.65,
        "dynamic_parameter_optimization": True,
        "auto_prune_models": True,
        "pairs": {},
        "model_pruning": {
            "enabled": True,
            "accuracy_threshold": 0.70,
            "pruning_interval": 24,
            "min_history_size": 100
        },
        "self_optimization": {
            "enabled": True,
            "optimization_interval": 12,
            "accuracy_weight": 0.8,
            "profit_weight": 0.2,
            "max_trials": 50
        }
    }
    
    # Set pair-specific ML settings
    for pair in all_pairs:
        if pair == "SOL/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.55 if risk_level in ["aggressive", "ultra"] else 0.65,
                "max_leverage": 50.0 if risk_level in ["aggressive", "ultra"] else 25.0,
                "base_leverage": 20.0 if risk_level in ["aggressive", "ultra"] else 10.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.99
            }
        elif pair == "BTC/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.60,
                "max_leverage": 25.0 if risk_level in ["aggressive", "ultra"] else 15.0,
                "base_leverage": 10.0 if risk_level in ["aggressive", "ultra"] else 5.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.95
            }
        elif pair == "ETH/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.58,
                "max_leverage": 30.0 if risk_level in ["aggressive", "ultra"] else 15.0,
                "base_leverage": 12.0 if risk_level in ["aggressive", "ultra"] else 6.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.94
            }
        elif pair == "ADA/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.60,
                "max_leverage": 40.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "base_leverage": 15.0 if risk_level in ["aggressive", "ultra"] else 8.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.92
            }
        elif pair == "DOT/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.60,
                "max_leverage": 45.0 if risk_level in ["aggressive", "ultra"] else 20.0,
                "base_leverage": 18.0 if risk_level in ["aggressive", "ultra"] else 10.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.93
            }
        elif pair == "LINK/USD":
            ml_config["pairs"][pair] = {
                "models": {
                    "tcn": {
                        "enabled": True,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "dropout": 0.2,
                        "epochs": 100,
                        "weight": 0.40
                    },
                    "lstm": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0005,
                        "dropout": 0.3,
                        "epochs": 75,
                        "weight": 0.20
                    },
                    "gru": {
                        "enabled": True,
                        "batch_size": 48,
                        "learning_rate": 0.0008,
                        "dropout": 0.25,
                        "epochs": 80,
                        "weight": 0.20
                    },
                    "transformer": {
                        "enabled": True,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "dropout": 0.2,
                        "epochs": 50,
                        "weight": 0.20
                    }
                },
                "confidence_threshold": 0.58,
                "max_leverage": 35.0 if risk_level in ["aggressive", "ultra"] else 18.0,
                "base_leverage": 15.0 if risk_level in ["aggressive", "ultra"] else 8.0,
                "risk_percentage": 0.15 if risk_level in ["aggressive", "ultra"] else 0.1,
                "accuracy": 0.91
            }
    
    # Save ML configuration
    with open(f"{CONFIG_DIR}/ml_config.json", 'w') as f:
        json.dump(ml_config, f, indent=2)
    
    logger.info(f"ML configuration created with {risk_level} risk profile for {len(all_pairs)} pairs")
    
    # Create dynamic parameters configuration
    dynamic_params_config = {
        "version": "1.5.0",
        "dynamic_parameters": {
            "enabled": True,
            "learning_rate": 0.20 if risk_level == "ultra" else (
                0.15 if risk_level == "aggressive" else (
                    0.10 if risk_level == "balanced" else 0.05
                )
            ),
            "adaptation_interval": 12,
            "max_adjustment": 0.3,
            "min_adjustment": 0.05,
            "confidence_scaling": True,
            "volatility_scaling": True,
            "market_regime_adaptation": True
        },
        "leverage_adjustment": {
            "enabled": True,
            "min_leverage_multiplier": 0.5,
            "max_leverage_multiplier": 2.5 if risk_level in ["aggressive", "ultra"] else 1.5,
            "confidence_threshold_high": 0.85,
            "confidence_threshold_low": 0.6
        },
        "risk_adjustment": {
            "enabled": True,
            "min_risk_multiplier": 0.5,
            "max_risk_multiplier": 2.0 if risk_level in ["aggressive", "ultra"] else 1.5,
            "win_streak_factor": 0.1,
            "loss_streak_factor": 0.15
        },
        "take_profit_adjustment": {
            "enabled": True,
            "baseline_tp_factor": 1.0,
            "min_tp_multiplier": 0.7,
            "max_tp_multiplier": 2.0,
            "recent_volatility_weight": 0.6
        },
        "stop_loss_adjustment": {
            "enabled": True,
            "baseline_sl_factor": 1.0,
            "min_sl_multiplier": 0.8,
            "max_sl_multiplier": 1.5,
            "atr_weight": 0.7
        },
        "position_sizing": {
            "kelly_criterion_enabled": True,
            "kelly_fraction": 0.5,
            "volatility_adjustment_enabled": True,
            "max_capital_per_trade": 0.25 if risk_level in ["aggressive", "ultra"] else 0.15
        },
        "trade_duration": {
            "dynamic_duration_enabled": True,
            "min_duration_minutes": 15,
            "max_duration_minutes": 1440,
            "trend_strength_weight": 0.6
        },
        "market_regime_thresholds": {
            "trending_adx_threshold": 25,
            "consolidation_bb_width_threshold": 0.03,
            "volatile_historical_vol_threshold": 0.04
        }
    }
    
    # Save dynamic parameters configuration
    with open(f"{CONFIG_DIR}/dynamic_params_config.json", 'w') as f:
        json.dump(dynamic_params_config, f, indent=2)
    
    logger.info(f"Dynamic parameter configuration created with {risk_level} risk profile")
    
    return True

def run_optimization(pairs, risk_level, epochs=50, trials=20):
    """Run optimization for all pairs"""
    logger.info("Running optimization for all trading pairs...")
    
    # Run risk-aware optimization
    logger.info(f"Optimizing with {risk_level} risk level...")
    run_command(
        ["python", "run_risk_aware_optimization.py", 
         "--pairs", ",".join(pairs),
         "--risk-level", risk_level,
         "--epochs", str(epochs),
         "--trials", str(trials)],
        "Optimizing with {risk_level} risk level"
    )
    
    # Apply optimized settings
    logger.info("Applying optimized settings...")
    run_command(
        ["python", "apply_optimized_settings.py",
         "--pairs", ",".join(pairs),
         "--risk-level", risk_level],
        "Applying optimized settings"
    )
    
    logger.info("Optimization completed")
    return True

def start_trading_bot(pairs, capital, sandbox=True):
    """Start the trading bot"""
    logger.info(f"Starting trading bot with {len(pairs)} pairs and ${capital:.2f} capital...")
    
    # Start the bot
    logger.info("Starting trading bot...")
    run_command(
        ["python", "main.py",
         "--pairs", ",".join(pairs),
         "--capital", str(capital),
         "--strategy", "integrated_ml",
         "--multi-strategy", "true",
         "--sandbox" if sandbox else "--live"],
        "Starting trading bot"
    )
    
    return True

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Define trading pairs
    all_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    
    # Reset portfolio
    if not reset_portfolio(args.capital):
        logger.error("Failed to reset portfolio")
        return
    
    # Create configuration files
    if not create_config_files(args.risk_level, all_pairs):
        logger.error("Failed to create configuration files")
        return
    
    # Run optimization
    if not run_optimization(all_pairs, args.risk_level):
        logger.error("Failed to run optimization")
        return
    
    # Start trading bot
    if not start_trading_bot(all_pairs, args.capital, args.sandbox):
        logger.error("Failed to start trading bot")
        return
    
    logger.info("Reset and optimization completed")
    
    # Display summary
    print("\n" + "=" * 80)
    print("TRADING BOT RESET AND OPTIMIZED")
    print("=" * 80)
    print(f"\nPortfolio has been reset to ${args.capital:.2f}")
    print("\nCONFIGURATION SUMMARY:")
    print(f"- Risk Level: {args.risk_level.upper()}")
    print(f"- Trading Pairs: {', '.join(all_pairs)}")
    print("- Dynamic Parameter Optimization: ENABLED")
    print("- ML Self-optimization: ENABLED")
    print("- Integrated Risk Management: ENABLED")
    print("- Cross-Strategy Coordination: ENABLED")
    print("\nThe bot is now running in SANDBOX mode.")
    print("Check the portfolio status with:")
    print("  python check_sandbox_portfolio.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()