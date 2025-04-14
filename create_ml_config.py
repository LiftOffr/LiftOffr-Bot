#!/usr/bin/env python3
"""
Create ML Configuration

This script generates and validates ML configuration files for the advanced trading system.
It creates both the standard ml_config.json and the enhanced ml_enhanced_config.json files
with optimized settings for all supported trading pairs.

The generated configurations include:
1. Training parameters for all ML models
2. Asset-specific hyperparameters
3. Capital allocation settings
4. Risk management configurations
5. Advanced model settings
"""

import os
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
DEFAULT_CONFIG_PATH = "ml_config.json"
ENHANCED_CONFIG_PATH = "ml_enhanced_config.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create ML configuration files")
    
    parser.add_argument("--trading-pairs", nargs="+", default=DEFAULT_TRADING_PAIRS,
                      help="Trading pairs to include in configuration")
    
    parser.add_argument("--output", type=str, default=DEFAULT_CONFIG_PATH,
                      help="Path to save the standard configuration")
    
    parser.add_argument("--enhanced-output", type=str, default=ENHANCED_CONFIG_PATH,
                      help="Path to save the enhanced configuration")
    
    parser.add_argument("--backup", action="store_true", default=True,
                      help="Create backup of existing configuration files")
    
    return parser.parse_args()

def create_standard_config(trading_pairs):
    """Create standard ML configuration"""
    config = {
        "version": "1.0.0",
        "updated_at": datetime.now().isoformat(),
        "global_settings": {
            "default_capital_allocation": {
                pair: 1.0 / len(trading_pairs) for pair in trading_pairs
            },
            "max_open_positions": len(trading_pairs),
            "portfolio_rebalance_interval": 24,  # hours
            "enable_dynamic_allocation": True,
            "ml_confidence_threshold": 0.65,
            "minimum_trading_data": 1000,  # candles
            "sandbox_mode": True,
            "log_level": "INFO"
        },
        "model_settings": {
            "gru": {
                "enabled": True,
                "lookback_period": 40,
                "epochs": 100,
                "batch_size": 64,
                "dropout_rate": 0.2,
                "validation_split": 0.2,
                "use_early_stopping": True,
                "patience": 10
            },
            "lstm": {
                "enabled": True,
                "lookback_period": 50,
                "epochs": 120,
                "batch_size": 32,
                "dropout_rate": 0.3,
                "validation_split": 0.2,
                "use_early_stopping": True,
                "patience": 15
            },
            "tcn": {
                "enabled": True,
                "lookback_period": 60,
                "epochs": 150,
                "batch_size": 32,
                "dropout_rate": 0.2,
                "validation_split": 0.2,
                "use_early_stopping": True,
                "patience": 15,
                "nb_filters": 64,
                "kernel_size": 3,
                "nb_stacks": 2,
                "dilations": [1, 2, 4, 8, 16, 32]
            }
        },
        "feature_settings": {
            "technical_indicators": [
                "rsi", "macd", "ema", "bollinger_bands", "atr", "obv", "stoch", "adx"
            ],
            "price_data": ["open", "high", "low", "close", "volume"],
            "normalization": "min_max",
            "feature_engineering": True,
            "auto_feature_selection": True
        },
        "asset_specific_settings": {}
    }
    
    # Add asset-specific settings
    for pair in trading_pairs:
        asset_symbol = pair.split('/')[0]
        config["asset_specific_settings"][pair] = {
            "trading_enabled": True,
            "min_required_data": 1000,  # candles
            "hyperparameters": {
                "gru": {
                    "learning_rate": 0.001,
                    "units": 64,
                    "dropout": 0.2,
                    "recurrent_dropout": 0.2
                },
                "lstm": {
                    "learning_rate": 0.001,
                    "units": 64,
                    "dropout": 0.3,
                    "recurrent_dropout": 0.3
                },
                "tcn": {
                    "learning_rate": 0.001,
                    "optimization": "adam"
                }
            },
            "regime_weights": {
                "bullish": 1.2,
                "bearish": 1.0,
                "sideways": 0.8,
                "volatile": 1.5
            },
            "risk_params": {
                "max_leverage": 20,
                "confidence_threshold": 0.7,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.05,
                "trailing_stop_pct": 0.02
            }
        }
    
    return config

def create_enhanced_config(trading_pairs):
    """Create enhanced ML configuration"""
    enhanced_config = {
        "version": "2.0.0",
        "updated_at": datetime.now().isoformat(),
        "global_settings": {
            "default_capital_allocation": {
                pair: 1.0 / len(trading_pairs) for pair in trading_pairs
            },
            "max_open_positions": len(trading_pairs) * 2,  # Allow multiple positions per pair
            "portfolio_rebalance_interval": 12,  # hours
            "enable_dynamic_allocation": True,
            "ml_confidence_threshold": 0.75,
            "minimum_trading_data": 2000,  # candles
            "sandbox_mode": True,
            "log_level": "INFO",
            "max_model_age_hours": 24,  # Force retraining after this period
            "enable_reinforcement_learning": True,
            "enable_transfer_learning": True,
            "enable_neural_architecture_search": False,  # Resource intensive, default off
            "enable_explainable_ai": True,
            "enable_sentiment_analysis": True,
            "use_cross_asset_features": True
        },
        "model_settings": {
            "attention_gru": {
                "enabled": True,
                "lookback_period": 60,
                "epochs": 200,
                "batch_size": 32,
                "dropout_rate": 0.3,
                "validation_split": 0.2,
                "use_early_stopping": True,
                "patience": 20,
                "attention_units": 32,
                "bidirectional": True
            },
            "temporal_fusion_transformer": {
                "enabled": True,
                "lookback_period": 100,
                "forecast_horizon": 10,
                "epochs": 250,
                "batch_size": 64,
                "hidden_layer_size": 64,
                "attention_heads": 4,
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "use_early_stopping": True,
                "patience": 25
            },
            "multi_asset_fusion": {
                "enabled": True,
                "max_lookback_period": 120,
                "correlation_threshold": 0.6,
                "granger_max_lag": 10,
                "use_pca": True,
                "pca_components": 3
            },
            "ensemble": {
                "enabled": True,
                "voting_method": "weighted",
                "model_weights": {
                    "attention_gru": 0.3,
                    "temporal_fusion_transformer": 0.4,
                    "tcn": 0.2,
                    "lstm": 0.1
                }
            }
        },
        "feature_settings": {
            "technical_indicators": [
                "rsi", "macd", "ema", "bollinger_bands", "atr", "obv", "stoch", "adx",
                "keltner", "volume_profile", "volatility", "ichimoku", "fibonacci"
            ],
            "price_data": ["open", "high", "low", "close", "volume"],
            "market_data": ["funding_rate", "open_interest", "liquidity"],
            "sentiment_features": ["news_sentiment", "social_sentiment", "market_sentiment"],
            "normalization": "robust_scaler",
            "feature_engineering": True,
            "auto_feature_selection": True,
            "dimensionality_reduction": True
        },
        "sentiment_settings": {
            "model": "finbert",
            "news_sources_weight": 0.6,
            "social_sources_weight": 0.4,
            "sentiment_update_interval": 6,  # hours
            "sentiment_influence_factor": 0.3,  # How much sentiment affects signals
            "cache_expiry": 12,  # hours
            "min_sources_required": 3
        },
        "correlation_settings": {
            "analysis_interval": 24,  # hours
            "correlation_window": 30,  # days
            "lead_lag_max_lag": 5,  # days
            "min_correlation_coefficient": 0.5,
            "use_cointegration": True,
            "use_feature_fusion": True
        },
        "hyperparameter_tuning": {
            "enabled": True,
            "optimization_metric": "val_loss",
            "tune_interval": 72,  # hours
            "max_trials": 30,
            "tuning_epochs": 50,
            "random_seed": 42,
            "dynamic_adaptation": True,
            "adaptation_threshold": 0.1  # Performance change to trigger adaptation
        },
        "risk_management": {
            "position_sizing": {
                "method": "kelly",
                "max_position_size": 0.25,  # Max 25% of portfolio per position
                "kelly_fraction": 0.5,  # Half-Kelly for conservativeness
                "volatility_scaling": True
            },
            "dynamic_leverage": {
                "enabled": True,
                "base_leverage": 3.0,
                "max_leverage": 20.0,
                "confidence_multiplier": 1.5,
                "volatility_divider": 2.0
            },
            "adaptive_exit": {
                "enabled": True,
                "exit_probability_threshold": 0.6,
                "profit_taking_multiplier": 1.5,
                "dynamic_trailing_stop": True
            }
        },
        "training_parameters": {},
        "asset_specific_settings": {}
    }
    
    # Add training parameters for each pair
    for pair in trading_pairs:
        asset_symbol = pair.split('/')[0]
        
        # Customize based on asset volatility
        if asset_symbol in ["SOL", "DOT"]:
            volatility = "high"
        elif asset_symbol in ["BTC", "ETH"]:
            volatility = "medium"
        else:
            volatility = "low"
        
        # Set training parameters based on volatility profile
        if volatility == "high":
            epochs_multiplier = 1.5
            learning_rate = 0.0005
            max_leverage = 20.0
        elif volatility == "medium":
            epochs_multiplier = 1.2
            learning_rate = 0.001
            max_leverage = 10.0
        else:
            epochs_multiplier = 1.0
            learning_rate = 0.001
            max_leverage = 5.0
        
        enhanced_config["training_parameters"][pair] = {
            "epochs": int(200 * epochs_multiplier),
            "batch_size": 32,
            "learning_rate": learning_rate,
            "validation_split": 0.2,
            "early_stopping_patience": 20,
            "test_size": 0.1,
            "weight_regimes": True,
            "use_asymmetric_loss": True
        }
        
        # Add asset-specific settings
        enhanced_config["asset_specific_settings"][pair] = {
            "trading_enabled": True,
            "volatility_profile": volatility,
            "hyperparameters": {
                "attention_gru": {
                    "learning_rate": learning_rate,
                    "units": 128,
                    "dropout": 0.3,
                    "recurrent_dropout": 0.3,
                    "attention_units": 64 if volatility == "high" else 32
                },
                "temporal_fusion_transformer": {
                    "learning_rate": learning_rate,
                    "hidden_size": 128 if volatility == "high" else 64,
                    "attention_heads": 4,
                    "dropout": 0.2
                }
            },
            "regime_weights": {
                "bullish": 1.2,
                "bearish": 1.0,
                "sideways": 0.8,
                "volatile": 2.0 if volatility == "high" else 1.5
            },
            "risk_params": {
                "max_leverage": max_leverage,
                "confidence_threshold": 0.8 if volatility == "high" else 0.7,
                "stop_loss_pct": 0.05 if volatility == "high" else 0.03,
                "take_profit_pct": 0.10 if volatility == "high" else 0.05,
                "trailing_stop_pct": 0.03 if volatility == "high" else 0.02
            },
            "retraining_trigger": {
                "performance_threshold": 0.1,  # Retrain if accuracy drops by 10%
                "volatility_change_threshold": 0.5,  # Retrain if volatility changes by 50%
                "max_age_days": 5
            }
        }
    
    return enhanced_config

def create_backup(file_path):
    """Create backup of existing file"""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"
        
        try:
            with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup of {file_path} at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup of {file_path}: {e}")
            return False
    
    return True  # No file to backup

def save_config(config, file_path, create_backup_file=True):
    """Save configuration to file"""
    # Create backup if requested
    if create_backup_file:
        create_backup(file_path)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info(f"Creating ML configuration for {len(args.trading_pairs)} trading pairs")
    
    # Create standard configuration
    standard_config = create_standard_config(args.trading_pairs)
    saved_standard = save_config(standard_config, args.output, args.backup)
    
    # Create enhanced configuration
    enhanced_config = create_enhanced_config(args.trading_pairs)
    saved_enhanced = save_config(enhanced_config, args.enhanced_output, args.backup)
    
    if saved_standard and saved_enhanced:
        logger.info("Configuration files created successfully!")
        return 0
    else:
        logger.error("Failed to create one or more configuration files")
        return 1

if __name__ == "__main__":
    main()