#!/usr/bin/env python3
"""
Update ML Configuration Script

This script updates the ML configuration to better integrate with the trading system
by optimizing hyperparameters, adjusting leverage settings, and fine-tuning the
strategy integration parameters.

Usage:
    python update_ml_config.py [--update-ensemble] [--reset-weights] [--backtest]
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_config_update.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Update ML configuration for better trading integration')
    parser.add_argument('--update-ensemble', action='store_true', 
                        help='Update ensemble configurations with current model weights')
    parser.add_argument('--reset-weights', action='store_true', 
                        help='Reset model weights based on recent performance')
    parser.add_argument('--backtest', action='store_true', 
                        help='Run backtest after updating configuration')
    parser.add_argument('--pairs', type=str, default='SOLUSD,BTCUSD,ETHUSD', 
                        help='Comma-separated list of trading pairs to update')
    
    return parser.parse_args()

def load_ml_config():
    """Load the current ML configuration."""
    config_path = 'ml_config.json'
    
    if not os.path.exists(config_path):
        logger.error(f"ML configuration file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def save_ml_config(config):
    """Save the updated ML configuration."""
    config_path = 'ml_config.json'
    
    # Update timestamp
    config['updated_at'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ML configuration saved to {config_path}")
    
    return True

def update_global_settings(config):
    """Update global ML settings for improved performance."""
    logger.info("Updating global ML settings")
    
    global_settings = config['global_settings']
    
    # Fine-tune confidence threshold for better signal quality
    global_settings['ml_confidence_threshold'] = 0.65
    
    # Adjust minimum trading data requirement based on our dataset size
    global_settings['minimum_trading_data'] = 456
    
    # Enable extreme leverage for maximum returns
    global_settings['extreme_leverage_enabled'] = True
    
    # Adjust model pruning threshold for better model selection
    global_settings['model_pruning_threshold'] = 0.5
    
    # Set model selection frequency (hours) for more frequent updates
    global_settings['model_selection_frequency'] = 6
    
    # Enable continuous learning for ongoing improvement
    global_settings['continuous_learning'] = True
    
    # Set training priority to critical
    global_settings['training_priority'] = "critical"
    
    logger.info("Global ML settings updated")
    
    return config

def update_model_settings(config):
    """Update model-specific settings for improved training."""
    logger.info("Updating model-specific settings")
    
    model_settings = config['model_settings']
    
    # Update LSTM settings
    if 'lstm' in model_settings:
        model_settings['lstm']['lookback_period'] = 60
        model_settings['lstm']['epochs'] = 250
        model_settings['lstm']['batch_size'] = 32
        model_settings['lstm']['dropout_rate'] = 0.3
        model_settings['lstm']['patience'] = 20
    
    # Update TCN settings
    if 'tcn' in model_settings:
        model_settings['tcn']['lookback_period'] = 60
        model_settings['tcn']['epochs'] = 300
        model_settings['tcn']['batch_size'] = 32
        model_settings['tcn']['dropout_rate'] = 0.2
        model_settings['tcn']['nb_filters'] = 64
    
    # Add transformer settings if not present
    if 'transformer' not in model_settings:
        model_settings['transformer'] = {
            "enabled": True,
            "lookback_period": 60,
            "epochs": 300,
            "batch_size": 32,
            "dropout_rate": 0.2,
            "validation_split": 0.2,
            "use_early_stopping": True,
            "patience": 20,
            "d_model": 64,
            "n_heads": 4,
            "ff_dim": 128,
            "n_encoder_layers": 2
        }
    
    logger.info("Model-specific settings updated")
    
    return config

def update_feature_settings(config):
    """Update feature selection and engineering settings."""
    logger.info("Updating feature settings")
    
    feature_settings = config['feature_settings']
    
    # Ensure all important technical indicators are included
    all_indicators = [
        "rsi", "macd", "ema", "bollinger_bands", "atr", "obv", "stoch", "adx",
        "kc", "ichimoku", "vwap", "psar", "heikin_ashi", "cmf", "wma", "pivot_points"
    ]
    
    feature_settings['technical_indicators'] = all_indicators
    
    # Add volume delta if not present
    if 'volume_delta' not in feature_settings['price_data']:
        feature_settings['price_data'].append('volume_delta')
    
    # Set optimal normalization method
    feature_settings['normalization'] = "robust_scaler"
    
    # Enable feature engineering and auto selection
    feature_settings['feature_engineering'] = True
    feature_settings['auto_feature_selection'] = True
    
    # Add feature importance tracking
    feature_settings['track_feature_importance'] = True
    
    logger.info("Feature settings updated")
    
    return config

def update_asset_specific_settings(config, pairs):
    """Update asset-specific settings for specified pairs."""
    logger.info(f"Updating asset-specific settings for {', '.join(pairs)}")
    
    asset_settings = config['asset_specific_settings']
    
    for pair in pairs:
        # Convert pair format (e.g., SOLUSD to SOL/USD)
        if '/' not in pair:
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
        else:
            formatted_pair = pair
        
        # Skip if asset not in config
        if formatted_pair not in asset_settings:
            logger.warning(f"Asset {formatted_pair} not found in configuration, skipping")
            continue
        
        # Update asset-specific settings
        asset_config = asset_settings[formatted_pair]
        
        # Enable trading
        asset_config['trading_enabled'] = True
        
        # Set minimum required data based on our dataset
        asset_config['min_required_data'] = 456
        
        # Update model hyperparameters
        asset_config['hyperparameters'] = {
            "lstm": {
                "learning_rate": 0.0005,
                "units": 128,
                "dropout": 0.3,
                "recurrent_dropout": 0.3
            },
            "gru": {
                "learning_rate": 0.0005,
                "units": 128,
                "dropout": 0.3,
                "recurrent_dropout": 0.3
            },
            "tcn": {
                "learning_rate": 0.0005,
                "optimization": "adam"
            },
            "transformer": {
                "learning_rate": 0.0005,
                "d_model": 64,
                "n_heads": 4
            }
        }
        
        # Update regime weights for adaptive behavior
        asset_config['regime_weights'] = {
            "bullish": 1.5,
            "bearish": 1.2,
            "sideways": 0.8,
            "volatile": 2.0
        }
        
        # Update risk parameters
        asset_config['risk_params'] = {
            "max_leverage": 125,
            "min_leverage": 20,
            "confidence_threshold": 0.65,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.06,
            "trailing_stop_pct": 0.02
        }
        
        logger.info(f"Updated settings for {formatted_pair}")
    
    return config

def update_training_parameters(config):
    """Update training parameters for improved model performance."""
    logger.info("Updating training parameters")
    
    training_params = config['training_parameters']
    
    # Set training priority
    training_params['priority'] = "high"
    
    # Optimize training parameters
    training_params['max_epochs'] = 300
    training_params['early_stopping_patience'] = 30
    training_params['learning_rate_schedule'] = "cosine_decay"
    training_params['batch_size'] = 32
    training_params['validation_split'] = 0.2
    
    # Adjust for smaller dataset
    training_params['training_data_min_samples'] = 456
    
    # Enable data augmentation for better generalization
    training_params['augment_data'] = True
    training_params['shuffle_training_data'] = True
    
    # Use class weights for imbalanced data
    training_params['use_class_weights'] = True
    
    # Use focal loss for better classification
    training_params['use_focal_loss'] = True
    training_params['focal_loss_gamma'] = 2.0
    
    # Enable asymmetric loss
    training_params['use_asymmetric_loss'] = True
    training_params['asymmetric_loss_gamma_pos'] = 1.0
    training_params['asymmetric_loss_gamma_neg'] = 4.0
    
    logger.info("Training parameters updated")
    
    return config

def update_strategy_integration(config):
    """Update strategy integration parameters for optimal trading signals."""
    logger.info("Updating strategy integration parameters")
    
    strategy_params = config['strategy_integration']
    
    # Enable integration of ARIMA and Adaptive strategies
    strategy_params['integrate_arima_adaptive'] = True
    
    # Adjust strategy weights
    strategy_params['arima_weight'] = 0.6  # Slightly favor ARIMA for its trending accuracy
    strategy_params['adaptive_weight'] = 0.4
    
    # Enable combined signals
    strategy_params['use_combined_signals'] = True
    
    # Set signal priority based on confidence
    strategy_params['signal_priority'] = "confidence"
    
    # Adjust signal threshold
    strategy_params['signal_threshold'] = 0.65
    
    # Add conflict resolution strategy if not present
    if 'conflict_resolution' not in strategy_params:
        strategy_params['conflict_resolution'] = "favor_strongest"
    
    logger.info("Strategy integration parameters updated")
    
    return config

def update_leverage_optimization(config):
    """Update leverage optimization parameters for risk-adjusted returns."""
    logger.info("Updating leverage optimization parameters")
    
    leverage_params = config['leverage_optimization']
    
    # Set leverage limits
    leverage_params['base_leverage'] = 20
    leverage_params['max_leverage'] = 125
    leverage_params['min_leverage'] = 5
    
    # Update confidence multiplier
    leverage_params['confidence_multiplier'] = 1.2
    
    # Adjust streak bonuses/penalties
    leverage_params['win_streak_bonus'] = 0.3
    leverage_params['loss_streak_penalty'] = 0.6
    
    # Enable volatility scaling
    leverage_params['volatility_scaling'] = True
    
    # Add market regime adjustment if not present
    if 'regime_adjustment' not in leverage_params:
        leverage_params['regime_adjustment'] = {
            "trending_up": 1.2,
            "trending_down": 0.8,
            "volatile": 0.6,
            "sideways": 1.0
        }
    
    logger.info("Leverage optimization parameters updated")
    
    return config

def update_ensemble_configurations(pairs, reset_weights=False):
    """Update ensemble configurations for specified pairs."""
    logger.info(f"Updating ensemble configurations for {', '.join(pairs)}")
    
    for pair in pairs:
        ensemble_dir = 'models/ensemble'
        weights_path = f'{ensemble_dir}/{pair}_weights.json'
        ensemble_path = f'{ensemble_dir}/{pair}_ensemble.json'
        
        # Skip if ensemble config does not exist
        if not os.path.exists(ensemble_path):
            logger.warning(f"Ensemble configuration not found for {pair}, skipping")
            continue
        
        # Load current configurations
        with open(ensemble_path, 'r') as f:
            ensemble_config = json.load(f)
        
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                weights = json.load(f)
        else:
            logger.warning(f"Weights configuration not found for {pair}, skipping")
            continue
        
        # Reset weights if requested
        if reset_weights:
            logger.info(f"Resetting weights for {pair}")
            
            # Get all available models
            model_types = list(weights.keys())
            
            # Set equal weights
            weight_value = 1.0 / len(model_types)
            weights = {model_type: weight_value for model_type in model_types}
        
        # Update ensemble models with current weights
        models = {}
        sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:3]  # Top 3 models by weight
        
        # Normalize weights for top models
        top_weight_sum = sum(weight for _, weight in top_models)
        
        for model_type, weight in top_models:
            model_path = f"models/{model_type}/{pair}_{model_type}.h5"
            
            # Skip if model file doesn't exist
            if not os.path.exists(model_path):
                model_path = f"models/{model_type}/{pair}.h5"  # Try alternative path
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found for {model_type}, skipping")
                    continue
            
            models[model_type] = {
                "path": model_path,
                "weight": weight / top_weight_sum  # Normalize
            }
        
        # Update ensemble configuration
        ensemble_config["models"] = models
        ensemble_config["parameters"]["confidence_threshold"] = 0.65
        ensemble_config["parameters"]["voting_method"] = "weighted"
        ensemble_config["parameters"]["trained_date"] = datetime.now().isoformat()
        
        # Save updated ensemble configuration
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        # Save weights
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Updated ensemble configuration for {pair}")
    
    return True

def run_backtest(pairs):
    """Run backtest to verify configuration changes."""
    logger.info(f"Running backtest for {', '.join(pairs)}")
    
    # Import backtest module
    try:
        import subprocess
        
        for pair in pairs:
            # Convert pair format
            if '/' not in pair:
                formatted_pair = f"{pair[:3]}/{pair[3:]}"
            else:
                formatted_pair = pair
            
            # Run backtest command
            cmd = [
                "python", "comprehensive_backtest.py",
                "--pair", formatted_pair,
                "--use-ml",
                "--days", "30"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Backtest completed successfully for {formatted_pair}")
                logger.info(result.stdout)
            else:
                logger.error(f"Backtest failed for {formatted_pair}: {result.stderr}")
        
        logger.info("All backtests completed")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return False
    
    return True

def main():
    """Main function to update ML configuration."""
    args = parse_arguments()
    
    # Parse pairs
    pairs = [pair.strip().upper() for pair in args.pairs.split(',')]
    
    # Load current configuration
    config = load_ml_config()
    if config is None:
        return
    
    # Update configuration sections
    config = update_global_settings(config)
    config = update_model_settings(config)
    config = update_feature_settings(config)
    config = update_asset_specific_settings(config, pairs)
    config = update_training_parameters(config)
    config = update_strategy_integration(config)
    config = update_leverage_optimization(config)
    
    # Save updated configuration
    if not save_ml_config(config):
        logger.error("Failed to save ML configuration")
        return
    
    # Update ensemble configurations if requested
    if args.update_ensemble:
        if not update_ensemble_configurations(pairs, args.reset_weights):
            logger.error("Failed to update ensemble configurations")
            return
    
    # Run backtest if requested
    if args.backtest:
        if not run_backtest(pairs):
            logger.error("Failed to run backtest")
            return
    
    logger.info("ML configuration update completed successfully")

if __name__ == "__main__":
    main()