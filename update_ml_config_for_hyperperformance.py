#!/usr/bin/env python3
"""
ML Configuration Updater for Hyperperformance

This script updates the ML configuration files (ml_config.json and ml_enhanced_config.json)
to optimize for achieving 90% win rate and 1000% returns. It implements:

1. Ultra-aggressive parameter settings for high performance
2. Increased training epochs and hyperparameter optimization
3. Strategy integration between ARIMA and Adaptive strategies
4. Optimized leverage and risk management settings
5. Enhanced multi-asset correlation analysis

Usage:
    python update_ml_config_for_hyperperformance.py [--max-leverage MAX_LEVERAGE]
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_config_updates.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
STANDARD_CONFIG_PATH = "ml_config.json"
ENHANCED_CONFIG_PATH = "ml_enhanced_config.json"
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]

def backup_config(config_path: str) -> str:
    """
    Create a backup of the config file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Path to backup file
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} does not exist, cannot create backup")
        return None
    
    backup_path = f"{config_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    try:
        with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup of {config_path} at {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Config file {config_path} does not exist")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def update_standard_config(config: Dict[str, Any], max_leverage: int = 125) -> Dict[str, Any]:
    """
    Update standard ML configuration for hyperperformance
    
    Args:
        config: Current configuration
        max_leverage: Maximum leverage setting
        
    Returns:
        Updated configuration
    """
    # Make a copy of the config to avoid modifying the original
    updated_config = config.copy() if config else {}
    
    # Update timestamp
    updated_config["updated_at"] = datetime.now().isoformat()
    
    # Global settings
    global_settings = updated_config.setdefault("global_settings", {})
    global_settings["ml_confidence_threshold"] = 0.75  # Increased from default
    global_settings["enable_dynamic_allocation"] = True
    global_settings["extreme_leverage_enabled"] = True
    global_settings["model_pruning_threshold"] = 0.6  # Higher threshold for pruning
    global_settings["model_selection_frequency"] = 12  # More frequent model selection
    global_settings["continuous_learning"] = True
    global_settings["training_priority"] = "critical"
    
    # Model settings
    model_settings = updated_config.setdefault("model_settings", {})
    
    # GRU settings
    gru = model_settings.setdefault("gru", {})
    gru["enabled"] = True
    gru["lookback_period"] = 60  # Increased lookback
    gru["epochs"] = 200  # More epochs for better training
    gru["batch_size"] = 32
    gru["dropout_rate"] = 0.3
    gru["validation_split"] = 0.2
    gru["use_early_stopping"] = True
    gru["patience"] = 15
    
    # LSTM settings
    lstm = model_settings.setdefault("lstm", {})
    lstm["enabled"] = True
    lstm["lookback_period"] = 80  # Increased lookback
    lstm["epochs"] = 250  # More epochs for better training
    lstm["batch_size"] = 32
    lstm["dropout_rate"] = 0.3
    lstm["validation_split"] = 0.2
    lstm["use_early_stopping"] = True
    lstm["patience"] = 20
    
    # TCN settings
    tcn = model_settings.setdefault("tcn", {})
    tcn["enabled"] = True
    tcn["lookback_period"] = 100  # Increased lookback
    tcn["epochs"] = 300  # More epochs for better training
    tcn["batch_size"] = 32
    tcn["dropout_rate"] = 0.2
    tcn["validation_split"] = 0.2
    tcn["use_early_stopping"] = True
    tcn["patience"] = 20
    tcn["nb_filters"] = 64
    tcn["kernel_size"] = 3
    tcn["nb_stacks"] = 2
    tcn["dilations"] = [1, 2, 4, 8, 16, 32, 64]  # Extended dilations for longer-term patterns
    
    # Feature settings
    feature_settings = updated_config.setdefault("feature_settings", {})
    feature_settings["normalization"] = "robust_scaler"  # Better for financial data
    feature_settings["feature_engineering"] = True
    feature_settings["auto_feature_selection"] = True
    
    # Make sure we have all important technical indicators
    feature_settings.setdefault("technical_indicators", [
        "rsi", "macd", "ema", "bollinger_bands", "atr", "obv", "stoch", "adx",
        "keltner", "volume_profile", "volatility", "ichimoku", "fibonacci"
    ])
    
    # Update trading pairs
    trading_pairs = global_settings.get("default_capital_allocation", {}).keys()
    if not trading_pairs:
        # Default allocation if none exists
        global_settings["default_capital_allocation"] = {pair: 1.0 / len(DEFAULT_TRADING_PAIRS) for pair in DEFAULT_TRADING_PAIRS}
        trading_pairs = DEFAULT_TRADING_PAIRS
    
    # Update asset-specific settings for each pair
    asset_specific = updated_config.setdefault("asset_specific_settings", {})
    
    for pair in trading_pairs:
        pair_settings = asset_specific.setdefault(pair, {})
        pair_settings["trading_enabled"] = True
        pair_settings["min_required_data"] = 2000  # More data required for training
        
        # Hyperparameters
        hyperparams = pair_settings.setdefault("hyperparameters", {})
        
        # GRU hyperparameters
        gru_hp = hyperparams.setdefault("gru", {})
        gru_hp["learning_rate"] = 0.0005  # Lower learning rate for better convergence
        gru_hp["units"] = 128  # More units
        gru_hp["dropout"] = 0.3
        gru_hp["recurrent_dropout"] = 0.3
        
        # LSTM hyperparameters
        lstm_hp = hyperparams.setdefault("lstm", {})
        lstm_hp["learning_rate"] = 0.0005
        lstm_hp["units"] = 128
        lstm_hp["dropout"] = 0.3
        lstm_hp["recurrent_dropout"] = 0.3
        
        # TCN hyperparameters
        tcn_hp = hyperparams.setdefault("tcn", {})
        tcn_hp["learning_rate"] = 0.0005
        tcn_hp["optimization"] = "adam"
        
        # Regime weights - higher weight for volatile markets
        regime_weights = pair_settings.setdefault("regime_weights", {})
        regime_weights["bullish"] = 1.5  # Increased weight for bullish markets
        regime_weights["bearish"] = 1.2  # Increased weight for bearish markets
        regime_weights["sideways"] = 0.8
        regime_weights["volatile"] = 2.0  # Much higher weight for volatile markets
        
        # Risk parameters - aggressive settings
        risk_params = pair_settings.setdefault("risk_params", {})
        risk_params["max_leverage"] = max_leverage
        risk_params["confidence_threshold"] = 0.7
        risk_params["stop_loss_pct"] = 0.03
        risk_params["take_profit_pct"] = 0.05
        risk_params["trailing_stop_pct"] = 0.02
    
    # Create training parameters section if not exists
    training_params = updated_config.setdefault("training_parameters", {})
    training_params["priority"] = "high"
    training_params["max_epochs"] = 500
    training_params["early_stopping_patience"] = 25
    training_params["learning_rate_schedule"] = "cosine_decay"
    training_params["batch_size"] = 32
    training_params["validation_split"] = 0.2
    training_params["training_data_min_samples"] = 5000
    training_params["augment_data"] = True
    training_params["shuffle_training_data"] = True
    training_params["stratify_samples"] = True
    
    # Strategy integration settings - crucial for our dual-strategy goal
    strategy_integration = updated_config.setdefault("strategy_integration", {})
    strategy_integration["integrate_arima_adaptive"] = True
    strategy_integration["arima_weight"] = 0.5  # Equal weight to ARIMA
    strategy_integration["adaptive_weight"] = 0.5  # Equal weight to Adaptive
    strategy_integration["use_combined_signals"] = True
    strategy_integration["signal_priority"] = "confidence"  # Prioritize signals based on confidence
    strategy_integration["signal_threshold"] = 0.7
    
    # Leverage optimization settings
    leverage_optimization = updated_config.setdefault("leverage_optimization", {})
    leverage_optimization["base_leverage"] = 5
    leverage_optimization["max_leverage"] = max_leverage
    leverage_optimization["min_leverage"] = 1
    leverage_optimization["confidence_multiplier"] = 1.5
    leverage_optimization["win_streak_bonus"] = 0.2
    leverage_optimization["loss_streak_penalty"] = 0.5
    leverage_optimization["volatility_scaling"] = True
    
    return updated_config

def update_enhanced_config(config: Dict[str, Any], max_leverage: int = 125) -> Dict[str, Any]:
    """
    Update enhanced ML configuration for hyperperformance
    
    Args:
        config: Current configuration
        max_leverage: Maximum leverage setting
        
    Returns:
        Updated configuration
    """
    # Make a copy of the config to avoid modifying the original
    updated_config = config.copy() if config else {}
    
    # Update timestamp and version
    updated_config["updated_at"] = datetime.now().isoformat()
    updated_config["version"] = "2.0.0"
    
    # Global settings
    global_settings = updated_config.setdefault("global_settings", {})
    global_settings["ml_confidence_threshold"] = 0.8  # Higher threshold for enhanced version
    global_settings["enable_dynamic_allocation"] = True
    global_settings["max_open_positions"] = 15  # Allow more concurrent positions
    global_settings["portfolio_rebalance_interval"] = 6  # More frequent rebalancing
    global_settings["minimum_trading_data"] = 3000  # More data required
    global_settings["max_model_age_hours"] = 12  # More frequent model retraining
    global_settings["enable_reinforcement_learning"] = True
    global_settings["enable_transfer_learning"] = True
    global_settings["enable_neural_architecture_search"] = True  # Enable NAS for optimal architectures
    global_settings["enable_explainable_ai"] = True
    global_settings["enable_sentiment_analysis"] = True
    global_settings["use_cross_asset_features"] = True
    global_settings["training_priority"] = "critical"
    global_settings["continuous_learning"] = True
    
    # Trading pairs
    trading_pairs = global_settings.get("default_capital_allocation", {}).keys()
    if not trading_pairs:
        # Default allocation if none exists
        global_settings["default_capital_allocation"] = {pair: 1.0 / len(DEFAULT_TRADING_PAIRS) for pair in DEFAULT_TRADING_PAIRS}
        trading_pairs = DEFAULT_TRADING_PAIRS
    
    # Enhanced model settings
    model_settings = updated_config.setdefault("model_settings", {})
    
    # Attention GRU settings
    attention_gru = model_settings.setdefault("attention_gru", {})
    attention_gru["enabled"] = True
    attention_gru["lookback_period"] = 100
    attention_gru["epochs"] = 300
    attention_gru["batch_size"] = 32
    attention_gru["dropout_rate"] = 0.3
    attention_gru["validation_split"] = 0.2
    attention_gru["use_early_stopping"] = True
    attention_gru["patience"] = 20
    attention_gru["attention_units"] = 64
    attention_gru["bidirectional"] = True
    
    # Temporal Fusion Transformer settings
    tft = model_settings.setdefault("temporal_fusion_transformer", {})
    tft["enabled"] = True
    tft["lookback_period"] = 200  # Extended lookback
    tft["forecast_horizon"] = 24  # Longer forecast horizon
    tft["epochs"] = 500  # Much more epochs for this advanced model
    tft["batch_size"] = 64
    tft["hidden_layer_size"] = 128
    tft["attention_heads"] = 8
    tft["dropout_rate"] = 0.2
    tft["learning_rate"] = 0.0003
    tft["use_early_stopping"] = True
    tft["patience"] = 30
    
    # Multi-asset fusion settings
    maf = model_settings.setdefault("multi_asset_fusion", {})
    maf["enabled"] = True
    maf["max_lookback_period"] = 240  # Extended lookback for correlation analysis
    maf["correlation_threshold"] = 0.5
    maf["granger_max_lag"] = 20
    maf["use_pca"] = True
    maf["pca_components"] = 5
    
    # Ensemble settings
    ensemble = model_settings.setdefault("ensemble", {})
    ensemble["enabled"] = True
    ensemble["voting_method"] = "weighted"
    ensemble["model_weights"] = {
        "attention_gru": 0.3,
        "temporal_fusion_transformer": 0.4,
        "tcn": 0.2,
        "lstm": 0.1
    }
    
    # Feature settings
    feature_settings = updated_config.setdefault("feature_settings", {})
    feature_settings["normalization"] = "robust_scaler"
    feature_settings["feature_engineering"] = True
    feature_settings["auto_feature_selection"] = True
    feature_settings["dimensionality_reduction"] = True
    
    # Comprehensive technical indicators
    feature_settings.setdefault("technical_indicators", [
        "rsi", "macd", "ema", "bollinger_bands", "atr", "obv", "stoch", "adx",
        "keltner", "volume_profile", "volatility", "ichimoku", "fibonacci",
        "parabolic_sar", "supertrend", "vwap", "elder_ray", "squeeze_momentum",
        "williams_r", "fractals", "pivot_points", "market_cipher"
    ])
    
    # Price data
    feature_settings.setdefault("price_data", ["open", "high", "low", "close", "volume"])
    
    # Market data
    feature_settings.setdefault("market_data", ["funding_rate", "open_interest", "liquidity"])
    
    # Sentiment features
    feature_settings.setdefault("sentiment_features", [
        "news_sentiment", "social_sentiment", "market_sentiment", "fear_greed_index"
    ])
    
    # Sentiment settings
    sentiment_settings = updated_config.setdefault("sentiment_settings", {})
    sentiment_settings["model"] = "finbert"
    sentiment_settings["news_sources_weight"] = 0.6
    sentiment_settings["social_sources_weight"] = 0.4
    sentiment_settings["sentiment_update_interval"] = 4  # More frequent updates
    sentiment_settings["sentiment_influence_factor"] = 0.4  # Increased influence
    sentiment_settings["cache_expiry"] = 6
    sentiment_settings["min_sources_required"] = 3
    
    # Correlation settings
    correlation_settings = updated_config.setdefault("correlation_settings", {})
    correlation_settings["analysis_interval"] = 12  # More frequent analysis
    correlation_settings["correlation_window"] = 60
    correlation_settings["lead_lag_max_lag"] = 10
    correlation_settings["min_correlation_coefficient"] = 0.4
    correlation_settings["use_cointegration"] = True
    correlation_settings["use_feature_fusion"] = True
    
    # Hyperparameter tuning
    hyperparameter_tuning = updated_config.setdefault("hyperparameter_tuning", {})
    hyperparameter_tuning["enabled"] = True
    hyperparameter_tuning["optimization_metric"] = "val_loss"
    hyperparameter_tuning["tune_interval"] = 48  # More frequent tuning
    hyperparameter_tuning["max_trials"] = 50  # More trials
    hyperparameter_tuning["tuning_epochs"] = 100
    hyperparameter_tuning["random_seed"] = 42
    hyperparameter_tuning["dynamic_adaptation"] = True
    hyperparameter_tuning["adaptation_threshold"] = 0.1
    
    # Risk management
    risk_management = updated_config.setdefault("risk_management", {})
    
    # Position sizing
    position_sizing = risk_management.setdefault("position_sizing", {})
    position_sizing["method"] = "kelly"
    position_sizing["max_position_size"] = 0.35  # Larger max position size
    position_sizing["kelly_fraction"] = 0.8  # More aggressive Kelly fraction
    position_sizing["volatility_scaling"] = True
    
    # Dynamic leverage
    dynamic_leverage = risk_management.setdefault("dynamic_leverage", {})
    dynamic_leverage["enabled"] = True
    dynamic_leverage["base_leverage"] = 10  # Higher base leverage
    dynamic_leverage["max_leverage"] = max_leverage
    dynamic_leverage["confidence_multiplier"] = 2.0  # Higher multiplier
    dynamic_leverage["volatility_divider"] = 1.5  # Less reduction for volatility
    
    # Asset specific settings
    asset_specific = updated_config.setdefault("asset_specific_settings", {})
    
    for pair in trading_pairs:
        pair_settings = asset_specific.setdefault(pair, {})
        pair_settings["trading_enabled"] = True
        pair_settings["volatility_profile"] = "adaptive"  # Adapt to changing volatility
        
        # Hyperparameters
        hyperparams = pair_settings.setdefault("hyperparameters", {})
        
        # Attention GRU hyperparameters
        attention_gru_hp = hyperparams.setdefault("attention_gru", {})
        attention_gru_hp["learning_rate"] = 0.0003
        attention_gru_hp["units"] = 128
        attention_gru_hp["dropout"] = 0.3
        attention_gru_hp["recurrent_dropout"] = 0.3
        attention_gru_hp["use_bidirectional"] = True
        attention_gru_hp["attention_size"] = 64
        
        # TFT hyperparameters
        tft_hp = hyperparams.setdefault("tft", {})
        tft_hp["learning_rate"] = 0.0003
        tft_hp["hidden_size"] = 128
        tft_hp["attention_heads"] = 8
        tft_hp["dropout"] = 0.2
        tft_hp["hidden_continuous_size"] = 64
        
        # Aggressive risk parameters
        risk_params = pair_settings.setdefault("risk_params", {})
        risk_params["max_leverage"] = max_leverage
        risk_params["confidence_threshold"] = 0.7
        risk_params["stop_loss_pct"] = 0.04  # Wider stop loss
        risk_params["take_profit_pct"] = 0.08  # Higher take profit
        risk_params["trailing_stop_pct"] = 0.03
        
        # Retraining triggers
        retraining = pair_settings.setdefault("retraining_trigger", {})
        retraining["performance_threshold"] = 0.05  # Retrain if performance drops by 5%
        retraining["volatility_change_threshold"] = 0.4
        retraining["max_age_days"] = 3  # More frequent retraining
    
    # Strategy integration settings - crucial for our dual-strategy goal
    strategy_integration = updated_config.setdefault("strategy_integration", {})
    strategy_integration["integrate_arima_adaptive"] = True
    strategy_integration["arima_weight"] = 0.5  # Equal weight to ARIMA
    strategy_integration["adaptive_weight"] = 0.5  # Equal weight to Adaptive
    strategy_integration["use_combined_signals"] = True
    strategy_integration["signal_priority"] = "confidence"
    strategy_integration["signal_threshold"] = 0.7
    strategy_integration["use_reinforcement_learning"] = True
    strategy_integration["dynamic_weight_adjustment"] = True
    strategy_integration["performance_based_weighting"] = True
    
    # Add asymmetric loss function settings for training
    loss_function = updated_config.setdefault("loss_function", {})
    loss_function["type"] = "asymmetric"
    loss_function["false_positive_penalty"] = 2.0  # Higher penalty for false buy signals
    loss_function["false_negative_penalty"] = 1.5  # Penalty for missed opportunities
    loss_function["use_focal_loss"] = True
    loss_function["gamma"] = 2.0  # Focus parameter for focal loss
    
    # Enhanced ML strategy settings
    ml_strategy = updated_config.setdefault("ml_strategy", {})
    ml_strategy["predict_direction"] = True
    ml_strategy["predict_magnitude"] = True
    ml_strategy["predict_leverage"] = True
    ml_strategy["predict_stop_loss"] = True
    ml_strategy["predict_take_profit"] = True
    ml_strategy["predict_holding_period"] = True
    ml_strategy["use_market_regime_detection"] = True
    ml_strategy["use_dynamic_exit_strategy"] = True
    ml_strategy["confidence_weighting"] = True
    
    # Training data configuration
    training_data = updated_config.setdefault("training_data", {})
    training_data["timeframes"] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    training_data["min_samples"] = 5000
    training_data["historical_window_days"] = 365  # Use a full year of data
    training_data["validation_split"] = 0.2
    training_data["test_split"] = 0.1
    training_data["use_walk_forward_validation"] = True
    training_data["data_augmentation"] = True
    training_data["balance_classes"] = True
    
    # Adaptive training schedule
    training_schedule = updated_config.setdefault("training_schedule", {})
    training_schedule["continuous_training"] = True
    training_schedule["incremental_training"] = True
    training_schedule["scheduled_retraining_hours"] = 12  # Retrain every 12 hours
    training_schedule["performance_triggered_retraining"] = True  # Retrain when performance drops
    training_schedule["market_regime_triggered_retraining"] = True  # Retrain on market regime changes
    
    return updated_config

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Update ML configurations for hyperperformance")
    parser.add_argument("--max-leverage", type=int, default=125, help="Maximum leverage setting (default: 125)")
    args = parser.parse_args()
    
    # Backup existing configs
    backup_config(STANDARD_CONFIG_PATH)
    backup_config(ENHANCED_CONFIG_PATH)
    
    # Load existing configs
    standard_config = load_config(STANDARD_CONFIG_PATH)
    enhanced_config = load_config(ENHANCED_CONFIG_PATH)
    
    # Update configs
    standard_config = update_standard_config(standard_config, args.max_leverage)
    enhanced_config = update_enhanced_config(enhanced_config, args.max_leverage)
    
    # Save updated configs
    save_config(standard_config, STANDARD_CONFIG_PATH)
    save_config(enhanced_config, ENHANCED_CONFIG_PATH)
    
    logger.info("ML configuration updated for hyperperformance")
    logger.info(f"Maximum leverage set to {args.max_leverage}")
    logger.info("Next steps:")
    logger.info("1. Run 'python enhanced_strategy_training.py' to train integrated ARIMA-Adaptive models")
    logger.info("2. Run 'python run_optimized_ml_trading.py --reset --sandbox' to start trading with the new settings")

if __name__ == "__main__":
    main()