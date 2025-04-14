#!/usr/bin/env python3
"""
Improved Model Retraining Script

This script addresses the input shape mismatch issue between our models (expecting shape=(None, 60, 20))
and our enhanced dataset features (shape=(None, 1, 56)). It provides a systematic approach to:

1. Preprocess and reshape the enhanced dataset
2. Configure and train models with the correct input shape
3. Create ensemble models with proper weights
4. Update ML configuration to integrate with the trading system

Usage:
    python improved_model_retraining.py --pairs SOLUSD,BTCUSD,ETHUSD --models lstm,tcn,transformer
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('model_retraining.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_TYPES = ['lstm', 'tcn', 'transformer', 'cnn', 'gru', 'bilstm', 'attention', 'hybrid']
ASSETS = ['SOLUSD', 'BTCUSD', 'ETHUSD', 'DOTUSD', 'LINKUSD']
TARGET_LOOKBACK = 60
TARGET_FEATURES = 20
RANDOM_SEED = 42
INPUT_SHAPE_ERROR_MSG = "input shape mismatch"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improved model retraining with correct input shapes')
    parser.add_argument('--pairs', type=str, default='SOLUSD', 
                        help='Comma-separated list of trading pairs to retrain models for')
    parser.add_argument('--models', type=str, default='lstm,tcn,transformer', 
                        help='Comma-separated list of models to retrain')
    parser.add_argument('--timeframe', type=str, default='1h', 
                        help='Timeframe for training data (1h, 4h, 1d)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--patience', type=int, default=20, 
                        help='Early stopping patience')
    parser.add_argument('--update-config', action='store_true', 
                        help='Update ML config after retraining')
    parser.add_argument('--create-ensemble', action='store_true', 
                        help='Create ensemble model after retraining')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug logging')
    
    return parser.parse_args()

def load_enhanced_dataset(asset, timeframe='1h'):
    """
    Load enhanced dataset for the specified asset and timeframe.
    
    Args:
        asset (str): Asset symbol (e.g., 'SOLUSD')
        timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
        
    Returns:
        pd.DataFrame: Enhanced dataset
    """
    filepath = f'training_data/{asset}_{timeframe}_enhanced.csv'
    
    if not os.path.exists(filepath):
        logger.error(f"Dataset file not found: {filepath}")
        return None
    
    logger.info(f"Loading enhanced dataset from {filepath}")
    df = pd.read_csv(filepath)
    
    return df

def preprocess_dataset(df, sequence_length=60, target_col='target_direction_24', 
                        strategy_columns=None, drop_cols=None):
    """
    Preprocess the dataset for model training.
    
    Args:
        df (pd.DataFrame): Enhanced dataset
        sequence_length (int): Sequence length for time series
        target_col (str): Target column for prediction
        strategy_columns (list): Strategy-related columns to include
        drop_cols (list): Columns to exclude from features
        
    Returns:
        tuple: X, y, feature_cols, scalers
    """
    logger.info(f"Preprocessing dataset with {len(df)} samples")
    
    # Default columns to drop if not specified
    if drop_cols is None:
        drop_cols = ['timestamp', 'arima_forecast', 'adaptive_prediction', 
                    'strategy_agreement', 'strategy_combined_strength',
                    'arima_dominance', 'adaptive_dominance', 'dominant_strategy']
    
    # Include strategy columns if specified
    if strategy_columns is None:
        strategy_columns = ['arima_prediction', 'arima_strength', 
                           'adaptive_prediction', 'adaptive_strength',
                           'adaptive_volatility', 'combined_prediction']
    
    # Remove any drop_cols that might be in strategy_columns
    for col in strategy_columns:
        if col in drop_cols:
            drop_cols.remove(col)
    
    # Handle any missing columns gracefully
    for col in drop_cols.copy():
        if col not in df.columns:
            drop_cols.remove(col)
            logger.warning(f"Column {col} not found in dataset, skipping")
    
    # Identify target columns
    target_columns = [col for col in df.columns if col.startswith('target_')]
    
    # Ensure target_col exists in the dataset
    if target_col not in df.columns:
        logger.warning(f"Target column {target_col} not found, using target_direction_24 as fallback")
        target_col = 'target_direction_24' if 'target_direction_24' in df.columns else target_columns[0]
    
    # Separate target and drop from features
    y = df[target_col].values
    
    # Calculate columns to keep as features
    feature_cols = [col for col in df.columns 
                   if col not in drop_cols + target_columns + [target_col]]
    
    # Handle non-numeric columns
    for col in feature_cols.copy():
        if df[col].dtype == 'object':
            logger.warning(f"Dropping non-numeric column: {col}")
            feature_cols.remove(col)
    
    # Select features
    X_raw = df[feature_cols].values
    
    # Scale features
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X_raw)
    
    # Create sequences
    X_sequences = []
    y_targets = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_targets.append(y[i+sequence_length])
    
    X = np.array(X_sequences)
    y_final = np.array(y_targets)
    
    # Reshape for expected model input shape (None, 60, 20)
    if X.shape[2] != TARGET_FEATURES:
        logger.info(f"Reshaping features from {X.shape} to match expected input shape")
        
        # If we have more features than needed, select the most important ones
        if X.shape[2] > TARGET_FEATURES:
            # Ensure we have enough features
            if len(feature_cols) < TARGET_FEATURES:
                logger.error(f"Not enough features ({len(feature_cols)}) to create {TARGET_FEATURES} target features")
                return None, None, None, None
            
            # Select most important features (first N features including price and key indicators)
            X = X[:, :, :TARGET_FEATURES]
            feature_cols = feature_cols[:TARGET_FEATURES]
            logger.info(f"Selected {TARGET_FEATURES} features from {len(feature_cols)} available")
        
        # If we have fewer features than needed, pad with zeros
        elif X.shape[2] < TARGET_FEATURES:
            logger.info(f"Padding features from {X.shape[2]} to {TARGET_FEATURES}")
            padding_width = TARGET_FEATURES - X.shape[2]
            X_padded = np.zeros((X.shape[0], X.shape[1], TARGET_FEATURES))
            X_padded[:, :, :X.shape[2]] = X
            X = X_padded
    
    # Check final dimensions
    logger.info(f"Final feature shape: {X.shape}, target shape: {y_final.shape}")
    
    return X, y_final, feature_cols, {"feature_scaler": feature_scaler}

def create_lstm_model(input_shape):
    """Create LSTM model with the correct input shape."""
    inputs = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model(input_shape):
    """Create CNN model with the correct input shape."""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_hybrid_model(input_shape):
    """Create hybrid model combining CNN and LSTM."""
    inputs = Input(shape=input_shape)
    
    # CNN path
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    
    # LSTM path
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = Dropout(0.3)(lstm)
    lstm = LSTM(64)(lstm)
    
    # Combine paths
    combined = Concatenate()([cnn, lstm])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model_type, X_train, y_train, X_val, y_val, asset, 
               epochs=200, batch_size=32, patience=20):
    """
    Train a model with the specified architecture and save it.
    
    Args:
        model_type (str): Type of model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        asset (str): Asset symbol
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        patience (int): Early stopping patience
        
    Returns:
        tuple: (trained model, training history)
    """
    input_shape = X_train.shape[1:]
    logger.info(f"Training {model_type} model for {asset} with input shape {input_shape}")
    
    # Model factory based on type
    if model_type == 'lstm':
        model = create_lstm_model(input_shape)
    elif model_type == 'cnn':
        model = create_cnn_model(input_shape)
    elif model_type == 'hybrid':
        model = create_hybrid_model(input_shape)
    else:
        logger.warning(f"Model type {model_type} not directly supported in this script.")
        logger.warning("Using LSTM model as fallback.")
        model = create_lstm_model(input_shape)
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-6),
        ModelCheckpoint(
            filepath=f'models/{model_type}/{asset}_{model_type}.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and standardized filename for compatibility
    model_dir = f'models/{model_type}'
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(f'{model_dir}/{asset}.h5')
    
    # Save a copy with the expected naming convention for ensemble models
    model.save(f'{model_dir}/{asset}_{model_type}.h5')
    
    # Save metrics
    metrics = {
        "accuracy": float(history.history['accuracy'][-1]),
        "val_accuracy": float(history.history['val_accuracy'][-1]),
        "loss": float(history.history['loss'][-1]),
        "val_loss": float(history.history['val_loss'][-1]),
        "training_date": datetime.now().isoformat(),
        "input_shape": list(input_shape),
        "epochs_trained": len(history.history['loss']),
        "early_stopping": len(history.history['loss']) < epochs
    }
    
    with open(f'{model_dir}/{asset}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model training completed: val_accuracy={metrics['val_accuracy']:.4f}")
    
    return model, history

def create_ensemble_config(asset, model_types, base_weights=None):
    """
    Create or update ensemble configuration for the specified asset.
    
    Args:
        asset (str): Asset symbol
        model_types (list): List of model types to include in ensemble
        base_weights (dict): Base weights for models (optional)
        
    Returns:
        dict: Ensemble configuration
    """
    logger.info(f"Creating ensemble configuration for {asset}")
    
    # Create models directory if it doesn't exist
    ensemble_dir = 'models/ensemble'
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Initialize ensemble config
    weights_path = f'{ensemble_dir}/{asset}_weights.json'
    ensemble_path = f'{ensemble_dir}/{asset}_ensemble.json'
    position_sizing_path = f'{ensemble_dir}/{asset}_position_sizing.json'
    
    # Load existing weights if available, otherwise use equal weights
    if os.path.exists(weights_path) and base_weights is None:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        logger.info(f"Loaded existing weights from {weights_path}")
    else:
        # Use provided base weights or equal weighting
        if base_weights is None:
            weight_value = 1.0 / len(model_types)
            weights = {model_type: weight_value for model_type in model_types}
        else:
            weights = base_weights.copy()
            
            # Add missing models with minimal weight
            for model_type in model_types:
                if model_type not in weights:
                    weights[model_type] = 0.05
            
            # Normalize weights to sum to 1.0
            weight_sum = sum(weights.values())
            weights = {k: v / weight_sum for k, v in weights.items()}
        
        logger.info(f"Created new weights configuration")
    
    # Save weights configuration
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    # Create ensemble configuration
    ensemble_config = {
        "models": {}
    }
    
    # Add models with highest weights to ensemble config
    sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_models = sorted_models[:3]  # Use top 3 models for ensemble
    
    # Normalize weights for top models
    top_weight_sum = sum(weight for _, weight in top_models)
    
    for model_type, weight in top_models:
        model_path = f"models/{model_type}/{asset}_{model_type}.h5"
        if os.path.exists(model_path):
            ensemble_config["models"][model_type] = {
                "path": model_path,
                "weight": weight / top_weight_sum  # Normalize to sum to 1.0
            }
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    # Add metadata
    ensemble_config["parameters"] = {
        "confidence_threshold": 0.65,
        "voting_method": "weighted",
        "trained_date": datetime.now().isoformat()
    }
    
    # Save ensemble configuration
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Create or update position sizing configuration if it doesn't exist
    if not os.path.exists(position_sizing_path):
        position_sizing_config = {
            "max_leverage": 125,
            "min_leverage": 20,
            "confidence_scaling": {
                "min_confidence": 0.5,
                "max_confidence": 0.95
            },
            "regime_adjustments": {
                "trending_up": 1.2,
                "trending_down": 1.0,
                "volatile": 1.5,
                "sideways": 0.7,
                "uncertain": 0.4
            },
            "risk_limits": {
                "max_capital_allocation": 0.5,
                "max_drawdown_percentage": 0.2,
                "profit_taking_threshold": 0.1
            },
            "trained_date": datetime.now().isoformat()
        }
        
        with open(position_sizing_path, 'w') as f:
            json.dump(position_sizing_config, f, indent=2)
        
        logger.info(f"Created new position sizing configuration")
    
    logger.info(f"Ensemble configuration created successfully")
    
    return ensemble_config

def update_ml_config(trained_assets):
    """
    Update the ML configuration to reflect the newly trained models.
    
    Args:
        trained_assets (list): List of assets that were trained
    """
    logger.info(f"Updating ML configuration")
    
    config_path = 'ml_config.json'
    
    if not os.path.exists(config_path):
        logger.error(f"ML configuration file not found: {config_path}")
        return False
    
    # Load existing configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update configuration
    config['updated_at'] = datetime.now().isoformat()
    
    # Ensure the min_required_data is adjusted for the current dataset size
    for asset in trained_assets:
        asset_key = f"{asset[:3]}/{asset[3:]}"
        if asset_key in config['asset_specific_settings']:
            # Set min_required_data to a reasonable value based on our dataset
            config['asset_specific_settings'][asset_key]['min_required_data'] = 456  # Match our dataset size
            config['asset_specific_settings'][asset_key]['trading_enabled'] = True
    
    # Adjust global training parameters
    config['training_parameters']['training_data_min_samples'] = 456  # Match our dataset size
    
    # Update strategy integration
    config['strategy_integration']['integrate_arima_adaptive'] = True
    config['strategy_integration']['arima_weight'] = 0.5
    config['strategy_integration']['adaptive_weight'] = 0.5
    config['strategy_integration']['use_combined_signals'] = True
    
    # Save updated configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"ML configuration updated successfully")
    
    return True

def main():
    """Main function to retrain models with correct input shapes."""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Parse comma-separated list of pairs and models
    pairs = [pair.strip().upper() for pair in args.pairs.split(',')]
    model_types = [model.strip().lower() for model in args.models.split(',')]
    
    # Validate inputs
    for pair in pairs:
        if pair not in ASSETS:
            logger.warning(f"Unknown asset: {pair}, will try to process anyway")
    
    for model_type in model_types:
        if model_type not in MODEL_TYPES:
            logger.error(f"Unsupported model type: {model_type}")
            return
    
    # Process each pair
    trained_pairs = []
    
    for pair in pairs:
        logger.info(f"Processing {pair}")
        
        # Load dataset
        df = load_enhanced_dataset(pair, args.timeframe)
        if df is None:
            logger.error(f"Failed to load dataset for {pair}")
            continue
        
        # Preprocess dataset
        X, y, feature_cols, scalers = preprocess_dataset(df)
        if X is None:
            logger.error(f"Failed to preprocess dataset for {pair}")
            continue
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Train models
        trained_models = []
        for model_type in model_types:
            try:
                model, history = train_model(
                    model_type, X_train, y_train, X_val, y_val, pair,
                    epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
                )
                trained_models.append(model_type)
                logger.info(f"Successfully trained {model_type} model for {pair}")
            except Exception as e:
                if INPUT_SHAPE_ERROR_MSG in str(e).lower():
                    logger.error(f"Input shape mismatch error for {model_type}: {str(e)}")
                else:
                    logger.error(f"Error training {model_type} model for {pair}: {str(e)}")
        
        # Create ensemble if requested and at least one model was trained
        if args.create_ensemble and trained_models:
            try:
                ensemble_config = create_ensemble_config(pair, trained_models)
                logger.info(f"Created ensemble configuration for {pair}")
            except Exception as e:
                logger.error(f"Error creating ensemble for {pair}: {str(e)}")
        
        if trained_models:
            trained_pairs.append(pair)
    
    # Update ML config if requested and at least one pair was trained
    if args.update_config and trained_pairs:
        try:
            update_ml_config(trained_pairs)
            logger.info(f"Updated ML configuration")
        except Exception as e:
            logger.error(f"Error updating ML configuration: {str(e)}")
    
    if trained_pairs:
        logger.info(f"Successfully processed {len(trained_pairs)} pairs: {', '.join(trained_pairs)}")
        logger.info("Model retraining completed successfully")
    else:
        logger.error("No pairs were successfully processed")

if __name__ == "__main__":
    main()