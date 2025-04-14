#!/usr/bin/env python3
"""
Improved Model Retraining Script

This script retrains our ML models to match the reshaped input data.
It addresses the issue where our models expect input shape (None, 1, 56)
but our reshaped data now has shape (None, 60, 20).

Usage:
    python improved_model_retraining.py --pair SOL/USD --model lstm,tcn,transformer
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model, save_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

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
DEFAULT_PAIRS = ['SOL/USD', 'BTC/USD', 'ETH/USD']
DEFAULT_MODELS = ['lstm', 'tcn', 'transformer']
DEFAULT_LOOKBACK = 60
DEFAULT_FEATURES = 20
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_PATIENCE = 15
DEFAULT_LEARNING_RATE = 0.001

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improved Model Retraining')
    parser.add_argument('--pair', type=str, default='SOL/USD',
                        help='Trading pair to retrain models for')
    parser.add_argument('--models', type=str, default=','.join(DEFAULT_MODELS),
                        help='Comma-separated list of models to retrain')
    parser.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK,
                        help=f'Lookback period for model training (default: {DEFAULT_LOOKBACK})')
    parser.add_argument('--features', type=int, default=DEFAULT_FEATURES,
                        help=f'Number of features for model training (default: {DEFAULT_FEATURES})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of epochs for training (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--validation-split', type=float, default=DEFAULT_VALIDATION_SPLIT,
                        help=f'Validation split for training (default: {DEFAULT_VALIDATION_SPLIT})')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                        help=f'Patience for early stopping (default: {DEFAULT_PATIENCE})')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help=f'Learning rate for model training (default: {DEFAULT_LEARNING_RATE})')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    return parser.parse_args()

def load_dataset(pair: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load dataset for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        Tuple of (DataFrame with dataset, list of feature column names)
    """
    pair_formatted = pair.replace('/', '')
    input_file = f'training_data/{pair_formatted}_1h_enhanced_reshaped.csv'
    
    if not os.path.exists(input_file):
        logger.warning(f"Reshaped file {input_file} not found, trying to find alternative files")
        
        alt_files = []
        training_dir = 'training_data'
        if os.path.exists(training_dir):
            for file in os.listdir(training_dir):
                if file.startswith(pair_formatted) and file.endswith('.csv'):
                    alt_files.append(os.path.join(training_dir, file))
        
        if not alt_files:
            logger.error(f"No files found for {pair}")
            return None, []
        
        # Use the first alternative file
        input_file = alt_files[0]
        logger.info(f"Using alternative file: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Get feature column names (excluding target columns)
        feature_cols = [col for col in df.columns if not col.startswith('target_') and col != 'timestamp']
        
        return df, feature_cols
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None, []

def load_feature_metadata(pair: str) -> Dict[str, Any]:
    """
    Load feature metadata for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        Dictionary with feature metadata
    """
    pair_formatted = pair.replace('/', '')
    metadata_file = f'training_data/{pair_formatted}_features_metadata.json'
    
    if not os.path.exists(metadata_file):
        logger.warning(f"Feature metadata file {metadata_file} not found")
        
        # Try in models directory
        model_dirs = ['lstm', 'tcn', 'transformer']
        for model_dir in model_dirs:
            alt_file = f'models/{model_dir}/feature_names.json'
            if os.path.exists(alt_file):
                logger.info(f"Using alternative metadata file: {alt_file}")
                try:
                    with open(alt_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading alternative metadata file: {str(e)}")
                break
        
        # Return empty metadata
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded feature metadata from {metadata_file}")
        return metadata
    except Exception as e:
        logger.error(f"Error loading feature metadata: {str(e)}")
        return {}

def prepare_training_data(df: pd.DataFrame, feature_cols: List[str], 
                          lookback: int = DEFAULT_LOOKBACK,
                          target_col: str = 'target_direction_24') -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare training data from dataset.
    
    Args:
        df: DataFrame with dataset
        feature_cols: List of feature column names
        lookback: Lookback period
        target_col: Target column name
        
    Returns:
        Tuple of (X, y, scaler)
    """
    if df is None or feature_cols is None or len(feature_cols) == 0:
        logger.error("Invalid input data for preparing training data")
        return None, None, None
    
    try:
        # Select features and target
        X_data = df[feature_cols].values
        y_data = df[target_col].values if target_col in df.columns else None
        
        if y_data is None:
            logger.error(f"Target column {target_col} not found in dataset")
            return None, None, None
        
        # Normalize features
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)
        
        # Create sequences - check if data is already in sequence format
        if len(X_data.shape) == 3:
            # Data is already in sequence format (samples, timesteps, features)
            logger.info(f"Data already in sequence format with shape {X_data.shape}")
            X = X_data
            y = y_data[:len(X)]
        else:
            # Need to create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(len(df) - lookback):
                X_sequences.append(X_data[i:i+lookback])
                y_sequences.append(y_data[i+lookback])
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
        
        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return X, y, scaler
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        return None, None, None

def create_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """
    Create LSTM model.
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        Compiled model
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_tcn_model(input_shape: Tuple[int, int]) -> Model:
    """
    Create TCN model.
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        Compiled model
    """
    # Simple convolutional model as a substitute for TCN
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transformer_model(input_shape: Tuple[int, int]) -> Model:
    """
    Create Transformer model.
    
    Args:
        input_shape: Input shape for the model
        
    Returns:
        Compiled model
    """
    inputs = Input(shape=input_shape)
    
    # Transformer layer
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=32
    )(inputs, inputs)
    x = LayerNormalization()(attention_output + inputs)
    
    # Conv layer
    conv_output = Conv1D(32, kernel_size=3, activation='relu')(x)
    pool_output = GlobalAveragePooling1D()(conv_output)
    
    # Dense layers
    x = Dense(32, activation='relu')(pool_output)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model(model_type: str, input_shape: Tuple[int, int]) -> Model:
    """
    Create model of the specified type.
    
    Args:
        model_type: Model type (lstm, tcn, transformer)
        input_shape: Input shape for the model
        
    Returns:
        Compiled model
    """
    if model_type == 'lstm':
        return create_lstm_model(input_shape)
    elif model_type == 'tcn':
        return create_tcn_model(input_shape)
    elif model_type == 'transformer':
        return create_transformer_model(input_shape)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

def train_model(model: Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
               patience: int = DEFAULT_PATIENCE) -> Tuple[Model, Dict[str, Any]]:
    """
    Train model.
    
    Args:
        model: Model to train
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        patience: Patience for early stopping
        
    Returns:
        Tuple of (trained model, training history)
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history.history

def save_trained_model(model: Model, model_type: str, pair: str) -> bool:
    """
    Save trained model.
    
    Args:
        model: Trained model
        model_type: Model type (lstm, tcn, transformer)
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        True if successful, False otherwise
    """
    pair_formatted = pair.replace('/', '')
    model_dir = f'models/{model_type}'
    model_path = f'{model_dir}/{pair_formatted}.h5'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    try:
        model.save(model_path)
        logger.info(f"Saved {model_type} model to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving {model_type} model: {str(e)}")
        return False

def save_model_metrics(metrics: Dict[str, Any], model_type: str, pair: str) -> bool:
    """
    Save model metrics.
    
    Args:
        metrics: Dictionary with model metrics
        model_type: Model type (lstm, tcn, transformer)
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        True if successful, False otherwise
    """
    pair_formatted = pair.replace('/', '')
    metrics_path = f'models/{model_type}/{pair_formatted}_metrics.json'
    
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved {model_type} metrics to {metrics_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving {model_type} metrics: {str(e)}")
        return False

def save_feature_metadata(feature_cols: List[str], scaler: StandardScaler, model_type: str) -> bool:
    """
    Save feature metadata.
    
    Args:
        feature_cols: List of feature column names
        scaler: Fitted scaler
        model_type: Model type (lstm, tcn, transformer)
        
    Returns:
        True if successful, False otherwise
    """
    model_dir = f'models/{model_type}'
    
    # Save feature names
    try:
        with open(f'{model_dir}/feature_names.json', 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        logger.info(f"Saved feature names to {model_dir}/feature_names.json")
    except Exception as e:
        logger.error(f"Error saving feature names: {str(e)}")
        return False
    
    # Save normalization parameters
    try:
        norm_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
        
        with open(f'{model_dir}/norm_params.json', 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        logger.info(f"Saved normalization parameters to {model_dir}/norm_params.json")
        return True
    except Exception as e:
        logger.error(f"Error saving normalization parameters: {str(e)}")
        return False

def update_ensemble_config(pair: str, models: List[str]) -> bool:
    """
    Update ensemble configuration for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        models: List of model types
        
    Returns:
        True if successful, False otherwise
    """
    pair_formatted = pair.replace('/', '')
    ensemble_dir = 'models/ensemble'
    ensemble_config_path = f'{ensemble_dir}/{pair_formatted}_ensemble.json'
    weights_path = f'{ensemble_dir}/{pair_formatted}_weights.json'
    
    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir)
    
    # Create/update ensemble configuration
    try:
        if os.path.exists(ensemble_config_path):
            with open(ensemble_config_path, 'r') as f:
                ensemble_config = json.load(f)
        else:
            ensemble_config = {
                'models': [],
                'parameters': {}
            }
        
        # Update models list
        ensemble_models = []
        for model_type in models:
            model_config = {
                'type': model_type,
                'enabled': True,
                'confidence_scaling': 1.0
            }
            
            if model_type == 'lstm':
                model_config['confidence_scaling'] = 1.2
            elif model_type == 'tcn':
                model_config['confidence_scaling'] = 1.1
            elif model_type == 'transformer':
                model_config['confidence_scaling'] = 1.15
            
            ensemble_models.append(model_config)
        
        ensemble_config['models'] = ensemble_models
        
        # Update parameters
        ensemble_config['parameters'] = {
            'method': 'weighted_voting',
            'confidence_threshold': 0.65,
            'use_confidence_scaling': True,
            'use_dynamic_weights': True,
            'update_frequency': 24,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        with open(ensemble_config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"Updated ensemble configuration in {ensemble_config_path}")
    except Exception as e:
        logger.error(f"Error updating ensemble configuration: {str(e)}")
        return False
    
    # Create/update weights configuration
    try:
        weights = {}
        
        # Assign weights based on model type
        total_weight = len(models)
        for model_type in models:
            if model_type == 'lstm':
                weights[model_type] = 0.2
            elif model_type == 'tcn':
                weights[model_type] = 0.2
            elif model_type == 'transformer':
                weights[model_type] = 0.15
            else:
                weights[model_type] = 0.1
        
        # Ensure weights sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for model_type in weights:
                weights[model_type] /= weight_sum
        
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
        
        logger.info(f"Updated ensemble weights in {weights_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating ensemble weights: {str(e)}")
        return False

def retrain_models_for_pair(pair: str, models: List[str], args) -> bool:
    """
    Retrain models for a specific pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        models: List of model types to retrain
        args: Command line arguments
        
    Returns:
        True if successful, False otherwise
    """
    # Load dataset
    df, feature_cols = load_dataset(pair)
    
    if df is None or len(feature_cols) == 0:
        logger.error(f"Failed to load dataset for {pair}")
        return False
    
    # Load feature metadata
    feature_metadata = load_feature_metadata(pair)
    
    # Prepare training data
    X, y, scaler = prepare_training_data(df, feature_cols, args.lookback)
    
    if X is None or y is None:
        logger.error(f"Failed to prepare training data for {pair}")
        return False
    
    logger.info(f"Prepared training data with shape {X.shape}")
    
    # Get actual features from loaded data, not command line args
    actual_timesteps = X.shape[1]
    actual_features = X.shape[2]
    logger.info(f"Using actual input shape: ({actual_timesteps}, {actual_features})")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.validation_split, random_state=42
    )
    
    logger.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    success = True
    
    # Train models
    for model_type in models:
        logger.info(f"Training {model_type} model for {pair}")
        
        # Create model with actual dimensions from the data
        model = create_model(model_type, input_shape=(actual_timesteps, actual_features))
        
        if model is None:
            logger.error(f"Failed to create {model_type} model for {pair}")
            success = False
            continue
        
        # Train model
        trained_model, history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
        )
        
        # Calculate metrics
        val_accuracy = max(history.get('val_accuracy', [0]))
        val_loss = min(history.get('val_loss', [1]))
        
        metrics = {
            'validation_accuracy': val_accuracy,
            'validation_loss': val_loss,
            'epochs_trained': len(history.get('loss', [])),
            'training_date': datetime.datetime.now().isoformat()
        }
        
        logger.info(f"{model_type} model for {pair}: val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")
        
        # Save model
        if not save_trained_model(trained_model, model_type, pair):
            logger.error(f"Failed to save {model_type} model for {pair}")
            success = False
        
        # Save metrics
        if not save_model_metrics(metrics, model_type, pair):
            logger.error(f"Failed to save {model_type} metrics for {pair}")
            success = False
        
        # Save feature metadata
        if not save_feature_metadata(feature_cols, scaler, model_type):
            logger.error(f"Failed to save feature metadata for {model_type}")
            success = False
    
    # Update ensemble configuration
    if not update_ensemble_config(pair, models):
        logger.error(f"Failed to update ensemble configuration for {pair}")
        success = False
    
    return success

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not available, please install TensorFlow and required dependencies")
        return 1
    
    # Parse models
    models = [model.strip() for model in args.models.split(',')]
    
    logger.info(f"Starting model retraining for {args.pair} with models: {', '.join(models)}")
    logger.info(f"Using input shape: ({args.lookback}, {args.features})")
    
    # Retrain models
    success = retrain_models_for_pair(args.pair, models, args)
    
    if success:
        logger.info(f"Model retraining for {args.pair} completed successfully")
    else:
        logger.error(f"Model retraining for {args.pair} failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())