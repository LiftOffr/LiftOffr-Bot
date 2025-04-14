#!/usr/bin/env python3
"""
Retrain Models with Correct Input Shape

This script retrains ML models with the correct input shape to match our enhanced datasets.
It handles:
1. Loading enhanced datasets
2. Creating models with correct input shapes
3. Training the models
4. Evaluating and saving the models

Usage:
    python retrain_models_with_correct_shape.py --pairs SOLUSD BTCUSD --epochs 50
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization, Input, 
    Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional,
    Attention, MultiHeadAttention, LayerNormalization, 
    Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_retraining.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAME = "1h"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
TRAINING_DATA_DIR = "training_data"
MODEL_DIRS = {
    "lstm": "models/lstm",
    "gru": "models/gru",
    "tcn": "models/tcn",
    "transformer": "models/transformer",
    "cnn": "models/cnn",
    "bilstm": "models/bilstm",
    "attention": "models/attention",
    "hybrid": "models/hybrid"
}
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Retrain models with correct input shape")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to train (default: {' '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help=f"Timeframe to use (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--model-types", nargs="+", default=list(MODEL_DIRS.keys()),
                        help=f"Model types to train (default: all)")
    return parser.parse_args()

def load_dataset(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load enhanced dataset for a trading pair"""
    dataset_path = os.path.join(TRAINING_DATA_DIR, f"{pair}_{timeframe}_enhanced.csv")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset for {pair} with {len(df)} samples and {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset for {pair}: {e}")
        return None

def prepare_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare dataset for model training"""
    # Remove timestamp and other non-numeric columns
    columns_to_drop = ["timestamp"]
    
    # Add string/categorical columns to drop list
    for col in df.columns:
        if df[col].dtype == 'object':
            logger.info(f"Dropping non-numeric column: {col}")
            columns_to_drop.append(col)
    
    # Drop non-numeric columns
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Get target variable (short-term direction, 8 hours ahead)
    target_col = "target_direction_8"
    if target_col not in df.columns:
        logger.error(f"Target column not found in dataset: {target_col}")
        return None, None, None, None, None
    
    # Get features and target
    y = df[target_col].values
    X = df.drop([col for col in df.columns if col.startswith("target_")], axis=1)
    
    # Save feature names
    feature_names = X.columns.tolist()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Verify all columns are numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            logger.error(f"Non-numeric column found after processing: {col}")
            return None, None, None, None, None
    
    logger.info(f"Processing {len(X.columns)} numeric features")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=VALIDATION_SPLIT, shuffle=False, random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def create_lstm_model(input_shape: Tuple[int, int]) -> Model:
    """Create LSTM model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_gru_model(input_shape: Tuple[int, int]) -> Model:
    """Create GRU model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    x = GRU(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = GRU(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_transformer_model(input_shape: Tuple[int, int]) -> Model:
    """Create Transformer model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    # Multi-head self-attention
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Feed-forward network
    feed_forward = Dense(128, activation='relu')(x)
    feed_forward = Dropout(0.3)(feed_forward)
    feed_forward = Dense(input_shape[-1])(feed_forward)
    
    # Add & normalize
    x = LayerNormalization(epsilon=1e-6)(x + feed_forward)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_cnn_model(input_shape: Tuple[int, int]) -> Model:
    """Create CNN model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_bilstm_model(input_shape: Tuple[int, int]) -> Model:
    """Create Bidirectional LSTM model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_attention_model(input_shape: Tuple[int, int]) -> Model:
    """Create Attention model with correct input shape"""
    inputs = Input(shape=input_shape)
    
    # LSTM layer with self-attention
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
    x = LayerNormalization(epsilon=1e-6)(attention_output + lstm_out)
    
    # Output layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_hybrid_model(input_shape: Tuple[int, int]) -> Model:
    """Create Hybrid model (CNN + LSTM + Attention) with correct input shape"""
    inputs = Input(shape=input_shape)
    
    # CNN branch
    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    
    # LSTM branch
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = Dropout(0.3)(lstm)
    
    # Combine branches
    combined = Concatenate()([cnn, lstm])
    
    # Self-attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(combined, combined)
    x = LayerNormalization(epsilon=1e-6)(attention_output + combined)
    
    # Output layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_tcn_model(input_shape: Tuple[int, int]) -> Model:
    """Create TCN model with correct input shape"""
    # Since TCN requires a specific library, we'll create a simpler version using Conv1D layers
    inputs = Input(shape=input_shape)
    
    # Using dilated convolutions to mimic TCN behavior
    x = Conv1D(64, kernel_size=2, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = Conv1D(64, kernel_size=2, padding='causal', dilation_rate=2, activation='relu')(x)
    x = Conv1D(64, kernel_size=2, padding='causal', dilation_rate=4, activation='relu')(x)
    x = Conv1D(64, kernel_size=2, padding='causal', dilation_rate=8, activation='relu')(x)
    
    # Output layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def get_model_builder(model_type: str):
    """Get the appropriate model builder function for the given model type"""
    model_builders = {
        "lstm": create_lstm_model,
        "gru": create_gru_model,
        "transformer": create_transformer_model,
        "cnn": create_cnn_model,
        "bilstm": create_bilstm_model,
        "attention": create_attention_model,
        "hybrid": create_hybrid_model,
        "tcn": create_tcn_model
    }
    
    return model_builders.get(model_type)

def train_model(
    model_type: str,
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[Optional[Model], Dict[str, Any]]:
    """Train a model with the given data"""
    # Get model builder
    model_builder = get_model_builder(model_type)
    if not model_builder:
        logger.error(f"Unknown model type: {model_type}")
        return None, {}
    
    # Reshape data for RNN models
    seq_length = 1  # Single time step per sample
    n_features = X_train.shape[1]  # Number of features
    
    X_train_reshaped = X_train.reshape((X_train.shape[0], seq_length, n_features))
    X_test_reshaped = X_test.reshape((X_test.shape[0], seq_length, n_features))
    
    input_shape = (seq_length, n_features)
    logger.info(f"Input shape for {model_type} model: {input_shape}")
    
    # Create and compile model
    model = model_builder(input_shape)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    logger.info(f"Training {model_type} model...")
    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_test_reshaped, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    evaluation = model.evaluate(X_test_reshaped, y_test, verbose=1)
    metrics = {
        "loss": float(evaluation[0]),
        "accuracy": float(evaluation[1])
    }
    
    logger.info(f"{model_type.upper()} model results: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    return model, metrics

def save_model(
    model: Model,
    model_type: str,
    pair: str,
    metrics: Dict[str, Any],
    feature_names: List[str]
):
    """Save the trained model and associated metadata"""
    # Create model directory if it doesn't exist
    model_dir = MODEL_DIRS.get(model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{pair}.h5")
    model.save(model_path)
    logger.info(f"Saved {model_type} model to {model_path}")
    
    # Save normalization parameters (placeholder, actual values would come from dataset)
    norm_params = {
        "mean": 0.0,
        "std": 1.0,
        "created_at": datetime.now().isoformat()
    }
    norm_params_path = os.path.join(model_dir, "norm_params.json")
    with open(norm_params_path, 'w') as f:
        json.dump(norm_params, f, indent=4)
    
    # Save feature names
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=4)
    
    # Save model metrics
    metrics_path = os.path.join(model_dir, f"{pair}_metrics.json")
    metrics_data = {
        "pair": pair,
        "model_type": model_type,
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "created_at": datetime.now().isoformat()
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

def create_ensemble_config(pair: str, model_metrics: Dict[str, Dict[str, Any]]):
    """Create ensemble configuration based on model performances"""
    # Calculate weights based on accuracy
    total_accuracy = sum(metrics["accuracy"] for metrics in model_metrics.values())
    
    if total_accuracy == 0:
        logger.error(f"No models with positive accuracy for {pair}")
        return
    
    weights = {model_type: metrics["accuracy"] / total_accuracy 
               for model_type, metrics in model_metrics.items()}
    
    # Create ensemble directory if it doesn't exist
    ensemble_dir = "models/ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Create ensemble configuration
    ensemble_config = {
        "pair": pair,
        "weights": weights,
        "performances": {model_type: {"accuracy": metrics["accuracy"], "loss": metrics["loss"]} 
                         for model_type, metrics in model_metrics.items()},
        "model_types": list(weights.keys()),
        "created_at": datetime.now().isoformat()
    }
    
    # Save ensemble configuration
    ensemble_config_path = os.path.join(ensemble_dir, f"{pair}_ensemble.json")
    with open(ensemble_config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    
    # Save weights separately for easy access
    weights_path = os.path.join(ensemble_dir, f"{pair}_weights.json")
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=4)
    
    # Create position sizing configuration
    position_sizing = {
        "pair": pair,
        "base_confidence": 0.75,
        "model_confidences": {model_type: metrics["accuracy"] for model_type, metrics in model_metrics.items()},
        "leverage_scaling": {
            "min_leverage": 1,
            "max_leverage": 5,
            "confidence_threshold": 0.8,
            "scaling_factor": 1.5
        },
        "position_sizing": {
            "base_size": 0.2,
            "min_size": 0.05,
            "max_size": 0.5,
            "confidence_scaling": True
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Save position sizing configuration
    position_sizing_path = os.path.join(ensemble_dir, f"{pair}_position_sizing.json")
    with open(position_sizing_path, 'w') as f:
        json.dump(position_sizing, f, indent=4)
    
    logger.info(f"Created ensemble configuration for {pair}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    logger.info("Starting model retraining with correct input shapes...")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Model types: {args.model_types}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Process each trading pair
    for pair in args.pairs:
        logger.info(f"Processing {pair}...")
        
        # Load dataset
        df = load_dataset(pair, args.timeframe)
        if df is None:
            logger.error(f"Failed to load dataset for {pair}, skipping")
            continue
        
        # Prepare dataset
        X_train, X_test, y_train, y_test, feature_names = prepare_dataset(df)
        if X_train is None:
            logger.error(f"Failed to prepare dataset for {pair}, skipping")
            continue
        
        # Train models and save metrics
        model_metrics = {}
        
        for model_type in args.model_types:
            logger.info(f"Training {model_type} model for {pair}...")
            
            model, metrics = train_model(
                model_type, X_train, y_train, X_test, y_test, 
                epochs=args.epochs, batch_size=args.batch_size
            )
            
            if model is not None:
                save_model(model, model_type, pair, metrics, feature_names)
                model_metrics[model_type] = metrics
            else:
                logger.error(f"Failed to train {model_type} model for {pair}")
        
        # Create ensemble configuration
        if model_metrics:
            create_ensemble_config(pair, model_metrics)
        else:
            logger.error(f"No models trained successfully for {pair}, skipping ensemble creation")
    
    logger.info("Model retraining completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())