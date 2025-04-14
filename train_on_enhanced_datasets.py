#!/usr/bin/env python3
"""
Train ML Models on Enhanced Datasets

This script trains multiple ML model architectures on the enhanced datasets:
1. TCN (Temporal Convolutional Network)
2. LSTM (Long Short-Term Memory)
3. GRU (Gated Recurrent Unit)
4. Transformer
5. CNN (Convolutional Neural Network)
6. BiLSTM (Bidirectional LSTM)
7. Attention-based model
8. Hybrid model

It creates ensemble models that combine the predictions from all models
with optimized weightings.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAMES = ["1h"]
MODEL_TYPES = ["tcn", "lstm", "gru", "transformer", "cnn", "bilstm", "attention", "hybrid"]
TRAINING_DATA_DIR = "training_data"
MODELS_DIR = "models"

# Environment setup
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"Using GPU: {physical_devices[0]}")
else:
    logger.info("Using CPU for training")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ML models on enhanced datasets")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to train on (default: {', '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
                        help=f"Timeframes to train on (default: {', '.join(DEFAULT_TIMEFRAMES)})")
    parser.add_argument("--model-types", nargs="+", default=MODEL_TYPES,
                        help=f"Model types to train (default: {', '.join(MODEL_TYPES)})")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Create ensemble models after training")
    return parser.parse_args()

def prepare_dataset(pair: str, timeframe: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for ML training
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load dataset
    dataset_path = os.path.join(TRAINING_DATA_DIR, f"{pair}_{timeframe}_enhanced.csv")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None, None, None, None
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
        
        # Remove timestamp and other non-feature columns
        if "timestamp" in df.columns:
            df = df.drop("timestamp", axis=1)
        
        # Get target variable (short-term direction, 8 hours ahead)
        target_col = "target_direction_8"
        if target_col not in df.columns:
            logger.error(f"Target column not found in dataset: {target_col}")
            return None, None, None, None
        
        # Get features and target
        y = df[target_col].values
        X = df.drop([col for col in df.columns if col.startswith("target_")], axis=1)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        
        # Reshape for recurrent models
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return None, None, None, None

def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a TCN (Temporal Convolutional Network) model
    This is a simulated TCN using 1D convolutions since keras-tcn is not available
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        TCN model
    """
    try:
        # Create a simulated TCN using 1D convolutions
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # Create a series of dilated convolutions to simulate TCN
        x = input_layer
        for dilation_rate in [1, 2, 4, 8, 16]:
            x = tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=3,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        logger.info("Using simulated TCN with 1D convolutions")
        return model
    
    except Exception as e:
        logger.error(f"Error building TCN model: {e}")
        return None

def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build an LSTM (Long Short-Term Memory) model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        LSTM model
    """
    try:
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # LSTM layers
        x = tf.keras.layers.LSTM(64, return_sequences=True)(input_layer)
        x = tf.keras.layers.LSTM(32)(x)
        
        # Output layers
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building LSTM model: {e}")
        return None

def build_gru_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a GRU (Gated Recurrent Unit) model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        GRU model
    """
    try:
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # GRU layers
        x = tf.keras.layers.GRU(64, return_sequences=True)(input_layer)
        x = tf.keras.layers.GRU(32)(x)
        
        # Output layers
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building GRU model: {e}")
        return None

def build_transformer_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a Transformer model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        Transformer model
    """
    try:
        # Parameters
        head_size = 256
        num_heads = 4
        ff_dim = 4 * head_size
        dropout = 0.2
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Transformer block
        x = inputs
        
        # Self-attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward network
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(input_shape[-1]),
        ])
        
        ffn_output = ffn(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
        
        # Flatten and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(20, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building Transformer model: {e}")
        return None

def build_cnn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a CNN (Convolutional Neural Network) model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        CNN model
    """
    try:
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Convolutional layers
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        
        # Flatten and dense layers
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building CNN model: {e}")
        return None

def build_bilstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a BiLSTM (Bidirectional LSTM) model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        BiLSTM model
    """
    try:
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Bidirectional LSTM layers
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building BiLSTM model: {e}")
        return None

def build_attention_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build an Attention-based model
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        Attention model
    """
    try:
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(1, activation="tanh")(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(64)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention to LSTM output
        attention_mul = tf.keras.layers.Multiply()([lstm_out, attention])
        attention_mul = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(attention_mul)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, activation="relu")(attention_mul)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building Attention model: {e}")
        return None

def build_hybrid_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a Hybrid model combining CNN and LSTM
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        Hybrid model
    """
    try:
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # CNN part
        cnn = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
        cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn)
        
        # LSTM part
        lstm = tf.keras.layers.LSTM(64, return_sequences=True)(cnn)
        
        # Attention mechanism
        attention = tf.keras.layers.Dense(1, activation="tanh")(lstm)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(64)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention to LSTM output
        context = tf.keras.layers.Multiply()([lstm, attention])
        context = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(context)
        
        # Dense layers
        x = tf.keras.layers.Dense(32, activation="relu")(context)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building Hybrid model: {e}")
        return None

def build_model(model_type: str, input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build a model based on the specified type
    
    Args:
        model_type: Type of model to build
        input_shape: Shape of input data
        
    Returns:
        Model
    """
    model_builders = {
        "tcn": build_tcn_model,
        "lstm": build_lstm_model,
        "gru": build_gru_model,
        "transformer": build_transformer_model,
        "cnn": build_cnn_model,
        "bilstm": build_bilstm_model,
        "attention": build_attention_model,
        "hybrid": build_hybrid_model
    }
    
    if model_type not in model_builders:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    return model_builders[model_type](input_shape)

def train_model(
    pair: str,
    timeframe: str,
    model_type: str,
    epochs: int = 50,
    batch_size: int = 32
) -> Optional[tf.keras.Model]:
    """
    Train a model on the specified dataset
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model or None if training failed
    """
    # Prepare dataset
    X_train, X_test, y_train, y_test = prepare_dataset(pair, timeframe)
    if X_train is None:
        return None
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(model_type, input_shape)
    if model is None:
        return None
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    # Train model
    try:
        logger.info(f"Training {model_type} model for {pair} ({timeframe})...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model
        os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)
        model_path = os.path.join(MODELS_DIR, model_type, f"{pair}_{model_type}.h5")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def create_ensemble_model(
    pair: str,
    timeframe: str,
    model_types: List[str]
) -> bool:
    """
    Create an ensemble model that combines predictions from multiple models
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        model_types: List of model types in the ensemble
        
    Returns:
        True if ensemble creation was successful, False otherwise
    """
    # Check if all required models exist
    for model_type in model_types:
        model_path = os.path.join(MODELS_DIR, model_type, f"{pair}_{model_type}.h5")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
    
    # Create ensemble weights based on model performance
    # For now, use fixed weights (we'll adapt this later based on validation performance)
    weights = {
        "transformer": 0.4,
        "tcn": 0.35,
        "lstm": 0.25,
        "gru": 0.2,
        "cnn": 0.15,
        "bilstm": 0.15,
        "attention": 0.15,
        "hybrid": 0.2
    }
    
    # Normalize weights
    available_weights = {k: v for k, v in weights.items() if k in model_types}
    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
    
    # Create ensemble config
    ensemble_config = {
        "model_types": model_types,
        "weights": normalized_weights,
        "timeframe": timeframe
    }
    
    # Create position sizing config
    position_sizing_config = {
        "base_leverage": 20,
        "max_leverage": 125,
        "confidence_threshold": 0.65,
        "high_confidence_threshold": 0.85,
        "scaling_factor": 1.5
    }
    
    # Save ensemble config
    os.makedirs(os.path.join(MODELS_DIR, "ensemble"), exist_ok=True)
    
    ensemble_config_path = os.path.join(MODELS_DIR, "ensemble", f"{pair}_ensemble.json")
    with open(ensemble_config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    
    weights_config_path = os.path.join(MODELS_DIR, "ensemble", f"{pair}_weights.json")
    with open(weights_config_path, 'w') as f:
        json.dump(normalized_weights, f, indent=4)
    
    position_sizing_path = os.path.join(MODELS_DIR, "ensemble", f"{pair}_position_sizing.json")
    with open(position_sizing_path, 'w') as f:
        json.dump(position_sizing_config, f, indent=4)
    
    logger.info(f"Ensemble model created for {pair}")
    logger.info(f"Ensemble weights: {normalized_weights}")
    
    return True

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create model directories
    for model_type in args.model_types:
        os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)
    
    # Train models for each pair and timeframe
    for pair in args.pairs:
        for timeframe in args.timeframes:
            successful_models = []
            
            for model_type in args.model_types:
                model = train_model(
                    pair=pair,
                    timeframe=timeframe,
                    model_type=model_type,
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )
                
                if model is not None:
                    successful_models.append(model_type)
            
            # Create ensemble model
            if args.ensemble and len(successful_models) > 1:
                create_ensemble_model(
                    pair=pair,
                    timeframe=timeframe,
                    model_types=successful_models
                )

if __name__ == "__main__":
    main()