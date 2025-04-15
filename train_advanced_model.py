#!/usr/bin/env python3
"""
Advanced Model Training

This script trains advanced deep learning and machine learning models 
optimized for cryptocurrency price prediction with extremely high accuracy.

Supported model types:
1. TCN (Temporal Convolutional Network)
2. LSTM (with attention)
3. Attention GRU
4. Transformer
5. XGBoost
6. LightGBM

Features:
- Hyperparameter optimization integration
- Cross-validation with time series splits
- Advanced training techniques
- Early stopping and learning rate scheduling
- Performance evaluation with trading-specific metrics
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_model_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_advanced_model")

# Constants
CONFIG_PATH = "config/new_coins_training_config.json"
MODEL_DIR = "ml_models"
TRAINING_DATA_DIR = "training_data"
OPTIMIZATION_RESULTS_DIR = "optimization_results"

# Create necessary directories
Path(MODEL_DIR).mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train advanced models for cryptocurrency trading.')
    parser.add_argument('--pair', type=str, required=True,
                      help='Trading pair to train model for (e.g., "AVAX/USD")')
    parser.add_argument('--model', type=str, required=True,
                      choices=['tcn', 'lstm', 'attention_gru', 'transformer', 'xgboost', 'lightgbm'],
                      help='Model type to train')
    parser.add_argument('--stage', type=int, default=1,
                      help='Training stage (default: 1)')
    parser.add_argument('--timeframe', type=str, default='1h',
                      help='Timeframe to use for training (default: 1h)')
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                      help='Target accuracy to achieve (default: 0.95)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size for training (default: from config)')
    parser.add_argument('--use-hyperopt', action='store_true',
                      help='Use hyperparameter optimization results')
    parser.add_argument('--force', action='store_true',
                      help='Force retrain even if model exists')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()


def load_config():
    """Load configuration from config file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def load_hyperopt_results(pair: str, model_type: str) -> Optional[Dict]:
    """
    Load hyperparameter optimization results.
    
    Args:
        pair: Trading pair
        model_type: Model type
        
    Returns:
        hyperparams: Hyperparameter dictionary or None if not available
    """
    pair_filename = pair.replace("/", "_")
    results_file = f"{OPTIMIZATION_RESULTS_DIR}/{model_type}_{pair_filename}_results.json"
    
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if "best_params" in results:
                logger.info(f"Loaded hyperopt results for {model_type} {pair}")
                return results["best_params"]
        
        logger.warning(f"No hyperopt results found for {model_type} {pair}")
        return None
    
    except Exception as e:
        logger.error(f"Error loading hyperopt results: {e}")
        return None


def load_data(pair: str, timeframe: str = '1h') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data for a specific pair and timeframe.
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        
    Returns:
        X: Features
        y: Target
    """
    pair_filename = pair.replace("/", "_")
    file_path = f"{TRAINING_DATA_DIR}/{pair_filename}_{timeframe}_features.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Training data file not found: {file_path}")
            return None, None
        
        df = pd.read_csv(file_path)
        
        # Extract features and target
        X = df.drop(['timestamp', 'target'], axis=1, errors='ignore')
        if 'target' in df.columns:
            y = df['target']
        else:
            logger.error(f"Target column not found in {file_path}")
            return None, None
        
        logger.info(f"Loaded training data for {pair} with {len(X)} rows and {X.shape[1]} features")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error loading data for {pair}: {e}")
        return None, None


def prepare_data(X: pd.DataFrame, y: pd.Series, model_type: str, config: Dict, pair: str) -> Tuple:
    """
    Prepare data for model training.
    
    Args:
        X: Features
        y: Target
        model_type: Model type
        config: Configuration
        pair: Trading pair
        
    Returns:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        X_test: Test features (optional)
        y_test: Test target (optional)
    """
    from sklearn.model_selection import train_test_split
    
    # Get pair-specific settings
    pair_settings = config.get("pair_specific_settings", {}).get(pair, {})
    common_settings = config.get("training_config", {}).get("common", {})
    
    # Get data split parameters
    validation_split = common_settings.get("validation_split", 0.2)
    test_split = common_settings.get("test_split", 0.1)
    
    # For sequence models, create sequences
    is_sequence_model = model_type in ['tcn', 'lstm', 'attention_gru', 'transformer']
    
    if is_sequence_model:
        # Get lookback window
        lookback_window = pair_settings.get("lookback_window", 120)
        
        # Convert to numpy arrays
        X_values = X.values
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = create_sequences(X_values, y_values, lookback_window)
        
        # Split the data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=test_split, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_split, shuffle=False
        )
        
        logger.info(f"Prepared {model_type} sequence data with shape: {X_seq.shape}")
        
        return X_train, X_val, y_train, y_val, X_test, y_test
    
    else:
        # For tree-based models, use flat data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=validation_split, shuffle=False
        )
        
        logger.info(f"Prepared {model_type} data with shape: {X.shape}")
        
        return X_train, X_val, y_train, y_val, X_test, y_test


def create_sequences(X: np.ndarray, y: np.ndarray, lookback_window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences for time series models.
    
    Args:
        X: Feature matrix
        y: Target vector
        lookback_window: Number of lookback periods
        
    Returns:
        X_seq: Sequence input features
        y_seq: Target values
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - lookback_window):
        X_seq.append(X[i:i+lookback_window])
        y_seq.append(y[i+lookback_window])
    
    return np.array(X_seq), np.array(y_seq)


def train_tcn_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                 hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train a TCN (Temporal Convolutional Network) model.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        # Try to import TCN layer
        try:
            from tcn import TCN
        except ImportError:
            try:
                from tensorflow.keras.layers import TCN
            except ImportError:
                logger.error("TCN layer not available. Please install keras-tcn")
                return None, {}
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        tcn_config = config.get("training_config", {}).get("tcn", {})
        
        # Override with hyperparams if available
        if hyperparams:
            for key, value in hyperparams.items():
                if key in tcn_config:
                    tcn_config[key] = value
        
        # Get training parameters
        epochs = args.epochs if args.epochs is not None else common_config.get("epochs", 150)
        batch_size = args.batch_size if args.batch_size is not None else common_config.get("batch_size", 32)
        
        # Get model hyperparameters
        num_filters = tcn_config.get("num_filters", 64)
        kernel_size = tcn_config.get("kernel_size", 3)
        nb_stacks = tcn_config.get("nb_stacks", 2)
        dilations = tcn_config.get("dilations", [1, 2, 4, 8, 16, 32])
        padding = tcn_config.get("padding", "causal")
        use_skip_connections = tcn_config.get("use_skip_connections", True)
        dropout_rate = tcn_config.get("dropout_rate", 0.2)
        return_sequences = tcn_config.get("return_sequences", False)
        activation = tcn_config.get("activation", "relu")
        use_batch_norm = tcn_config.get("use_batch_norm", True)
        use_layer_norm = tcn_config.get("use_layer_norm", False)
        use_weight_norm = tcn_config.get("use_weight_norm", True)
        
        # Create model
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        
        x = TCN(
            nb_filters=num_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            kernel_initializer='he_normal',
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm
        )(inputs)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation="relu")(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save path
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/tcn_{pair_filename}_model"
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=common_config.get("early_stopping_patience", 15),
                min_delta=common_config.get("early_stopping_min_delta", 0.0005),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=common_config.get("learning_rate_reduction_factor", 0.5),
                patience=common_config.get("learning_rate_patience", 10),
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if args.verbose else 2
        )
        
        # Get best accuracy
        best_epoch = np.argmax(history.history["val_accuracy"])
        best_val_accuracy = history.history["val_accuracy"][best_epoch]
        
        # Load best model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects={"TCN": TCN})
        
        # Final evaluation
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate additional metrics
        y_pred = model.predict(X_val)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        
        # Log results
        logger.info(f"TCN model training for {pair} completed:")
        logger.info(f"  Best epoch: {best_epoch + 1}/{epochs}")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_epoch": int(best_epoch)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training TCN model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def train_lstm_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                  hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train an LSTM model with attention.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        lstm_config = config.get("training_config", {}).get("lstm", {})
        
        # Override with hyperparams if available
        if hyperparams:
            for key, value in hyperparams.items():
                if key in lstm_config:
                    lstm_config[key] = value
        
        # Get training parameters
        epochs = args.epochs if args.epochs is not None else common_config.get("epochs", 150)
        batch_size = args.batch_size if args.batch_size is not None else common_config.get("batch_size", 32)
        
        # Get model hyperparameters
        units = lstm_config.get("units", [128, 64, 32])
        activation = lstm_config.get("activation", "tanh")
        recurrent_activation = lstm_config.get("recurrent_activation", "sigmoid")
        use_bias = lstm_config.get("use_bias", True)
        dropout = lstm_config.get("dropout", 0.3)
        recurrent_dropout = lstm_config.get("recurrent_dropout", 0.3)
        bidirectional = lstm_config.get("bidirectional", True)
        use_batch_norm = lstm_config.get("use_batch_norm", True)
        attention_mechanism = lstm_config.get("attention_mechanism", True)
        
        # Create model
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        
        x = inputs
        
        # LSTM layers
        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1 or attention_mechanism
            
            if bidirectional:
                lstm_layer = Bidirectional(LSTM(
                    units=unit,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=return_sequences
                ))(x)
            else:
                lstm_layer = LSTM(
                    units=unit,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    return_sequences=return_sequences
                )(x)
            
            x = lstm_layer
            
            if use_batch_norm:
                x = BatchNormalization()(x)
        
        # Add attention mechanism if specified
        if attention_mechanism:
            # Simple attention mechanism
            from tensorflow.keras.layers import Attention, Lambda, Multiply
            
            attention_layer = Dense(units[-1] * (2 if bidirectional else 1), activation="tanh")(x)
            attention_weights = Dense(1, activation="softmax")(attention_layer)
            context_vector = Multiply()([x, attention_weights])
            x = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        
        x = Dropout(dropout)(x)
        x = Dense(64, activation="relu")(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save path
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/lstm_{pair_filename}_model"
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=common_config.get("early_stopping_patience", 15),
                min_delta=common_config.get("early_stopping_min_delta", 0.0005),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=common_config.get("learning_rate_reduction_factor", 0.5),
                patience=common_config.get("learning_rate_patience", 10),
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if args.verbose else 2
        )
        
        # Get best accuracy
        best_epoch = np.argmax(history.history["val_accuracy"])
        best_val_accuracy = history.history["val_accuracy"][best_epoch]
        
        # Load best model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        
        # Final evaluation
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate additional metrics
        y_pred = model.predict(X_val)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        
        # Log results
        logger.info(f"LSTM model training for {pair} completed:")
        logger.info(f"  Best epoch: {best_epoch + 1}/{epochs}")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_epoch": int(best_epoch)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def train_attention_gru_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                           hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train an Attention GRU model.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Bidirectional, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        gru_config = config.get("training_config", {}).get("attention_gru", {})
        
        # Override with hyperparams if available
        if hyperparams:
            for key, value in hyperparams.items():
                if key in gru_config:
                    gru_config[key] = value
        
        # Get training parameters
        epochs = args.epochs if args.epochs is not None else common_config.get("epochs", 150)
        batch_size = args.batch_size if args.batch_size is not None else common_config.get("batch_size", 32)
        
        # Get model hyperparameters
        gru_units = gru_config.get("gru_units", [96, 64, 32])
        attention_units = gru_config.get("attention_units", 64)
        dropout_rate = gru_config.get("dropout_rate", 0.3)
        activation = gru_config.get("activation", "tanh")
        recurrent_activation = gru_config.get("recurrent_activation", "sigmoid")
        attention_activation = gru_config.get("attention_activation", "tanh")
        use_bias = gru_config.get("use_bias", True)
        bidirectional = gru_config.get("bidirectional", True)
        use_batch_norm = gru_config.get("use_batch_norm", True)
        
        # Create model
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        
        x = inputs
        
        # GRU layers
        for i, unit in enumerate(gru_units):
            return_sequences = i < len(gru_units) - 1 or True  # Always return sequences for attention
            
            if bidirectional:
                gru_layer = Bidirectional(GRU(
                    units=unit,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    return_sequences=return_sequences
                ))(x)
            else:
                gru_layer = GRU(
                    units=unit,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    use_bias=use_bias,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    return_sequences=return_sequences
                )(x)
            
            x = gru_layer
            
            if use_batch_norm:
                x = BatchNormalization()(x)
        
        # Add attention mechanism
        from tensorflow.keras.layers import Lambda, Multiply
        
        attention_layer = Dense(attention_units, activation=attention_activation)(x)
        attention_weights = Dense(1, activation="softmax")(attention_layer)
        context_vector = Multiply()([x, attention_weights])
        x = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
        
        # Final layers
        x = Dense(64, activation="relu")(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
        
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save path
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/attention_gru_{pair_filename}_model"
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=common_config.get("early_stopping_patience", 15),
                min_delta=common_config.get("early_stopping_min_delta", 0.0005),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=common_config.get("learning_rate_reduction_factor", 0.5),
                patience=common_config.get("learning_rate_patience", 10),
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if args.verbose else 2
        )
        
        # Get best accuracy
        best_epoch = np.argmax(history.history["val_accuracy"])
        best_val_accuracy = history.history["val_accuracy"][best_epoch]
        
        # Load best model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        
        # Final evaluation
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate additional metrics
        y_pred = model.predict(X_val)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        
        # Log results
        logger.info(f"Attention GRU model training for {pair} completed:")
        logger.info(f"  Best epoch: {best_epoch + 1}/{epochs}")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_epoch": int(best_epoch)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training Attention GRU model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def train_transformer_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                         hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train a Transformer model.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        transformer_config = config.get("training_config", {}).get("transformer", {})
        
        # Override with hyperparams if available
        if hyperparams:
            for key, value in hyperparams.items():
                if key in transformer_config:
                    transformer_config[key] = value
        
        # Get training parameters
        epochs = args.epochs if args.epochs is not None else common_config.get("epochs", 150)
        batch_size = args.batch_size if args.batch_size is not None else common_config.get("batch_size", 32)
        
        # Get model hyperparameters
        num_layers = transformer_config.get("num_layers", 3)
        d_model = transformer_config.get("d_model", 128)
        num_heads = transformer_config.get("num_heads", 8)
        dff = transformer_config.get("dff", 512)
        dropout_rate = transformer_config.get("dropout_rate", 0.2)
        use_batch_norm = transformer_config.get("use_batch_norm", True)
        layer_norm = transformer_config.get("layer_norm", True)
        positional_encoding = transformer_config.get("positional_encoding", True)
        
        # Import or create MultiHeadAttention layer
        try:
            from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        except ImportError:
            logger.error("MultiHeadAttention layer not available. Using TensorFlow 2.x is required.")
            return None, {}
        
        # Define Transformer encoder block
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head Self Attention
            if layer_norm:
                attention_input = LayerNormalization(epsilon=1e-6)(inputs)
            else:
                attention_input = inputs
            
            attention_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(attention_input, attention_input)
            
            # Skip connection
            attention_output = Dropout(dropout)(attention_output)
            attention_output = tf.keras.layers.add([inputs, attention_output])
            
            # Feed Forward
            if layer_norm:
                ffn_input = LayerNormalization(epsilon=1e-6)(attention_output)
            else:
                ffn_input = attention_output
            
            ffn_output = Dense(ff_dim, activation="relu")(ffn_input)
            ffn_output = Dropout(dropout)(ffn_output)
            ffn_output = Dense(inputs.shape[-1])(ffn_output)
            
            # Skip connection
            ffn_output = Dropout(dropout)(ffn_output)
            return tf.keras.layers.add([attention_output, ffn_output])
        
        # Add positional encoding
        def add_positional_encoding(inputs):
            input_shape = tf.shape(inputs)
            batch_size, seq_length = input_shape[0], input_shape[1]
            input_dim = inputs.shape[-1]
            
            # Generate positional encodings
            positions = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)
            positions = tf.expand_dims(positions, axis=1)
            
            # Use sine and cosine functions of different frequencies
            div_term = tf.pow(10000.0, tf.range(0, input_dim, 2, dtype=tf.float32) / input_dim)
            div_term = tf.expand_dims(div_term, axis=0)
            
            pos_encoding = tf.zeros((seq_length, input_dim))
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.range(seq_length), tf.range(0, input_dim, 2)], axis=1),
                tf.sin(positions / div_term)
            )
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.range(seq_length), tf.range(1, input_dim, 2)], axis=1),
                tf.cos(positions / div_term)
            )
            
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)
            pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
            
            return inputs + pos_encoding
        
        # Create model
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        
        # Add positional encoding if specified
        if positional_encoding:
            x = tf.keras.layers.Lambda(add_positional_encoding)(inputs)
        else:
            x = inputs
        
        # Transformer encoder blocks
        for _ in range(num_layers):
            x = transformer_encoder(
                x,
                head_size=d_model // num_heads,
                num_heads=num_heads,
                ff_dim=dff,
                dropout=dropout_rate
            )
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Final layers
        x = Dense(64, activation="relu")(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(32, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=1e-3)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save path
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/transformer_{pair_filename}_model"
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=common_config.get("early_stopping_patience", 15),
                min_delta=common_config.get("early_stopping_min_delta", 0.0005),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=common_config.get("learning_rate_reduction_factor", 0.5),
                patience=common_config.get("learning_rate_patience", 10),
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if args.verbose else 2
        )
        
        # Get best accuracy
        best_epoch = np.argmax(history.history["val_accuracy"])
        best_val_accuracy = history.history["val_accuracy"][best_epoch]
        
        # Load best model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        
        # Final evaluation
        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate additional metrics
        y_pred = model.predict(X_val)
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_pred_binary = (y_pred > 0.5).astype(int)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        
        # Log results
        logger.info(f"Transformer model training for {pair} completed:")
        logger.info(f"  Best epoch: {best_epoch + 1}/{epochs}")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_epoch": int(best_epoch)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training Transformer model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def train_xgboost_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                     hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train an XGBoost model.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        xgboost_config = config.get("hyperparameter_optimization", {}).get("parameters", {}).get("xgboost", {})
        
        # Override with hyperparams if available
        if hyperparams:
            xgboost_params = {k: v for k, v in hyperparams.items() if k in xgboost_config}
        else:
            # Default parameters
            xgboost_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'scale_pos_weight': 1
            }
        
        # Create model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            **xgboost_params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=args.verbose
        )
        
        # Save model
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/xgboost_{pair_filename}_model.json"
        model.save_model(model_path)
        
        # Evaluate model
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        # Additional metrics
        precision = precision_score(y_val, y_pred_val)
        recall = recall_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Log results
        logger.info(f"XGBoost model training for {pair} completed:")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def train_lightgbm_model(X_train, X_val, y_train, y_val, config: Dict, pair: str, 
                      hyperparams: Dict = None, args=None) -> Tuple[Any, Dict]:
    """
    Train a LightGBM model.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        config: Configuration
        pair: Trading pair
        hyperparams: Hyperparameters from optimization
        args: Command line arguments
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Get model configuration
        common_config = config.get("training_config", {}).get("common", {})
        lightgbm_config = config.get("hyperparameter_optimization", {}).get("parameters", {}).get("lightgbm", {})
        
        # Override with hyperparams if available
        if hyperparams:
            lightgbm_params = {k: v for k, v in hyperparams.items() if k in lightgbm_config}
        else:
            # Default parameters
            lightgbm_params = {
                'n_estimators': 100,
                'num_leaves': 31,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 20,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'bagging_freq': 0
            }
        
        # Create model
        model = lgb.LGBMClassifier(
            objective='binary',
            **lightgbm_params,
            random_state=42
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=args.verbose
        )
        
        # Save model
        pair_filename = pair.replace("/", "_")
        model_path = f"{MODEL_DIR}/lightgbm_{pair_filename}_model.txt"
        model.booster_.save_model(model_path)
        
        # Evaluate model
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        
        # Additional metrics
        precision = precision_score(y_val, y_pred_val)
        recall = recall_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Log results
        logger.info(f"LightGBM model training for {pair} completed:")
        logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Return metrics
        metrics = {
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Error training LightGBM model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def save_training_metrics(metrics: Dict, pair: str, model_type: str, stage: int, args) -> bool:
    """
    Save training metrics to a JSON file.
    
    Args:
        metrics: Training metrics
        pair: Trading pair
        model_type: Model type
        stage: Training stage
        args: Command line arguments
        
    Returns:
        success: Whether the save succeeded
    """
    try:
        # Create directory if it doesn't exist
        results_dir = f"{MODEL_DIR}/metrics"
        Path(results_dir).mkdir(exist_ok=True)
        
        # Save metrics
        pair_filename = pair.replace("/", "_")
        metrics_file = f"{results_dir}/{model_type}_{pair_filename}_stage{stage}_metrics.json"
        
        # Add metadata
        metrics["pair"] = pair
        metrics["model_type"] = model_type
        metrics["stage"] = stage
        metrics["training_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics["target_accuracy"] = args.target_accuracy
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved training metrics to {metrics_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving training metrics: {e}")
        return False


def evaluate_stage_success(metrics: Dict, target_accuracy: float = 0.95) -> bool:
    """
    Evaluate if the training stage was successful.
    
    Args:
        metrics: Training metrics
        target_accuracy: Target accuracy
        
    Returns:
        success: Whether the stage was successful
    """
    val_accuracy = metrics.get("val_accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    f1 = metrics.get("f1", 0)
    
    # Success criteria
    accuracy_success = val_accuracy >= target_accuracy
    precision_success = precision >= 0.9  # High precision for trading
    recall_success = recall >= 0.9  # High recall for capturing opportunities
    f1_success = f1 >= 0.9  # High F1 score for overall performance
    
    # Overall success
    success = accuracy_success and precision_success and recall_success and f1_success
    
    logger.info(f"Stage evaluation:")
    logger.info(f"  Accuracy: {val_accuracy:.4f} (Target: {target_accuracy:.4f}) - {'Success' if accuracy_success else 'Needs improvement'}")
    logger.info(f"  Precision: {precision:.4f} (Target: 0.90) - {'Success' if precision_success else 'Needs improvement'}")
    logger.info(f"  Recall: {recall:.4f} (Target: 0.90) - {'Success' if recall_success else 'Needs improvement'}")
    logger.info(f"  F1 Score: {f1:.4f} (Target: 0.90) - {'Success' if f1_success else 'Needs improvement'}")
    logger.info(f"  Overall: {'Success' if success else 'Needs improvement'}")
    
    return success


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Training {args.model_type} model for {args.pair} (Stage {args.stage})")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return 1
    
    # Check if model already exists
    pair_filename = args.pair.replace("/", "_")
    model_path = f"{MODEL_DIR}/{args.model_type}_{pair_filename}_model"
    
    if os.path.exists(model_path) and not args.force:
        logger.info(f"Model already exists at {model_path}. Use --force to retrain.")
        return 0
    
    # Load data
    X, y = load_data(args.pair, args.timeframe)
    if X is None or y is None:
        logger.error("Failed to load data. Exiting.")
        return 1
    
    # Prepare data
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data(X, y, args.model_type, config, args.pair)
    
    # Load hyperparameter optimization results
    hyperparams = None
    if args.use_hyperopt:
        hyperparams = load_hyperopt_results(args.pair, args.model_type)
    
    # Train model
    model = None
    metrics = {}
    
    if args.model_type == 'tcn':
        model, metrics = train_tcn_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    elif args.model_type == 'lstm':
        model, metrics = train_lstm_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    elif args.model_type == 'attention_gru':
        model, metrics = train_attention_gru_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    elif args.model_type == 'transformer':
        model, metrics = train_transformer_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    elif args.model_type == 'xgboost':
        model, metrics = train_xgboost_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    elif args.model_type == 'lightgbm':
        model, metrics = train_lightgbm_model(X_train, X_val, y_train, y_val, config, args.pair, hyperparams, args)
    
    if model is None:
        logger.error(f"Failed to train {args.model_type} model for {args.pair}.")
        return 1
    
    # Save training metrics
    save_training_metrics(metrics, args.pair, args.model_type, args.stage, args)
    
    # Evaluate stage success
    stage_success = evaluate_stage_success(metrics, args.target_accuracy)
    
    # Print final results
    val_accuracy = metrics.get("val_accuracy", 0)
    logger.info(f"Final accuracy: {val_accuracy:.4f} (Target: {args.target_accuracy:.4f})")
    logger.info(f"Stage {args.stage} {'successful' if stage_success else 'needs improvement'}")
    
    # Return success/failure
    return 0 if val_accuracy >= args.target_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())