#!/usr/bin/env python3
"""
Advanced ML Integration for Kraken Trading Bot

This module implements an advanced machine learning integration system that:
1. Trains multiple model types (TCN, LSTM, GRU, Transformer, etc.) in ensemble
2. Continuously optimizes and retrains models with both historical and live data
3. Dynamically adjusts trading parameters based on real-time prediction confidence
4. Maintains separate models for entry/exit timing, direction prediction, and risk management
5. Implements adaptive feature selection based on market regimes
"""
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, TimeDistributed, LayerNormalization
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import custom TCN implementation
from implement_tcn_architecture import TCNLayer, create_tcn_model
from custom_transformer import TransformerBlock, PositionalEncoding
from hybrid_models import create_cnn_lstm_model, create_tcn_gru_attention_model, create_transformer_lstm_model, create_ultra_ensemble_model

# Import our modules
import kraken_api_client as kraken
from trade_optimizer import TradeOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
MODEL_DIR = "ml_models"
HISTORICAL_DIR = f"{DATA_DIR}/historical_data"
FEATURE_IMPORTANCE_DIR = f"{DATA_DIR}/feature_importance"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
TRAINING_LOG_FILE = f"{DATA_DIR}/training_log.json"

# ML parameters
SEQUENCE_LENGTH = 60  # Number of time steps to look back
PREDICTION_HORIZON = 12  # Number of steps to predict ahead
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
N_FEATURES = 50  # Will be determined dynamically based on input data

# Default trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Model architectures to use in ensemble
MODEL_ARCHITECTURES = [
    "tcn", "lstm", "gru", "attention_gru", "transformer", "hybrid"
]

class MLModelEnsemble:
    """
    Ensemble of multiple ML models for trading predictions
    """
    def __init__(self, pair: str, task: str = "direction", model_dir: str = MODEL_DIR):
        """
        Initialize ML model ensemble
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            task: Prediction task (direction, entry, exit, volatility, etc.)
            model_dir: Directory to save models
        """
        self.pair = pair
        self.task = task
        self.model_dir = model_dir
        self.safe_pair = pair.replace('/', '_')
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.hyperparameters = {}
        self.performance_metrics = {}
        self.is_trained = False
        self.last_training_time = None
        
        # Ensure model directory exists
        os.makedirs(f"{model_dir}/{self.safe_pair}", exist_ok=True)
        
        # Initialize ensemble models
        self._initialize_models()
        
        logger.info(f"Initialized ML model ensemble for {pair} ({task})")
    
    def _initialize_models(self):
        """Initialize all models in the ensemble"""
        for architecture in MODEL_ARCHITECTURES:
            model_path = f"{self.model_dir}/{self.safe_pair}/{architecture}_{self.task}_model"
            
            # Check if model already exists
            if os.path.exists(model_path):
                try:
                    self.models[architecture] = load_model(model_path)
                    self.is_trained = True
                    logger.info(f"Loaded existing {architecture} model for {self.pair} ({self.task})")
                except Exception as e:
                    logger.error(f"Error loading {architecture} model for {self.pair}: {e}")
            else:
                self.is_trained = False
                logger.info(f"Will train new {architecture} model for {self.pair} ({self.task})")
        
        # Load scalers if they exist
        scaler_path = f"{self.model_dir}/{self.safe_pair}/scalers_{self.task}.pkl"
        if os.path.exists(scaler_path):
            try:
                self.scalers = pd.read_pickle(scaler_path)
                logger.info(f"Loaded existing scalers for {self.pair} ({self.task})")
            except Exception as e:
                logger.error(f"Error loading scalers for {self.pair}: {e}")
        
        # Load feature importance if it exists
        fi_path = f"{FEATURE_IMPORTANCE_DIR}/{self.safe_pair}_{self.task}_feature_importance.json"
        if os.path.exists(fi_path):
            try:
                with open(fi_path, 'r') as f:
                    self.feature_importance = json.load(f)
                logger.info(f"Loaded feature importance for {self.pair} ({self.task})")
            except Exception as e:
                logger.error(f"Error loading feature importance for {self.pair}: {e}")
        
        # Load hyperparameters if they exist
        hp_path = f"{self.model_dir}/{self.safe_pair}/hyperparams_{self.task}.json"
        if os.path.exists(hp_path):
            try:
                with open(hp_path, 'r') as f:
                    self.hyperparameters = json.load(f)
                logger.info(f"Loaded hyperparameters for {self.pair} ({self.task})")
            except Exception as e:
                logger.error(f"Error loading hyperparameters for {self.pair}: {e}")
        
        # Load performance metrics if they exist
        metrics_path = f"{self.model_dir}/{self.safe_pair}/metrics_{self.task}.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
                logger.info(f"Loaded performance metrics for {self.pair} ({self.task})")
            except Exception as e:
                logger.error(f"Error loading performance metrics for {self.pair}: {e}")
    
    def _create_model(self, architecture: str, input_shape: Tuple[int, int], hyperparams: Dict[str, Any] = None) -> Model:
        """
        Create model based on specified architecture
        
        Args:
            architecture: Model architecture (tcn, lstm, gru, etc.)
            input_shape: Input shape (sequence_length, n_features)
            hyperparams: Optional hyperparameters
        
        Returns:
            Keras model
        """
        if hyperparams is None:
            hyperparams = {}
        
        # Set default hyperparameters if not provided
        hp = {
            # Common hyperparameters
            'learning_rate': hyperparams.get('learning_rate', 0.001),
            'dropout_rate': hyperparams.get('dropout_rate', 0.2),
            'recurrent_dropout': hyperparams.get('recurrent_dropout', 0.2),
            'l2_reg': hyperparams.get('l2_reg', 0.001),
            
            # Architecture-specific hyperparameters
            'lstm_units': hyperparams.get('lstm_units', [128, 64]),
            'gru_units': hyperparams.get('gru_units', [128, 64]),
            'tcn_nb_filters': hyperparams.get('tcn_nb_filters', 64),
            'tcn_kernel_size': hyperparams.get('tcn_kernel_size', 3),
            'tcn_dilations': hyperparams.get('tcn_dilations', [1, 2, 4, 8]),
            'transformer_heads': hyperparams.get('transformer_heads', 8),
            'transformer_dim': hyperparams.get('transformer_dim', 64),
            'dense_units': hyperparams.get('dense_units', [32, 16])
        }
        
        # Store hyperparameters
        self.hyperparameters[architecture] = hp
        
        # Create model based on architecture
        if architecture == 'tcn':
            # Temporal Convolutional Network
            model = Sequential([
                TCN(
                    nb_filters=hp['tcn_nb_filters'],
                    kernel_size=hp['tcn_kernel_size'],
                    dilations=hp['tcn_dilations'],
                    activation='relu',
                    padding='causal',
                    return_sequences=False,
                    dropout_rate=hp['dropout_rate'],
                    kernel_regularizer=regularizers.l2(hp['l2_reg']),
                    input_shape=input_shape
                ),
                BatchNormalization(),
                Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
                Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
            ])
            
            # Add final layer based on task
            if self.task == 'direction':
                model.add(Dense(1, activation='sigmoid'))  # Binary classification (up/down)
            elif self.task == 'entry' or self.task == 'exit':
                model.add(Dense(1, activation='sigmoid'))  # Binary classification (enter/don't enter)
            elif self.task == 'volatility':
                model.add(Dense(1, activation='linear'))   # Regression (volatility prediction)
            else:
                model.add(Dense(1, activation='linear'))   # Default to regression
        
        elif architecture == 'lstm':
            # LSTM network
            model = Sequential([
                LSTM(
                    hp['lstm_units'][0],
                    return_sequences=True,
                    dropout=hp['dropout_rate'],
                    recurrent_dropout=hp['recurrent_dropout'],
                    kernel_regularizer=regularizers.l2(hp['l2_reg']),
                    input_shape=input_shape
                ),
                BatchNormalization(),
                LSTM(
                    hp['lstm_units'][1],
                    return_sequences=False,
                    dropout=hp['dropout_rate'],
                    recurrent_dropout=hp['recurrent_dropout'],
                    kernel_regularizer=regularizers.l2(hp['l2_reg'])
                ),
                BatchNormalization(),
                Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
                Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
            ])
            
            # Add final layer based on task
            if self.task == 'direction':
                model.add(Dense(1, activation='sigmoid'))
            elif self.task == 'entry' or self.task == 'exit':
                model.add(Dense(1, activation='sigmoid'))
            elif self.task == 'volatility':
                model.add(Dense(1, activation='linear'))
            else:
                model.add(Dense(1, activation='linear'))
        
        elif architecture == 'gru':
            # GRU network
            model = Sequential([
                GRU(
                    hp['gru_units'][0],
                    return_sequences=True,
                    dropout=hp['dropout_rate'],
                    recurrent_dropout=hp['recurrent_dropout'],
                    kernel_regularizer=regularizers.l2(hp['l2_reg']),
                    input_shape=input_shape
                ),
                BatchNormalization(),
                GRU(
                    hp['gru_units'][1],
                    return_sequences=False,
                    dropout=hp['dropout_rate'],
                    recurrent_dropout=hp['recurrent_dropout'],
                    kernel_regularizer=regularizers.l2(hp['l2_reg'])
                ),
                BatchNormalization(),
                Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
                Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dropout(hp['dropout_rate']),
            ])
            
            # Add final layer based on task
            if self.task == 'direction':
                model.add(Dense(1, activation='sigmoid'))
            elif self.task == 'entry' or self.task == 'exit':
                model.add(Dense(1, activation='sigmoid'))
            elif self.task == 'volatility':
                model.add(Dense(1, activation='linear'))
            else:
                model.add(Dense(1, activation='linear'))
        
        elif architecture == 'attention_gru':
            # Implement attention mechanism with GRU
            # This is a more advanced architecture
            inputs = Input(shape=input_shape)
            
            # GRU layers
            gru1 = GRU(
                hp['gru_units'][0],
                return_sequences=True,
                dropout=hp['dropout_rate'],
                recurrent_dropout=hp['recurrent_dropout'],
                kernel_regularizer=regularizers.l2(hp['l2_reg'])
            )(inputs)
            
            # Self-attention mechanism
            attention_scores = TimeDistributed(Dense(1))(gru1)
            attention_weights = tf.nn.softmax(attention_scores, axis=1)
            context_vector = tf.reduce_sum(attention_weights * gru1, axis=1)
            
            # Dense layers
            x = BatchNormalization()(context_vector)
            x = Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            x = Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            
            # Output layer
            if self.task == 'direction':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'entry' or self.task == 'exit':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'volatility':
                outputs = Dense(1, activation='linear')(x)
            else:
                outputs = Dense(1, activation='linear')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        elif architecture == 'transformer':
            # For simplicity, we'll implement a simplified transformer-based model
            inputs = Input(shape=input_shape)
            
            # Simple multi-head attention (you could expand this for a full transformer)
            # This is a simplified version of the transformer architecture
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=hp['transformer_heads'],
                key_dim=hp['transformer_dim']
            )(inputs, inputs)
            
            # Add & Normalize
            attention_output = tf.keras.layers.LayerNormalization()(attention_output + inputs)
            
            # Feed Forward Network
            ffn = tf.keras.Sequential([
                Dense(hp['transformer_dim'] * 4, activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg'])),
                Dense(hp['transformer_dim'])
            ])
            
            # Add & Normalize
            x = tf.keras.layers.LayerNormalization()(ffn(attention_output) + attention_output)
            
            # Flatten the sequence dimension
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Dense layers
            x = Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            x = Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            
            # Output layer
            if self.task == 'direction':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'entry' or self.task == 'exit':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'volatility':
                outputs = Dense(1, activation='linear')(x)
            else:
                outputs = Dense(1, activation='linear')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        elif architecture == 'hybrid':
            # Hybrid model combining TCN, LSTM, and attention mechanisms
            inputs = Input(shape=input_shape)
            
            # TCN branch
            tcn_branch = TCN(
                nb_filters=hp['tcn_nb_filters'],
                kernel_size=hp['tcn_kernel_size'],
                dilations=hp['tcn_dilations'],
                activation='relu',
                padding='causal',
                return_sequences=True,
                dropout_rate=hp['dropout_rate'],
                kernel_regularizer=regularizers.l2(hp['l2_reg'])
            )(inputs)
            
            # LSTM branch
            lstm_branch = LSTM(
                hp['lstm_units'][0],
                return_sequences=True,
                dropout=hp['dropout_rate'],
                recurrent_dropout=hp['recurrent_dropout'],
                kernel_regularizer=regularizers.l2(hp['l2_reg'])
            )(inputs)
            
            # Attention mechanism for both branches
            tcn_attention = TimeDistributed(Dense(1))(tcn_branch)
            tcn_weights = tf.nn.softmax(tcn_attention, axis=1)
            tcn_context = tf.reduce_sum(tcn_weights * tcn_branch, axis=1)
            
            lstm_attention = TimeDistributed(Dense(1))(lstm_branch)
            lstm_weights = tf.nn.softmax(lstm_attention, axis=1)
            lstm_context = tf.reduce_sum(lstm_weights * lstm_branch, axis=1)
            
            # Combine branches
            combined = Concatenate()([tcn_context, lstm_context])
            x = BatchNormalization()(combined)
            
            # Dense layers
            x = Dense(hp['dense_units'][0], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            x = Dense(hp['dense_units'][1], activation='relu', kernel_regularizer=regularizers.l2(hp['l2_reg']))(x)
            x = Dropout(hp['dropout_rate'])(x)
            
            # Output layer
            if self.task == 'direction':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'entry' or self.task == 'exit':
                outputs = Dense(1, activation='sigmoid')(x)
            elif self.task == 'volatility':
                outputs = Dense(1, activation='linear')(x)
            else:
                outputs = Dense(1, activation='linear')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        else:
            # Default to simple LSTM if architecture not recognized
            logger.warning(f"Architecture {architecture} not recognized, defaulting to LSTM")
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                LSTM(32),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid' if self.task in ['direction', 'entry', 'exit'] else 'linear')
            ])
        
        # Compile model
        if self.task in ['direction', 'entry', 'exit']:
            # Binary classification tasks
            model.compile(
                optimizer=Adam(learning_rate=hp['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        else:
            # Regression tasks
            model.compile(
                optimizer=Adam(learning_rate=hp['learning_rate']),
                loss='mean_squared_error',
                metrics=['mae', 'mse']
            )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_col: str, scaler_x: Optional[StandardScaler] = None, 
                    scaler_y: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
        """
        Prepare data for ML model training
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            scaler_x: Optional scaler for features
            scaler_y: Optional scaler for target
            
        Returns:
            X, y, scaler_x, scaler_y
        """
        # Select features (all columns except target)
        features = data.drop(columns=[target_col] if target_col in data.columns else [])
        
        # Extract target
        if target_col in data.columns:
            target = data[target_col].values
        else:
            # If target column doesn't exist, create it
            # For 'direction' task, target is 1 if price went up, 0 if down
            if self.task == 'direction':
                # Create target based on price change
                if 'close' in data.columns:
                    target = (data['close'].shift(-1) > data['close']).astype(int).values
                else:
                    raise ValueError("Cannot create 'direction' target: 'close' column not found")
            
            # For 'entry' task, we'd need a more complex rule (this is simplified)
            elif self.task == 'entry':
                # Create target based on whether entering now would be profitable
                if 'close' in data.columns:
                    # Check if price rises by at least 2% in next N periods
                    lookahead = 6  # Number of periods to look ahead
                    pct_change = (data['close'].shift(-lookahead) / data['close'] - 1) * 100
                    target = (pct_change > 2.0).astype(int).values
                else:
                    raise ValueError("Cannot create 'entry' target: 'close' column not found")
            
            # For 'exit' task, also need a complex rule
            elif self.task == 'exit':
                # Create target based on whether exiting now would be profitable
                if 'close' in data.columns:
                    # Check if price falls by at least 1% in next N periods
                    lookahead = 3  # Number of periods to look ahead
                    pct_change = (data['close'].shift(-lookahead) / data['close'] - 1) * 100
                    target = (pct_change < -1.0).astype(int).values
                else:
                    raise ValueError("Cannot create 'exit' target: 'close' column not found")
            
            # For 'volatility' task
            elif self.task == 'volatility':
                # Create target based on future price volatility
                if 'close' in data.columns:
                    # Calculate future volatility as standard deviation of returns
                    lookahead = 12  # Number of periods to look ahead
                    future_returns = []
                    for i in range(1, lookahead + 1):
                        future_returns.append(data['close'].pct_change(i).shift(-i))
                    
                    future_returns_df = pd.concat(future_returns, axis=1)
                    target = future_returns_df.std(axis=1).values
                else:
                    raise ValueError("Cannot create 'volatility' target: 'close' column not found")
            
            else:
                raise ValueError(f"Cannot create target for unknown task: {self.task}")
        
        # Create or use scalers
        if scaler_x is None:
            scaler_x = StandardScaler()
            features_scaled = scaler_x.fit_transform(features)
        else:
            features_scaled = scaler_x.transform(features)
        
        if scaler_y is None and self.task not in ['direction', 'entry', 'exit']:
            # Only scale target for regression tasks
            scaler_y = StandardScaler()
            target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
        elif self.task not in ['direction', 'entry', 'exit']:
            target_scaled = scaler_y.transform(target.reshape(-1, 1)).flatten()
        else:
            # For classification tasks, don't scale target
            target_scaled = target
            scaler_y = None
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled)
        
        return X, y, scaler_x, scaler_y
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray,
                        sequence_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            features: Scaled features
            target: Scaled target
            sequence_length: Length of sequences
            
        Returns:
            X, y sequences
        """
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, target_col: str, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train all models in the ensemble
        
        Args:
            data: DataFrame with features and target
            target_col: Target column name
            force_retrain: Whether to retrain even if already trained
            
        Returns:
            Dictionary with training results
        """
        if self.is_trained and not force_retrain:
            logger.info(f"Models for {self.pair} ({self.task}) already trained. Use force_retrain=True to retrain.")
            return self.performance_metrics
        
        # Prepare data
        X, y, scaler_x, scaler_y = self.prepare_data(data, target_col)
        
        # Save scalers
        self.scalers['x'] = scaler_x
        self.scalers['y'] = scaler_y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, shuffle=False)
        
        # Train each model in the ensemble
        for architecture in MODEL_ARCHITECTURES:
            logger.info(f"Training {architecture} model for {self.pair} ({self.task})")
            
            try:
                # Create model
                model = self._create_model(architecture, (X.shape[1], X.shape[2]))
                
                # Set up callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                    ModelCheckpoint(
                        f"{self.model_dir}/{self.safe_pair}/{architecture}_{self.task}_model_best", 
                        save_best_only=True, 
                        monitor='val_loss'
                    )
                ]
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SPLIT,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate model
                results = model.evaluate(X_test, y_test, verbose=0)
                
                # Save performance metrics
                metrics = {
                    'loss': float(results[0]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'training_time': datetime.now().isoformat()
                }
                
                # For classification tasks, add additional metrics
                if self.task in ['direction', 'entry', 'exit']:
                    y_pred = (model.predict(X_test) > 0.5).astype(int)
                    metrics.update({
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                        'f1': float(f1_score(y_test, y_pred, zero_division=0))
                    })
                    
                    # Log detailed metrics
                    logger.info(f"{architecture} model for {self.pair} ({self.task}) metrics:")
                    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
                else:
                    metrics.update({
                        'mae': float(results[1]),
                        'mse': float(results[2])
                    })
                    
                    # Log detailed metrics
                    logger.info(f"{architecture} model for {self.pair} ({self.task}) metrics:")
                    logger.info(f"  MAE: {metrics['mae']:.4f}")
                    logger.info(f"  MSE: {metrics['mse']:.4f}")
                
                # Store metrics
                self.performance_metrics[architecture] = metrics
                
                # Save model
                model.save(f"{self.model_dir}/{self.safe_pair}/{architecture}_{self.task}_model")
                self.models[architecture] = model
                
                logger.info(f"Successfully trained and saved {architecture} model for {self.pair} ({self.task})")
            
            except Exception as e:
                logger.error(f"Error training {architecture} model for {self.pair} ({self.task}): {e}")
        
        # Save scalers
        scaler_path = f"{self.model_dir}/{self.safe_pair}/scalers_{self.task}.pkl"
        pd.to_pickle(self.scalers, scaler_path)
        
        # Save performance metrics
        metrics_path = f"{self.model_dir}/{self.safe_pair}/metrics_{self.task}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save hyperparameters
        hp_path = f"{self.model_dir}/{self.safe_pair}/hyperparams_{self.task}.json"
        with open(hp_path, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
        
        # Update training status
        self.is_trained = True
        self.last_training_time = datetime.now()
        
        # Return performance metrics
        return self.performance_metrics
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using the ensemble
        
        Args:
            features: DataFrame with features
        
        Returns:
            Dictionary with predictions and confidence
        """
        if not self.is_trained or not self.models:
            logger.error(f"Models for {self.pair} ({self.task}) not trained")
            return {'prediction': None, 'confidence': 0.0, 'ensemble_agreement': 0.0}
        
        # Scale features
        if 'x' not in self.scalers:
            logger.error(f"Feature scaler for {self.pair} ({self.task}) not found")
            return {'prediction': None, 'confidence': 0.0, 'ensemble_agreement': 0.0}
        
        # Check if feature columns match the scaler
        if len(features.columns) != self.scalers['x'].n_features_in_:
            logger.error(f"Feature count mismatch: expected {self.scalers['x'].n_features_in_}, got {len(features.columns)}")
            # Try to use a subset of columns if possible
            if len(features.columns) > self.scalers['x'].n_features_in_:
                logger.warning(f"Attempting to use first {self.scalers['x'].n_features_in_} columns")
                features = features.iloc[:, :self.scalers['x'].n_features_in_]
            else:
                return {'prediction': None, 'confidence': 0.0, 'ensemble_agreement': 0.0}
        
        features_scaled = self.scalers['x'].transform(features)
        
        # Create sequence
        if len(features_scaled) < SEQUENCE_LENGTH:
            logger.error(f"Not enough data points for sequence: got {len(features_scaled)}, need {SEQUENCE_LENGTH}")
            return {'prediction': None, 'confidence': 0.0, 'ensemble_agreement': 0.0}
        
        # Take the last SEQUENCE_LENGTH data points
        sequence = features_scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, -1)
        
        # Make predictions with each model
        predictions = {}
        for architecture, model in self.models.items():
            try:
                # Make prediction
                pred = model.predict(sequence, verbose=0)
                
                # Post-process prediction
                if self.task in ['direction', 'entry', 'exit']:
                    # For classification tasks, output is probability
                    pred_value = float(pred[0][0])
                    pred_class = 1 if pred_value > 0.5 else 0
                    predictions[architecture] = {
                        'value': pred_value,
                        'class': pred_class,
                        'confidence': abs(pred_value - 0.5) * 2  # Scale to 0-1
                    }
                else:
                    # For regression tasks, unscale the output
                    if 'y' in self.scalers and self.scalers['y'] is not None:
                        pred_value = float(self.scalers['y'].inverse_transform(pred)[0][0])
                    else:
                        pred_value = float(pred[0][0])
                    
                    predictions[architecture] = {
                        'value': pred_value,
                        'confidence': 0.8  # Default confidence for regression
                    }
            except Exception as e:
                logger.error(f"Error making prediction with {architecture} model: {e}")
        
        # Ensemble the predictions
        if not predictions:
            return {'prediction': None, 'confidence': 0.0, 'ensemble_agreement': 0.0}
        
        # For classification tasks
        if self.task in ['direction', 'entry', 'exit']:
            # Get votes
            votes = [p['class'] for p in predictions.values()]
            vote_count = sum(votes)
            majority_vote = 1 if vote_count > len(votes) / 2 else 0
            
            # Calculate agreement/confidence
            ensemble_agreement = vote_count / len(votes) if majority_vote == 1 else (len(votes) - vote_count) / len(votes)
            
            # Weight by individual model confidence
            confidence_weighted_sum = sum(p['confidence'] for p in predictions.values())
            avg_confidence = confidence_weighted_sum / len(predictions)
            
            # Combine agreement and average confidence
            final_confidence = (ensemble_agreement + avg_confidence) / 2
            
            return {
                'prediction': majority_vote,
                'confidence': final_confidence,
                'ensemble_agreement': ensemble_agreement,
                'individual_predictions': predictions
            }
        
        # For regression tasks
        else:
            # Weighted average based on model performance
            weights = {}
            for architecture in predictions:
                # Use inverse MSE as weight if available, otherwise use 1.0
                if architecture in self.performance_metrics and 'mse' in self.performance_metrics[architecture]:
                    weights[architecture] = 1.0 / max(0.0001, self.performance_metrics[architecture]['mse'])
                else:
                    weights[architecture] = 1.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in weights.items()}
            else:
                normalized_weights = {k: 1.0 / len(weights) for k in weights}
            
            # Calculate weighted average
            weighted_sum = sum(predictions[arch]['value'] * normalized_weights[arch] for arch in predictions)
            
            # Calculate confidence based on agreement
            values = [p['value'] for p in predictions.values()]
            mean_value = sum(values) / len(values)
            std_dev = np.std(values) if len(values) > 1 else 0
            
            # Normalize std_dev to a confidence score (higher std_dev = lower confidence)
            agreement_factor = max(0, 1 - (std_dev / (abs(mean_value) + 0.0001)))
            
            return {
                'prediction': weighted_sum,
                'confidence': agreement_factor,
                'ensemble_agreement': agreement_factor,
                'individual_predictions': predictions
            }
    
    def update_with_new_data(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Update models with new data (incremental training)
        
        Args:
            data: DataFrame with new data
            target_col: Target column name
            
        Returns:
            Dictionary with updated performance metrics
        """
        if not self.is_trained or not self.models:
            logger.info(f"Models for {self.pair} ({self.task}) not trained yet. Running full training.")
            return self.train(data, target_col)
        
        # Prepare data using existing scalers
        X, y, _, _ = self.prepare_data(data, target_col, self.scalers.get('x'), self.scalers.get('y'))
        
        # Update each model
        for architecture, model in self.models.items():
            try:
                logger.info(f"Updating {architecture} model for {self.pair} ({self.task}) with new data")
                
                # Set up callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=PATIENCE // 2, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
                ]
                
                # Fine-tune model with new data
                history = model.fit(
                    X, y,
                    epochs=EPOCHS // 2,  # Fewer epochs for fine-tuning
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SPLIT,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Update performance metrics
                self.performance_metrics[architecture]['last_update_time'] = datetime.now().isoformat()
                self.performance_metrics[architecture]['val_loss'] = float(history.history['val_loss'][-1])
                
                # Save updated model
                model.save(f"{self.model_dir}/{self.safe_pair}/{architecture}_{self.task}_model")
                
                logger.info(f"Successfully updated {architecture} model for {self.pair} ({self.task})")
            
            except Exception as e:
                logger.error(f"Error updating {architecture} model for {self.pair} ({self.task}): {e}")
        
        # Save updated performance metrics
        metrics_path = f"{self.model_dir}/{self.safe_pair}/metrics_{self.task}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Update training time
        self.last_training_time = datetime.now()
        
        return self.performance_metrics
    
    def get_feature_importance(self, data: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Calculate feature importance for the models
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_trained or not self.models:
            logger.error(f"Models for {self.pair} ({self.task}) not trained")
            return {}
        
        try:
            # Prepare data using existing scalers
            X, y, _, _ = self.prepare_data(data, target_col, self.scalers.get('x'), self.scalers.get('y'))
            
            # Extract feature names
            feature_names = data.drop(columns=[target_col] if target_col in data.columns else []).columns.tolist()
            
            # Calculate permutation importance for each model
            importance_scores = {}
            
            for architecture, model in self.models.items():
                # Use a simplified approach to estimate feature importance
                # We'll use a permutation-based approach
                baseline_prediction = model.predict(X, verbose=0)
                baseline_loss = ((baseline_prediction - y.reshape(-1, 1)) ** 2).mean()
                
                feature_importance = {}
                for i, feature in enumerate(feature_names):
                    # Create a copy of the data and permute one feature
                    X_permuted = X.copy()
                    X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
                    
                    # Make predictions with permuted data
                    permuted_prediction = model.predict(X_permuted, verbose=0)
                    permuted_loss = ((permuted_prediction - y.reshape(-1, 1)) ** 2).mean()
                    
                    # Calculate importance as the increase in loss
                    importance = max(0, permuted_loss - baseline_loss)
                    feature_importance[feature] = float(importance)
                
                # Normalize importance scores
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
                
                importance_scores[architecture] = feature_importance
            
            # Aggregate importance scores across architectures
            aggregated_importance = {}
            for feature in feature_names:
                scores = [model_scores.get(feature, 0) for model_scores in importance_scores.values()]
                aggregated_importance[feature] = sum(scores) / len(scores)
            
            # Save feature importance
            self.feature_importance = aggregated_importance
            
            # Save to file
            os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
            fi_path = f"{FEATURE_IMPORTANCE_DIR}/{self.safe_pair}_{self.task}_feature_importance.json"
            with open(fi_path, 'w') as f:
                json.dump(aggregated_importance, f, indent=2)
            
            return aggregated_importance
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
        
    def get_best_model(self) -> Tuple[str, Model]:
        """
        Get the best performing model in the ensemble
        
        Returns:
            Tuple of (architecture, model)
        """
        if not self.is_trained or not self.models:
            return None, None
        
        # Find the best model based on performance metrics
        best_architecture = None
        best_score = float('inf')
        
        for architecture, metrics in self.performance_metrics.items():
            if architecture not in self.models:
                continue
            
            # Use validation loss as the primary metric
            score = metrics.get('val_loss', float('inf'))
            
            if score < best_score:
                best_score = score
                best_architecture = architecture
        
        if best_architecture is None:
            return None, None
        
        return best_architecture, self.models[best_architecture]

class AdvancedMLTrader:
    """
    Advanced ML-based trading system that uses multiple ML models
    to make trading decisions and optimize parameters
    """
    def __init__(self, trading_pairs: List[str] = DEFAULT_PAIRS):
        """
        Initialize the advanced ML trader
        
        Args:
            trading_pairs: List of trading pairs to trade
        """
        self.trading_pairs = trading_pairs
        
        # Dictionary to hold model ensembles for each pair and task
        self.model_ensembles = {}
        
        # Initialize directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(HISTORICAL_DIR, exist_ok=True)
        os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)
        
        # Initialize model ensembles for each pair and task
        self._initialize_models()
        
        # Optimizer for trade parameters
        self.optimizer = TradeOptimizer(trading_pairs)
        
        logger.info(f"Initialized Advanced ML Trader for {len(trading_pairs)} pairs")
    
    def _initialize_models(self):
        """Initialize model ensembles for each pair and task"""
        tasks = ['direction', 'entry', 'exit', 'volatility']
        
        for pair in self.trading_pairs:
            self.model_ensembles[pair] = {}
            
            for task in tasks:
                try:
                    self.model_ensembles[pair][task] = MLModelEnsemble(pair, task)
                except Exception as e:
                    logger.error(f"Error initializing models for {pair} ({task}): {e}")
    
    def _load_historical_data(self, pair: str) -> pd.DataFrame:
        """
        Load historical data for a pair
        
        Args:
            pair: Trading pair
            
        Returns:
            DataFrame with historical data
        """
        pair_file = pair.replace('/', '_')
        csv_file = f"{HISTORICAL_DIR}/{pair_file}_60m.csv"
        
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                return df
            except Exception as e:
                logger.error(f"Error loading historical data for {pair}: {e}")
        
        # If CSV file doesn't exist, return empty DataFrame
        logger.warning(f"Historical data file for {pair} not found: {csv_file}")
        return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as features
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with added features
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Required columns missing from data: {[col for col in required_cols if col not in df.columns]}")
            
            # Try to use reasonable defaults for missing columns
            if 'open' not in df.columns and 'close' in df.columns:
                df['open'] = df['close'].shift(1)
            if 'high' not in df.columns and 'close' in df.columns and 'open' in df.columns:
                df['high'] = df[['open', 'close']].max(axis=1)
            if 'low' not in df.columns and 'close' in df.columns and 'open' in df.columns:
                df['low'] = df[['open', 'close']].min(axis=1)
            if 'volume' not in df.columns:
                df['volume'] = 0
        
        try:
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Calculate various moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            # Calculate price distances from moving averages
            for window in [20, 50, 100, 200]:
                df[f'dist_sma_{window}'] = (df['close'] / df[f'sma_{window}'] - 1) * 100
                df[f'dist_ema_{window}'] = (df['close'] / df[f'ema_{window}'] - 1) * 100
            
            # Calculate Bollinger Bands
            for window in [20, 50]:
                df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
                df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
                df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
                df[f'bb_pct_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            for window in [6, 14, 28]:
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss
                df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            for window in [7, 14, 28]:
                df[f'atr_{window}'] = true_range.rolling(window=window).mean()
                # Normalized ATR (ATR as percentage of price)
                df[f'natr_{window}'] = (df[f'atr_{window}'] / df['close']) * 100
            
            # Calculate Stochastic Oscillator
            for window in [14, 28]:
                df[f'lowest_{window}'] = df['low'].rolling(window=window).min()
                df[f'highest_{window}'] = df['high'].rolling(window=window).max()
                df[f'stoch_k_{window}'] = 100 * ((df['close'] - df[f'lowest_{window}']) / 
                                                (df[f'highest_{window}'] - df[f'lowest_{window}']))
                df[f'stoch_d_{window}'] = df[f'stoch_k_{window}'].rolling(window=3).mean()
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Additional features
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility indicators
            for window in [10, 20, 30]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * 100
            
            # Trend indicators
            df['adx_up'] = df['high'] - df['high'].shift(1)
            df['adx_down'] = df['low'].shift(1) - df['low']
            df['adx_up'] = np.where((df['adx_up'] > df['adx_down']) & (df['adx_up'] > 0), df['adx_up'], 0)
            df['adx_down'] = np.where((df['adx_down'] > df['adx_up']) & (df['adx_down'] > 0), df['adx_down'], 0)
            df['adx_up_sum'] = df['adx_up'].rolling(window=14).sum()
            df['adx_down_sum'] = df['adx_down'].rolling(window=14).sum()
            
            # Price momentum
            for window in [1, 3, 5, 10, 20, 50]:
                df[f'momentum_{window}'] = df['close'].pct_change(periods=window) * 100
            
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return df
    
    def train_all_models(self, force_retrain: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Train all models in the ensemble
        
        Args:
            force_retrain: Whether to retrain all models even if already trained
            
        Returns:
            Dictionary with training results for each pair and task
        """
        results = {}
        
        for pair in self.trading_pairs:
            results[pair] = {}
            
            # Load historical data
            logger.info(f"Loading and preparing data for {pair}")
            data = self._load_historical_data(pair)
            
            if data.empty:
                logger.error(f"No historical data available for {pair}")
                continue
            
            # Add technical features
            data = self._add_technical_features(data)
            
            if data.empty:
                logger.error(f"Failed to add technical features for {pair}")
                continue
            
            # Train models for each task
            for task in self.model_ensembles[pair]:
                logger.info(f"Training models for {pair} ({task})")
                
                try:
                    # Train the ensemble
                    task_results = self.model_ensembles[pair][task].train(data, task, force_retrain)
                    results[pair][task] = task_results
                    
                    # Calculate feature importance
                    importance = self.model_ensembles[pair][task].get_feature_importance(data, task)
                    logger.info(f"Top 5 features for {pair} ({task}): " + 
                               str(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]))
                
                except Exception as e:
                    logger.error(f"Error training models for {pair} ({task}): {e}")
                    results[pair][task] = {"error": str(e)}
        
        # Log overall results
        self._log_training_results(results)
        
        return results
    
    def _log_training_results(self, results: Dict[str, Dict[str, Any]]):
        """
        Log training results to file
        
        Args:
            results: Dictionary with training results
        """
        # Load existing log if available
        log_data = {}
        if os.path.exists(TRAINING_LOG_FILE):
            try:
                with open(TRAINING_LOG_FILE, 'r') as f:
                    log_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading training log: {e}")
        
        # Add new results
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        if 'history' not in log_data:
            log_data['history'] = []
        
        log_data['history'].append(log_entry)
        log_data['latest'] = results
        
        # Calculate summary statistics
        summary = {
            'pairs_trained': len(results),
            'models_trained': sum(len(pair_results) for pair_results in results.values()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add accuracy metrics for classification tasks
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for pair, pair_results in results.items():
            for task, task_results in pair_results.items():
                if task in ['direction', 'entry', 'exit']:
                    for arch, metrics in task_results.items():
                        if 'accuracy' in metrics:
                            accuracies.append(metrics['accuracy'])
                        if 'precision' in metrics:
                            precisions.append(metrics['precision'])
                        if 'recall' in metrics:
                            recalls.append(metrics['recall'])
                        if 'f1' in metrics:
                            f1_scores.append(metrics['f1'])
        
        if accuracies:
            summary['avg_accuracy'] = sum(accuracies) / len(accuracies)
        if precisions:
            summary['avg_precision'] = sum(precisions) / len(precisions)
        if recalls:
            summary['avg_recall'] = sum(recalls) / len(recalls)
        if f1_scores:
            summary['avg_f1'] = sum(f1_scores) / len(f1_scores)
        
        log_data['summary'] = summary
        
        # Save log
        with open(TRAINING_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Training complete. Average accuracy: {summary.get('avg_accuracy', 'N/A')}")
    
    def predict(self, pair: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions for a trading pair
        
        Args:
            pair: Trading pair
            features: DataFrame with features
            
        Returns:
            Dictionary with predictions for each task
        """
        if pair not in self.model_ensembles:
            logger.error(f"No models available for {pair}")
            return {}
        
        predictions = {}
        
        # Add technical features if needed
        if features.shape[1] < 20:  # Rough estimate that features need to be added
            features = self._add_technical_features(features)
        
        # Make prediction for each task
        for task, ensemble in self.model_ensembles[pair].items():
            try:
                task_prediction = ensemble.predict(features)
                predictions[task] = task_prediction
            except Exception as e:
                logger.error(f"Error making prediction for {pair} ({task}): {e}")
        
        return predictions
    
    def get_trading_decision(self, pair: str, features: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Get trading decision based on predictions
        
        Args:
            pair: Trading pair
            features: DataFrame with features
            current_price: Current market price
            
        Returns:
            Dictionary with trading decision
        """
        predictions = self.predict(pair, features)
        
        if not predictions:
            return {'action': 'hold', 'confidence': 0.0}
        
        # Extract predictions for each task
        direction_pred = predictions.get('direction', {}).get('prediction')
        direction_conf = predictions.get('direction', {}).get('confidence', 0.0)
        
        entry_pred = predictions.get('entry', {}).get('prediction')
        entry_conf = predictions.get('entry', {}).get('confidence', 0.0)
        
        exit_pred = predictions.get('exit', {}).get('prediction')
        exit_conf = predictions.get('exit', {}).get('confidence', 0.0)
        
        volatility_pred = predictions.get('volatility', {}).get('prediction')
        
        # Determine direction (long/short)
        if direction_pred is None:
            return {'action': 'hold', 'confidence': 0.0}
        
        direction = 'long' if direction_pred == 1 else 'short'
        
        # Determine action based on entry/exit signals
        action = 'hold'
        confidence = direction_conf
        
        if entry_pred == 1 and entry_conf > 0.6:
            action = f'enter_{direction}'
            confidence = (direction_conf + entry_conf) / 2
        elif exit_pred == 1 and exit_conf > 0.6:
            action = 'exit'
            confidence = exit_conf
        
        # Adjust parameters based on predictions
        params = {}
        
        # Set leverage based on confidence and volatility
        if action.startswith('enter'):
            # Get optimal leverage from optimizer
            leverage = self.optimizer.get_optimal_leverage(pair, confidence, direction)
            params['leverage'] = leverage
            
            # Get optimal entry price
            entry_price = self.optimizer.calculate_optimal_entry_price(pair, current_price, direction, confidence)
            params['entry_price'] = entry_price
            
            # Get optimal stop loss and take profit levels
            stop_loss = self.optimizer.get_optimal_stop_loss(pair, direction, leverage)
            take_profit = self.optimizer.get_optimal_take_profit(pair, direction)
            
            params['stop_loss_pct'] = stop_loss
            params['take_profit_pct'] = take_profit
            
            # Calculate liquidation price
            if direction == 'long':
                liquidation_price = entry_price * (1 - (95.0 / leverage) / 100)
            else:
                liquidation_price = entry_price * (1 + (95.0 / leverage) / 100)
            
            params['liquidation_price'] = liquidation_price
            
            # Get trailing stop parameters
            activation, distance = self.optimizer.get_trailing_stop_params(pair)
            params['trailing_stop_activation'] = activation
            params['trailing_stop_distance'] = distance
        
        return {
            'action': action,
            'confidence': confidence,
            'direction': direction,
            'predictions': predictions,
            'parameters': params
        }
    
    def update_models_with_new_data(self, recent_data: Dict[str, pd.DataFrame]):
        """
        Update models with new market data
        
        Args:
            recent_data: Dictionary of DataFrames with recent data for each pair
        """
        for pair, data in recent_data.items():
            if pair not in self.model_ensembles:
                logger.warning(f"No models available for {pair}, skipping update")
                continue
            
            if data.empty:
                logger.warning(f"No data available for {pair}, skipping update")
                continue
            
            # Add technical features
            data = self._add_technical_features(data)
            
            if data.empty:
                logger.warning(f"Failed to add technical features for {pair}, skipping update")
                continue
            
            # Update models for each task
            for task, ensemble in self.model_ensembles[pair].items():
                try:
                    logger.info(f"Updating models for {pair} ({task}) with new data")
                    ensemble.update_with_new_data(data, task)
                except Exception as e:
                    logger.error(f"Error updating models for {pair} ({task}): {e}")
    
    def save_ml_config(self):
        """Save ML configuration to file"""
        config = {
            'pairs': self.trading_pairs,
            'models': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Collect model information
        for pair in self.trading_pairs:
            if pair not in self.model_ensembles:
                continue
            
            config['models'][pair] = {}
            
            for task, ensemble in self.model_ensembles[pair].items():
                if not ensemble.is_trained:
                    continue
                
                # Get best model for each task
                best_arch, _ = ensemble.get_best_model()
                
                if best_arch is None:
                    continue
                
                config['models'][pair][task] = {
                    'best_model': best_arch,
                    'last_training': ensemble.last_training_time.isoformat() if ensemble.last_training_time else None,
                    'performance': ensemble.performance_metrics.get(best_arch, {})
                }
        
        # Save config
        with open(ML_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ML configuration to {ML_CONFIG_FILE}")

class ContinuousLearningManager:
    """
    Manager for continuous learning and model improvement
    """
    def __init__(self, trader: AdvancedMLTrader, training_interval: int = 24 * 60 * 60):
        """
        Initialize continuous learning manager
        
        Args:
            trader: AdvancedMLTrader instance
            training_interval: Interval between full retraining in seconds (default: 24 hours)
        """
        self.trader = trader
        self.training_interval = training_interval
        self.last_full_training = None
        self.last_update = None
        self.recent_data = {}
        
        logger.info("Initialized Continuous Learning Manager")
    
    def fetch_historical_data(self, pair: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch historical data for a pair
        
        Args:
            pair: Trading pair
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Create Kraken API client
            client = kraken.KrakenAPIClient()
            
            # Convert pair to Kraken format
            kraken_pair = pair.replace("/", "")
            
            # Calculate start and end time
            end_time = int(time.time())
            start_time = end_time - (days * 24 * 60 * 60)
            
            # Get OHLC data from Kraken API
            ohlc_data = client._request(
                kraken.PUBLIC_ENDPOINTS["ohlc"],
                {
                    "pair": kraken_pair,
                    "interval": 60,  # 1-hour candles
                    "since": start_time
                }
            )
            
            # Find the key corresponding to the pair data
            pair_key = None
            for key in ohlc_data.keys():
                if key != "last":
                    pair_key = key
                    break
            
            if not pair_key:
                logger.warning(f"No data found for {pair}")
                return pd.DataFrame()
            
            # Extract OHLC data
            candles = ohlc_data[pair_key]
            if not candles:
                logger.warning(f"No candles found for {pair}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                ]
            )
            
            # Convert types
            df['timestamp'] = df['timestamp'].astype(float)
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                df[col] = df[col].astype(float)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Sort index
            df.sort_index(inplace=True)
            
            # Save to disk
            pair_file = pair.replace('/', '_')
            df.to_csv(f"{HISTORICAL_DIR}/{pair_file}_60m.csv")
            
            logger.info(f"Fetched {len(df)} historical candles for {pair}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {pair}: {e}")
            return pd.DataFrame()
    
    def fetch_recent_data(self, pair: str, hours: int = 24) -> pd.DataFrame:
        """
        Fetch recent data for a pair
        
        Args:
            pair: Trading pair
            hours: Number of hours of recent data
            
        Returns:
            DataFrame with recent data
        """
        try:
            # Create Kraken API client
            client = kraken.KrakenAPIClient()
            
            # Convert pair to Kraken format
            kraken_pair = pair.replace("/", "")
            
            # Calculate start time
            end_time = int(time.time())
            start_time = end_time - (hours * 60 * 60)
            
            # Get OHLC data from Kraken API
            ohlc_data = client._request(
                kraken.PUBLIC_ENDPOINTS["ohlc"],
                {
                    "pair": kraken_pair,
                    "interval": 60,  # 1-hour candles
                    "since": start_time
                }
            )
            
            # Find the key corresponding to the pair data
            pair_key = None
            for key in ohlc_data.keys():
                if key != "last":
                    pair_key = key
                    break
            
            if not pair_key:
                logger.warning(f"No data found for {pair}")
                return pd.DataFrame()
            
            # Extract OHLC data
            candles = ohlc_data[pair_key]
            if not candles:
                logger.warning(f"No candles found for {pair}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                ]
            )
            
            # Convert types
            df['timestamp'] = df['timestamp'].astype(float)
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
                df[col] = df[col].astype(float)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Sort index
            df.sort_index(inplace=True)
            
            logger.info(f"Fetched {len(df)} recent candles for {pair}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching recent data for {pair}: {e}")
            return pd.DataFrame()
    
    def run_full_training(self):
        """Run full training on all models"""
        logger.info("Starting full model training")
        
        # Fetch historical data for all pairs
        for pair in self.trader.trading_pairs:
            try:
                logger.info(f"Fetching historical data for {pair}")
                df = self.fetch_historical_data(pair)
                
                if not df.empty:
                    self.recent_data[pair] = df
            except Exception as e:
                logger.error(f"Error fetching historical data for {pair}: {e}")
        
        # Train all models
        self.trader.train_all_models(force_retrain=True)
        
        # Save ML config
        self.trader.save_ml_config()
        
        # Update timestamp
        self.last_full_training = datetime.now()
        logger.info("Full model training complete")
    
    def update_models(self):
        """Update models with recent data"""
        logger.info("Updating models with recent data")
        
        # Fetch recent data for all pairs
        recent_data = {}
        
        for pair in self.trader.trading_pairs:
            try:
                logger.info(f"Fetching recent data for {pair}")
                df = self.fetch_recent_data(pair)
                
                if not df.empty:
                    recent_data[pair] = df
            except Exception as e:
                logger.error(f"Error fetching recent data for {pair}: {e}")
        
        # Update models
        self.trader.update_models_with_new_data(recent_data)
        
        # Save ML config
        self.trader.save_ml_config()
        
        # Update timestamp
        self.last_update = datetime.now()
        logger.info("Model update complete")
    
    def check_training_schedule(self):
        """Check if models need to be trained or updated"""
        now = datetime.now()
        
        # Check if full training is needed
        if (self.last_full_training is None or 
                (now - self.last_full_training).total_seconds() >= self.training_interval):
            logger.info("Full training scheduled")
            self.run_full_training()
        
        # Check if update is needed
        elif (self.last_update is None or 
               (now - self.last_update).total_seconds() >= self.training_interval / 8):
            logger.info("Model update scheduled")
            self.update_models()

def initialize_ml_system():
    """Initialize the ML system"""
    logger.info("Initializing ML system")
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HISTORICAL_DIR, exist_ok=True)
    
    # Create Advanced ML Trader
    trader = AdvancedMLTrader()
    
    # Create Continuous Learning Manager
    learning_manager = ContinuousLearningManager(trader)
    
    return trader, learning_manager

def main():
    """Main function"""
    logger.info("Starting Advanced ML Integration")
    
    # Initialize ML system
    trader, learning_manager = initialize_ml_system()
    
    # Run initial training
    learning_manager.run_full_training()
    
    # Start continuous learning and training
    try:
        while True:
            # Check training schedule
            learning_manager.check_training_schedule()
            
            # Sleep for a while
            logger.info("Sleeping for 10 minutes")
            time.sleep(600)  # 10 minutes
    
    except KeyboardInterrupt:
        logger.info("Stopping due to user interrupt")
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    
    finally:
        logger.info("Advanced ML Integration stopped")

if __name__ == "__main__":
    main()