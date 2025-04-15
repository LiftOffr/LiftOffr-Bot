#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization

This script performs sophisticated hyperparameter optimization for ML models
using Bayesian optimization, genetic algorithms, and other techniques to
achieve optimal performance metrics including accuracy, profit factor,
and robustness under different market conditions.

Features:
1. Multi-objective optimization
2. Cross-validation with time series splits
3. Custom evaluation metrics
4. Parallelized optimization
5. Transfer learning from prior optimizations
6. Adaptive parameter spaces
7. Warmstart from previous results
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
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("hyperparameter_optimization")

# Constants
CONFIG_PATH = "config/new_coins_training_config.json"
OUTPUT_DIR = "optimization_results"
TRAINING_DATA_DIR = "training_data"
MODEL_WEIGHTS_DIR = "model_weights"

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_WEIGHTS_DIR]:
    Path(directory).mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced hyperparameter optimization for trading models.')
    parser.add_argument('--pair', type=str, required=True,
                      help='Trading pair to optimize for (e.g., "AVAX/USD")')
    parser.add_argument('--model', type=str, required=True,
                      help='Model type (tcn, lstm, attention_gru, transformer, xgboost, lightgbm)')
    parser.add_argument('--trials', type=int, default=100,
                      help='Number of optimization trials')
    parser.add_argument('--objective', type=str, default='val_accuracy',
                      help='Optimization objective (val_accuracy, profit_factor, custom_metric)')
    parser.add_argument('--multi-objective', action='store_true',
                      help='Enable multi-objective optimization')
    parser.add_argument('--cv-folds', type=int, default=5,
                      help='Number of cross-validation folds')
    parser.add_argument('--time-series-split', action='store_true',
                      help='Use time series cross-validation')
    parser.add_argument('--parallel', action='store_true',
                      help='Run optimization in parallel')
    parser.add_argument('--n-jobs', type=int, default=4,
                      help='Number of parallel jobs')
    parser.add_argument('--transfer-learning', action='store_true',
                      help='Use transfer learning from other pairs')
    parser.add_argument('--warm-start', action='store_true',
                      help='Warm start from previous optimization')
    parser.add_argument('--early-stopping', action='store_true',
                      help='Enable early stopping for trials')
    
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


def load_data(pair: str, timeframe: str = '1h') -> Tuple[np.ndarray, np.ndarray]:
    """Load training data for a specific pair and timeframe."""
    pair_filename = pair.replace("/", "_")
    file_path = f"{TRAINING_DATA_DIR}/{pair_filename}_{timeframe}_features.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Training data file not found: {file_path}")
            return None, None
        
        df = pd.read_csv(file_path)
        
        # Extract features and target
        features = df.drop(['timestamp', 'target'], axis=1, errors='ignore')
        if 'target' in df.columns:
            target = df['target']
        else:
            logger.error(f"Target column not found in {file_path}")
            return None, None
        
        logger.info(f"Loaded training data for {pair} with {len(features)} rows and {features.shape[1]} features")
        
        return features.values, target.values
    
    except Exception as e:
        logger.error(f"Error loading data for {pair}: {e}")
        return None, None


def create_sequences(features: np.ndarray, target: np.ndarray, 
                     lookback_window: int, sequence_model: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences for time series models.
    
    Args:
        features: Feature matrix
        target: Target vector
        lookback_window: Number of lookback periods
        sequence_model: Whether the model expects sequence input
        
    Returns:
        X: Sequence input features
        y: Target values
    """
    if not sequence_model:
        return features, target
    
    X, y = [], []
    
    for i in range(len(features) - lookback_window):
        X.append(features[i:i+lookback_window])
        y.append(target[i+lookback_window])
    
    return np.array(X), np.array(y)


def get_parameter_space(model_type: str, config: Dict) -> Dict:
    """Get hyperparameter space for the specified model type."""
    try:
        param_config = config.get("hyperparameter_optimization", {}).get("parameters", {})
        
        if model_type in param_config:
            return param_config[model_type]
        else:
            logger.warning(f"Parameter space not found for model type: {model_type}")
            return {}
    
    except Exception as e:
        logger.error(f"Error getting parameter space: {e}")
        return {}


def custom_metric(y_true: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray = None) -> float:
    """
    Custom evaluation metric that balances accuracy with profitability metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        X_test: Test features (optional, for additional context)
        
    Returns:
        score: Balanced score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Calculate trading-specific metrics
    # True positives (correctly predicted uptrends)
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    
    # False positives (incorrectly predicted uptrends)
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    
    # True negatives (correctly predicted downtrends)
    tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
    
    # False negatives (incorrectly predicted downtrends)
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
    
    # Win rate (assuming we trade every signal)
    win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Balance of trades (ratio of long to short)
    balance = (tp + fp) / (tn + fn) if (tn + fn) > 0 else 0
    target_balance = 1.0  # We want balanced trading
    balance_penalty = max(0, 1 - abs(balance - target_balance))
    
    # Confidence calibration (higher confidence should correlate with correctness)
    high_conf_correct = ((y_pred > 0.7) & (y_pred_binary == y_true)).sum()
    high_conf_total = (y_pred > 0.7).sum()
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    
    # Combine metrics with weights
    score = (
        0.35 * accuracy +
        0.20 * precision +
        0.15 * recall +
        0.10 * f1 +
        0.10 * win_rate +
        0.05 * balance_penalty +
        0.05 * high_conf_accuracy
    )
    
    return score


def optimize_tcn_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize TCN model hyperparameters using Bayesian optimization.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Try to import TCN layer
        try:
            from tcn import TCN
        except ImportError:
            try:
                from tensorflow.keras.layers import TCN
            except ImportError:
                logger.error("TCN layer not available. Please install keras-tcn")
                return {}
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            num_filters = trial.suggest_categorical('num_filters', param_space.get('num_filters', [64]))
            kernel_size = trial.suggest_categorical('kernel_size', param_space.get('kernel_size', [3]))
            nb_stacks = trial.suggest_categorical('nb_stacks', param_space.get('nb_stacks', [2]))
            
            # Handle different parameter types
            if isinstance(param_space.get('dilations', [1, 2, 4, 8, 16, 32]), list) and all(isinstance(x, list) for x in param_space.get('dilations', [[1, 2, 4, 8, 16, 32]])):
                dilations = trial.suggest_categorical('dilations', param_space.get('dilations', [[1, 2, 4, 8, 16, 32]]))
            else:
                dilations = param_space.get('dilations', [1, 2, 4, 8, 16, 32])
            
            activation = trial.suggest_categorical('activation', param_space.get('activation', ['relu']))
            dropout_rate = trial.suggest_categorical('dropout_rate', param_space.get('dropout_rate', [0.2]))
            use_skip_connections = trial.suggest_categorical('use_skip_connections', param_space.get('use_skip_connections', [True]))
            use_batch_norm = trial.suggest_categorical('use_batch_norm', param_space.get('use_batch_norm', [True]))
            use_layer_norm = trial.suggest_categorical('use_layer_norm', param_space.get('use_layer_norm', [False]))
            use_weight_norm = trial.suggest_categorical('use_weight_norm', param_space.get('use_weight_norm', [True]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [1e-3]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                input_shape = X_train.shape[1:]
                inputs = Input(shape=input_shape)
                
                x = TCN(
                    nb_filters=num_filters,
                    kernel_size=kernel_size,
                    nb_stacks=nb_stacks,
                    dilations=dilations,
                    padding='causal',
                    use_skip_connections=use_skip_connections,
                    dropout_rate=dropout_rate,
                    return_sequences=False,
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
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                
                # Early stopping callback
                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    restore_best_weights=True
                )
                
                # Train model
                batch_size = 32
                epochs = 50 if args.early_stopping else 5  # Use fewer epochs for optimization
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping] if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_val)
                
                if args.objective == 'val_accuracy':
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred)
                else:
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Default to accuracy
                
                cv_scores.append(score)
                
                # Clean up
                tf.keras.backend.clear_session()
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"tcn_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best TCN parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/tcn_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in TCN optimization: {e}")
        return {}


def optimize_lstm_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize LSTM model hyperparameters using Bayesian optimization.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            units = trial.suggest_categorical('units', param_space.get('units', [[128, 64]]))
            dropout = trial.suggest_categorical('dropout', param_space.get('dropout', [0.2]))
            recurrent_dropout = trial.suggest_categorical('recurrent_dropout', param_space.get('recurrent_dropout', [0.2]))
            activation = trial.suggest_categorical('activation', param_space.get('activation', ['tanh']))
            recurrent_activation = trial.suggest_categorical('recurrent_activation', param_space.get('recurrent_activation', ['sigmoid']))
            bidirectional = trial.suggest_categorical('bidirectional', param_space.get('bidirectional', [True]))
            use_batch_norm = trial.suggest_categorical('use_batch_norm', param_space.get('use_batch_norm', [True]))
            attention_mechanism = trial.suggest_categorical('attention_mechanism', param_space.get('attention_mechanism', [True]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [1e-3]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                input_shape = X_train.shape[1:]
                inputs = Input(shape=input_shape)
                
                x = inputs
                
                # LSTM layers
                for i, unit in enumerate(units):
                    return_sequences = i < len(units) - 1
                    
                    if bidirectional:
                        lstm_layer = Bidirectional(LSTM(
                            units=unit,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=True,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            return_sequences=return_sequences
                        ))(x)
                    else:
                        lstm_layer = LSTM(
                            units=unit,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=True,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            return_sequences=return_sequences
                        )(x)
                    
                    x = lstm_layer
                    
                    if use_batch_norm:
                        x = BatchNormalization()(x)
                
                x = Dropout(dropout)(x)
                
                # Add attention mechanism if specified and appropriate
                if attention_mechanism and len(units) > 1:
                    # Simplified attention mechanism
                    from tensorflow.keras.layers import Attention, Lambda, Multiply
                    
                    attention_units = units[-1] * (2 if bidirectional else 1)
                    
                    attention_layer = Dense(attention_units, activation="tanh")(x)
                    attention_weights = Dense(1, activation="softmax")(attention_layer)
                    context_vector = Multiply()([x, attention_weights])
                    x = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
                
                x = Dense(64, activation="relu")(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = Dropout(dropout)(x)
                x = Dense(32, activation="relu")(x)
                outputs = Dense(1, activation="sigmoid")(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                
                # Compile model
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                
                # Early stopping callback
                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    restore_best_weights=True
                )
                
                # Train model
                batch_size = 32
                epochs = 50 if args.early_stopping else 5  # Use fewer epochs for optimization
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping] if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_val)
                
                if args.objective == 'val_accuracy':
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred)
                else:
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Default to accuracy
                
                cv_scores.append(score)
                
                # Clean up
                tf.keras.backend.clear_session()
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"lstm_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best LSTM parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/lstm_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in LSTM optimization: {e}")
        return {}


def optimize_attention_gru_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize Attention GRU model hyperparameters.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Bidirectional, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            gru_units = trial.suggest_categorical('gru_units', param_space.get('gru_units', [[96, 64]]))
            attention_units = trial.suggest_categorical('attention_units', param_space.get('attention_units', [64]))
            dropout_rate = trial.suggest_categorical('dropout_rate', param_space.get('dropout_rate', [0.2]))
            activation = trial.suggest_categorical('activation', param_space.get('activation', ['tanh']))
            recurrent_activation = trial.suggest_categorical('recurrent_activation', param_space.get('recurrent_activation', ['sigmoid']))
            attention_activation = trial.suggest_categorical('attention_activation', param_space.get('attention_activation', ['tanh']))
            bidirectional = trial.suggest_categorical('bidirectional', param_space.get('bidirectional', [True]))
            use_batch_norm = trial.suggest_categorical('use_batch_norm', param_space.get('use_batch_norm', [True]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [1e-3]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                input_shape = X_train.shape[1:]
                inputs = Input(shape=input_shape)
                
                x = inputs
                
                # GRU layers
                for i, unit in enumerate(gru_units):
                    return_sequences = i < len(gru_units) - 1
                    
                    if bidirectional:
                        gru_layer = Bidirectional(GRU(
                            units=unit,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=True,
                            dropout=dropout_rate,
                            recurrent_dropout=dropout_rate,
                            return_sequences=return_sequences
                        ))(x)
                    else:
                        gru_layer = GRU(
                            units=unit,
                            activation=activation,
                            recurrent_activation=recurrent_activation,
                            use_bias=True,
                            dropout=dropout_rate,
                            recurrent_dropout=dropout_rate,
                            return_sequences=return_sequences
                        )(x)
                    
                    x = gru_layer
                    
                    if use_batch_norm:
                        x = BatchNormalization()(x)
                
                x = Dropout(dropout_rate)(x)
                
                # Add attention mechanism
                from tensorflow.keras.layers import Lambda, Multiply
                
                # Calculate attention weights
                attention_layer = Dense(attention_units, activation=attention_activation)(x)
                attention_weights = Dense(1, activation="softmax")(attention_layer)
                
                # Apply attention weights to GRU output
                context_vector = Multiply()([x, attention_weights])
                x = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
                
                x = Dense(64, activation="relu")(x)
                
                if use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = Dropout(dropout_rate)(x)
                x = Dense(32, activation="relu")(x)
                outputs = Dense(1, activation="sigmoid")(x)
                
                model = Model(inputs=inputs, outputs=outputs)
                
                # Compile model
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                
                # Early stopping callback
                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    restore_best_weights=True
                )
                
                # Train model
                batch_size = 32
                epochs = 50 if args.early_stopping else 5  # Use fewer epochs for optimization
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping] if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_val)
                
                if args.objective == 'val_accuracy':
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred)
                else:
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Default to accuracy
                
                cv_scores.append(score)
                
                # Clean up
                tf.keras.backend.clear_session()
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"attention_gru_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best Attention GRU parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/attention_gru_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in Attention GRU optimization: {e}")
        return {}


def optimize_transformer_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize Transformer model hyperparameters.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Define a simplified Transformer encoder
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head Self Attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            attention_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = attention_output + inputs
            
            # Feed Forward
            ff_output = LayerNormalization(epsilon=1e-6)(x)
            ff_output = Dense(ff_dim, activation="relu")(ff_output)
            ff_output = Dropout(dropout)(ff_output)
            ff_output = Dense(inputs.shape[-1])(ff_output)
            return x + ff_output
        
        # Import or create MultiHeadAttention layer
        try:
            from tensorflow.keras.layers import MultiHeadAttention
        except ImportError:
            # Define a simplified MultiHeadAttention layer for older TF versions
            from tensorflow.keras.layers import Layer
            
            class MultiHeadAttention(Layer):
                def __init__(self, key_dim, num_heads, dropout=0):
                    super().__init__()
                    self.key_dim = key_dim
                    self.num_heads = num_heads
                    self.dropout = dropout
                
                def build(self, input_shape):
                    self.query_dense = Dense(self.key_dim * self.num_heads)
                    self.key_dense = Dense(self.key_dim * self.num_heads)
                    self.value_dense = Dense(self.key_dim * self.num_heads)
                    self.combine_heads = Dense(input_shape[-1])
                
                def attention(self, query, key, value):
                    score = tf.matmul(query, key, transpose_b=True)
                    dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
                    scaled_score = score / tf.math.sqrt(dim_key)
                    weights = tf.nn.softmax(scaled_score, axis=-1)
                    output = tf.matmul(weights, value)
                    return output
                
                def separate_heads(self, x, batch_size):
                    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
                    return tf.transpose(x, perm=[0, 2, 1, 3])
                
                def call(self, inputs, mask=None):
                    batch_size = tf.shape(inputs)[0]
                    
                    query = self.query_dense(inputs)
                    key = self.key_dense(inputs)
                    value = self.value_dense(inputs)
                    
                    query = self.separate_heads(query, batch_size)
                    key = self.separate_heads(key, batch_size)
                    value = self.separate_heads(value, batch_size)
                    
                    attention = self.attention(query, key, value)
                    attention = tf.transpose(attention, perm=[0, 2, 1, 3])
                    concat_attention = tf.reshape(attention, (batch_size, -1, self.key_dim * self.num_heads))
                    output = self.combine_heads(concat_attention)
                    return output
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            num_layers = trial.suggest_categorical('num_layers', param_space.get('num_layers', [3]))
            d_model = trial.suggest_categorical('d_model', param_space.get('d_model', [128]))
            num_heads = trial.suggest_categorical('num_heads', param_space.get('num_heads', [8]))
            dff = trial.suggest_categorical('dff', param_space.get('dff', [512]))
            dropout_rate = trial.suggest_categorical('dropout_rate', param_space.get('dropout_rate', [0.2]))
            use_batch_norm = trial.suggest_categorical('use_batch_norm', param_space.get('use_batch_norm', [True]))
            layer_norm = trial.suggest_categorical('layer_norm', param_space.get('layer_norm', [True]))
            positional_encoding = trial.suggest_categorical('positional_encoding', param_space.get('positional_encoding', [True]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [1e-3]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                input_shape = X_train.shape[1:]
                inputs = Input(shape=input_shape)
                
                # Add positional encoding if specified
                x = inputs
                if positional_encoding:
                    x = add_positional_encoding(x, d_model)
                
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
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(
                    optimizer=optimizer,
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )
                
                # Early stopping callback
                early_stopping = EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    restore_best_weights=True
                )
                
                # Train model
                batch_size = 32
                epochs = 50 if args.early_stopping else 5  # Use fewer epochs for optimization
                
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping] if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred = model.predict(X_val)
                
                if args.objective == 'val_accuracy':
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Accuracy
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred)
                else:
                    score = model.evaluate(X_val, y_val, verbose=0)[1]  # Default to accuracy
                
                cv_scores.append(score)
                
                # Clean up
                tf.keras.backend.clear_session()
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"transformer_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best Transformer parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/transformer_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in Transformer optimization: {e}")
        return {}


def optimize_xgboost_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize XGBoost model hyperparameters.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import xgboost as xgb
        from sklearn.metrics import accuracy_score
        
        # Reshape input if necessary (from 3D to 2D)
        if len(X.shape) == 3:
            # Flatten the sequence dimension for tree-based models
            X = X.reshape(X.shape[0], -1)
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            n_estimators = trial.suggest_categorical('n_estimators', param_space.get('n_estimators', [100]))
            max_depth = trial.suggest_categorical('max_depth', param_space.get('max_depth', [3]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [0.1]))
            gamma = trial.suggest_categorical('gamma', param_space.get('gamma', [0]))
            subsample = trial.suggest_categorical('subsample', param_space.get('subsample', [0.8]))
            colsample_bytree = trial.suggest_categorical('colsample_bytree', param_space.get('colsample_bytree', [0.8]))
            min_child_weight = trial.suggest_categorical('min_child_weight', param_space.get('min_child_weight', [1]))
            scale_pos_weight = trial.suggest_categorical('scale_pos_weight', param_space.get('scale_pos_weight', [1]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
                
                # Train model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10 if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                if args.objective == 'val_accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred_proba)
                else:
                    score = accuracy_score(y_val, y_pred)
                
                cv_scores.append(score)
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"xgboost_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best XGBoost parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/xgboost_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in XGBoost optimization: {e}")
        return {}


def optimize_lightgbm_model(X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """
    Optimize LightGBM model hyperparameters.
    
    Args:
        X: Input features
        y: Target labels
        param_space: Parameter space definition
        args: Command line arguments
        
    Returns:
        best_params: Best hyperparameters
    """
    try:
        import optuna
        from sklearn.model_selection import KFold, TimeSeriesSplit
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score
        
        # Reshape input if necessary (from 3D to 2D)
        if len(X.shape) == 3:
            # Flatten the sequence dimension for tree-based models
            X = X.reshape(X.shape[0], -1)
        
        # Set up cross-validation
        if args.time_series_split:
            cv = TimeSeriesSplit(n_splits=args.cv_folds)
        else:
            cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # Define the objective function
        def objective(trial):
            # Sample hyperparameters from the parameter space
            n_estimators = trial.suggest_categorical('n_estimators', param_space.get('n_estimators', [100]))
            num_leaves = trial.suggest_categorical('num_leaves', param_space.get('num_leaves', [31]))
            max_depth = trial.suggest_categorical('max_depth', param_space.get('max_depth', [3]))
            learning_rate = trial.suggest_categorical('learning_rate', param_space.get('learning_rate', [0.1]))
            subsample = trial.suggest_categorical('subsample', param_space.get('subsample', [0.8]))
            colsample_bytree = trial.suggest_categorical('colsample_bytree', param_space.get('colsample_bytree', [0.8]))
            min_child_samples = trial.suggest_categorical('min_child_samples', param_space.get('min_child_samples', [20]))
            reg_alpha = trial.suggest_categorical('reg_alpha', param_space.get('reg_alpha', [0]))
            reg_lambda = trial.suggest_categorical('reg_lambda', param_space.get('reg_lambda', [0]))
            bagging_freq = trial.suggest_categorical('bagging_freq', param_space.get('bagging_freq', [0]))
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                model = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_samples=min_child_samples,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    bagging_freq=bagging_freq,
                    random_state=42
                )
                
                # Train model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10 if args.early_stopping else None,
                    verbose=0
                )
                
                # Evaluate model
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                if args.objective == 'val_accuracy':
                    score = accuracy_score(y_val, y_pred)
                elif args.objective == 'custom_metric':
                    score = custom_metric(y_val, y_pred_proba)
                else:
                    score = accuracy_score(y_val, y_pred)
                
                cv_scores.append(score)
            
            # Return mean score across CV folds
            return np.mean(cv_scores)
        
        # Create study for hyperparameter optimization
        study_name = f"lightgbm_{args.pair.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        if args.parallel:
            n_jobs = min(args.n_jobs, 4)  # Limit to reasonable number
            study.optimize(objective, n_trials=args.trials, n_jobs=n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best LightGBM parameters found: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Save optimization results
        results_file = f"{OUTPUT_DIR}/lightgbm_{args.pair.replace('/', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": best_value,
                "optimization_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "n_trials": args.trials,
                "objective": args.objective
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
        
        return best_params
    
    except Exception as e:
        logger.error(f"Error in LightGBM optimization: {e}")
        return {}


def optimize_model(model_type: str, X: np.ndarray, y: np.ndarray, param_space: Dict, args) -> Dict:
    """Optimize model hyperparameters based on model type."""
    logger.info(f"Optimizing {model_type} model for {args.pair}")
    
    if model_type == 'tcn':
        return optimize_tcn_model(X, y, param_space, args)
    elif model_type == 'lstm':
        return optimize_lstm_model(X, y, param_space, args)
    elif model_type == 'attention_gru':
        return optimize_attention_gru_model(X, y, param_space, args)
    elif model_type == 'transformer':
        return optimize_transformer_model(X, y, param_space, args)
    elif model_type == 'xgboost':
        return optimize_xgboost_model(X, y, param_space, args)
    elif model_type == 'lightgbm':
        return optimize_lightgbm_model(X, y, param_space, args)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return {}


def update_config_with_best_params(config: Dict, model_type: str, best_params: Dict, pair: str) -> Dict:
    """Update configuration with best parameters found during optimization."""
    if not best_params:
        logger.warning(f"No parameters to update for {model_type}")
        return config
    
    logger.info(f"Updating configuration with best parameters for {model_type}")
    
    try:
        # Make a copy of the config to avoid modifying the original
        updated_config = config.copy()
        
        # Update model-specific configuration
        model_config = updated_config.get("training_config", {}).get(model_type, {})
        for param, value in best_params.items():
            model_config[param] = value
        
        updated_config["training_config"][model_type] = model_config
        
        # Save updated configuration
        output_file = f"{OUTPUT_DIR}/{model_type}_{pair.replace('/', '_')}_config.json"
        with open(output_file, 'w') as f:
            json.dump(updated_config, f, indent=2)
        
        logger.info(f"Updated configuration saved to {output_file}")
        
        return updated_config
    
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return config


def add_positional_encoding(inputs, d_model):
    """Add positional encoding to input tensor."""
    import tensorflow as tf
    
    # Get input dimensions
    batch_size = tf.shape(inputs)[0]
    seq_length = tf.shape(inputs)[1]
    
    # Create position indices
    position_indices = tf.range(seq_length, dtype=tf.float32)[tf.newaxis, :]
    
    # Create sine and cosine values for different dimensions
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * 
        (-tf.math.log(10000.0) / d_model)
    )
    
    # Calculate sine and cosine position encodings
    pe_sin = tf.sin(position_indices * div_term)
    pe_cos = tf.cos(position_indices * div_term)
    
    # Interleave sine and cosine values
    pe = tf.stack([pe_sin, pe_cos], axis=-1)
    pe = tf.reshape(pe, [1, seq_length, d_model])
    
    # Repeat for batch dimension
    pe = tf.tile(pe, [batch_size, 1, 1])
    
    return inputs + pe


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Starting hyperparameter optimization for {args.pair} {args.model_type}")
    logger.info(f"Optimization objective: {args.objective}")
    logger.info(f"Number of trials: {args.trials}")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return 1
    
    # Get pair-specific settings
    pair_settings = config.get("pair_specific_settings", {}).get(args.pair, {})
    lookback_window = pair_settings.get("lookback_window", 120)
    
    # Load data
    X, y = load_data(args.pair, timeframe='1h')
    if X is None or y is None:
        logger.error("Failed to load data. Exiting.")
        return 1
    
    # Determine if model requires sequence input
    sequence_model = args.model_type in ['tcn', 'lstm', 'attention_gru', 'transformer']
    
    # Create sequences for time series models
    if sequence_model:
        X, y = create_sequences(X, y, lookback_window, sequence_model)
        logger.info(f"Created sequences with shape: {X.shape}")
    
    # Get parameter space for the specified model type
    param_space = get_parameter_space(args.model_type, config)
    if not param_space:
        logger.warning(f"Using default parameter space for {args.model_type}")
        param_space = {}
    
    # Run optimization
    best_params = optimize_model(args.model_type, X, y, param_space, args)
    
    # Update configuration with best parameters
    if best_params:
        update_config_with_best_params(config, args.model_type, best_params, args.pair)
    
    logger.info("Hyperparameter optimization completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())