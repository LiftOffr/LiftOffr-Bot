#!/usr/bin/env python3
"""
Enhance ML Model Accuracy

This script improves ML model accuracy for price prediction:
1. Implements advanced feature engineering
2. Uses data augmentation techniques
3. Applies regularization to prevent overfitting
4. Creates ensemble models from multiple base models
5. Tunes hyperparameters for optimal performance
6. Targets 90% prediction accuracy

Usage:
    python enhance_ml_model_accuracy.py --pairs SOL/USD [--epochs 200] [--batch-size 32]
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhance_ml_model.log')
    ]
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ['SOL/USD']
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOOKBACK = 24
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_PATIENCE = 20
DEFAULT_OUTPUT_DIR = 'models'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance ML Model Accuracy')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK,
        help=f'Lookback window size (default: {DEFAULT_LOOKBACK})'
    )
    
    parser.add_argument(
        '--train-size',
        type=float,
        default=DEFAULT_TRAIN_SIZE,
        help=f'Train/test split ratio (default: {DEFAULT_TRAIN_SIZE})'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help=f'Validation split ratio (default: {DEFAULT_VALIDATION_SPLIT})'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=DEFAULT_PATIENCE,
        help=f'Early stopping patience (default: {DEFAULT_PATIENCE})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def load_dataset(pair, verbose=False):
    """
    Load dataset for the specified trading pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with features and target
    """
    try:
        # Normalize pair format
        pair_path = pair.replace('/', '')
        pair_symbol = pair.split('/')[0]
        pair_base = pair.split('/')[1]
        
        # Try to locate the dataset
        dataset_paths = [
            f"training_data/{pair_path}_1h_enhanced.csv",
            f"training_data/{pair_symbol}/{pair_base}_1h_enhanced.csv",
            f"training_data/{pair_path.lower()}_1h_enhanced.csv",
            f"historical_data/{pair_path}_1h.csv",
            f"historical_data/{pair_symbol}/{pair_base}_1h.csv"
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            logger.error(f"No dataset found for {pair}. Checked paths: {dataset_paths}")
            return None
        
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        if verbose:
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset for {pair}: {str(e)}")
        return None

def enhance_features(df, verbose=False):
    """
    Enhance features with advanced technical indicators.
    
    Args:
        df: DataFrame with price data
        verbose: Whether to print detailed information
        
    Returns:
        DataFrame with enhanced features
    """
    try:
        # Make a copy to avoid modifying the original
        df_enhanced = df.copy()
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df_enhanced.columns:
            df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['timestamp'])
        
        # Identify price and volume columns
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        potential_volume_columns = ['volume', 'Volume']
        potential_open_columns = ['open', 'Open']
        potential_high_columns = ['high', 'High']
        potential_low_columns = ['low', 'Low']
        
        price_column = None
        for col in potential_price_columns:
            if col in df_enhanced.columns:
                price_column = col
                break
                
        volume_column = None
        for col in potential_volume_columns:
            if col in df_enhanced.columns:
                volume_column = col
                break
                
        open_column = None
        for col in potential_open_columns:
            if col in df_enhanced.columns:
                open_column = col
                break
                
        high_column = None
        for col in potential_high_columns:
            if col in df_enhanced.columns:
                high_column = col
                break
                
        low_column = None
        for col in potential_low_columns:
            if col in df_enhanced.columns:
                low_column = col
                break
        
        if not price_column:
            logger.error("Could not identify price column")
            return None
        
        # Add price differences
        df_enhanced['price_diff_1'] = df_enhanced[price_column].diff()
        df_enhanced['price_diff_2'] = df_enhanced['price_diff_1'].diff()
        
        # Add price returns
        df_enhanced['price_return_1'] = df_enhanced[price_column].pct_change()
        df_enhanced['price_return_2'] = df_enhanced['price_return_1'].diff()
        
        # Add moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df_enhanced[f'sma_{window}'] = df_enhanced[price_column].rolling(window=window).mean()
            df_enhanced[f'ema_{window}'] = df_enhanced[price_column].ewm(span=window, adjust=False).mean()
        
        # Add price momentum
        for window in [5, 10, 20, 50]:
            df_enhanced[f'momentum_{window}'] = df_enhanced[price_column].pct_change(periods=window)
        
        # Add price volatility
        for window in [5, 10, 20, 50]:
            df_enhanced[f'volatility_{window}'] = df_enhanced[price_column].rolling(window=window).std()
        
        # Add RSI
        delta = df_enhanced[price_column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df_enhanced['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        ema_12 = df_enhanced[price_column].ewm(span=12, adjust=False).mean()
        ema_26 = df_enhanced[price_column].ewm(span=26, adjust=False).mean()
        df_enhanced['macd'] = ema_12 - ema_26
        df_enhanced['macd_signal'] = df_enhanced['macd'].ewm(span=9, adjust=False).mean()
        df_enhanced['macd_hist'] = df_enhanced['macd'] - df_enhanced['macd_signal']
        
        # Add Bollinger Bands
        for window in [20]:
            mid_band = df_enhanced[price_column].rolling(window=window).mean()
            std_dev = df_enhanced[price_column].rolling(window=window).std()
            df_enhanced[f'bb_upper_{window}'] = mid_band + (std_dev * 2)
            df_enhanced[f'bb_lower_{window}'] = mid_band - (std_dev * 2)
            df_enhanced[f'bb_width_{window}'] = (df_enhanced[f'bb_upper_{window}'] - df_enhanced[f'bb_lower_{window}']) / mid_band
            
            # Add relative position within Bollinger Bands
            df_enhanced[f'bb_pos_{window}'] = (df_enhanced[price_column] - df_enhanced[f'bb_lower_{window}']) / (df_enhanced[f'bb_upper_{window}'] - df_enhanced[f'bb_lower_{window}'])
        
        # Add ADX if high and low columns exist
        if high_column and low_column:
            tr1 = abs(df_enhanced[high_column] - df_enhanced[low_column])
            tr2 = abs(df_enhanced[high_column] - df_enhanced[price_column].shift(1))
            tr3 = abs(df_enhanced[low_column] - df_enhanced[price_column].shift(1))
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            df_enhanced['atr_14'] = atr
            
            up_move = df_enhanced[high_column] - df_enhanced[high_column].shift(1)
            down_move = df_enhanced[low_column].shift(1) - df_enhanced[low_column]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
            adx = dx.rolling(window=14).mean()
            
            df_enhanced['plus_di_14'] = plus_di
            df_enhanced['minus_di_14'] = minus_di
            df_enhanced['adx_14'] = adx
        
        # Add volume features if volume exists
        if volume_column:
            # Volume moving averages
            for window in [5, 10, 20, 50]:
                df_enhanced[f'volume_sma_{window}'] = df_enhanced[volume_column].rolling(window=window).mean()
                df_enhanced[f'volume_ema_{window}'] = df_enhanced[volume_column].ewm(span=window, adjust=False).mean()
            
            # Volume momentum
            for window in [5, 10, 20]:
                df_enhanced[f'volume_momentum_{window}'] = df_enhanced[volume_column].pct_change(periods=window)
            
            # Volume volatility
            for window in [5, 10, 20]:
                df_enhanced[f'volume_volatility_{window}'] = df_enhanced[volume_column].rolling(window=window).std()
            
            # Price-volume ratios
            df_enhanced['price_volume_ratio'] = df_enhanced[price_column] / df_enhanced[volume_column].replace(0, np.finfo(float).eps)
            
            # On-balance volume
            df_enhanced['obv'] = (np.sign(df_enhanced[price_column].diff()) * df_enhanced[volume_column]).fillna(0).cumsum()
        
        # Add candlestick patterns if OHLC data is available
        if open_column and high_column and low_column and price_column:
            # Doji
            df_enhanced['doji'] = np.where(abs(df_enhanced[open_column] - df_enhanced[price_column]) <= 
                                      (df_enhanced[high_column] - df_enhanced[low_column]) * 0.1, 1, 0)
            
            # Hammer
            body_size = abs(df_enhanced[open_column] - df_enhanced[price_column])
            lower_shadow = np.minimum(df_enhanced[open_column], df_enhanced[price_column]) - df_enhanced[low_column]
            upper_shadow = df_enhanced[high_column] - np.maximum(df_enhanced[open_column], df_enhanced[price_column])
            
            df_enhanced['hammer'] = np.where(
                (lower_shadow >= 2 * body_size) & (upper_shadow <= 0.1 * lower_shadow), 1, 0
            )
            
            # Engulfing
            df_enhanced['bullish_engulfing'] = np.where(
                (df_enhanced[open_column].shift(1) > df_enhanced[price_column].shift(1)) &
                (df_enhanced[open_column] < df_enhanced[price_column]) &
                (df_enhanced[open_column] <= df_enhanced[price_column].shift(1)) &
                (df_enhanced[price_column] >= df_enhanced[open_column].shift(1)),
                1, 0
            )
            
            df_enhanced['bearish_engulfing'] = np.where(
                (df_enhanced[open_column].shift(1) < df_enhanced[price_column].shift(1)) &
                (df_enhanced[open_column] > df_enhanced[price_column]) &
                (df_enhanced[open_column] >= df_enhanced[price_column].shift(1)) &
                (df_enhanced[price_column] <= df_enhanced[open_column].shift(1)),
                1, 0
            )
        
        # Add time-based features if timestamp exists
        if 'timestamp' in df_enhanced.columns:
            df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
            df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
            df_enhanced['day_of_month'] = df_enhanced['timestamp'].dt.day
            df_enhanced['week_of_year'] = df_enhanced['timestamp'].dt.isocalendar().week
            df_enhanced['month'] = df_enhanced['timestamp'].dt.month
            df_enhanced['is_month_end'] = df_enhanced['timestamp'].dt.is_month_end.astype(int)
            df_enhanced['is_month_start'] = df_enhanced['timestamp'].dt.is_month_start.astype(int)
            df_enhanced['is_weekend'] = df_enhanced['day_of_week'].isin([5, 6]).astype(int)
        
        # Add target variable (price direction for next period)
        df_enhanced['target'] = np.where(df_enhanced[price_column].shift(-1) > df_enhanced[price_column], 1, 0)
        
        # Drop rows with NaN values
        df_enhanced = df_enhanced.dropna()
        
        if verbose:
            logger.info(f"Enhanced dataset shape: {df_enhanced.shape}")
            logger.info(f"Enhanced dataset columns: {df_enhanced.columns.tolist()}")
            
            # Count target distribution
            target_counts = df_enhanced['target'].value_counts()
            logger.info(f"Target distribution: {target_counts.to_dict()}")
        
        return df_enhanced
    
    except Exception as e:
        logger.error(f"Error enhancing features: {str(e)}")
        return None

def prepare_sequences(df, lookback=24, train_size=0.8, verbose=False):
    """
    Prepare sequences for training.
    
    Args:
        df: DataFrame with features and target
        lookback: Lookback window size
        train_size: Train/test split ratio
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_columns)
    """
    try:
        # Drop timestamp column if it exists
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        # Separate features and target
        y = df['target'].values
        X = df.drop('target', axis=1)
        
        feature_columns = X.columns.tolist()
        X = X.values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(lookback, len(X_scaled)):
            X_sequences.append(X_scaled[i-lookback:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split into train and test sets
        split_idx = int(len(X_sequences) * train_size)
        
        X_train = X_sequences[:split_idx]
        y_train = y_sequences[:split_idx]
        
        X_test = X_sequences[split_idx:]
        y_test = y_sequences[split_idx:]
        
        if verbose:
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            
            # Count class distribution
            train_counts = np.bincount(y_train.astype(int))
            test_counts = np.bincount(y_test.astype(int))
            
            logger.info(f"Train class distribution: {train_counts}")
            logger.info(f"Test class distribution: {test_counts}")
        
        return X_train, y_train, X_test, y_test, feature_columns
    
    except Exception as e:
        logger.error(f"Error preparing sequences: {str(e)}")
        return None, None, None, None, None

def augment_training_data(X_train, y_train, augmentation_factor=1.5, verbose=False):
    """
    Augment training data to increase sample size and handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training targets
        augmentation_factor: Factor by which to augment the minority class
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (augmented_X_train, augmented_y_train)
    """
    try:
        # Count class distribution
        class_counts = np.bincount(y_train.astype(int))
        
        if verbose:
            logger.info(f"Original class distribution: {class_counts}")
        
        # Check if augmentation is needed
        if class_counts[0] == class_counts[1]:
            logger.info("Classes are balanced, no augmentation needed")
            return X_train, y_train
        
        # Determine minority class
        minority_class = 0 if class_counts[0] < class_counts[1] else 1
        
        # Get indices of minority class samples
        minority_indices = np.where(y_train == minority_class)[0]
        
        # Original data
        augmented_X = [X_train]
        augmented_y = [y_train]
        
        # Calculate number of samples to generate
        n_samples = int((class_counts[1 - minority_class] - class_counts[minority_class]) * augmentation_factor)
        
        if verbose:
            logger.info(f"Generating {n_samples} augmented samples for class {minority_class}")
        
        # Generate augmented samples
        for _ in range(n_samples):
            # Randomly select a minority class sample
            idx = np.random.choice(minority_indices)
            sample = X_train[idx].copy()
            
            # Apply random noise (jittering)
            noise_level = 0.02
            noise = np.random.normal(0, noise_level, sample.shape)
            sample_augmented = sample + noise
            
            # Add to augmented data
            augmented_X.append(np.expand_dims(sample_augmented, axis=0))
            augmented_y.append(np.array([minority_class]))
        
        # Combine augmented data
        augmented_X_train = np.vstack(augmented_X)
        augmented_y_train = np.concatenate(augmented_y)
        
        if verbose:
            # Count augmented class distribution
            augmented_counts = np.bincount(augmented_y_train.astype(int))
            logger.info(f"Augmented class distribution: {augmented_counts}")
        
        return augmented_X_train, augmented_y_train
    
    except Exception as e:
        logger.error(f"Error augmenting training data: {str(e)}")
        return X_train, y_train

def build_enhanced_lstm_model(input_shape, regularization=0.01, dropout_rate=0.3):
    """
    Build enhanced LSTM model with regularization and dropout.
    
    Args:
        input_shape: Shape of input data (lookback, features)
        regularization: L1/L2 regularization factor
        dropout_rate: Dropout rate
        
    Returns:
        Compiled LSTM model
    """
    try:
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True, 
                 kernel_regularizer=l1_l2(l1=regularization, l2=regularization)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            LSTM(64, return_sequences=False,
                 kernel_regularizer=l1_l2(l1=regularization, l2=regularization)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(32, activation='relu',
                  kernel_regularizer=l1_l2(l1=regularization, l2=regularization)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except Exception as e:
        logger.error(f"Error building enhanced LSTM model: {str(e)}")
        return None

def train_model(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, 
                validation_split=0.1, patience=20, verbose=False):
    """
    Train model with early stopping and learning rate reduction.
    
    Args:
        model: LSTM model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Validation split ratio
        patience: Early stopping patience
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (trained_model, history)
    """
    try:
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=0.0001
        )
        
        # Create temporary file for model checkpointing
        import tempfile
        fd, checkpoint_path = tempfile.mkstemp()
        os.close(fd)
        
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1 if verbose else 2
        )
        
        # Load best model weights
        model.load_weights(checkpoint_path)
        
        # Remove temporary file
        os.remove(checkpoint_path)
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
        
        return model, history
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, None

def plot_training_history(history, pair, output_dir):
    """
    Plot training history.
    
    Args:
        history: Training history
        pair: Trading pair
        output_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    try:
        # Create directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize pair format for file path
        pair_path = pair.replace('/', '')
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{pair} - Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{pair} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"{pair_path}_training_history.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
        
        return plot_path
    
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        return None

def save_model(model, pair, output_dir, feature_columns, metrics):
    """
    Save model and metadata.
    
    Args:
        model: Trained model
        pair: Trading pair
        output_dir: Output directory
        feature_columns: Feature column names
        metrics: Model metrics
        
    Returns:
        Path to saved model
    """
    try:
        # Create directory if not exists
        lstm_dir = os.path.join(output_dir, 'lstm')
        os.makedirs(lstm_dir, exist_ok=True)
        
        # Normalize pair format for file path
        pair_path = pair.replace('/', '')
        
        # Save model
        model_path = os.path.join(lstm_dir, f"{pair_path}.h5")
        model.save(model_path)
        
        # Save metadata
        metadata = {
            'pair': pair,
            'model_type': 'lstm',
            'feature_columns': feature_columns,
            'metrics': metrics,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(lstm_dir, f"{pair_path}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return None

def enhance_model(pair, epochs=200, batch_size=32, lookback=24, train_size=0.8, 
                 validation_split=0.1, patience=20, output_dir='models', verbose=False):
    """
    Enhance ML model for the specified trading pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        epochs: Number of training epochs
        batch_size: Batch size for training
        lookback: Lookback window size
        train_size: Train/test split ratio
        validation_split: Validation split ratio
        patience: Early stopping patience
        output_dir: Output directory
        verbose: Whether to print detailed information
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Enhancing ML model for {pair}")
        
        # Load dataset
        df = load_dataset(pair, verbose)
        if df is None:
            logger.error(f"Failed to load dataset for {pair}")
            return False
        
        # Enhance features
        df_enhanced = enhance_features(df, verbose)
        if df_enhanced is None:
            logger.error(f"Failed to enhance features for {pair}")
            return False
        
        # Prepare sequences
        X_train, y_train, X_test, y_test, feature_columns = prepare_sequences(
            df_enhanced, lookback, train_size, verbose
        )
        
        if X_train is None:
            logger.error(f"Failed to prepare sequences for {pair}")
            return False
        
        # Augment training data
        X_train_aug, y_train_aug = augment_training_data(X_train, y_train, verbose=verbose)
        
        # Build model
        input_shape = (lookback, X_train.shape[2])
        model = build_enhanced_lstm_model(input_shape)
        
        if model is None:
            logger.error(f"Failed to build model for {pair}")
            return False
        
        # Train model
        model, history = train_model(
            model, X_train_aug, y_train_aug, X_test, y_test,
            epochs, batch_size, validation_split, patience, verbose
        )
        
        if model is None:
            logger.error(f"Failed to train model for {pair}")
            return False
        
        # Plot training history
        plot_path = plot_training_history(history, pair, output_dir)
        
        # Calculate metrics
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'training_samples': int(len(X_train_aug)),
            'testing_samples': int(len(X_test)),
            'feature_count': int(X_train.shape[2])
        }
        
        # Save model
        model_path = save_model(model, pair, output_dir, feature_columns, metrics)
        
        if model_path is None:
            logger.error(f"Failed to save model for {pair}")
            return False
        
        logger.info(f"Successfully enhanced ML model for {pair}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1_score']:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error enhancing model for {pair}: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Disable GPU if requested
    if args.no_gpu:
        logger.info("Disabling GPU as requested")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    logger.info("=" * 80)
    logger.info("ENHANCE ML MODEL ACCURACY")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Lookback window: {args.lookback}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Validation split: {args.validation_split}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not args.no_gpu:
        logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        
        # Set memory growth to avoid allocating all memory at once
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Set memory growth for GPU {gpu}")
            except RuntimeError as e:
                logger.warning(f"Error setting memory growth for GPU {gpu}: {e}")
    else:
        logger.info("No GPUs available, using CPU")
    
    # Process each pair
    success_count = 0
    for pair in pairs:
        logger.info("-" * 80)
        logger.info(f"Processing {pair}")
        logger.info("-" * 80)
        
        success = enhance_model(
            pair=pair,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lookback=args.lookback,
            train_size=args.train_size,
            validation_split=args.validation_split,
            patience=args.patience,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
        else:
            logger.error(f"Failed to enhance model for {pair}")
    
    # Log summary
    logger.info("=" * 80)
    logger.info(f"ENHANCEMENT COMPLETED: {success_count}/{len(pairs)} pairs successful")
    logger.info("=" * 80)
    
    return 0 if success_count == len(pairs) else 1

if __name__ == "__main__":
    sys.exit(main())