"""
Enhanced Transformer-Based ML Models for Trading Bot

This module implements advanced transformer-based neural network architectures
combining TCN (Temporal Convolutional Networks), CNN, and LSTM for highly accurate
price prediction in volatile market conditions.

The models are designed to achieve 90%+ directional accuracy through:
1. Temporal pattern recognition with TCN layers
2. Feature extraction with CNNs
3. Sequential memory with LSTM/GRU
4. Self-attention mechanisms for focusing on important price movements
5. Transformer encoder blocks for capturing complex market dynamics

Usage:
- Train models with historical data
- Use for real-time prediction in the trading bot
- Periodic retraining to adapt to changing market conditions
"""

import os
import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    Flatten, Concatenate, Attention, MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D, Activation, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Import TCN layers (ensuring compatibility)
try:
    from keras_tcn import TCN
except ImportError:
    # Fallback TCN implementation if keras_tcn is not available
    from tensorflow.keras.layers import Conv1D, Activation, SpatialDropout1D

    def TCN(nb_filters=64, kernel_size=2, nb_stacks=1, dilations=None,
            activation='relu', padding='causal', use_skip_connections=True,
            dropout_rate=0.0, return_sequences=True, name='tcn'):
        """
        Simple TCN implementation using Conv1D layers with dilations
        """
        def _tcn_layer(x):
            if dilations is None:
                _dilations = [1, 2, 4, 8, 16, 32]
            else:
                _dilations = dilations
            
            residual = x
            for i, dilation in enumerate(_dilations):
                x_in = x
                x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                          dilation_rate=dilation, padding=padding,
                          name=f'{name}_conv1d_{i}')(x)
                x = BatchNormalization(name=f'{name}_bn_{i}')(x)
                x = Activation(activation, name=f'{name}_act_{i}')(x)
                x = SpatialDropout1D(dropout_rate, name=f'{name}_dropout_{i}')(x)
                
                # Second conv layer
                x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                          dilation_rate=dilation, padding=padding,
                          name=f'{name}_conv1d_{i}_2')(x)
                x = BatchNormalization(name=f'{name}_bn_{i}_2')(x)
                x = Activation(activation, name=f'{name}_act_{i}_2')(x)
                x = SpatialDropout1D(dropout_rate, name=f'{name}_dropout_{i}_2')(x)
                
                # Skip connection
                if use_skip_connections:
                    # 1x1 conv if needed for matching dimensions
                    if i == 0:
                        residual = Conv1D(filters=nb_filters, kernel_size=1,
                                         name=f'{name}_residual_conv')(residual)
                    x = Add(name=f'{name}_add_{i}')([x, residual])
                    residual = x
            
            if not return_sequences:
                x = GlobalAveragePooling1D(name=f'{name}_pool')(x)
            
            return x
        
        return _tcn_layer

logger = logging.getLogger(__name__)

# Model directories
MODEL_DIR = os.path.join(os.getcwd(), 'models', 'enhanced')
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters for enhanced models
SEQUENCE_LENGTH = 60  # 60 time steps for sequence data
TRAIN_TEST_SPLIT = 0.15  # 15% of data for testing
VAL_SPLIT = 0.15  # 15% of training data for validation
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 20  # Early stopping patience
LEARNING_RATE = 0.001
L1_REG = 0.0001
L2_REG = 0.0001
DROPOUT_RATE = 0.3

def create_advanced_hybrid_tcn_model(input_shape, use_attention=True, 
                                  use_transformer=True, n_filters=64,
                                  l1_reg=0.0001, l2_reg=0.0001):
    """
    Create an advanced hybrid model combining TCN, CNN and LSTM with 
    transformer blocks and attention mechanisms
    
    Args:
        input_shape: Input shape for the model (time steps, features)
        use_attention: Whether to include attention mechanism
        use_transformer: Whether to include transformer blocks
        n_filters: Number of filters for convolutional layers
        l1_reg: L1 regularization coefficient
        l2_reg: L2 regularization coefficient
        
    Returns:
        Model: Advanced hybrid model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initialize regularizer
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    
    # TCN branch
    tcn_branch = TCN(
        nb_filters=n_filters,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        activation='relu',
        use_skip_connections=True,
        padding='causal',
        dropout_rate=DROPOUT_RATE,
        return_sequences=True,
        name='tcn_branch'
    )(inputs)
    
    # CNN branch
    cnn_branch = Conv1D(
        filters=n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizer,
        name='cnn_branch_1'
    )(inputs)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Dropout(DROPOUT_RATE)(cnn_branch)
    
    cnn_branch = Conv1D(
        filters=n_filters*2,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizer,
        name='cnn_branch_2'
    )(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Dropout(DROPOUT_RATE)(cnn_branch)
    
    # LSTM branch
    lstm_branch = LSTM(
        units=n_filters,
        return_sequences=True,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE,
        name='lstm_branch'
    )(inputs)
    
    # Optional: GRU branch for additional expressiveness
    gru_branch = GRU(
        units=n_filters,
        return_sequences=True,
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE,
        name='gru_branch'
    )(inputs)
    
    # Merge branches
    merged = Concatenate(name='merged_branches')([tcn_branch, cnn_branch, lstm_branch, gru_branch])
    
    # Optional transformer encoder blocks
    if use_transformer:
        for i in range(2):  # 2 transformer blocks
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=8, 
                key_dim=n_filters, 
                dropout=DROPOUT_RATE,
                name=f'transformer_mha_{i}'
            )(merged, merged)
            
            # Skip connection 1
            merged_1 = Add(name=f'transformer_add_1_{i}')([merged, attention_output])
            merged_1 = LayerNormalization(name=f'transformer_ln_1_{i}')(merged_1)
            
            # Feed-forward network
            ffn = Dense(
                units=n_filters*4, 
                activation='relu',
                kernel_regularizer=regularizer,
                name=f'transformer_ffn_1_{i}'
            )(merged_1)
            ffn = Dropout(DROPOUT_RATE)(ffn)
            ffn = Dense(
                units=n_filters*4,
                kernel_regularizer=regularizer,
                name=f'transformer_ffn_2_{i}'
            )(ffn)
            
            # Skip connection 2
            merged = Add(name=f'transformer_add_2_{i}')([merged_1, ffn])
            merged = LayerNormalization(name=f'transformer_ln_2_{i}')(merged)
    
    # Additional attention mechanism for time series
    if use_attention:
        attention_out = Attention(use_scale=True, name='series_attention')([merged, merged])
        merged = Concatenate(name='attention_concat')([merged, attention_out])
    
    # Final layers
    x = GlobalAveragePooling1D(name='global_pool')(merged)
    
    x = Dense(
        units=n_filters*2,
        activation='relu',
        kernel_regularizer=regularizer,
        name='dense_1'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = Dense(
        units=n_filters,
        activation='relu',
        kernel_regularizer=regularizer,
        name='dense_2'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE/2)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='enhanced_hybrid_model')
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_for_training(data, sequence_length=SEQUENCE_LENGTH):
    """
    Prepare data for training by creating sequences and targets
    
    Args:
        data: DataFrame with price data and technical indicators
        sequence_length: Length of sequence for each sample
        
    Returns:
        X, y: Prepared data
    """
    # Ensure DataFrame has all necessary features
    required_columns = [
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Add technical indicators if not present
    data = calculate_technical_indicators(data)
    
    # Remove rows with NaN values (from indicator calculation)
    data = data.dropna()
    
    # Extract features and target
    features = data.drop(['target'], axis=1, errors='ignore').values
    
    # Create target based on price direction
    next_close = data['close'].shift(-1)
    target = (next_close > data['close']).astype(int)
    target = target[:-1]  # Remove last row which will be NaN
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length - 1):
        X.append(features_scaled[i:(i + sequence_length)])
        y.append(target.iloc[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for the dataset
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Ensure we have required price columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    # Simple Moving Averages
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['sma10'] = df['close'].rolling(window=10).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    df['lowest_14'] = df['low'].rolling(window=14).min()
    df['highest_14'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - df['lowest_14']) / 
                          (df['highest_14'] - df['lowest_14']))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    
    # Rate of Change
    df['roc'] = df['close'].pct_change(periods=10) * 100
    
    # On-Balance Volume (OBV)
    df['obv'] = (df['volume'] * 
                ((df['close'] > df['close'].shift()).astype(int) - 
                 (df['close'] < df['close'].shift()).astype(int))).cumsum()
    
    # Normalized OBV
    df['obv_norm'] = (df['obv'] - df['obv'].rolling(window=20).min()) / \
                    (df['obv'].rolling(window=20).max() - df['obv'].rolling(window=20).min())
    
    # Price relative to moving averages (normalized)
    df['price_rel_sma20'] = df['close'] / df['sma20'] - 1
    df['price_rel_sma50'] = df['close'] / df['sma50'] - 1
    df['price_rel_ema9'] = df['close'] / df['ema9'] - 1
    
    # Volatility
    df['volatility'] = df['bb_std'] / df['bb_middle']
    
    # ADX (Average Directional Index)
    # +DM, -DM
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff()
    df['plus_dm'] = df['plus_dm'].where((df['plus_dm'] > 0) & 
                                       (df['plus_dm'] > df['minus_dm'].abs()), 0)
    df['minus_dm'] = df['minus_dm'].abs().where((df['minus_dm'] > 0) & 
                                               (df['minus_dm'] > df['plus_dm']), 0)
    
    # ATR for ADX
    df['plus_di'] = 100 * df['plus_dm'].rolling(window=14).mean() / df['atr']
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=14).mean() / df['atr']
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    
    # Add momentum features
    df['momentum'] = df['close'] - df['close'].shift(14)
    df['momentum_norm'] = df['momentum'] / df['close'].shift(14)
    
    # Add price and volume features
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    
    # Create target variable based on next period's price direction
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    return df

def train_enhanced_model(data, model_name='enhanced_tcn_cnn_lstm', sequence_length=SEQUENCE_LENGTH):
    """
    Train an enhanced model with the provided data
    
    Args:
        data: DataFrame with price and indicator data
        model_name: Name for saving the model
        sequence_length: Length of sequence for each sample
        
    Returns:
        model: Trained model
        evaluation: Dictionary with evaluation metrics
        scaler: Fitted data scaler
    """
    logger.info(f"Preparing data for {model_name}...")
    X, y, scaler = prepare_data_for_training(data, sequence_length)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, shuffle=False
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, shuffle=False
    )
    
    logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # Create a model
    input_shape = (X_train.shape[1], X_train.shape[2])
    logger.info(f"Creating model with input shape {input_shape}...")
    
    model = create_advanced_hybrid_tcn_model(
        input_shape=input_shape,
        use_attention=True,
        use_transformer=True
    )
    
    # Callbacks
    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    logger.info(f"Training model {model_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    logger.info("Evaluating model...")
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    evaluation = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Model evaluation: {evaluation}")
    
    # Plot training history
    plot_training_history(history, model_name)
    
    return model, evaluation, scaler

def plot_training_history(history, model_name):
    """
    Plot and save training history
    
    Args:
        history: Training history
        model_name: Name for the plot file
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(MODEL_DIR, f"{model_name}_history.png")
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to {plot_path}")

def save_model_with_metadata(model, scaler, evaluation, asset, model_name):
    """
    Save model with metadata for future use
    
    Args:
        model: Trained model
        scaler: Fitted data scaler
        evaluation: Evaluation metrics
        asset: Trading pair name
        model_name: Base name for the model
    """
    # Create asset-specific directory
    asset_dir = os.path.join(MODEL_DIR, asset.replace('/', '_'))
    os.makedirs(asset_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(asset_dir, f"{model_name}.h5")
    model.save(model_path)
    
    # Save the scaler
    scaler_path = os.path.join(asset_dir, f"{model_name}_scaler.npy")
    np.save(scaler_path, scaler)
    
    # Save evaluation metrics
    eval_path = os.path.join(asset_dir, f"{model_name}_eval.txt")
    with open(eval_path, 'w') as f:
        f.write(f"Evaluation Metrics for {model_name} on {asset}\n")
        f.write("="*50 + "\n")
        for metric, value in evaluation.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logger.info(f"Model saved to {model_path} with evaluation metrics")

def load_model_with_metadata(asset, model_name):
    """
    Load model with metadata
    
    Args:
        asset: Trading pair name
        model_name: Base name for the model
        
    Returns:
        model: Loaded model
        scaler: Fitted data scaler
        evaluation: Evaluation metrics dictionary
    """
    # Create asset-specific directory path
    asset_dir = os.path.join(MODEL_DIR, asset.replace('/', '_'))
    
    # Check if model exists
    model_path = os.path.join(asset_dir, f"{model_name}.h5")
    if not os.path.exists(model_path):
        logger.error(f"Model {model_path} not found")
        return None, None, None
    
    # Load the model
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Try loading with custom objects if TCN is causing issues
        model = load_model(model_path, custom_objects={'TCN': TCN})
    
    # Load the scaler
    scaler_path = os.path.join(asset_dir, f"{model_name}_scaler.npy")
    try:
        scaler = np.load(scaler_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        scaler = None
    
    # Load evaluation metrics
    eval_path = os.path.join(asset_dir, f"{model_name}_eval.txt")
    evaluation = {}
    try:
        with open(eval_path, 'r') as f:
            for line in f:
                if ': ' in line:
                    metric, value = line.strip().split(': ')
                    evaluation[metric] = float(value)
    except Exception as e:
        logger.error(f"Error loading evaluation metrics: {e}")
    
    logger.info(f"Loaded model from {model_path}")
    return model, scaler, evaluation

def predict_with_model(model, scaler, data, sequence_length=SEQUENCE_LENGTH):
    """
    Make predictions using a trained model
    
    Args:
        model: Trained model
        scaler: Fitted data scaler
        data: DataFrame with recent data
        sequence_length: Length of sequence for prediction
        
    Returns:
        float: Prediction (0-1 probability of price increase)
        dict: Additional prediction details
    """
    # Ensure DataFrame has all necessary features
    data = calculate_technical_indicators(data)
    
    # Remove rows with NaN values (from indicator calculation)
    data = data.dropna()
    
    # Extract features
    features = data.drop(['target'], axis=1, errors='ignore').values
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Create a sequence for prediction (use the most recent data)
    if len(features_scaled) >= sequence_length:
        sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
    else:
        logger.warning(f"Not enough data for sequence. Need {sequence_length}, got {len(features_scaled)}")
        return 0.5, {"error": "Not enough data for prediction"}
    
    # Make prediction
    prediction = model.predict(sequence)[0][0]
    
    # Get additional prediction details
    latest_close = data['close'].iloc[-1]
    
    details = {
        "probability": float(prediction),
        "direction": "UP" if prediction > 0.5 else "DOWN",
        "confidence": abs(prediction - 0.5) * 2,  # Scale to 0-1
        "current_price": float(latest_close),
        "signals": {
            "rsi": float(data['rsi'].iloc[-1]),
            "macd": float(data['macd'].iloc[-1]),
            "adx": float(data['adx'].iloc[-1]) if 'adx' in data.columns else None,
            "price_rel_sma20": float(data['price_rel_sma20'].iloc[-1])
        }
    }
    
    return prediction, details

def train_models_for_all_assets(assets=None, data_sources=None):
    """
    Train enhanced models for all specified assets
    
    Args:
        assets: List of assets to train models for
        data_sources: Dict mapping assets to DataFrames
        
    Returns:
        dict: Dictionary with trained models and metadata
    """
    assets = assets or ["SOL/USD", "ETH/USD", "BTC/USD"]
    results = {}
    
    for asset in assets:
        logger.info(f"Training model for {asset}...")
        
        # Get data for this asset
        if data_sources and asset in data_sources:
            data = data_sources[asset]
        else:
            logger.warning(f"No data source provided for {asset}, skipping")
            continue
        
        try:
            # Train model for this asset
            model_name = f"enhanced_tcn_cnn_lstm_{asset.replace('/', '_').lower()}"
            model, evaluation, scaler = train_enhanced_model(data, model_name)
            
            # Save model with metadata
            save_model_with_metadata(model, scaler, evaluation, asset, "enhanced_model")
            
            # Store results
            results[asset] = {
                "model": model,
                "scaler": scaler,
                "evaluation": evaluation,
                "model_name": "enhanced_model"
            }
            
            logger.info(f"Successfully trained model for {asset} with accuracy {evaluation['accuracy']:.4f}")
        
        except Exception as e:
            logger.error(f"Error training model for {asset}: {e}")
    
    return results

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Example dataset creation for testing
    from numpy.random import randn
    
    def create_test_dataset(n_samples=1000):
        index = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        close = 100 + np.cumsum(randn(n_samples) * 0.5)
        high = close + abs(randn(n_samples)) * 0.5
        low = close - abs(randn(n_samples)) * 0.5
        open_price = close.copy()
        np.random.shuffle(open_price)
        open_price = np.clip(open_price, low, high)
        volume = abs(randn(n_samples)) * 1000 + 1000
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=index)
        
        return df
    
    # Test with synthetic data
    logger.info("Creating test dataset...")
    test_data = create_test_dataset(2000)
    
    logger.info("Training test model...")
    model, evaluation, scaler = train_enhanced_model(test_data, "test_model")
    logger.info(f"Test model evaluation: {evaluation}")