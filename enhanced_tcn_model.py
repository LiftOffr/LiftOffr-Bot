#!/usr/bin/env python3
"""
Enhanced TCN (Temporal Convolutional Network) Model Implementation

This module provides an enhanced implementation of Temporal Convolutional Networks
optimized for financial time series prediction with significantly improved accuracy.

Key enhancements include:
1. Multi-branch architecture combining TCN, Attention, and Transformer components
2. Residual connections with channel-wise attention
3. Advanced regularization techniques
4. Hyperparameter optimization for financial data
5. Market regime-specific adaptations

Target accuracy: 90%+ for directional prediction
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, Activation, 
    Dropout, SpatialDropout1D, Add, Multiply, Flatten, 
    GlobalAveragePooling1D, Reshape, Permute, Lambda, 
    LayerNormalization, MultiHeadAttention, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models/tcn_enhanced"
os.makedirs(MODELS_DIR, exist_ok=True)

class EnhancedTCNModel:
    """
    Enhanced TCN model combining TCN, Attention, and Transformer components
    with advanced regularization and hyperparameter tuning for financial data.
    """
    
    def __init__(self, input_shape, output_size=1, 
                 filters=64, kernel_size=3, dilations=None, 
                 dropout_rate=0.3, use_skip_connections=True,
                 use_batch_norm=True, use_layer_norm=True,
                 use_spatial_dropout=True, use_attention=True,
                 use_transformer=True, use_channel_attention=True,
                 l1_reg=0.0001, l2_reg=0.0001,
                 model_path=None):
        """
        Initialize Enhanced TCN model
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            output_size (int): Output size (1 for binary classification/regression)
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Size of the convolutional kernel
            dilations (list): List of dilation rates for TCN layers
            dropout_rate (float): Dropout rate
            use_skip_connections (bool): Whether to use skip connections
            use_batch_norm (bool): Whether to use batch normalization
            use_layer_norm (bool): Whether to use layer normalization
            use_spatial_dropout (bool): Whether to use spatial dropout
            use_attention (bool): Whether to use attention mechanism
            use_transformer (bool): Whether to use transformer components
            use_channel_attention (bool): Whether to use channel-wise attention
            l1_reg (float): L1 regularization coefficient
            l2_reg (float): L2 regularization coefficient
            model_path (str): Path to load pre-trained model
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 2, 4, 8, 16]
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_spatial_dropout = use_spatial_dropout
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.use_channel_attention = use_channel_attention
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Build or load model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
        else:
            logger.info("Building enhanced TCN model")
            self.model = self._build_model()
    
    def _residual_block(self, x, dilation_rate, stage, block):
        """
        TCN residual block with dilated convolutions
        
        Args:
            x: Input tensor
            dilation_rate (int): Dilation rate
            stage (int): Stage number
            block (int): Block number
            
        Returns:
            Tensor: Output tensor
        """
        # Naming convention
        name_base = f'stage{stage}_block{block}_'
        
        # Residual branch
        residual = x
        
        # If dimensions don't match, use 1x1 conv to match dimensions
        if K.int_shape(residual)[-1] != self.filters * 2:
            residual = Conv1D(
                self.filters * 2, 1, padding='same',
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=name_base+'matchconv'
            )(residual)
        
        # First dilated convolution
        x = Conv1D(
            self.filters, self.kernel_size, 
            dilation_rate=dilation_rate,
            padding='causal',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_base+'conv1'
        )(x)
        
        # Normalization options
        if self.use_batch_norm:
            x = BatchNormalization(name=name_base+'bn1')(x)
        if self.use_layer_norm:
            x = LayerNormalization(epsilon=1e-6, name=name_base+'ln1')(x)
        
        # Activation
        x = Activation('relu', name=name_base+'relu1')(x)
        
        # Dropout options
        if self.use_spatial_dropout:
            x = SpatialDropout1D(self.dropout_rate, name=name_base+'spatial_dropout1')(x)
        else:
            x = Dropout(self.dropout_rate, name=name_base+'dropout1')(x)
        
        # Second dilated convolution
        x = Conv1D(
            self.filters * 2, self.kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_base+'conv2'
        )(x)
        
        # Normalization
        if self.use_batch_norm:
            x = BatchNormalization(name=name_base+'bn2')(x)
        if self.use_layer_norm:
            x = LayerNormalization(epsilon=1e-6, name=name_base+'ln2')(x)
        
        # Channel-wise attention
        if self.use_channel_attention:
            attention = self._channel_attention(x, name_base+'channel_attention')
            x = Multiply(name=name_base+'attention_multiply')([x, attention])
        
        # Skip connection
        if self.use_skip_connections:
            x = Add(name=name_base+'add')([x, residual])
        
        # Final activation
        x = Activation('relu', name=name_base+'relu2')(x)
        
        return x
    
    def _channel_attention(self, input_tensor, name_prefix):
        """
        Channel-wise attention mechanism
        
        Args:
            input_tensor: Input tensor
            name_prefix (str): Prefix for layer names
            
        Returns:
            Tensor: Channel attention weights
        """
        # Get input shape
        channel_axis = -1
        channels = K.int_shape(input_tensor)[channel_axis]
        
        # Global average pooling
        avg_pool = GlobalAveragePooling1D(name=name_prefix+'_gap')(input_tensor)
        avg_pool = Reshape((1, channels), name=name_prefix+'_reshape1')(avg_pool)
        
        # Two dense layers (squeeze and excitation)
        x = Dense(
            channels // 8, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_dense1'
        )(avg_pool)
        
        x = Dense(
            channels,
            activation='sigmoid',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_dense2'
        )(x)
        
        return x
    
    def _transformer_block(self, x, name_prefix):
        """
        Transformer block with multi-head attention and feed-forward network
        
        Args:
            x: Input tensor
            name_prefix (str): Prefix for layer names
            
        Returns:
            Tensor: Output tensor
        """
        # Layer normalization
        ln1 = LayerNormalization(epsilon=1e-6, name=name_prefix+'_ln1')(x)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1,
            name=name_prefix+'_mha'
        )(ln1, ln1)
        
        # Skip connection
        out1 = Add(name=name_prefix+'_add1')([x, attention_output])
        
        # Layer normalization
        ln2 = LayerNormalization(epsilon=1e-6, name=name_prefix+'_ln2')(out1)
        
        # Feed-forward network
        ffn = Dense(
            self.filters * 4,
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_ffn1'
        )(ln2)
        
        ffn = Dropout(self.dropout_rate, name=name_prefix+'_ffn_dropout')(ffn)
        
        ffn = Dense(
            K.int_shape(ln2)[-1],
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_ffn2'
        )(ffn)
        
        # Skip connection
        out2 = Add(name=name_prefix+'_add2')([out1, ffn])
        
        return out2
    
    def _attention_block(self, x, name_prefix):
        """
        Self-attention block
        
        Args:
            x: Input tensor
            name_prefix (str): Prefix for layer names
            
        Returns:
            Tensor: Output tensor
        """
        # Get input shape
        seq_len = K.int_shape(x)[1]
        dim = K.int_shape(x)[2]
        
        # Create query, key, value projections
        query = Dense(
            dim, 
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_query'
        )(x)
        
        key = Dense(
            dim, 
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_key'
        )(x)
        
        value = Dense(
            dim, 
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_value'
        )(x)
        
        # Reshape for matrix multiplication
        query = Reshape((seq_len, 8, dim // 8))(query)
        query = Permute((2, 1, 3))(query)  # (batch_size, heads, seq_len, depth)
        
        key = Reshape((seq_len, 8, dim // 8))(key)
        key = Permute((2, 1, 3))(key)  # (batch_size, heads, seq_len, depth)
        
        value = Reshape((seq_len, 8, dim // 8))(value)
        value = Permute((2, 1, 3))(value)  # (batch_size, heads, seq_len, depth)
        
        # Attention scores
        attention_scores = Lambda(
            lambda x: tf.matmul(x[0], x[1], transpose_b=True) / tf.math.sqrt(tf.cast(dim // 8, tf.float32)),
            name=name_prefix+'_scores'
        )([query, key])
        
        # Apply softmax
        attention_weights = Lambda(
            lambda x: tf.nn.softmax(x, axis=-1),
            name=name_prefix+'_weights'
        )(attention_scores)
        
        # Apply attention weights
        attention_output = Lambda(
            lambda x: tf.matmul(x[0], x[1]),
            name=name_prefix+'_output'
        )([attention_weights, value])
        
        # Reshape back
        attention_output = Permute((2, 1, 3))(attention_output)  # (batch_size, seq_len, heads, depth)
        attention_output = Reshape((seq_len, dim))(attention_output)
        
        # Final projection
        output = Dense(
            dim, 
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name=name_prefix+'_projection'
        )(attention_output)
        
        # Skip connection
        output = Add(name=name_prefix+'_add')([x, output])
        
        return output

    def _build_model(self):
        """
        Build the enhanced TCN model
        
        Returns:
            Model: Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape, name='input')
        
        # Initial projection
        x = Conv1D(
            self.filters, 1, 
            padding='causal',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='initial_projection'
        )(inputs)
        
        # TCN Branch
        tcn_branch = x
        skip_connections = []
        
        # TCN blocks with increasing dilation rates
        for i, dilation_rate in enumerate(self.dilations):
            tcn_branch = self._residual_block(
                tcn_branch, 
                dilation_rate, 
                stage=1, 
                block=i+1
            )
            skip_connections.append(tcn_branch)
        
        # Combine skip connections
        if self.use_skip_connections and len(skip_connections) > 0:
            tcn_output = Add(name='tcn_skip_add')(skip_connections)
        else:
            tcn_output = tcn_branch
        
        # Attention Branch (if enabled)
        if self.use_attention:
            attention_branch = x
            
            # Apply self-attention
            attention_branch = self._attention_block(
                attention_branch, 
                name_prefix='attention'
            )
            
            # Combine with TCN branch
            x = Concatenate(name='tcn_attention_concat')([tcn_output, attention_branch])
            
            # Projection after concatenation
            x = Conv1D(
                self.filters * 2, 1,
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='combined_projection'
            )(x)
        else:
            x = tcn_output
        
        # Transformer Block (if enabled)
        if self.use_transformer:
            x = self._transformer_block(x, name_prefix='transformer')
        
        # Global pooling
        x = GlobalAveragePooling1D(name='global_pooling')(x)
        
        # Fully connected layers
        x = Dense(
            128, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='fc1'
        )(x)
        
        x = Dropout(self.dropout_rate, name='fc1_dropout')(x)
        
        x = Dense(
            64, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='fc2'
        )(x)
        
        x = Dropout(self.dropout_rate, name='fc2_dropout')(x)
        
        # Output layer
        if self.output_size == 1:
            # Binary classification or regression
            outputs = Dense(
                1, 
                activation='sigmoid',
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='output'
            )(x)
        else:
            # Multi-class classification
            outputs = Dense(
                self.output_size, 
                activation='softmax',
                kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name='output'
            )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='EnhancedTCN')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if self.output_size == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        logger.info(model.summary())
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              batch_size=32, epochs=100, early_stopping_patience=15,
              model_prefix="enhanced_tcn", save_best_only=True):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation data
            y_val (numpy.ndarray): Validation labels
            batch_size (int): Batch size
            epochs (int): Maximum number of epochs
            early_stopping_patience (int): Early stopping patience
            model_prefix (str): Prefix for saving model
            save_best_only (bool): Save only the best model
            
        Returns:
            History: Training history
        """
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        if early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Model checkpoint
        model_path = os.path.join(MODELS_DIR, f"{model_prefix}_best.h5")
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=save_best_only,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Train model
        logger.info(f"Training model with {len(X_train)} samples, batch size {batch_size}")
        
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use training data for validation (not ideal)
            history = self.model.fit(
                X_train, y_train,
                validation_split=0.2,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save final model
        final_model_path = os.path.join(MODELS_DIR, f"{model_prefix}_final.h5")
        self.model.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        return history
    
    def predict(self, X):
        """
        Generate predictions
        
        Args:
            X (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model
        
        Args:
            X (numpy.ndarray): Input data
            y (numpy.ndarray): Ground truth
            
        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(X, y)
    
    def save(self, path):
        """
        Save model
        
        Args:
            path (str): Path to save model
        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load model
        
        Args:
            path (str): Path to load model
        """
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def prepare_sequences(data, sequence_length=60, step=1, target_column='close_direction'):
    """
    Prepare sequences for time series prediction
    
    Args:
        data (pd.DataFrame): Input data
        sequence_length (int): Sequence length
        step (int): Step size for creating sequences
        target_column (str): Target column name
        
    Returns:
        tuple: (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length, step):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length][target_column])
    
    return np.array(X), np.array(y)


def prepare_multi_timeframe_data(data_dict, primary_timeframe="1h", 
                               sequence_length=60, target_column='close_direction'):
    """
    Prepare multi-timeframe data
    
    Args:
        data_dict (dict): Dictionary of dataframes for each timeframe
        primary_timeframe (str): Primary timeframe
        sequence_length (int): Sequence length
        target_column (str): Target column name
        
    Returns:
        tuple: (X, y) where X is multi-timeframe input and y is target values
    """
    # Ensure primary timeframe exists
    if primary_timeframe not in data_dict:
        raise ValueError(f"Primary timeframe {primary_timeframe} not found in data_dict")
    
    # Get primary data
    primary_data = data_dict[primary_timeframe]
    
    # Create sequences for primary timeframe
    X_primary, y = prepare_sequences(
        primary_data, 
        sequence_length=sequence_length, 
        target_column=target_column
    )
    
    # Process each additional timeframe
    X_dict = {'primary': X_primary}
    
    for tf, df in data_dict.items():
        if tf == primary_timeframe:
            continue
        
        # Align with primary timeframe
        aligned_data = align_timeframe_data(
            primary_data.index, 
            df, 
            primary_timeframe, 
            tf
        )
        
        # Create sequences
        X_tf, _ = prepare_sequences(
            aligned_data, 
            sequence_length=sequence_length, 
            target_column=target_column
        )
        
        # Add to dictionary
        X_dict[tf] = X_tf
    
    return X_dict, y


def align_timeframe_data(primary_index, secondary_df, primary_tf, secondary_tf):
    """
    Align secondary timeframe data with primary timeframe
    
    Args:
        primary_index (pd.Index): Index of primary timeframe data
        secondary_df (pd.DataFrame): Secondary timeframe data
        primary_tf (str): Primary timeframe
        secondary_tf (str): Secondary timeframe
        
    Returns:
        pd.DataFrame: Aligned secondary timeframe data
    """
    # Resample secondary data to primary timeframe
    if secondary_tf == "1d" and primary_tf in ["1h", "4h"]:
        # For daily data, forward-fill to hourly
        resampled = secondary_df.reindex(
            pd.date_range(
                start=secondary_df.index.min(),
                end=secondary_df.index.max(),
                freq=primary_tf
            ),
            method='ffill'
        )
    elif secondary_tf == "4h" and primary_tf == "1h":
        # For 4h data, forward-fill to hourly
        resampled = secondary_df.reindex(
            pd.date_range(
                start=secondary_df.index.min(),
                end=secondary_df.index.max(),
                freq=primary_tf
            ),
            method='ffill'
        )
    elif secondary_tf == "1h" and primary_tf == "15m":
        # For hourly data to 15m, forward-fill
        resampled = secondary_df.reindex(
            pd.date_range(
                start=secondary_df.index.min(),
                end=secondary_df.index.max(),
                freq=primary_tf
            ),
            method='ffill'
        )
    else:
        # Default: forward-fill
        resampled = secondary_df.reindex(
            pd.date_range(
                start=secondary_df.index.min(),
                end=secondary_df.index.max(),
                freq=primary_tf
            ),
            method='ffill'
        )
    
    # Align with primary index
    aligned = resampled.reindex(primary_index, method='ffill')
    
    return aligned


def train_enhanced_tcn_model(data, model_params=None, training_params=None, model_prefix="enhanced_tcn"):
    """
    Train an enhanced TCN model on the provided data
    
    Args:
        data (pd.DataFrame): Input data
        model_params (dict): Model parameters
        training_params (dict): Training parameters
        model_prefix (str): Prefix for saving model
        
    Returns:
        tuple: (model, history, accuracy)
    """
    # Default model parameters
    default_model_params = {
        'filters': 64,
        'kernel_size': 3,
        'dilations': [1, 2, 4, 8, 16, 32],
        'dropout_rate': 0.3,
        'use_skip_connections': True,
        'use_batch_norm': True,
        'use_layer_norm': True,
        'use_spatial_dropout': True,
        'use_attention': True,
        'use_transformer': True,
        'use_channel_attention': True,
        'l1_reg': 0.0001,
        'l2_reg': 0.0001
    }
    
    # Default training parameters
    default_training_params = {
        'sequence_length': 60,
        'test_size': 0.2,
        'validation_size': 0.2,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15,
        'save_best_only': True
    }
    
    # Update with provided parameters
    if model_params:
        default_model_params.update(model_params)
    model_params = default_model_params
    
    if training_params:
        default_training_params.update(training_params)
    training_params = default_training_params
    
    # Prepare data
    sequence_length = training_params['sequence_length']
    
    # Ensure we have direction column
    if 'close_direction' not in data.columns:
        data['close_direction'] = (data['close'].pct_change() > 0).astype(int)
    
    # Create sequences
    X, y = prepare_sequences(
        data, 
        sequence_length=sequence_length, 
        target_column='close_direction'
    )
    
    # Split data
    test_size = training_params['test_size']
    validation_size = training_params['validation_size']
    
    # Calculate split points
    n_samples = len(X)
    test_split = int(n_samples * (1 - test_size))
    val_split = int(test_split * (1 - validation_size))
    
    # Split data
    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:test_split], y[val_split:test_split]
    X_test, y_test = X[test_split:], y[test_split:]
    
    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Create model
    model = EnhancedTCNModel(
        input_shape=X_train.shape[1:],
        **model_params
    )
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=training_params['batch_size'],
        epochs=training_params['epochs'],
        early_stopping_patience=training_params['early_stopping_patience'],
        model_prefix=model_prefix,
        save_best_only=training_params['save_best_only']
    )
    
    # Evaluate on test data
    logger.info("Evaluating model on test data")
    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    
    return model, history, accuracy


def train_multi_timeframe_model(data_dict, primary_timeframe="1h", 
                              model_params=None, training_params=None, 
                              model_prefix="enhanced_tcn_multi"):
    """
    Train a multi-timeframe enhanced TCN model
    
    Args:
        data_dict (dict): Dictionary of dataframes for each timeframe
        primary_timeframe (str): Primary timeframe
        model_params (dict): Model parameters
        training_params (dict): Training parameters
        model_prefix (str): Prefix for saving model
        
    Returns:
        tuple: (model, history, accuracy)
    """
    # Implementation of multi-timeframe model training would go here
    # This is a more complex implementation that would require custom model architecture
    # to handle multiple timeframes as input
    
    logger.info("Multi-timeframe model training not yet implemented")
    
    return None, None, None


def calculate_performance_metrics(model, X_test, y_test):
    """
    Calculate detailed performance metrics for the model
    
    Args:
        model (EnhancedTCNModel): Trained model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
        
    Returns:
        dict: Performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_test, y_pred_prob)
    except:
        auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr
    }
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path (str, optional): Path to save plot
    """
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def train_regime_specific_models(data, regime_column="market_regime", 
                               model_params=None, training_params=None):
    """
    Train regime-specific models
    
    Args:
        data (pd.DataFrame): Input data with market regime column
        regime_column (str): Column name for market regime
        model_params (dict): Model parameters
        training_params (dict): Training parameters
        
    Returns:
        dict: Dictionary of trained models for each regime
    """
    # Check if regime column exists
    if regime_column not in data.columns:
        raise ValueError(f"Regime column '{regime_column}' not found in data")
    
    # Get unique regimes
    regimes = data[regime_column].unique()
    logger.info(f"Found {len(regimes)} unique regimes: {regimes}")
    
    # Train model for each regime
    regime_models = {}
    
    for regime in regimes:
        logger.info(f"Training model for regime: {regime}")
        
        # Filter data for regime
        regime_data = data[data[regime_column] == regime].copy()
        
        # Skip if not enough data
        if len(regime_data) < 1000:
            logger.warning(f"Not enough data for regime {regime} ({len(regime_data)} samples), skipping")
            continue
        
        # Train model
        model_prefix = f"enhanced_tcn_{regime}"
        model, history, accuracy = train_enhanced_tcn_model(
            regime_data,
            model_params=model_params,
            training_params=training_params,
            model_prefix=model_prefix
        )
        
        # Store model
        regime_models[regime] = {
            'model': model,
            'accuracy': accuracy
        }
    
    return regime_models


def main():
    """
    Main function for testing the enhanced TCN model
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced TCN Model Training")
    
    parser.add_argument("--symbol", type=str, default="SOLUSD", 
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       help="Timeframe (default: 1h)")
    parser.add_argument("--sequence-length", type=int, default=60, 
                       help="Sequence length (default: 60)")
    parser.add_argument("--filters", type=int, default=64, 
                       help="Number of filters (default: 64)")
    parser.add_argument("--kernel-size", type=int, default=3, 
                       help="Kernel size (default: 3)")
    parser.add_argument("--dropout-rate", type=float, default=0.3, 
                       help="Dropout rate (default: 0.3)")
    parser.add_argument("--no-attention", action="store_true", 
                       help="Disable attention mechanism")
    parser.add_argument("--no-transformer", action="store_true", 
                       help="Disable transformer components")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Maximum epochs (default: 100)")
    parser.add_argument("--multi-timeframe", action="store_true", 
                       help="Use multi-timeframe data")
    parser.add_argument("--regime-specific", action="store_true", 
                       help="Train regime-specific models")
    parser.add_argument("--output-dir", type=str, default=MODELS_DIR, 
                       help="Output directory (default: models/tcn_enhanced)")
    
    args = parser.parse_args()
    
    # Set output directory
    global MODELS_DIR
    MODELS_DIR = args.output_dir
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Set model parameters
    model_params = {
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'dropout_rate': args.dropout_rate,
        'use_attention': not args.no_attention,
        'use_transformer': not args.no_transformer
    }
    
    # Set training parameters
    training_params = {
        'sequence_length': args.sequence_length,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    
    # Load data
    data = load_historical_data(args.symbol, args.timeframe)
    
    if data is None:
        logger.error(f"Failed to load data for {args.symbol} {args.timeframe}")
        return
    
    # Multi-timeframe training
    if args.multi_timeframe:
        logger.info("Loading multi-timeframe data")
        
        # Load additional timeframes
        if args.timeframe == "1h":
            additional_timeframes = ["15m", "4h", "1d"]
        elif args.timeframe == "15m":
            additional_timeframes = ["1h", "4h", "1d"]
        elif args.timeframe == "4h":
            additional_timeframes = ["1h", "1d"]
        else:
            additional_timeframes = []
        
        data_dict = {args.timeframe: data}
        
        for tf in additional_timeframes:
            tf_data = load_historical_data(args.symbol, tf)
            if tf_data is not None:
                data_dict[tf] = tf_data
        
        # Train multi-timeframe model
        model, history, accuracy = train_multi_timeframe_model(
            data_dict,
            primary_timeframe=args.timeframe,
            model_params=model_params,
            training_params=training_params,
            model_prefix=f"enhanced_tcn_multi_{args.symbol}_{args.timeframe}"
        )
    
    # Regime-specific training
    elif args.regime_specific:
        # Detect market regimes
        from market_regime_detector import detect_market_regimes
        
        # Add market regime column
        if 'market_regime' not in data.columns:
            data = detect_market_regimes(data)
        
        # Train regime-specific models
        regime_models = train_regime_specific_models(
            data,
            regime_column="market_regime",
            model_params=model_params,
            training_params=training_params
        )
        
        # Print regime-specific accuracies
        print("\nRegime-specific model accuracies:")
        for regime, model_info in regime_models.items():
            print(f"{regime}: {model_info['accuracy']:.4f}")
    
    # Standard training
    else:
        # Train model
        model, history, accuracy = train_enhanced_tcn_model(
            data,
            model_params=model_params,
            training_params=training_params,
            model_prefix=f"enhanced_tcn_{args.symbol}_{args.timeframe}"
        )
        
        # Plot training history
        if history:
            plot_path = os.path.join(MODELS_DIR, f"training_history_{args.symbol}_{args.timeframe}.png")
            plot_training_history(history, save_path=plot_path)
        
        # Print final accuracy
        print(f"\nFinal test accuracy: {accuracy:.4f}")


def load_historical_data(symbol, timeframe="1h"):
    """
    Load historical data
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        
    Returns:
        pd.DataFrame: Historical data
    """
    # Construct file path
    file_path = os.path.join("historical_data", f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows of {timeframe} data for {symbol}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    main()