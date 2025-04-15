#!/usr/bin/env python3
"""
Hybrid ML Models for Trading Bot

This module implements hybrid machine learning models that combine multiple
architectures (CNN, LSTM, GRU, TCN) for improved prediction accuracy in
the trading bot. These architectures leverage the strengths of different
neural network types to capture various patterns in the price data.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dropout, BatchNormalization, Input, Concatenate, Flatten,
    Bidirectional, TimeDistributed
)
from tensorflow.keras import regularizers
from typing import Dict, Any, List, Optional, Tuple

# Import custom implementations to avoid package dependency issues
from implement_tcn_architecture import TCNLayer
from custom_transformer import TransformerBlock, PositionalEncoding

class AttentionLayer(layers.Layer):
    """Custom attention layer for sequence data"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention weights
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights to input
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

def create_cnn_lstm_model(input_shape: Tuple[int, int],
                          cnn_filters: List[int] = [64, 128],
                          lstm_units: List[int] = [128, 64],
                          dropout_rate: float = 0.2,
                          l2_reg: float = 0.001,
                          learning_rate: float = 0.001,
                          is_classification: bool = True) -> Model:
    """
    Create a CNN-LSTM hybrid model for time series prediction
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        cnn_filters: List of filter sizes for CNN layers
        lstm_units: List of unit sizes for LSTM layers
        dropout_rate: Dropout rate
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer
        is_classification: Whether this is a classification task (True) or regression (False)
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = inputs
    for filters in cnn_filters:
        x = Conv1D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # Return sequences for all but the last layer
        x = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = BatchNormalization()(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    if is_classification:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae', 'mse']
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_tcn_gru_attention_model(input_shape: Tuple[int, int],
                                  tcn_filters: int = 64,
                                  tcn_kernel_size: int = 3,
                                  tcn_dilations: List[int] = None,
                                  gru_units: List[int] = [128, 64],
                                  dropout_rate: float = 0.2,
                                  l2_reg: float = 0.001,
                                  learning_rate: float = 0.001,
                                  is_classification: bool = True) -> Model:
    """
    Create a TCN-GRU-Attention hybrid model for time series prediction
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        tcn_filters: Number of filters for TCN
        tcn_kernel_size: Kernel size for TCN
        tcn_dilations: List of dilation rates (default [1, 2, 4, 8])
        gru_units: List of unit sizes for GRU layers
        dropout_rate: Dropout rate
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer
        is_classification: Whether this is a classification task (True) or regression (False)
        
    Returns:
        Compiled Keras model
    """
    if tcn_dilations is None:
        tcn_dilations = [1, 2, 4, 8]
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # TCN branch
    tcn_branch = TCNLayer(
        nb_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        dilations=tcn_dilations,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        return_sequences=True
    )(inputs)
    
    # GRU branch
    gru_branch = inputs
    for i, units in enumerate(gru_units):
        gru_branch = GRU(
            units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(gru_branch)
        gru_branch = BatchNormalization()(gru_branch)
    
    # Concatenate branches
    combined = Concatenate()([tcn_branch, gru_branch])
    
    # Attention layer
    x = AttentionLayer()(combined)
    
    # Dense layers
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    if is_classification:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae', 'mse']
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_transformer_lstm_model(input_shape: Tuple[int, int],
                                d_model: int = 64,
                                num_heads: int = 4,
                                transformer_layers: int = 2,
                                lstm_units: List[int] = [128, 64],
                                dropout_rate: float = 0.2,
                                l2_reg: float = 0.001,
                                learning_rate: float = 0.001,
                                is_classification: bool = True) -> Model:
    """
    Create a Transformer-LSTM hybrid model for time series prediction
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        d_model: Dimensionality of the transformer model
        num_heads: Number of attention heads
        transformer_layers: Number of transformer layers
        lstm_units: List of unit sizes for LSTM layers
        dropout_rate: Dropout rate
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer
        is_classification: Whether this is a classification task (True) or regression (False)
        
    Returns:
        Compiled Keras model
    """
    sequence_length, n_features = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Project inputs to d_model dimensions
    transformer_branch = Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    
    # Add positional encoding
    transformer_branch = PositionalEncoding(sequence_length, d_model)(transformer_branch)
    
    # Transformer layers
    for _ in range(transformer_layers):
        transformer_branch = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=d_model*4,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )(transformer_branch)
    
    # LSTM branch
    lstm_branch = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)  # Return sequences for all but the last layer
        lstm_branch = LSTM(
            units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate/2,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
    
    # Global pooling for transformer branch
    transformer_pooled = GlobalAveragePooling1D()(transformer_branch)
    
    # Concatenate branches
    combined = Concatenate()([transformer_pooled, lstm_branch])
    
    # Dense layers
    x = BatchNormalization()(combined)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    if is_classification:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae', 'mse']
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_ultra_ensemble_model(input_shape: Tuple[int, int],
                              dropout_rate: float = 0.2,
                              l2_reg: float = 0.001,
                              learning_rate: float = 0.001,
                              is_classification: bool = True) -> Model:
    """
    Create an ultra ensemble model combining TCN, LSTM, GRU, and Transformer
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        dropout_rate: Dropout rate
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer
        is_classification: Whether this is a classification task (True) or regression (False)
        
    Returns:
        Compiled Keras model
    """
    sequence_length, n_features = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # TCN branch
    tcn_branch = TCNLayer(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        return_sequences=True
    )(inputs)
    tcn_pooled = GlobalAveragePooling1D()(tcn_branch)
    
    # LSTM branch
    lstm_branch = LSTM(
        128,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = LSTM(
        64,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(lstm_branch)
    lstm_branch = BatchNormalization()(lstm_branch)
    
    # GRU branch
    gru_branch = GRU(
        128,
        return_sequences=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    gru_branch = BatchNormalization()(gru_branch)
    gru_branch = GRU(
        64,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate/2,
        kernel_regularizer=regularizers.l2(l2_reg)
    )(gru_branch)
    gru_branch = BatchNormalization()(gru_branch)
    
    # Transformer branch
    transformer_branch = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    transformer_branch = PositionalEncoding(sequence_length, 64)(transformer_branch)
    transformer_branch = TransformerBlock(
        d_model=64,
        num_heads=4,
        dff=256,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )(transformer_branch)
    transformer_pooled = GlobalAveragePooling1D()(transformer_branch)
    
    # CNN branch
    cnn_branch = Conv1D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_branch = Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(cnn_branch)
    cnn_branch = BatchNormalization()(cnn_branch)
    cnn_pooled = GlobalAveragePooling1D()(cnn_branch)
    
    # Concatenate all branches
    combined = Concatenate()([tcn_pooled, lstm_branch, gru_branch, transformer_pooled, cnn_pooled])
    
    # Dense layers with layer normalization
    x = LayerNormalization()(combined)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization()(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    if is_classification:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        outputs = Dense(1, activation='linear')(x)
        loss = 'mean_squared_error'
        metrics = ['mae', 'mse']
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

# Example of how to use
if __name__ == "__main__":
    # Example with small test data
    import numpy as np
    
    # Create sample data (64 samples, 60 time steps, 10 features)
    X = np.random.random((64, 60, 10))
    y = np.random.randint(0, 2, (64, 1))  # Binary targets
    
    # Create and compile CNN-LSTM model
    cnn_lstm_model = create_cnn_lstm_model(
        input_shape=(60, 10),
        is_classification=True
    )
    
    # Print model summary
    print("CNN-LSTM Model:")
    cnn_lstm_model.summary()
    
    # Train model for a few epochs
    cnn_lstm_model.fit(
        X, y,
        epochs=2,
        batch_size=8,
        validation_split=0.2
    )
    
    # Create and compile TCN-GRU-Attention model
    tcn_gru_model = create_tcn_gru_attention_model(
        input_shape=(60, 10),
        is_classification=True
    )
    
    # Print model summary
    print("\nTCN-GRU-Attention Model:")
    tcn_gru_model.summary()
    
    # Train model for a few epochs
    tcn_gru_model.fit(
        X, y,
        epochs=2,
        batch_size=8,
        validation_split=0.2
    )
    
    print("Training complete for sample data")