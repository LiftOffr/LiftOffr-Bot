#!/usr/bin/env python3
"""
Custom Transformer Architecture for Trading Bot

This module implements a custom Transformer architecture for time series prediction
in the trading bot, focusing on capturing long-range dependencies and temporal patterns
in price data.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras import regularizers
from typing import Dict, Any, List, Optional, Tuple

class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer models"""
    
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        """
        Initialize positional encoding
        
        Args:
            sequence_length: Length of input sequences
            d_model: Dimensionality of the model
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((1, sequence_length, d_model))
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices=[[0, i, j] for i in range(sequence_length) for j in range(0, d_model, 2)],
            updates=tf.reshape(tf.sin(position * div_term), [-1])
        )
        
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices=[[0, i, j] for i in range(sequence_length) for j in range(1, d_model, 2)],
            updates=tf.reshape(tf.cos(position * div_term), [-1])
        )
        
        self.pe = pe
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        """Get layer configuration"""
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config

class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dff: int, 
                 dropout_rate: float = 0.1, 
                 l2_reg: float = 0.001,
                 **kwargs):
        """
        Initialize transformer block
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
            dff: Dimensionality of feed-forward network
            dropout_rate: Dropout rate
            l2_reg: L2 regularization coefficient
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Multi-head attention
        self.mha = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))
        ])
        
        # Normalization and dropout layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass for the transformer block"""
        # Multi-head attention
        attention_output = self.mha(inputs, inputs, inputs, mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        """Get layer configuration"""
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config

def create_transformer_model(input_shape: Tuple[int, int],
                           d_model: int = 64,
                           num_heads: int = 4,
                           num_encoder_layers: int = 2,
                           dff: int = 128,
                           dropout_rate: float = 0.1,
                           l2_reg: float = 0.001,
                           learning_rate: float = 0.001,
                           is_classification: bool = True) -> Model:
    """
    Create a Transformer model for time series prediction
    
    Args:
        input_shape: Input shape (sequence_length, n_features)
        d_model: Dimensionality of the model
        num_heads: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dff: Dimensionality of feed-forward network
        dropout_rate: Dropout rate
        l2_reg: L2 regularization coefficient
        learning_rate: Learning rate for optimizer
        is_classification: Whether this is a classification task (True) or regression (False)
        
    Returns:
        Compiled Keras model
    """
    sequence_length, n_features = input_shape
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Project inputs to d_model dimensions
    x = Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    
    # Add positional encoding
    x = PositionalEncoding(sequence_length, d_model)(x)
    
    # Transformer encoder layers
    for _ in range(num_encoder_layers):
        x = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )(x)
    
    # Global average pooling across time dimension
    x = layers.GlobalAveragePooling1D()(x)
    
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

# Example of how to use
if __name__ == "__main__":
    # Example with small test data
    import numpy as np
    
    # Create sample data (64 samples, 60 time steps, 10 features)
    X = np.random.random((64, 60, 10))
    y = np.random.randint(0, 2, (64, 1))  # Binary targets
    
    # Create and compile model
    model = create_transformer_model(
        input_shape=(60, 10),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        dropout_rate=0.1,
        is_classification=True
    )
    
    # Print model summary
    model.summary()
    
    # Train model for a few epochs
    model.fit(
        X, y,
        epochs=5,
        batch_size=8,
        validation_split=0.2
    )
    
    print("Training complete for sample data")