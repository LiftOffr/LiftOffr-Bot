#!/usr/bin/env python3
"""
Attention Mechanism for Hybrid Models

This module implements various attention mechanisms that can be used
in the hybrid model architecture for improved performance:

1. Self-Attention: Learns relationships between different time steps
2. Scaled Dot-Product Attention: From the Transformer architecture
3. Multi-Head Attention: Parallel attention layers with different projections
4. Temporal Attention: Focus on the most relevant time steps
5. Feature Attention: Focus on the most relevant features

These attention mechanisms can be combined with LSTM, GRU, and TCN models
to improve the model's ability to learn long-term dependencies and
focus on the most important parts of the input sequence.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Reshape, Permute, Multiply
from tensorflow.keras import backend as K

class SelfAttention(Layer):
    """
    Self-attention layer that learns relationships between different time steps
    
    Args:
        units (int): Number of attention units
    """
    def __init__(self, units=128, **kwargs):
        self.units = units
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W1 = self.add_weight(
            name="W1", 
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W2 = self.add_weight(
            name="W2", 
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        # ut shape: (batch_size, time_steps, units)
        ut = K.tanh(K.dot(inputs, self.W1))
        
        # et shape: (batch_size, time_steps, 1)
        et = K.dot(ut, self.W2)
        
        # at shape: (batch_size, time_steps, 1)
        at = K.softmax(et, axis=1)
        
        # output shape: (batch_size, features)
        output = K.sum(inputs * at, axis=1)
        
        return output, at
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], 1)
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"units": self.units})
        return config

class ScaledDotProductAttention(Layer):
    """
    Scaled dot-product attention from the Transformer architecture
    
    Args:
        d_k (int): Dimensionality of the key vectors
    """
    def __init__(self, d_k=64, **kwargs):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_q = self.add_weight(
            name="W_q",
            shape=(input_shape[-1], self.d_k),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W_k = self.add_weight(
            name="W_k",
            shape=(input_shape[-1], self.d_k),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W_v = self.add_weight(
            name="W_v",
            shape=(input_shape[-1], self.d_k),
            initializer="glorot_uniform",
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        q = K.dot(inputs, self.W_q)  # (batch_size, time_steps, d_k)
        k = K.dot(inputs, self.W_k)  # (batch_size, time_steps, d_k)
        v = K.dot(inputs, self.W_v)  # (batch_size, time_steps, d_k)
        
        # Scaled dot-product attention
        scores = K.batch_dot(q, k, axes=[2, 2]) / K.sqrt(K.cast(self.d_k, dtype=K.floatx()))  # (batch_size, time_steps, time_steps)
        weights = K.softmax(scores, axis=-1)  # (batch_size, time_steps, time_steps)
        output = K.batch_dot(weights, v)  # (batch_size, time_steps, d_k)
        
        return output, weights
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_k), (input_shape[0], input_shape[1], input_shape[1])
    
    def get_config(self):
        config = super(ScaledDotProductAttention, self).get_config()
        config.update({"d_k": self.d_k})
        return config

class MultiHeadAttention(Layer):
    """
    Multi-head attention from the Transformer architecture
    
    Args:
        num_heads (int): Number of attention heads
        d_model (int): Dimensionality of the model
    """
    def __init__(self, num_heads=8, d_model=256, **kwargs):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        super(MultiHeadAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W_q = self.add_weight(
            name="W_q",
            shape=(input_shape[-1], self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W_k = self.add_weight(
            name="W_k",
            shape=(input_shape[-1], self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W_v = self.add_weight(
            name="W_v",
            shape=(input_shape[-1], self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.W_o = self.add_weight(
            name="W_o",
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )
        self.built = True
    
    def split_heads(self, x):
        # x shape: (batch_size, time_steps, d_model)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))  # (batch_size, time_steps, num_heads, d_k)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, time_steps, d_k)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        batch_size = tf.shape(inputs)[0]
        
        q = K.dot(inputs, self.W_q)  # (batch_size, time_steps, d_model)
        k = K.dot(inputs, self.W_k)  # (batch_size, time_steps, d_model)
        v = K.dot(inputs, self.W_v)  # (batch_size, time_steps, d_model)
        
        q = self.split_heads(q)  # (batch_size, num_heads, time_steps, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, time_steps, d_k)
        v = self.split_heads(v)  # (batch_size, num_heads, time_steps, d_k)
        
        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) / K.sqrt(K.cast(self.d_k, dtype=K.floatx()))  # (batch_size, num_heads, time_steps, time_steps)
        weights = K.softmax(scores, axis=-1)  # (batch_size, num_heads, time_steps, time_steps)
        attention = tf.matmul(weights, v)  # (batch_size, num_heads, time_steps, d_k)
        
        # Concatenate heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, time_steps, num_heads, d_k)
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))  # (batch_size, time_steps, d_model)
        
        # Final linear projection
        output = K.dot(attention, self.W_o)  # (batch_size, time_steps, d_model)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "d_model": self.d_model
        })
        return config

class TemporalAttention(Layer):
    """
    Temporal attention layer that focuses on the most relevant time steps
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape = (batch_size, time_steps, features)
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[2], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            shape=(input_shape[1],),
            initializer="zeros",
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Calculate attention weights
        e = K.tanh(K.dot(inputs, self.W) + self.b)  # (batch_size, time_steps, 1)
        a = K.softmax(e, axis=1)  # (batch_size, time_steps, 1)
        
        # Apply attention weights
        output = inputs * a  # (batch_size, time_steps, features)
        
        return output, a
    
    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], input_shape[1], 1)
    
    def get_config(self):
        return super(TemporalAttention, self).get_config()

class FeatureAttention(Layer):
    """
    Feature attention layer that focuses on the most relevant features
    """
    def __init__(self, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape = (batch_size, time_steps, features)
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            shape=(input_shape[2],),
            initializer="zeros",
            trainable=True
        )
        self.built = True
    
    def call(self, inputs):
        # Permute input to (batch_size, features, time_steps)
        x = Permute((2, 1))(inputs)
        
        # Calculate attention weights
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch_size, features, 1)
        a = K.softmax(e, axis=1)  # (batch_size, features, 1)
        
        # Apply attention weights
        output = x * a  # (batch_size, features, time_steps)
        
        # Permute back to original shape
        output = Permute((2, 1))(output)  # (batch_size, time_steps, features)
        
        return output, a
    
    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], input_shape[2], 1)
    
    def get_config(self):
        return super(FeatureAttention, self).get_config()

def build_attention_lstm_model(input_shape, attention_type='self', return_sequences=False):
    """
    Build a LSTM model with the specified attention mechanism
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, features)
        attention_type (str): Type of attention mechanism to use
        return_sequences (bool): Whether to return the full sequence or just the last output
        
    Returns:
        model: Keras model with the specified attention mechanism
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # LSTM layer
    lstm_units = 128
    if return_sequences:
        lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    else:
        lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    
    # Apply attention
    if attention_type == 'self':
        attention_output, _ = SelfAttention()(lstm)
    elif attention_type == 'scaled_dot':
        attention_output, _ = ScaledDotProductAttention()(lstm)
    elif attention_type == 'multi_head':
        attention_output = MultiHeadAttention()(lstm)
    elif attention_type == 'temporal':
        attention_output, _ = TemporalAttention()(lstm)
    elif attention_type == 'feature':
        attention_output, _ = FeatureAttention()(lstm)
    else:
        attention_output = lstm
    
    # Dense layers
    if not return_sequences and attention_type not in ['self', 'scaled_dot']:
        attention_output = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    
    dense = tf.keras.layers.Dense(64, activation='relu')(attention_output)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(3, activation='softmax')(dropout)  # 3 classes: -1, 0, 1
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model

def build_hybrid_attention_model(input_shape, output_shape=3):
    """
    Build a hybrid model with multiple attention mechanisms
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, features)
        output_shape (int): Number of output classes
        
    Returns:
        model: Keras hybrid model with multiple attention mechanisms
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # 1. LSTM with Self-Attention branch
    lstm = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    lstm_attention_output, _ = SelfAttention()(lstm)
    
    # 2. GRU with Temporal Attention branch
    gru = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
    gru_attention, _ = TemporalAttention()(gru)
    gru_attention_output = tf.keras.layers.GlobalAveragePooling1D()(gru_attention)
    
    # 3. TCN branch (if available)
    try:
        from tcn import TCN
        tcn = TCN(nb_filters=64, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8], 
                  return_sequences=False)(inputs)
        has_tcn = True
    except ImportError:
        # Fallback to 1D CNN if TCN is not available
        conv1d = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
        pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1d)
        conv1d = tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu')(pool)
        tcn = tf.keras.layers.GlobalAveragePooling1D()(conv1d)
        has_tcn = False
    
    # 4. Transformer-style Multi-Head Attention branch
    transformer = MultiHeadAttention(num_heads=8, d_model=256)(inputs)
    transformer_output = tf.keras.layers.GlobalAveragePooling1D()(transformer)
    
    # Merge branches
    if has_tcn:
        merged = tf.keras.layers.Concatenate()([lstm_attention_output, gru_attention_output, tcn, transformer_output])
    else:
        merged = tf.keras.layers.Concatenate()([lstm_attention_output, gru_attention_output, tcn, transformer_output])
    
    # Meta-learner (Dense layers)
    dense = tf.keras.layers.Dense(128, activation='relu')(merged)
    bn = tf.keras.layers.BatchNormalization()(dense)
    dropout = tf.keras.layers.Dropout(0.5)(bn)
    dense = tf.keras.layers.Dense(64, activation='relu')(dropout)
    bn = tf.keras.layers.BatchNormalization()(dense)
    dropout = tf.keras.layers.Dropout(0.3)(bn)
    
    # Output layer
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(dropout)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model