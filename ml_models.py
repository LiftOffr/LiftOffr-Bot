"""
Machine Learning Models for Trading Bot
Includes TCN, CNN, and LSTM models for market prediction.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Conv1D, Dropout, BatchNormalization, 
    Input, Concatenate, GlobalAveragePooling1D, MaxPooling1D,
    Bidirectional, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)

# Constants
LOOK_BACK = 60  # Number of time steps to look back
FORECAST_STEPS = 5  # Number of steps to forecast
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

class TCNBlock(tf.keras.layers.Layer):
    """Temporal Convolutional Network block with causal convolutions"""
    
    def __init__(self, filters, kernel_size, dilation_rate=1, dropout_rate=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.batch_norm1 = BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.dropout1 = Dropout(dropout_rate)
        
        self.conv2 = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate
        )
        self.batch_norm2 = BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        self.dropout2 = Dropout(dropout_rate)
        
        self.residual = tf.keras.layers.Conv1D(filters=filters, kernel_size=1) if True else None
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        if self.residual is not None:
            res = self.residual(inputs)
            x = tf.keras.layers.add([x, res])
        
        return x

class MarketMLModel:
    """Combined model with TCN, CNN, and LSTM branches"""
    
    def __init__(self, sequence_length=LOOK_BACK, n_features=8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.tcn_model = None
        self.cnn_model = None
        self.lstm_model = None
        self.combined_model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def build_tcn_model(self):
        """Build a Temporal Convolutional Network model"""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        x = TCNBlock(filters=64, kernel_size=3, dilation_rate=1)(inputs)
        x = TCNBlock(filters=64, kernel_size=3, dilation_rate=2)(x)
        x = TCNBlock(filters=64, kernel_size=3, dilation_rate=4)(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(FORECAST_STEPS, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.tcn_model = model
        return model
    
    def build_cnn_model(self):
        """Build a Convolutional Neural Network model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
                   input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(FORECAST_STEPS, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.cnn_model = model
        return model
    
    def build_lstm_model(self):
        """Build an LSTM model for sequential data"""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), 
                         input_shape=(self.sequence_length, self.n_features)),
            LayerNormalization(),
            Dropout(0.3),
            
            Bidirectional(LSTM(64, return_sequences=False)),
            LayerNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dense(FORECAST_STEPS, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.lstm_model = model
        return model
    
    def build_combined_model(self):
        """Build a combined model with TCN, CNN, and LSTM branches"""
        # Ensure individual models are built
        if not self.tcn_model:
            self.build_tcn_model()
        if not self.cnn_model:
            self.build_cnn_model()
        if not self.lstm_model:
            self.build_lstm_model()
            
        # Create combined model architecture
        input_layer = Input(shape=(self.sequence_length, self.n_features))
        
        # TCN branch
        tcn_layer = self.tcn_model.layers[1](input_layer)
        for i in range(2, len(self.tcn_model.layers)-2):
            tcn_layer = self.tcn_model.layers[i](tcn_layer)
        tcn_output = GlobalAveragePooling1D()(tcn_layer)
        
        # CNN branch
        cnn_layer = self.cnn_model.layers[0](input_layer)
        for i in range(1, len(self.cnn_model.layers)-4):  # Exclude final Dense layers
            cnn_layer = self.cnn_model.layers[i](cnn_layer)
        cnn_output = GlobalAveragePooling1D()(cnn_layer)
        
        # LSTM branch
        lstm_layer = self.lstm_model.layers[0](input_layer)
        for i in range(1, len(self.lstm_model.layers)-3):  # Exclude final Dense layers
            lstm_layer = self.lstm_model.layers[i](lstm_layer)
        lstm_output = lstm_layer  # Already flattened
        
        # Combine branches
        combined = Concatenate()([tcn_output, cnn_output, lstm_output])
        
        # Final layers
        x = Dense(64, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(FORECAST_STEPS, activation='linear')(x)
        
        # Create and compile model
        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        
        self.combined_model = model
        return model
    
    def prepare_data(self, market_data, target_column='close'):
        """
        Prepare market data for training
        
        Args:
            market_data (pd.DataFrame): Market data with columns like open, high, low, close, volume
            target_column (str): Column to predict (default: 'close')
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        # Make sure data is sorted by time
        data = market_data.copy()
        
        # Create features
        features = data.drop(columns=['timestamp'] if 'timestamp' in data.columns else [])
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Create target (future prices for next FORECAST_STEPS)
        targets = np.zeros((len(data) - FORECAST_STEPS, FORECAST_STEPS))
        
        target_idx = data.columns.get_loc(target_column)
        for i in range(len(data) - FORECAST_STEPS):
            targets[i] = data.iloc[i:i+FORECAST_STEPS, target_idx].values
        
        # Scale targets
        scaled_targets = self.target_scaler.fit_transform(targets)
        
        # Create sequences of LOOK_BACK length
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length - FORECAST_STEPS + 1):
            X.append(scaled_features[i:i + self.sequence_length])
            y.append(scaled_targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        return X_train, X_val, y_train, y_val
    
    def train_models(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train all models
        
        Args:
            X_train, y_train, X_val, y_val: Training and validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Training history
        """
        # Ensure all models are built
        if not self.tcn_model:
            self.build_tcn_model()
        if not self.cnn_model:
            self.build_cnn_model()
        if not self.lstm_model:
            self.build_lstm_model()
        if not self.combined_model:
            self.build_combined_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(f'{MODELS_DIR}/tcn_model.h5', save_best_only=True, monitor='val_loss'),
            ModelCheckpoint(f'{MODELS_DIR}/cnn_model.h5', save_best_only=True, monitor='val_loss'),
            ModelCheckpoint(f'{MODELS_DIR}/lstm_model.h5', save_best_only=True, monitor='val_loss'),
            ModelCheckpoint(f'{MODELS_DIR}/combined_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        # Train TCN model
        logging.info("Training TCN model...")
        tcn_history = self.tcn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks[0:2],
            verbose=1
        )
        
        # Train CNN model
        logging.info("Training CNN model...")
        cnn_history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks[0:1] + callbacks[2:3],
            verbose=1
        )
        
        # Train LSTM model
        logging.info("Training LSTM model...")
        lstm_history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks[0:1] + callbacks[3:4],
            verbose=1
        )
        
        # Train combined model
        logging.info("Training combined model...")
        combined_history = self.combined_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks[0:1] + callbacks[4:],
            verbose=1
        )
        
        return {
            'tcn': tcn_history.history,
            'cnn': cnn_history.history,
            'lstm': lstm_history.history,
            'combined': combined_history.history
        }
    
    def predict(self, market_data, model_type='combined'):
        """
        Make predictions with specified model type
        
        Args:
            market_data (pd.DataFrame): Market data, recent observations
            model_type (str): Model to use for prediction ('tcn', 'cnn', 'lstm', or 'combined')
            
        Returns:
            np.array: Predicted values
        """
        # Prepare data
        features = market_data.copy()
        if 'timestamp' in features.columns:
            features = features.drop(columns=['timestamp'])
        
        # Scale features
        scaled_features = self.feature_scaler.transform(features.values)
        
        # Create sequence
        if len(scaled_features) >= self.sequence_length:
            sequence = scaled_features[-self.sequence_length:]
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)
        else:
            # Pad if we don't have enough data
            padding = np.zeros((self.sequence_length - len(scaled_features), self.n_features))
            sequence = np.vstack([padding, scaled_features])
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)
        
        # Choose model
        if model_type.lower() == 'tcn':
            model = self.tcn_model
        elif model_type.lower() == 'cnn':
            model = self.cnn_model
        elif model_type.lower() == 'lstm':
            model = self.lstm_model
        else:
            model = self.combined_model
        
        # Make prediction
        scaled_prediction = model.predict(sequence)
        
        # Inverse transform
        prediction = self.target_scaler.inverse_transform(scaled_prediction)
        
        return prediction[0]  # Return as 1D array
    
    def load_models(self, models_dir=MODELS_DIR):
        """Load all models from disk"""
        try:
            self.tcn_model = tf.keras.models.load_model(f'{models_dir}/tcn_model.h5')
            self.cnn_model = tf.keras.models.load_model(f'{models_dir}/cnn_model.h5')
            self.lstm_model = tf.keras.models.load_model(f'{models_dir}/lstm_model.h5')
            self.combined_model = tf.keras.models.load_model(f'{models_dir}/combined_model.h5')
            logging.info("All models loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
    
    def save_scalers(self, models_dir=MODELS_DIR):
        """Save scalers to disk"""
        try:
            import joblib
            joblib.dump(self.feature_scaler, f'{models_dir}/feature_scaler.pkl')
            joblib.dump(self.target_scaler, f'{models_dir}/target_scaler.pkl')
            logging.info("Scalers saved successfully")
            return True
        except Exception as e:
            logging.error(f"Error saving scalers: {str(e)}")
            return False
    
    def load_scalers(self, models_dir=MODELS_DIR):
        """Load scalers from disk"""
        try:
            import joblib
            self.feature_scaler = joblib.load(f'{models_dir}/feature_scaler.pkl')
            self.target_scaler = joblib.load(f'{models_dir}/target_scaler.pkl')
            logging.info("Scalers loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading scalers: {str(e)}")
            return False


# Helper function to prepare training data from trades.csv
def prepare_market_data_from_trades(csv_file='trades.csv'):
    """
    Prepare market data from trades.csv for ML model training
    
    Args:
        csv_file (str): Path to trades.csv file
        
    Returns:
        pd.DataFrame: Prepared market data
    """
    try:
        # Load trades data
        trades_df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Sort by timestamp
        trades_df = trades_df.sort_values('timestamp')
        
        # Resample to consistent intervals (e.g., 15 minutes)
        trades_df.set_index('timestamp', inplace=True)
        market_data = trades_df.resample('15T').agg({
            'price': 'ohlc',
            'amount': 'sum'
        })
        
        # Flatten column names
        market_data.columns = ['_'.join(col).strip() for col in market_data.columns.values]
        
        # Rename columns to standard names
        market_data.rename(columns={
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'amount_sum': 'volume'
        }, inplace=True)
        
        # Reset index to get timestamp as column
        market_data.reset_index(inplace=True)
        
        # Calculate additional features
        market_data['price_change'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['high'] - market_data['low']
        market_data['range_pct'] = market_data['volatility'] / market_data['close']
        
        # Drop rows with NaN
        market_data.dropna(inplace=True)
        
        return market_data
    
    except Exception as e:
        logging.error(f"Error preparing market data: {str(e)}")
        return None


if __name__ == "__main__":
    # Test model creation
    model = MarketMLModel(sequence_length=LOOK_BACK, n_features=8)
    model.build_tcn_model()
    model.build_cnn_model()
    model.build_lstm_model()
    model.build_combined_model()
    
    # Print model summaries
    print("TCN Model Summary:")
    model.tcn_model.summary()
    
    print("\nCNN Model Summary:")
    model.cnn_model.summary()
    
    print("\nLSTM Model Summary:")
    model.lstm_model.summary()
    
    print("\nCombined Model Summary:")
    model.combined_model.summary()