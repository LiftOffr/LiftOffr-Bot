#!/usr/bin/env python3
"""
ML Data Integrator for Kraken Trading Bot

This module handles the integration of historical data from multiple sources and timeframes,
prepares it for model training, and generates unified datasets for all model types.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
MODELS_DIR = "models"
SEQUENCE_LENGTH = 60  # Default sequence length for time series data
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class MLDataIntegrator:
    """
    Handles the integration and processing of historical market data from multiple
    sources and timeframes for use in machine learning model training.
    """
    def __init__(self, trading_pair="SOL/USD"):
        """
        Initialize the ML data integrator
        
        Args:
            trading_pair (str): Trading pair to process data for
        """
        self.trading_pair = trading_pair
        self.ticker_symbol = trading_pair.replace("/", "")
        self.data = {}
        self.integrated_data = None
        self.feature_names = []
    
    def load_data(self, timeframes=None):
        """
        Load historical data for all specified timeframes
        
        Args:
            timeframes (list): List of timeframes to load (e.g. ["15m", "1h", "4h"])
                If None, loads all available timeframes
        
        Returns:
            dict: Dictionary of dataframes for each timeframe
        """
        # If no timeframes specified, detect available ones
        if timeframes is None:
            timeframes = []
            for filename in os.listdir(DATA_DIR):
                if filename.startswith(f"{self.ticker_symbol}_") and filename.endswith(".csv"):
                    timeframe = filename.split("_")[1].split(".")[0]
                    timeframes.append(timeframe)
        
        logger.info(f"Loading data for {self.trading_pair} with timeframes: {timeframes}")
        
        # Load each timeframe
        for timeframe in timeframes:
            filepath = os.path.join(DATA_DIR, f"{self.ticker_symbol}_{timeframe}.csv")
            
            if os.path.exists(filepath):
                try:
                    # Load CSV file
                    df = pd.read_csv(filepath)
                    
                    # Convert timestamp to datetime if needed
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    # Convert string values to float if needed
                    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Store the dataframe
                    self.data[timeframe] = df
                    logger.info(f"Loaded {len(df)} rows for {timeframe}")
                
                except Exception as e:
                    logger.error(f"Error loading data for {timeframe}: {e}")
                    traceback.print_exc()
            else:
                logger.warning(f"Data file for {timeframe} not found at {filepath}")
        
        return self.data
    
    def calculate_technical_indicators(self, df, timeframe):
        """
        Calculate technical indicators for a dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            timeframe (str): Timeframe of the data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Basic indicators
        df_copy[f'returns_{timeframe}'] = df_copy['close'].pct_change()
        df_copy[f'log_returns_{timeframe}'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Volatility (standard deviation of log returns)
        df_copy[f'volatility_{timeframe}'] = df_copy[f'log_returns_{timeframe}'].rolling(window=20).std()
        
        # Moving averages
        for period in [9, 21, 50, 100, 200]:
            df_copy[f'sma{period}_{timeframe}'] = df_copy['close'].rolling(window=period).mean()
            df_copy[f'ema{period}_{timeframe}'] = df_copy['close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df_copy[f'macd_{timeframe}'] = df_copy['close'].ewm(span=12, adjust=False).mean() - df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy[f'macd_signal_{timeframe}'] = df_copy[f'macd_{timeframe}'].ewm(span=9, adjust=False).mean()
        df_copy[f'macd_hist_{timeframe}'] = df_copy[f'macd_{timeframe}'] - df_copy[f'macd_signal_{timeframe}']
        
        # RSI
        delta = df_copy['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
        df_copy[f'rsi_{timeframe}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df_copy[f'bb_middle_{timeframe}'] = df_copy['close'].rolling(window=20).mean()
        df_copy[f'bb_std_{timeframe}'] = df_copy['close'].rolling(window=20).std()
        df_copy[f'bb_upper_{timeframe}'] = df_copy[f'bb_middle_{timeframe}'] + 2 * df_copy[f'bb_std_{timeframe}']
        df_copy[f'bb_lower_{timeframe}'] = df_copy[f'bb_middle_{timeframe}'] - 2 * df_copy[f'bb_std_{timeframe}']
        df_copy[f'bb_width_{timeframe}'] = (df_copy[f'bb_upper_{timeframe}'] - df_copy[f'bb_lower_{timeframe}']) / df_copy[f'bb_middle_{timeframe}']
        
        # Price relative to Bollinger Bands
        df_copy[f'bb_pct_b_{timeframe}'] = (df_copy['close'] - df_copy[f'bb_lower_{timeframe}']) / (df_copy[f'bb_upper_{timeframe}'] - df_copy[f'bb_lower_{timeframe}'])
        
        # ATR (Average True Range)
        tr1 = df_copy['high'] - df_copy['low']
        tr2 = abs(df_copy['high'] - df_copy['close'].shift(1))
        tr3 = abs(df_copy['low'] - df_copy['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_copy[f'atr_{timeframe}'] = tr.rolling(window=14).mean()
        
        # ADX (Average Directional Index)
        # +DM and -DM
        plus_dm = df_copy['high'].diff()
        minus_dm = df_copy['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # +DI and -DI
        plus_di = 100 * plus_dm.rolling(window=14).mean() / df_copy[f'atr_{timeframe}'].replace(0, 1e-9)
        minus_di = 100 * minus_dm.rolling(window=14).mean() / df_copy[f'atr_{timeframe}'].replace(0, 1e-9)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
        df_copy[f'adx_{timeframe}'] = dx.rolling(window=14).mean()
        
        # Clean up NaN values
        df_copy.dropna(inplace=True)
        
        return df_copy
    
    def process_datasets(self):
        """
        Process all loaded datasets by adding technical indicators
        
        Returns:
            dict: Dictionary of processed dataframes
        """
        # Process each timeframe dataset
        processed_data = {}
        
        for timeframe, df in self.data.items():
            logger.info(f"Processing data for {timeframe}...")
            processed_df = self.calculate_technical_indicators(df, timeframe)
            processed_data[timeframe] = processed_df
            logger.info(f"Processed {len(processed_df)} rows for {timeframe}")
        
        self.data = processed_data
        return processed_data
    
    def integrate_timeframes(self, primary_timeframe="1h"):
        """
        Integrate data from multiple timeframes into a single dataframe
        
        Args:
            primary_timeframe (str): Primary timeframe to use as the base
        
        Returns:
            pd.DataFrame: Integrated dataframe with data from all timeframes
        """
        if primary_timeframe not in self.data:
            logger.error(f"Primary timeframe {primary_timeframe} not loaded.")
            return None
        
        # Use the primary timeframe as the base
        integrated_df = self.data[primary_timeframe].copy()
        
        # Get column prefixes for the primary timeframe
        primary_prefixes = set()
        for col in integrated_df.columns:
            if f"_{primary_timeframe}" in col:
                prefix = col.split(f"_{primary_timeframe}")[0]
                primary_prefixes.add(prefix)
        
        # Add data from other timeframes
        for timeframe, df in self.data.items():
            if timeframe == primary_timeframe:
                continue
            
            # Get column prefixes for this timeframe
            timeframe_prefixes = set()
            for col in df.columns:
                if f"_{timeframe}" in col:
                    prefix = col.split(f"_{timeframe}")[0]
                    timeframe_prefixes.add(prefix)
            
            # Find common prefixes
            common_prefixes = primary_prefixes.intersection(timeframe_prefixes)
            
            # Reindex the dataset to match the primary timeframe
            reindexed_df = df.reindex(integrated_df.index, method='ffill')
            
            # Add the columns with common prefixes to the integrated dataframe
            for prefix in common_prefixes:
                col_name = f"{prefix}_{timeframe}"
                if col_name in reindexed_df.columns:
                    integrated_df[col_name] = reindexed_df[col_name]
        
        # Drop any rows with NaN values
        integrated_df.dropna(inplace=True)
        
        # Store the integrated data
        self.integrated_data = integrated_df
        logger.info(f"Integrated data has {len(integrated_df)} rows and {len(integrated_df.columns)} features")
        
        return integrated_df
    
    def prepare_training_data(self, integrated_df, target_column=None, sequence_length=SEQUENCE_LENGTH):
        """
        Prepare integrated data for model training
        
        Args:
            integrated_df (pd.DataFrame): Integrated dataframe with all features
            target_column (str): Column to use as prediction target (default: close_1h)
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        """
        if integrated_df is None:
            logger.error("No integrated data available.")
            return None, None, None, None, None, None, None
        
        # Default target column
        if target_column is None:
            # Look for close_1h column
            for col in integrated_df.columns:
                if col.startswith("close_") and col.endswith("h"):
                    target_column = col
                    break
            
            # If still not found, use the first 'close' column
            if target_column is None:
                for col in integrated_df.columns:
                    if col.startswith("close_"):
                        target_column = col
                        break
            
            # If still not found, use regular close
            if target_column is None and "close" in integrated_df.columns:
                target_column = "close"
        
        if target_column not in integrated_df.columns:
            logger.error(f"Target column {target_column} not found in data.")
            return None, None, None, None, None, None, None
        
        logger.info(f"Using {target_column} as target column for prediction")
        
        # Create return-based target (percentage change for next period)
        integrated_df["target_return"] = integrated_df[target_column].pct_change().shift(-1)
        
        # Create direction-based target (1 for up, 0 for down)
        integrated_df["target_direction"] = (integrated_df["target_return"] > 0).astype(int)
        
        # Remove NaN values
        integrated_df.dropna(inplace=True)
        
        # Collect feature columns (exclude target columns)
        features = integrated_df.columns.tolist()
        features.remove("target_return")
        features.remove("target_direction")
        
        # Store feature names
        self.feature_names = features
        
        # Create sequences
        X = []
        y_return = []
        y_direction = []
        
        for i in range(len(integrated_df) - sequence_length):
            X.append(integrated_df[features].iloc[i:i+sequence_length].values)
            y_return.append(integrated_df["target_return"].iloc[i+sequence_length])
            y_direction.append(integrated_df["target_direction"].iloc[i+sequence_length])
        
        # Convert to arrays
        X = np.array(X)
        y_return = np.array(y_return)
        y_direction = np.array(y_direction)
        
        # Scale X using MinMaxScaler for each feature
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape((n_samples * n_timesteps, n_features))
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape((n_samples, n_timesteps, n_features))
        
        # Save scaler for future use
        os.makedirs(os.path.join(MODELS_DIR, "scalers"), exist_ok=True)
        np.save(os.path.join(MODELS_DIR, "scalers", "feature_scaler_params.npy"), 
                (scaler.data_min_, scaler.data_max_))
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_return_temp, y_return_test, y_direction_temp, y_direction_test = train_test_split(
            X_scaled, y_return, y_direction, test_size=TEST_SPLIT, shuffle=False
        )
        
        X_train, X_val, y_train_return, y_val_return, y_train_direction, y_val_direction = train_test_split(
            X_temp, y_return_temp, y_direction_temp, 
            test_size=VAL_SPLIT/(TRAIN_SPLIT + VAL_SPLIT), 
            shuffle=False
        )
        
        logger.info(f"Created {len(X_train)} training samples, {len(X_val)} validation samples, and {len(X_test)} test samples")
        
        return (X_train, y_train_return, y_train_direction, 
                X_val, y_val_return, y_val_direction,
                X_test, y_return_test, y_direction_test,
                features)
    
    def save_processed_data(self, data_tuple, model_type):
        """
        Save processed data for a specific model type
        
        Args:
            data_tuple (tuple): Tuple of processed data
            model_type (str): Type of model (tcn, cnn, lstm, etc.)
            
        Returns:
            bool: Success status
        """
        if data_tuple is None or len(data_tuple) != 10:
            logger.error("Invalid data tuple for saving.")
            return False
        
        try:
            (X_train, y_train_return, y_train_direction, 
             X_val, y_val_return, y_val_direction,
             X_test, y_test_return, y_test_direction,
             features) = data_tuple
            
            # Create directory for this model type
            model_dir = os.path.join(MODELS_DIR, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save data
            np.save(os.path.join(model_dir, "X_train.npy"), X_train)
            np.save(os.path.join(model_dir, "y_train_return.npy"), y_train_return)
            np.save(os.path.join(model_dir, "y_train_direction.npy"), y_train_direction)
            np.save(os.path.join(model_dir, "X_val.npy"), X_val)
            np.save(os.path.join(model_dir, "y_val_return.npy"), y_val_return)
            np.save(os.path.join(model_dir, "y_val_direction.npy"), y_val_direction)
            np.save(os.path.join(model_dir, "X_test.npy"), X_test)
            np.save(os.path.join(model_dir, "y_test_return.npy"), y_test_return)
            np.save(os.path.join(model_dir, "y_test_direction.npy"), y_test_direction)
            
            # Save feature names
            with open(os.path.join(model_dir, "feature_names.json"), 'w') as f:
                json.dump(features, f)
            
            logger.info(f"Saved processed data for {model_type} model")
            return True
        
        except Exception as e:
            logger.error(f"Error saving processed data for {model_type}: {e}")
            traceback.print_exc()
            return False
    
    def save_train_stats(self, model, history, model_type, suffix=""):
        """
        Save model training statistics
        
        Args:
            model: Trained model
            history: Training history
            model_type (str): Type of model (tcn, cnn, lstm, etc.)
            suffix (str): Optional suffix for file names
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory for this model type
            model_dir = os.path.join(MODELS_DIR, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_file = os.path.join(model_dir, f"{self.ticker_symbol}{suffix}.h5")
            model.save(model_file)
            
            # Save training history
            history_file = os.path.join(model_dir, f"{self.ticker_symbol}_history{suffix}.json")
            with open(history_file, 'w') as f:
                json.dump(history.history, f)
            
            logger.info(f"Saved model and training history for {model_type}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model and history for {model_type}: {e}")
            traceback.print_exc()
            return False
    
    def save_evaluation_results(self, results, model_type, suffix=""):
        """
        Save model evaluation results
        
        Args:
            results (dict): Evaluation results
            model_type (str): Type of model (tcn, cnn, lstm, etc.)
            suffix (str): Optional suffix for file names
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory for this model type
            model_dir = os.path.join(MODELS_DIR, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save evaluation results
            eval_file = os.path.join(model_dir, f"{self.ticker_symbol}_evaluation{suffix}.json")
            with open(eval_file, 'w') as f:
                json.dump(results, f)
            
            logger.info(f"Saved evaluation results for {model_type}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving evaluation results for {model_type}: {e}")
            traceback.print_exc()
            return False


def main():
    """
    Process historical data and prepare it for model training
    """
    trading_pair = "SOL/USD"
    timeframe = "1h"
    
    # Initialize data integrator
    integrator = MLDataIntegrator(trading_pair)
    
    # Load historical data
    data = integrator.load_data()
    
    if not data:
        logger.error("No data loaded. Please run fetch_historical_data.sh first.")
        return
    
    # Process data with technical indicators
    processed_data = integrator.process_datasets()
    
    # Integrate data from all timeframes
    integrated_data = integrator.integrate_timeframes(primary_timeframe=timeframe)
    
    if integrated_data is None:
        logger.error("Failed to integrate data.")
        return
    
    # Prepare training data
    training_data = integrator.prepare_training_data(integrated_data)
    
    if training_data[0] is None:
        logger.error("Failed to prepare training data.")
        return
    
    # Save processed data for each model type
    model_types = ["tcn", "cnn", "lstm", "gru", "bilstm", "attention", "transformer", "hybrid"]
    
    for model_type in model_types:
        integrator.save_processed_data(training_data, model_type)
    
    logger.info("Data processing complete and ready for model training.")


if __name__ == "__main__":
    main()