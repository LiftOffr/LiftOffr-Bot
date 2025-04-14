#!/usr/bin/env python3
"""
Prepare Enhanced Dataset for Dual Strategy ML Training

This script prepares enhanced datasets that combine features from both ARIMA and Adaptive
strategies, creating rich training data for the ML models. It processes historical data
and generates integrated datasets that enable the ML models to learn from both strategies
simultaneously.

Features:
1. Processes historical price data
2. Generates ARIMA strategy features and predictions
3. Generates Adaptive strategy features and predictions 
4. Calculates technical indicators and additional features
5. Creates target variables for ML training
6. Saves enhanced datasets for training

Usage:
    python prepare_enhanced_dataset.py --pair PAIR --timeframe TIMEFRAME [--output-dir OUTPUT_DIR]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_dataset.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = "training_data"
HISTORICAL_DATA_DIR = "historical_data"

class EnhancedDatasetPreparation:
    """
    Class for preparing enhanced datasets that combine ARIMA and Adaptive strategy features
    """
    def __init__(
        self,
        pair: str,
        timeframe: str = "1h",
        output_dir: str = DEFAULT_OUTPUT_DIR,
        min_samples: int = 200
    ):
        """
        Initialize the dataset preparation
        
        Args:
            pair: Trading pair (e.g., "SOL/USD")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            output_dir: Output directory for the dataset
            min_samples: Minimum number of samples required
        """
        self.pair = pair
        self.timeframe = timeframe
        self.output_dir = output_dir
        self.min_samples = min_samples
        
        # Create filename versions of the pair
        self.pair_filename = pair.replace("/", "")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize strategies (import at runtime to avoid circular imports)
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize ARIMA and Adaptive strategies for feature extraction"""
        try:
            # Import at runtime to avoid circular imports
            from fixed_strategy import ARIMAStrategy, AdaptiveStrategy
            
            self.arima_strategy = ARIMAStrategy(
                trading_pair=self.pair,
                timeframe=self.timeframe,
                max_leverage=125
            )
            
            self.adaptive_strategy = AdaptiveStrategy(
                trading_pair=self.pair,
                timeframe=self.timeframe,
                max_leverage=125
            )
            
            logger.info(f"Initialized ARIMA and Adaptive strategies for {self.pair}")
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            self.arima_strategy = None
            self.adaptive_strategy = None

    def load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load historical data for the trading pair
        
        Returns:
            DataFrame with historical data or None if data not found
        """
        # Construct file path - check nested directory structure first
        pair_dir_path = os.path.join(HISTORICAL_DATA_DIR, self.pair_filename)
        data_path = os.path.join(pair_dir_path, f"{self.pair_filename}_{self.timeframe}.csv")
        
        # If not found in nested structure, try flat structure as fallback
        if not os.path.exists(data_path):
            data_path = os.path.join(HISTORICAL_DATA_DIR, f"{self.pair_filename}_{self.timeframe}.csv")
        
        # Check if file exists in either location
        if not os.path.exists(data_path):
            logger.error(f"Historical data file not found: {data_path}")
            return None
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Convert timestamp to datetime if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Check if we have enough data
            if len(df) < self.min_samples:
                logger.warning(f"Not enough data for {self.pair}: {len(df)} < {self.min_samples}")
                return df
            
            logger.info(f"Loaded {len(df)} samples for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Basic indicators
            df["rsi"] = self._calculate_rsi(df["close"], window=14)
            df["ema9"] = self._calculate_ema(df["close"], window=9)
            df["ema21"] = self._calculate_ema(df["close"], window=21)
            df["ema50"] = self._calculate_ema(df["close"], window=50)
            df["ema100"] = self._calculate_ema(df["close"], window=100)
            df["atr"] = self._calculate_atr(df, window=14)
            df["pct_change"] = df["close"].pct_change()
            df["pct_change_8h"] = df["close"].pct_change(8)
            df["pct_change_24h"] = df["close"].pct_change(24)
            
            # Bollinger Bands
            df["bb_middle"], df["bb_upper"], df["bb_lower"] = self._calculate_bollinger_bands(df["close"], window=20)
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            
            # Keltner Channels
            df["kc_middle"], df["kc_upper"], df["kc_lower"] = self._calculate_keltner_channels(df, window=20)
            df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / df["kc_middle"]
            
            # Volatility measures
            df["volatility_1h"] = df["pct_change"].rolling(window=24).std()
            df["volatility_4h"] = df["pct_change"].rolling(window=96).std()
            df["volatility_1d"] = df["pct_change"].rolling(window=24*7).std()
            
            # Volume profiles
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            
            # Trend strength
            df["adx"] = self._calculate_adx(df, window=14)
            
            # Market regime
            df["bull_market"] = ((df["ema9"] > df["ema21"]) & (df["ema21"] > df["ema50"])).astype(int)
            df["bear_market"] = ((df["ema9"] < df["ema21"]) & (df["ema21"] < df["ema50"])).astype(int)
            df["neutral_market"] = 1 - df["bull_market"] - df["bear_market"]
            df["market_regime"] = (df["bull_market"] * 1) + (df["bear_market"] * -1) + (df["neutral_market"] * 0)
            
            # Calculate returns on different timeframes
            for period in [1, 2, 3, 4, 6, 8, 12, 24, 48, 96]:
                df[f"future_return_{period}"] = df["close"].pct_change(period).shift(-period)
            
            logger.info(f"Calculated technical indicators for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def generate_arima_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ARIMA strategy features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ARIMA features
        """
        try:
            if not hasattr(self, "arima_strategy") or self.arima_strategy is None:
                logger.warning("ARIMA strategy not initialized, skipping ARIMA features")
                return df
            
            # Initialize feature columns
            df["arima_prediction"] = 0
            df["arima_forecast"] = 0
            df["arima_strength"] = 0
            
            # Process data in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size].copy()
                
                # Get ARIMA predictions for this chunk
                for j in range(len(chunk)):
                    try:
                        candle_data = chunk.iloc[j:j+1].copy()
                        signal, forecast, strength = self.arima_strategy.generate_signal_for_candle(candle_data)
                        
                        # Convert signal to numeric value
                        signal_value = 1 if signal == "buy" else (-1 if signal == "sell" else 0)
                        
                        # Update values in the main dataframe
                        df.loc[chunk.index[j], "arima_prediction"] = signal_value
                        # Convert forecast to float to avoid compatibility issues
                        df.loc[chunk.index[j], "arima_forecast"] = float(forecast)
                        df.loc[chunk.index[j], "arima_strength"] = strength
                    except Exception as e:
                        # Just log and continue on errors
                        logger.debug(f"Error generating ARIMA signal for candle {j}: {e}")
            
            logger.info(f"Generated ARIMA features for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error generating ARIMA features: {e}")
            return df
    
    def generate_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Adaptive strategy features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Adaptive features
        """
        try:
            if not hasattr(self, "adaptive_strategy") or self.adaptive_strategy is None:
                logger.warning("Adaptive strategy not initialized, skipping Adaptive features")
                return df
            
            # Initialize feature columns
            df["adaptive_prediction"] = 0
            df["adaptive_strength"] = 0
            df["adaptive_volatility"] = 0
            
            # Process data in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size].copy()
                
                # Get Adaptive predictions for this chunk
                for j in range(len(chunk)):
                    try:
                        candle_data = chunk.iloc[j:j+1].copy()
                        signal, strength, volatility = self.adaptive_strategy.generate_signal_for_candle(candle_data)
                        
                        # Convert signal to numeric value
                        signal_value = 1 if signal == "buy" else (-1 if signal == "sell" else 0)
                        
                        # Update values in the main dataframe
                        df.loc[chunk.index[j], "adaptive_prediction"] = signal_value
                        df.loc[chunk.index[j], "adaptive_strength"] = strength
                        df.loc[chunk.index[j], "adaptive_volatility"] = volatility
                    except Exception as e:
                        # Just log and continue on errors
                        logger.debug(f"Error generating Adaptive signal for candle {j}: {e}")
            
            logger.info(f"Generated Adaptive features for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error generating Adaptive features: {e}")
            return df

    def generate_strategy_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features that represent interactions between strategies
        
        Args:
            df: DataFrame with strategy features
            
        Returns:
            DataFrame with added interaction features
        """
        try:
            # Check if strategy columns exist
            if "arima_prediction" not in df.columns or "adaptive_prediction" not in df.columns:
                logger.warning("Strategy prediction columns not found, skipping interaction features")
                return df
            
            # Agreement between strategies (1 if they agree, 0 if neutral, -1 if they disagree)
            df["strategy_agreement"] = np.sign(df["arima_prediction"] * df["adaptive_prediction"])
            
            # Combined strength (product of strengths, becomes negative if disagreement)
            df["strategy_combined_strength"] = df["arima_strength"] * df["adaptive_strength"] * df["strategy_agreement"]
            
            # Signal dominance (which strategy has the stronger signal)
            df["arima_dominance"] = (df["arima_strength"] > df["adaptive_strength"]).astype(int)
            df["adaptive_dominance"] = (df["adaptive_strength"] > df["arima_strength"]).astype(int)
            
            # Create a combined signal based on strength
            df["dominant_strategy"] = np.where(df["arima_strength"] > df["adaptive_strength"], "arima", "adaptive")
            
            # Create a combined prediction that uses the stronger signal
            df["combined_prediction"] = np.where(
                df["arima_strength"] > df["adaptive_strength"],
                df["arima_prediction"],
                df["adaptive_prediction"]
            )
            
            # Combined leverage opportunity (high when both strategies agree and are confident)
            df["leverage_opportunity"] = df["strategy_combined_strength"] * df["strategy_agreement"]
            
            logger.info(f"Generated strategy interaction features for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error generating strategy interaction features: {e}")
            return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for ML training
        
        Args:
            df: DataFrame with price and feature data
            
        Returns:
            DataFrame with added target variables
        """
        try:
            # Calculate price direction targets
            for period in [1, 2, 3, 4, 8, 12, 24]:
                # Future price change
                future_change = df["close"].shift(-period) - df["close"]
                # Binary direction (1 for up, 0 for down)
                df[f"target_direction_{period}"] = (future_change > 0).astype(int)
                # Threshold-based direction (1 for significant up, -1 for significant down, 0 for flat)
                threshold = df["atr"] * 0.5  # Use half ATR as significance threshold
                df[f"target_direction_thresh_{period}"] = 0
                df.loc[future_change > threshold, f"target_direction_thresh_{period}"] = 1
                df.loc[future_change < -threshold, f"target_direction_thresh_{period}"] = -1
                
                # Percentage change target (for regression)
                df[f"target_pct_change_{period}"] = future_change / df["close"] * 100
                
                # Volatility-adjusted return (Sharpe ratio style)
                vol = df["volatility_1h"].rolling(window=period).mean().shift(-period)
                vol_safe = vol.copy()
                vol_safe.loc[vol_safe < 0.0001] = 0.0001  # Avoid division by zero
                df[f"target_vol_adj_{period}"] = (future_change / df["close"]) / vol_safe
            
            # Calculate trading signals based on combined strategy
            df["target_signal"] = 0
            # Buy signal when both strategies agree on buy or ARIMA buy with high confidence
            buy_condition = ((df["arima_prediction"] > 0) & (df["adaptive_prediction"] > 0)) | \
                           ((df["arima_prediction"] > 0) & (df["arima_strength"] > 0.8))
            # Sell signal when both strategies agree on sell or ARIMA sell with high confidence
            sell_condition = ((df["arima_prediction"] < 0) & (df["adaptive_prediction"] < 0)) | \
                            ((df["arima_prediction"] < 0) & (df["arima_strength"] > 0.8))
            
            df.loc[buy_condition, "target_signal"] = 1
            df.loc[sell_condition, "target_signal"] = -1
            
            # Calculate optimal leverage based on future returns and strategy confidence
            for period in [4, 8, 12, 24]:
                future_return = df[f"future_return_{period}"]
                # Calculate ideal leverage (high leverage for high-confidence, profitable trades)
                confidence = (df["arima_strength"] + df["adaptive_strength"]) / 2
                
                # For positive returns, leverage scales with confidence
                positive_leverage = np.minimum(125 * confidence * (future_return > 0), 125)
                
                # For negative returns, inverse leverage (but limit to smaller values)
                negative_leverage = np.minimum(-5 * confidence * (future_return < 0), 0)
                
                # Combine the two
                df[f"target_leverage_{period}"] = positive_leverage + negative_leverage
                
                # Leverage adjustments: when returns are near zero, leverage should be zero
                small_return_mask = np.abs(future_return) < 0.001
                df.loc[small_return_mask, f"target_leverage_{period}"] = 0
            
            logger.info(f"Created target variables for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
            return df

    def prepare_enhanced_dataset(self) -> Optional[pd.DataFrame]:
        """
        Prepare the enhanced dataset combining all features
        
        Returns:
            DataFrame with the prepared dataset or None if preparation failed
        """
        try:
            # Step 1: Load historical data
            df = self.load_historical_data()
            if df is None or len(df) < 100:  # Basic validation
                return None
            
            # Step 2: Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Step 3: Generate ARIMA features
            df = self.generate_arima_features(df)
            
            # Step 4: Generate Adaptive features
            df = self.generate_adaptive_features(df)
            
            # Step 5: Generate strategy interaction features
            df = self.generate_strategy_interaction_features(df)
            
            # Step 6: Create target variables
            df = self.create_target_variables(df)
            
            # Step 7: Drop rows with NaN values
            df.dropna(inplace=True)
            
            logger.info(f"Prepared enhanced dataset with {len(df)} samples for {self.pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error preparing enhanced dataset: {e}")
            return None

    def save_dataset(self, df: pd.DataFrame) -> str:
        """
        Save the prepared dataset to CSV
        
        Args:
            df: DataFrame with the prepared dataset
            
        Returns:
            Path to the saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Construct output path
            output_path = os.path.join(self.output_dir, f"{self.pair_filename}_{self.timeframe}_enhanced.csv")
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved enhanced dataset to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return ""

    # Helper methods for technical indicators
    def _calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for a price series"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        relative_strength = gain / loss
        rsi = 100 - (100 / (1 + relative_strength))
        return rsi
    
    def _calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate EMA for a price series"""
        return data.ewm(span=window, adjust=False).mean()
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR for price data"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _calculate_bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands for a price series"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return sma, upper_band, lower_band
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, window: int = 20, atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels for price data"""
        ema = df["close"].ewm(span=window, adjust=False).mean()
        atr = self._calculate_atr(df, window)
        upper_channel = ema + (atr * atr_multiplier)
        lower_channel = ema - (atr * atr_multiplier)
        return ema, upper_channel, lower_channel
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX for price data"""
        # Placeholder for ADX calculation
        return pd.Series(50, index=df.index)  # Simplified for demonstration

def main():
    """Main function to prepare the enhanced dataset"""
    parser = argparse.ArgumentParser(description="Prepare enhanced dataset for dual strategy ML training")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g., 'SOL/USD')")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g., '1h', '4h', '1d')")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for the dataset")
    parser.add_argument("--min-samples", type=int, default=200, help="Minimum number of samples required")
    args = parser.parse_args()
    
    # Create dataset preparation object
    preparation = EnhancedDatasetPreparation(
        pair=args.pair,
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        min_samples=args.min_samples
    )
    
    # Prepare and save the dataset
    df = preparation.prepare_enhanced_dataset()
    if df is not None:
        # Still warn about limited data but proceed with saving
        if len(df) < args.min_samples:
            logger.warning(f"Limited data for {args.pair}: {len(df)} < {args.min_samples} samples, but proceeding anyway")
        
        output_path = preparation.save_dataset(df)
        if output_path:
            logger.info(f"Successfully prepared and saved enhanced dataset for {args.pair} to {output_path}")
            logger.info(f"Dataset contains {len(df)} samples with {len(df.columns)} features")
        else:
            logger.error(f"Failed to save dataset for {args.pair}")
    else:
        logger.error(f"Failed to prepare enhanced dataset for {args.pair}")

if __name__ == "__main__":
    main()