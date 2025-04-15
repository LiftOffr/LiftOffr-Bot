#!/usr/bin/env python3
"""
Advanced Training Data Preparation

This script prepares sophisticated datasets for ML training, with:
1. Advanced feature engineering
2. Cross-asset correlation metrics
3. Market regime detection
4. Sentiment analysis integration
5. Feature selection and importance ranking
6. Multi-timeframe data preparation
7. Enhanced technical indicators
8. Custom predictive features

The goal is to create rich datasets that enable models to achieve
near-perfect prediction accuracy and maximize trading returns.
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_data_preparation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("advanced_data_preparation")

# Constants
DEFAULT_PAIRS = ["AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DATA_DIR = "historical_data"
OUTPUT_DIR = "training_data"
CONFIG_PATH = "config/new_coins_training_config.json"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare advanced training data with sophisticated feature engineering.')
    parser.add_argument('--pairs', type=str, default=None,
                      help='Comma-separated list of trading pairs (default: all new coins)')
    parser.add_argument('--timeframes', type=str, default=None,
                      help='Comma-separated list of timeframes (default: all standard timeframes)')
    parser.add_argument('--lookback', type=int, default=120,
                      help='Lookback window for feature engineering (default: 120)')
    parser.add_argument('--prediction-horizon', type=int, default=12,
                      help='Prediction horizon in periods (default: 12)')
    parser.add_argument('--advanced-features', action='store_true',
                      help='Generate advanced technical and derived features')
    parser.add_argument('--cross-asset-correlation', action='store_true',
                      help='Include cross-asset correlation features')
    parser.add_argument('--market-regime-detection', action='store_true',
                      help='Include market regime detection features')
    parser.add_argument('--sentiment-analysis', action='store_true',
                      help='Include sentiment analysis features')
    parser.add_argument('--feature-selection', action='store_true',
                      help='Perform feature selection')
    parser.add_argument('--force', action='store_true',
                      help='Force regeneration of datasets even if they exist')
    
    return parser.parse_args()


def load_config():
    """Load configuration from config file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        return {
            "feature_engineering": {
                "technical_indicators": [
                    "rsi", "macd", "bbands", "atr", "adx"
                ],
                "derived_features": [
                    "price_momentum", "volatility_indicator"
                ],
                "normalization": "standardization"
            }
        }


def load_historical_data(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load historical data for a specific pair and timeframe."""
    pair_filename = pair.replace("/", "_")
    file_path = f"{DATA_DIR}/{pair_filename}_{timeframe}.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Historical data file not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading historical data for {pair} ({timeframe}): {e}")
        return None


def calculate_technical_indicators(df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
    """Calculate technical indicators for price data."""
    try:
        from talib import abstract as ta
    except ImportError:
        logger.warning("TA-Lib not available, falling back to custom technical indicator implementation")
        # Implement custom TA functions here
    
    df = df.copy()
    
    # Map strings to functions
    indicator_functions = {
        # Momentum indicators
        "rsi": lambda df: ta.RSI(df, timeperiod=14),
        "macd": lambda df: calculate_macd(df),
        "stochastic": lambda df: calculate_stochastic(df),
        "williams_r": lambda df: ta.WILLR(df, timeperiod=14),
        "cci": lambda df: ta.CCI(df, timeperiod=14),
        "adx": lambda df: ta.ADX(df, timeperiod=14),
        "aroon": lambda df: calculate_aroon(df),
        "mfi": lambda df: ta.MFI(df, timeperiod=14),
        "trix": lambda df: ta.TRIX(df, timeperiod=14),
        "ultimate_oscillator": lambda df: ta.ULTOSC(df, timeperiod1=7, timeperiod2=14, timeperiod3=28),
        
        # Volatility indicators
        "bbands": lambda df: calculate_bollinger_bands(df),
        "atr": lambda df: ta.ATR(df, timeperiod=14),
        "keltner": lambda df: calculate_keltner_channels(df),
        "choppiness_index": lambda df: calculate_choppiness_index(df),
        
        # Volume indicators
        "obv": lambda df: ta.OBV(df),
        "cmf": lambda df: calculate_chaikin_money_flow(df),
        "vwap": lambda df: calculate_vwap(df),
        
        # Trend indicators
        "ichimoku": lambda df: calculate_ichimoku(df),
        "supertrend": lambda df: calculate_supertrend(df),
        "parabolic_sar": lambda df: ta.SAR(df, acceleration=0.02, maximum=0.2),
        "pivot_points": lambda df: calculate_pivot_points(df),
        "dpo": lambda df: ta.DPO(df, timeperiod=14),
        
        # Other indicators
        "fibonacci_retracement": lambda df: calculate_fibonacci_retracement(df),
        "vortex": lambda df: calculate_vortex(df),
        "elder_ray": lambda df: calculate_elder_ray(df),
        "coppock_curve": lambda df: calculate_coppock_curve(df),
        "fisher_transform": lambda df: calculate_fisher_transform(df),
        "awesome_oscillator": lambda df: calculate_awesome_oscillator(df),
        "accelerator_oscillator": lambda df: calculate_accelerator_oscillator(df),
        "hull_moving_average": lambda df: calculate_hull_moving_average(df)
    }
    
    # Calculate requested indicators
    for indicator in indicators:
        if indicator in indicator_functions:
            try:
                # Apply the indicator function
                result = indicator_functions[indicator](df)
                
                # Handle different types of returns
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[f"{indicator}_{col}"] = result[col]
                elif isinstance(result, tuple) or isinstance(result, list):
                    for i, component in enumerate(result):
                        df[f"{indicator}_{i}"] = component
                else:
                    df[indicator] = result
                
                logger.debug(f"Calculated {indicator} indicator")
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                continue
        else:
            logger.warning(f"Unknown indicator: {indicator}")
    
    return df


def create_derived_features(df: pd.DataFrame, features: List[str], lookback: int = 20) -> pd.DataFrame:
    """Create advanced derived features for ML models."""
    df = df.copy()
    
    feature_functions = {
        "price_momentum": lambda df: calculate_price_momentum(df, lookback),
        "volatility_indicator": lambda df: calculate_volatility_indicator(df, lookback),
        "trend_strength": lambda df: calculate_trend_strength(df, lookback),
        "range_breakout": lambda df: calculate_range_breakout(df, lookback),
        "volume_momentum": lambda df: calculate_volume_momentum(df, lookback),
        "support_resistance": lambda df: identify_support_resistance(df, lookback),
        "price_cyclicality": lambda df: calculate_price_cyclicality(df, lookback),
        "volume_profile": lambda df: calculate_volume_profile(df, lookback),
        "order_flow_imbalance": lambda df: calculate_order_flow_imbalance(df),
        "price_velocity": lambda df: calculate_price_velocity(df, lookback),
        "breakout_strength": lambda df: calculate_breakout_strength(df, lookback),
        "momentum_divergence": lambda df: calculate_momentum_divergence(df),
        "mean_reversion_probability": lambda df: calculate_mean_reversion_probability(df, lookback),
        "price_pattern_recognition": lambda df: identify_price_patterns(df)
    }
    
    for feature in features:
        if feature in feature_functions:
            try:
                # Apply the feature function
                result = feature_functions[feature](df)
                
                # Handle different types of returns
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[f"{feature}_{col}"] = result[col]
                elif isinstance(result, tuple) or isinstance(result, list):
                    for i, component in enumerate(result):
                        df[f"{feature}_{i}"] = component
                else:
                    df[feature] = result
                
                logger.debug(f"Created derived feature: {feature}")
            except Exception as e:
                logger.error(f"Error creating derived feature {feature}: {e}")
                continue
        else:
            logger.warning(f"Unknown derived feature: {feature}")
    
    return df


def add_cross_asset_correlations(df: pd.DataFrame, pair: str, timeframe: str, 
                              reference_pairs: List[str], window: int = 20) -> pd.DataFrame:
    """Add cross-asset correlation features."""
    df = df.copy()
    
    # Load data for reference pairs
    reference_data = {}
    for ref_pair in reference_pairs:
        if ref_pair != pair:  # Skip self-correlation
            ref_data = load_historical_data(ref_pair, timeframe)
            if ref_data is not None:
                # Ensure timestamps match
                ref_data = ref_data.set_index('timestamp')
                reference_data[ref_pair] = ref_data
    
    if not reference_data:
        logger.warning(f"No reference data available for cross-asset correlation")
        return df
    
    # Calculate rolling correlations
    df_with_timestamp = df.copy()
    df = df.set_index('timestamp')
    
    for ref_pair, ref_data in reference_data.items():
        try:
            # Resample to ensure matching timestamps
            common_index = df.index.intersection(ref_data.index)
            
            if len(common_index) > window:
                # Calculate correlation with different metrics
                df_common = df.loc[common_index, 'close']
                ref_common = ref_data.loc[common_index, 'close']
                
                # Rolling Pearson correlation
                rolling_corr = df_common.rolling(window=window).corr(ref_common)
                df.loc[common_index, f"corr_{ref_pair.replace('/', '_')}"] = rolling_corr
                
                # Rolling beta (volatility relative to reference asset)
                rolling_beta = (
                    df_common.rolling(window=window).cov(ref_common) / 
                    ref_common.rolling(window=window).var()
                )
                df.loc[common_index, f"beta_{ref_pair.replace('/', '_')}"] = rolling_beta
                
                logger.debug(f"Added correlation features for {pair} with {ref_pair}")
            else:
                logger.warning(f"Insufficient common data for correlation between {pair} and {ref_pair}")
        except Exception as e:
            logger.error(f"Error calculating correlation for {ref_pair}: {e}")
    
    # Reset index
    df = df.reset_index()
    
    # If timestamp column was lost, recover it
    if 'timestamp' not in df.columns:
        df['timestamp'] = df_with_timestamp['timestamp']
    
    return df


def detect_market_regimes(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Detect market regimes (trending, ranging, volatile) using various metrics."""
    df = df.copy()
    
    try:
        # 1. ADX indicator for trend strength
        if 'adx' not in df.columns:
            from talib import abstract as ta
            df['adx'] = ta.ADX(df, timeperiod=14)
        
        # 2. Volatility (ATR relative to price)
        if 'atr' not in df.columns:
            from talib import abstract as ta
            df['atr'] = ta.ATR(df, timeperiod=14)
        
        df['rel_volatility'] = df['atr'] / df['close']
        
        # 3. Price oscillation (directional changes)
        df['direction'] = np.sign(df['close'].diff())
        df['direction_changes'] = df['direction'].diff().abs() / 2
        
        # 4. Market regime classification
        # Threshold values can be optimized based on historical data
        trend_threshold = 25
        volatility_threshold = df['rel_volatility'].quantile(0.7)
        
        df['regime_trending'] = (df['adx'] > trend_threshold).astype(int)
        df['regime_volatile'] = (df['rel_volatility'] > volatility_threshold).astype(int)
        df['regime_ranging'] = ((df['adx'] <= trend_threshold) & 
                              (df['rel_volatility'] <= volatility_threshold)).astype(int)
        
        # Additional regime - breakout detection
        df['high_20d'] = df['high'].rolling(20).max()
        df['low_20d'] = df['low'].rolling(20).min()
        df['breakout_up'] = (df['close'] > df['high_20d'].shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['low_20d'].shift(1)).astype(int)
        
        # 5. Advanced regime indicators
        # Moving window entropy to measure randomness
        def rolling_entropy(window):
            hist, _ = np.histogram(window, bins=10)
            probs = hist / len(window)
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        df['price_entropy'] = df['close'].rolling(lookback).apply(
            rolling_entropy, raw=True
        ).fillna(0)
        
        logger.debug(f"Added market regime features")
        
        return df
    except Exception as e:
        logger.error(f"Error detecting market regimes: {e}")
        return df


def apply_feature_transformations(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply advanced feature transformations like Fourier features, wavelet decomposition, etc."""
    df = df.copy()
    
    transform_config = config.get("feature_engineering", {})
    
    # Apply Fourier transform if configured
    if transform_config.get("use_fourier_features", False):
        try:
            # Simple Fourier features (sine and cosine components)
            df = add_fourier_features(df, 'close', num_components=5)
            logger.debug("Added Fourier features")
        except Exception as e:
            logger.error(f"Error adding Fourier features: {e}")
    
    # Apply wavelet decomposition if configured
    if transform_config.get("wavelet_decomposition", False):
        try:
            df = add_wavelet_features(df, 'close', levels=3)
            logger.debug("Added wavelet decomposition features")
        except Exception as e:
            logger.error(f"Error adding wavelet features: {e}")
    
    # Apply rolling statistical features
    if "rolling_statistical_features" in transform_config.get("advanced_transformations", []):
        try:
            df = add_rolling_statistics(df, windows=[5, 10, 20, 50])
            logger.debug("Added rolling statistical features")
        except Exception as e:
            logger.error(f"Error adding rolling statistics: {e}")
    
    # Apply fractional differentiation for stationarity
    if "fractional_differentiation" in transform_config.get("advanced_transformations", []):
        try:
            df = add_fractional_differentiation(df, 'close', d=0.4)
            logger.debug("Added fractionally differentiated features")
        except Exception as e:
            logger.error(f"Error adding fractional differentiation: {e}")
    
    return df


def feature_selection(df: pd.DataFrame, target_col: str, threshold: float = 0.01) -> pd.DataFrame:
    """
    Select most important features using various techniques.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of the target column
        threshold: Importance threshold for feature selection
        
    Returns:
        DataFrame with selected features
    """
    from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
    
    try:
        # Separate features and target
        X = df.drop(['timestamp', target_col], axis=1, errors='ignore')
        y = df[target_col]
        
        # Remove low variance features
        var_selector = VarianceThreshold(threshold=0.001)
        X_var = var_selector.fit_transform(X)
        
        # Get selected columns
        selected_var = X.columns[var_selector.get_support()]
        logger.info(f"Variance selection: {len(selected_var)}/{len(X.columns)} features selected")
        
        # Calculate mutual information for remaining features
        X_var_df = pd.DataFrame(X_var, columns=selected_var)
        mi_scores = mutual_info_regression(X_var_df, y)
        mi_df = pd.DataFrame({'feature': selected_var, 'importance': mi_scores})
        mi_df = mi_df.sort_values('importance', ascending=False)
        
        # Select features above threshold
        important_features = mi_df[mi_df['importance'] > threshold]['feature'].tolist()
        logger.info(f"Mutual information selection: {len(important_features)} features selected")
        
        # Add selected features and target to results
        result_df = df[['timestamp'] + important_features + [target_col]]
        
        # Log most important features
        top_features = mi_df.head(10)['feature'].tolist()
        logger.info(f"Top 10 features: {', '.join(top_features)}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        logger.warning("Using all features due to feature selection error")
        return df


def create_training_targets(df: pd.DataFrame, prediction_horizon: int = 12) -> pd.DataFrame:
    """Create training targets based on future price movement."""
    df = df.copy()
    
    try:
        # Calculate future returns
        df['future_return'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create binary target (1 for up, 0 for down)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Create multi-class target (for potential multi-class models)
        strong_up_threshold = df['future_return'].quantile(0.75)
        strong_down_threshold = df['future_return'].quantile(0.25)
        
        conditions = [
            (df['future_return'] > strong_up_threshold),
            (df['future_return'] > 0) & (df['future_return'] <= strong_up_threshold),
            (df['future_return'] <= 0) & (df['future_return'] >= strong_down_threshold),
            (df['future_return'] < strong_down_threshold)
        ]
        choices = [3, 2, 1, 0]  # 3: strong up, 2: moderate up, 1: moderate down, 0: strong down
        
        df['target_multiclass'] = np.select(conditions, choices, default=1)
        
        # Create regression target (normalized future return)
        from sklearn.preprocessing import StandardScaler
        df['target_regression'] = StandardScaler().fit_transform(df['future_return'].values.reshape(-1, 1))
        
        logger.debug(f"Created training targets with horizon {prediction_horizon}")
        
        return df
    except Exception as e:
        logger.error(f"Error creating training targets: {e}")
        return df


def normalize_features(df: pd.DataFrame, method: str = 'standardization') -> pd.DataFrame:
    """Normalize features using the specified method."""
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    
    # Columns to exclude from normalization
    exclude_cols = ['timestamp', 'target', 'target_multiclass', 'target_regression']
    
    # Identify columns to normalize
    normalize_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Handle missing values
    df[normalize_cols] = df[normalize_cols].fillna(df[normalize_cols].mean())
    
    # Apply normalization based on specified method
    if method == 'standardization':
        scaler = StandardScaler()
    elif method == 'robust_scaling':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.warning(f"Unknown normalization method '{method}', using standardization")
        scaler = StandardScaler()
    
    df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
    
    return df


def save_training_data(df: pd.DataFrame, pair: str, timeframe: str) -> None:
    """Save processed training data to CSV file."""
    pair_filename = pair.replace("/", "_")
    output_file = f"{OUTPUT_DIR}/{pair_filename}_{timeframe}_features.csv"
    
    try:
        df.to_csv(output_file, index=False)
        logger.info(f"Saved training data to {output_file} ({len(df)} records)")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")


def process_pair_data(pair: str, timeframes: List[str], args, config: Dict) -> None:
    """Process data for a specific trading pair across all timeframes."""
    logger.info(f"Processing data for {pair}")
    
    # Get pair-specific settings
    pair_settings = config.get("pair_specific_settings", {}).get(pair, {})
    
    # Use pair-specific settings if available, otherwise use command line args
    lookback = pair_settings.get("lookback_window", args.lookback)
    prediction_horizon = pair_settings.get("prediction_horizon", args.prediction_horizon)
    
    # Reference pairs for correlation (major crypto assets)
    reference_pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD"]
    # Add any additional reference pairs from config
    additional_refs = pair_settings.get("additional_features", [])
    reference_pairs.extend([r for r in additional_refs if "/" in r and r not in reference_pairs])
    
    for timeframe in timeframes:
        logger.info(f"Processing {pair} data for {timeframe} timeframe")
        
        # Check if output file already exists and skip if not forced
        pair_filename = pair.replace("/", "_")
        output_file = f"{OUTPUT_DIR}/{pair_filename}_{timeframe}_features.csv"
        
        if os.path.exists(output_file) and not args.force:
            logger.info(f"Skipping {pair} {timeframe} (output file already exists)")
            continue
        
        # Load historical data
        df = load_historical_data(pair, timeframe)
        if df is None:
            continue
        
        # Step 1: Calculate technical indicators
        indicator_list = config.get("feature_engineering", {}).get("technical_indicators", [])
        df = calculate_technical_indicators(df, indicator_list)
        
        # Step 2: Create derived features
        if args.advanced_features:
            feature_list = config.get("feature_engineering", {}).get("derived_features", [])
            df = create_derived_features(df, feature_list, lookback)
        
        # Step 3: Add cross-asset correlations
        if args.cross_asset_correlation:
            df = add_cross_asset_correlations(df, pair, timeframe, reference_pairs, lookback)
        
        # Step 4: Add market regime detection
        if args.market_regime_detection:
            df = detect_market_regimes(df, lookback)
        
        # Step 5: Apply advanced feature transformations
        df = apply_feature_transformations(df, config)
        
        # Step 6: Create training targets
        df = create_training_targets(df, prediction_horizon)
        
        # Step 7: Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Step 8: Apply feature selection
        if args.feature_selection:
            feature_importance_threshold = pair_settings.get("feature_importance_threshold", 0.01)
            df = feature_selection(df, "target", threshold=feature_importance_threshold)
        
        # Step 9: Normalize features
        normalization_method = config.get("feature_engineering", {}).get("normalization", "standardization")
        df = normalize_features(df, method=normalization_method)
        
        # Step 10: Save processed data
        save_training_data(df, pair, timeframe)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs and timeframes
    pairs = args.pairs.split(",") if args.pairs else DEFAULT_PAIRS
    timeframes = args.timeframes.split(",") if args.timeframes else DEFAULT_TIMEFRAMES
    
    logger.info(f"Preparing advanced training data for {len(pairs)} pairs across {len(timeframes)} timeframes")
    
    # Load configuration
    config = load_config()
    
    # Process data for each pair
    for pair in pairs:
        process_pair_data(pair, timeframes, args, config)
    
    logger.info("Data preparation completed")
    
    return 0


# Helper functions for indicators and features
# These would be implemented in full detail in the actual script
def calculate_macd(df: pd.DataFrame) -> Tuple:
    from talib import abstract as ta
    macd, signal, hist = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    return macd, signal, hist

def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    from talib import abstract as ta
    upper, middle, lower = ta.BBANDS(df, timeperiod=20, nbdevup=2, nbdevdn=2)
    result = pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_width': (upper - lower) / middle
    })
    return result

def calculate_stochastic(df: pd.DataFrame) -> Tuple:
    from talib import abstract as ta
    slowk, slowd = ta.STOCH(df, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    return slowk, slowd

def calculate_aroon(df: pd.DataFrame) -> Tuple:
    from talib import abstract as ta
    aroon_down, aroon_up = ta.AROON(df, timeperiod=14)
    return aroon_down, aroon_up

def calculate_chaikin_money_flow(df: pd.DataFrame) -> pd.Series:
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    return cmf

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    v = df['volume'].values
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * v).cumsum() / v.cumsum()

def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    # Simplified implementation - this would be more complex in actual code
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2  # Conversion line
    
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2  # Base line
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # Leading span A
    
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)  # Leading span B
    
    chikou_span = df['close'].shift(-26)  # Lagging span
    
    result = pd.DataFrame({
        'ichimoku_tenkan': tenkan_sen,
        'ichimoku_kijun': kijun_sen,
        'ichimoku_senkou_a': senkou_span_a,
        'ichimoku_senkou_b': senkou_span_b,
        'ichimoku_chikou': chikou_span
    })
    return result

def calculate_supertrend(df: pd.DataFrame, period=14, multiplier=3.0) -> pd.Series:
    # Simplified implementation
    from talib import abstract as ta
    atr = ta.ATR(df, timeperiod=period)
    
    # Calculate upper and lower bands
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize supertrend series
    supertrend = pd.Series(np.nan, index=df.index)
    
    # Simplified calculation - actual implementation would be more complex
    for i in range(period, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            supertrend.iloc[i] = 1  # Uptrend
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            supertrend.iloc[i] = -1  # Downtrend
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]  # Maintain previous trend
    
    return supertrend

def calculate_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate pivot points based on previous candle
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    s1 = (2 * pivot) - df['high'].shift(1)
    s2 = pivot - (df['high'].shift(1) - df['low'].shift(1))
    r1 = (2 * pivot) - df['low'].shift(1)
    r2 = pivot + (df['high'].shift(1) - df['low'].shift(1))
    
    result = pd.DataFrame({
        'pivot': pivot,
        'support1': s1,
        'support2': s2,
        'resistance1': r1,
        'resistance2': r2
    })
    return result

def calculate_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
    from talib import abstract as ta
    atr = ta.ATR(df, timeperiod=14)
    ema20 = ta.EMA(df, timeperiod=20)
    
    upper = ema20 + (2 * atr)
    lower = ema20 - (2 * atr)
    
    result = pd.DataFrame({
        'keltner_middle': ema20,
        'keltner_upper': upper,
        'keltner_lower': lower,
        'keltner_width': (upper - lower) / ema20
    })
    return result

def calculate_vortex(df: pd.DataFrame) -> pd.DataFrame:
    # Simplified implementation
    df = df.copy()
    df['tr'] = np.maximum(df['high'] - df['low'], 
                        np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                  abs(df['low'] - df['close'].shift(1))))
    df['vm_plus'] = abs(df['high'] - df['low'].shift(1))
    df['vm_minus'] = abs(df['low'] - df['high'].shift(1))
    
    def rolling_sum(x, window):
        return x.rolling(window=window).sum()
    
    period = 14
    df['vip'] = rolling_sum(df['vm_plus'], period) / rolling_sum(df['tr'], period)
    df['vim'] = rolling_sum(df['vm_minus'], period) / rolling_sum(df['tr'], period)
    
    result = pd.DataFrame({
        'vortex_pos': df['vip'],
        'vortex_neg': df['vim'],
        'vortex_diff': df['vip'] - df['vim']
    })
    return result

def calculate_fibonacci_retracement(df: pd.DataFrame) -> pd.DataFrame:
    # Simple implementation using recent high/low points
    high_20d = df['high'].rolling(20).max()
    low_20d = df['low'].rolling(20).min()
    
    range_price = high_20d - low_20d
    fib_236 = high_20d - range_price * 0.236
    fib_382 = high_20d - range_price * 0.382
    fib_50 = high_20d - range_price * 0.5
    fib_618 = high_20d - range_price * 0.618
    fib_786 = high_20d - range_price * 0.786
    
    result = pd.DataFrame({
        'fib_high': high_20d,
        'fib_low': low_20d,
        'fib_236': fib_236,
        'fib_382': fib_382,
        'fib_50': fib_50,
        'fib_618': fib_618,
        'fib_786': fib_786
    })
    
    # Add binary indicators for price near Fibonacci levels
    level_margin = 0.003  # 0.3% margin
    for level in ['236', '382', '50', '618', '786']:
        fib_col = f'fib_{level}'
        df[f'near_{fib_col}'] = (
            (df['close'] > result[fib_col] * (1 - level_margin)) & 
            (df['close'] < result[fib_col] * (1 + level_margin))
        ).astype(int)
    
    return result

def calculate_choppiness_index(df: pd.DataFrame, window=14) -> pd.Series:
    # Measures market choppiness (sideways vs. trending)
    atr_sum = np.sum([np.max([df['high'].iloc[i] - df['low'].iloc[i],
                             abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                             abs(df['low'].iloc[i] - df['close'].iloc[i-1])])
                     for i in range(1, len(df))])
    
    high_low_range = np.max(df['high'][-window:]) - np.min(df['low'][-window:])
    
    chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(window)
    return pd.Series(chop, index=df.index)

def calculate_elder_ray(df: pd.DataFrame) -> pd.DataFrame:
    from talib import abstract as ta
    ema13 = ta.EMA(df, timeperiod=13)
    
    bull_power = df['high'] - ema13
    bear_power = df['low'] - ema13
    
    result = pd.DataFrame({
        'elder_bull': bull_power,
        'elder_bear': bear_power
    })
    return result

def calculate_coppock_curve(df: pd.DataFrame) -> pd.Series:
    # Coppock Curve - a momentum indicator
    roc1 = df['close'].pct_change(periods=14)
    roc2 = df['close'].pct_change(periods=11)
    coppock = (roc1 + roc2).rolling(window=10).sum()
    return coppock

def calculate_fisher_transform(df: pd.DataFrame, period=10) -> pd.Series:
    # Fisher Transform of price normalized to -1 to 1 range
    med = (df['high'] + df['low']) / 2
    ndaymin = med.rolling(window=period).min()
    ndaymax = med.rolling(window=period).max()
    
    raw = -1.0 + 2.0 * ((med - ndaymin) / (ndaymax - ndaymin))
    
    # Adjust out-of-bounds values
    raw = raw.clip(-0.999, 0.999)
    
    # Calculate Fisher transform
    fish = 0.5 * np.log((1.0 + raw) / (1.0 - raw))
    return fish

def calculate_awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    # Awesome Oscillator
    midpoint = (df['high'] + df['low']) / 2
    ao = midpoint.rolling(window=5).mean() - midpoint.rolling(window=34).mean()
    return ao

def calculate_accelerator_oscillator(df: pd.DataFrame) -> pd.Series:
    # Accelerator Oscillator (AC)
    awesome = calculate_awesome_oscillator(df)
    ac = awesome - awesome.rolling(window=5).mean()
    return ac

def calculate_hull_moving_average(df: pd.DataFrame, period=16) -> pd.Series:
    # Hull Moving Average
    wma1 = df['close'].rolling(window=period//2).apply(
        lambda x: np.average(x, weights=np.arange(len(x))+1), raw=True)
    wma2 = df['close'].rolling(window=period).apply(
        lambda x: np.average(x, weights=np.arange(len(x))+1), raw=True)
    
    raw = 2 * wma1 - wma2
    hma = raw.rolling(window=int(np.sqrt(period))).apply(
        lambda x: np.average(x, weights=np.arange(len(x))+1), raw=True)
    
    return hma

# Derived feature calculation functions
def calculate_price_momentum(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate various momentum metrics
    returns = df['close'].pct_change()
    
    result = pd.DataFrame({
        'mom_1d': df['close'] / df['close'].shift(1) - 1,
        'mom_5d': df['close'] / df['close'].shift(5) - 1,
        'mom_10d': df['close'] / df['close'].shift(10) - 1,
        'mom_20d': df['close'] / df['close'].shift(window) - 1,
        'mom_std': returns.rolling(window=window).std(),
        'mom_accel': returns.diff().rolling(window=window).mean()
    })
    
    return result

def calculate_volatility_indicator(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate volatility metrics
    returns = df['close'].pct_change()
    
    result = pd.DataFrame({
        'vol_std': returns.rolling(window=window).std(),
        'vol_range_mean': (df['high'] - df['low']).rolling(window=window).mean() / df['close'],
        'vol_max_range': (df['high'] - df['low']).rolling(window=window).max() / df['close'],
        'vol_min_range': (df['high'] - df['low']).rolling(window=window).min() / df['close'],
        'vol_parkinson': np.sqrt((1/(4*np.log(2))) * 
                                 (np.log(df['high']/df['low'])**2).rolling(window=window).mean()),
        'vol_ratio': returns.abs().rolling(window=window).mean() / returns.rolling(window=window).std()
    })
    
    return result

def calculate_trend_strength(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate trend strength metrics
    from scipy import stats
    
    close = df['close'].values
    result = pd.DataFrame(index=df.index)
    
    # Linear regression slope over rolling windows
    def rolling_slope(x):
        x = x[~np.isnan(x)]
        if len(x) < 5:  # Require at least 5 points
            return np.nan
        return stats.linregress(np.arange(len(x)), x)[0]
    
    result['trend_slope'] = df['close'].rolling(window=window).apply(rolling_slope, raw=True)
    
    # R-squared of linear fit
    def rolling_rsquared(x):
        x = x[~np.isnan(x)]
        if len(x) < 5:
            return np.nan
        _, _, r_value, _, _ = stats.linregress(np.arange(len(x)), x)
        return r_value**2
    
    result['trend_rsquared'] = df['close'].rolling(window=window).apply(rolling_rsquared, raw=True)
    
    # Directional consistency
    returns = df['close'].pct_change()
    result['trend_consistency'] = abs(returns.rolling(window=window).sum()) / returns.abs().rolling(window=window).sum()
    
    # Distance from major moving averages
    result['trend_ma_distance'] = (df['close'] / df['close'].rolling(window=window).mean() - 1)
    
    return result

def calculate_range_breakout(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate range breakout metrics
    high_window = df['high'].rolling(window=window).max()
    low_window = df['low'].rolling(window=window).min()
    
    result = pd.DataFrame({
        'breakout_up': (df['close'] > high_window.shift(1)).astype(int),
        'breakout_down': (df['close'] < low_window.shift(1)).astype(int),
        'distance_to_high': df['close'] / high_window - 1,
        'distance_to_low': df['close'] / low_window - 1,
        'range_position': (df['close'] - low_window) / (high_window - low_window)
    })
    
    return result

def calculate_volume_momentum(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate volume momentum metrics
    result = pd.DataFrame({
        'vol_change': df['volume'] / df['volume'].shift(1) - 1,
        'vol_ma_ratio': df['volume'] / df['volume'].rolling(window=window).mean(),
        'vol_std': df['volume'].rolling(window=window).std() / df['volume'].rolling(window=window).mean(),
        'vol_trend': (df['volume'] * df['close'].pct_change()).rolling(window=window).sum()
    })
    
    return result

def identify_support_resistance(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Identify support and resistance levels
    # This is a simplified implementation
    high_window = df['high'].rolling(window=window).max()
    low_window = df['low'].rolling(window=window).min()
    
    result = pd.DataFrame({
        'support': low_window,
        'resistance': high_window,
        'near_support': (df['close'] / low_window < 1.02).astype(int),
        'near_resistance': (df['close'] / high_window > 0.98).astype(int)
    })
    
    return result

def calculate_price_cyclicality(df: pd.DataFrame, window=20) -> pd.Series:
    # Measure price cyclicality
    # This is a simplified proxy using autocorrelation
    returns = df['close'].pct_change()
    
    def autocorr(x, lag=1):
        x = x[~np.isnan(x)]
        if len(x) < lag + 5:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    
    cycle_indicator = returns.rolling(window=window).apply(lambda x: autocorr(x, lag=5), raw=True)
    return cycle_indicator

def calculate_volume_profile(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate volume profile
    # This is a simplified version
    vol_levels = 10  # Number of price levels
    
    price_range = df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()
    bin_size = price_range / vol_levels
    
    result = pd.DataFrame(index=df.index)
    
    # Simplified implementation that creates binary indicators for price levels
    # with high volume concentration
    for i in range(vol_levels):
        level_low = df['low'].rolling(window=window).min() + i * bin_size
        level_high = level_low + bin_size
        
        # Check if current price is in this level
        in_level = ((df['close'] >= level_low) & (df['close'] < level_high)).astype(int)
        
        # Mark levels with high volume
        volume_in_level = df['volume'] * in_level
        high_vol_level = (volume_in_level.rolling(window=window).sum() > 
                         df['volume'].rolling(window=window).mean() * 1.5).astype(int)
        
        result[f'vol_level_{i}'] = high_vol_level
    
    return result

def calculate_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    # Calculate order flow imbalance (simplified proxy)
    # Actual implementation would use bid/ask data
    
    # Use price and volume as proxy
    price_delta = df['close'] - df['open']
    volume = df['volume']
    
    # Positive when upticks have more volume
    imbalance = np.sign(price_delta) * volume
    imbalance_normalized = imbalance / volume.rolling(window=20).mean()
    
    return imbalance_normalized

def calculate_price_velocity(df: pd.DataFrame, window=20) -> pd.DataFrame:
    # Calculate price velocity and acceleration
    returns = df['close'].pct_change()
    
    result = pd.DataFrame({
        'velocity': returns.rolling(window=window).mean(),
        'acceleration': returns.diff().rolling(window=window).mean(),
        'jerk': returns.diff().diff().rolling(window=window).mean()
    })
    
    return result

def calculate_breakout_strength(df: pd.DataFrame, window=20) -> pd.Series:
    # Calculate breakout strength
    high_window = df['high'].rolling(window=window).max().shift(1)
    low_window = df['low'].rolling(window=window).min().shift(1)
    
    # Strength based on close price distance from range and volume
    upper_breakout = (df['close'] > high_window).astype(int) * (df['close'] / high_window - 1) * df['volume']
    lower_breakout = (df['close'] < low_window).astype(int) * (1 - df['close'] / low_window) * df['volume']
    
    breakout_strength = upper_breakout - lower_breakout
    
    return breakout_strength

def calculate_momentum_divergence(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate divergence between price and momentum indicators
    from talib import abstract as ta
    
    # Get RSI if not already calculated
    if 'rsi' not in df.columns:
        df['rsi'] = ta.RSI(df, timeperiod=14)
    
    price_trend = df['close'].pct_change(periods=10)
    rsi_trend = df['rsi'].diff(periods=10)
    
    # Bearish divergence: price up, RSI down
    bearish_div = (price_trend > 0) & (rsi_trend < 0)
    
    # Bullish divergence: price down, RSI up
    bullish_div = (price_trend < 0) & (rsi_trend > 0)
    
    result = pd.DataFrame({
        'bullish_divergence': bullish_div.astype(int),
        'bearish_divergence': bearish_div.astype(int)
    })
    
    return result

def calculate_mean_reversion_probability(df: pd.DataFrame, window=20) -> pd.Series:
    # Calculate mean reversion probability
    ma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    
    # Z-score distance from mean
    z_score = (df['close'] - ma) / std
    
    # Higher values mean higher probability of mean reversion
    reversion_prob = np.abs(z_score) * (1 - df['adx'] / 100 if 'adx' in df.columns else 1)
    
    return reversion_prob

def identify_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # Identify common price patterns (very simplified implementation)
    # Real implementation would use more sophisticated pattern recognition
    
    result = pd.DataFrame(index=df.index)
    
    # Double top pattern (simplified)
    recent_high = df['high'].rolling(window=10).max()
    double_top = ((df['high'] > recent_high * 0.98) & 
                 (df['high'] < recent_high * 1.02) & 
                 (df['high'].shift(5) > recent_high * 0.98) &
                 (df['high'].shift(5) < recent_high * 1.02) &
                 (df['high'].rolling(window=5).min() < recent_high * 0.97)).astype(int)
    
    # Double bottom pattern (simplified)
    recent_low = df['low'].rolling(window=10).min()
    double_bottom = ((df['low'] > recent_low * 0.98) & 
                    (df['low'] < recent_low * 1.02) & 
                    (df['low'].shift(5) > recent_low * 0.98) &
                    (df['low'].shift(5) < recent_low * 1.02) &
                    (df['low'].rolling(window=5).max() > recent_low * 1.03)).astype(int)
    
    # Head and shoulders (extremely simplified)
    hs_pattern = ((df['high'].shift(10) > df['high'].shift(15)) &
                 (df['high'].shift(5) > df['high'].shift(10)) &
                 (df['high'] < df['high'].shift(5))).astype(int)
    
    result['double_top'] = double_top
    result['double_bottom'] = double_bottom
    result['head_shoulders'] = hs_pattern
    
    return result

def add_fourier_features(df: pd.DataFrame, col: str, num_components: int = 5) -> pd.DataFrame:
    """Add Fourier transform features to the dataframe."""
    from scipy.fftpack import fft
    
    df = df.copy()
    price_series = df[col].values
    
    # Apply FFT
    fft_values = fft(price_series)
    fft_magnitudes = np.abs(fft_values)[:len(price_series)//2]
    
    # Find dominant frequencies
    dominant_idxs = np.argsort(fft_magnitudes)[-num_components:]
    
    # Create sine and cosine components for dominant frequencies
    for i, idx in enumerate(dominant_idxs):
        freq = idx / len(price_series)
        t = np.arange(len(price_series))
        
        df[f'fourier_sin_{i+1}'] = np.sin(2 * np.pi * freq * t)
        df[f'fourier_cos_{i+1}'] = np.cos(2 * np.pi * freq * t)
    
    return df

def add_wavelet_features(df: pd.DataFrame, col: str, levels: int = 3) -> pd.DataFrame:
    """Add wavelet decomposition features."""
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not available, skipping wavelet features")
        return df
    
    df = df.copy()
    price_series = df[col].values
    
    # Apply wavelet decomposition
    wavelet = 'db4'  # Daubechies wavelet
    coeffs = pywt.wavedec(price_series, wavelet, level=levels)
    
    # Add coefficients as features
    for i, coeff in enumerate(coeffs):
        if i == 0:
            # Approximation coefficients (scaled appropriately)
            df[f'wavelet_a{levels}'] = np.interp(
                np.arange(len(price_series)), 
                np.linspace(0, len(price_series)-1, len(coeff)), 
                coeff
            )
        else:
            # Detail coefficients
            df[f'wavelet_d{levels-i+1}'] = np.interp(
                np.arange(len(price_series)), 
                np.linspace(0, len(price_series)-1, len(coeff)), 
                coeff
            )
    
    return df

def add_rolling_statistics(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Add rolling statistical features for price and volume."""
    df = df.copy()
    
    for window in windows:
        # Price statistics
        df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'rolling_skew_{window}'] = df['close'].rolling(window=window).skew()
        df[f'rolling_kurt_{window}'] = df['close'].rolling(window=window).kurt()
        df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
        df[f'rolling_quantile_25_{window}'] = df['close'].rolling(window=window).quantile(0.25)
        df[f'rolling_quantile_75_{window}'] = df['close'].rolling(window=window).quantile(0.75)
        
        # Volume statistics
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
        df[f'volume_rolling_std_{window}'] = df['volume'].rolling(window=window).std()
        
        # Return statistics
        returns = df['close'].pct_change()
        df[f'return_rolling_mean_{window}'] = returns.rolling(window=window).mean()
        df[f'return_rolling_std_{window}'] = returns.rolling(window=window).std()
        df[f'return_rolling_skew_{window}'] = returns.rolling(window=window).skew()
        df[f'return_rolling_kurt_{window}'] = returns.rolling(window=window).kurt()
    
    return df

def add_fractional_differentiation(df: pd.DataFrame, col: str, d: float = 0.4) -> pd.DataFrame:
    """Add fractionally differentiated price series for stationarity."""
    df = df.copy()
    price_series = df[col].values
    
    # Calculate weights for fractional differentiation
    def get_weights(d, size):
        weights = [1.0]
        for k in range(1, size):
            weights.append(-weights[-1] * (d - k + 1) / k)
        return np.array(weights)
    
    # Apply weights using convolution
    weights = get_weights(d, len(price_series))
    frac_diff = np.convolve(price_series, weights, mode='valid')
    
    # Pad the beginning with NaNs to maintain shape
    padding = len(price_series) - len(frac_diff)
    df[f'frac_diff_{d}'] = np.append(np.array([np.nan] * padding), frac_diff)
    
    return df


if __name__ == "__main__":
    sys.exit(main())