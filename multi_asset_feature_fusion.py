#!/usr/bin/env python3
"""
Multi-Asset Feature Fusion Module for Kraken Trading Bot

This module analyzes correlations and relationships between multiple assets
to enhance the prediction accuracy of the trading bot.

It identifies lead-lag relationships, correlations in volatility, and other
cross-asset patterns that can provide predictive signals for trading.
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CORRELATION_THRESHOLD = 0.6  # Minimum correlation to consider assets related
GRANGER_MAX_LAG = 5  # Maximum lag for Granger causality tests
CORRELATION_WINDOW = 60  # Window for rolling correlation (in periods)
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
DATA_DIR = "cross_asset_data"
os.makedirs(DATA_DIR, exist_ok=True)

class MultiAssetFeatureFusion:
    """
    Implements multi-asset feature fusion to enhance prediction accuracy
    by incorporating relationships between different assets.
    """
    def __init__(self, base_asset, related_assets=None, lookback_periods=90):
        """
        Initialize the multi-asset feature fusion module
        
        Args:
            base_asset (str): The primary asset for prediction (e.g., "SOL/USD")
            related_assets (list): List of potentially related assets to analyze
            lookback_periods (int): Number of periods to analyze for relationships
        """
        self.base_asset = base_asset
        self.related_assets = related_assets or self._get_default_related_assets()
        self.lookback_periods = lookback_periods
        self.correlation_matrix = None
        self.lead_lag_relationships = {}
        self.cointegration_results = {}
        self.selected_features = []
        self.scaler = StandardScaler()
        self.pca = None
        
        # Loading state if exists
        self._load_state()
        
        logger.info(f"Multi-Asset Feature Fusion initialized for {base_asset} with related assets: {self.related_assets}")
    
    def _get_default_related_assets(self):
        """Get default related assets excluding the base asset"""
        return [asset for asset in SUPPORTED_ASSETS if asset != self.base_asset]
    
    def _load_state(self):
        """Load previously calculated state if available"""
        state_path = os.path.join(DATA_DIR, f"{self.base_asset.replace('/', '_')}_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.correlation_matrix = state.get('correlation_matrix')
                self.lead_lag_relationships = state.get('lead_lag_relationships', {})
                self.cointegration_results = state.get('cointegration_results', {})
                self.selected_features = state.get('selected_features', [])
                
                logger.info(f"Loaded existing state for {self.base_asset}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save current state for future use"""
        state_path = os.path.join(DATA_DIR, f"{self.base_asset.replace('/', '_')}_state.json")
        
        # Convert correlation matrix to list if it's a DataFrame
        corr_matrix = self.correlation_matrix
        if isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = corr_matrix.to_dict()
        
        state = {
            'correlation_matrix': corr_matrix,
            'lead_lag_relationships': self.lead_lag_relationships,
            'cointegration_results': self.cointegration_results,
            'selected_features': self.selected_features,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f)
        
        logger.info(f"Saved state for {self.base_asset}")
    
    def analyze_relationships(self, data_dict):
        """
        Analyze relationships between assets
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
                Each DataFrame should have at least 'close' and 'timestamp' columns
                
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing relationships between {self.base_asset} and related assets")
        
        # Check that we have data for all assets
        all_assets = [self.base_asset] + self.related_assets
        missing_assets = [asset for asset in all_assets if asset not in data_dict]
        if missing_assets:
            logger.warning(f"Missing data for assets: {missing_assets}")
            # Filter out missing assets
            self.related_assets = [asset for asset in self.related_assets if asset in data_dict]
        
        # Calculate correlation matrix
        correlation_df = self._calculate_correlation_matrix(data_dict)
        self.correlation_matrix = correlation_df
        
        # Find lead-lag relationships using Granger causality
        self.lead_lag_relationships = self._analyze_lead_lag_relationships(data_dict)
        
        # Test for cointegration
        self.cointegration_results = self._test_cointegration(data_dict)
        
        # Select most predictive features based on the relationships
        self.selected_features = self._select_predictive_features()
        
        # Save the state
        self._save_state()
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'lead_lag_relationships': self.lead_lag_relationships,
            'cointegration_results': self.cointegration_results,
            'selected_features': self.selected_features
        }
    
    def _calculate_correlation_matrix(self, data_dict):
        """
        Calculate correlation matrix between assets
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        all_assets = [self.base_asset] + self.related_assets
        available_assets = [asset for asset in all_assets if asset in data_dict]
        
        # Create a DataFrame with close prices for all assets
        price_data = {}
        for asset in available_assets:
            price_data[asset] = data_dict[asset]['close'].values
        
        price_df = pd.DataFrame(price_data)
        
        # Calculate Pearson correlation
        correlation_matrix = price_df.corr(method='pearson')
        
        logger.info(f"Calculated correlation matrix for {len(available_assets)} assets")
        
        return correlation_matrix
    
    def _analyze_lead_lag_relationships(self, data_dict):
        """
        Analyze lead-lag relationships using Granger causality tests
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
            
        Returns:
            dict: Lead-lag relationships for each asset pair
        """
        lead_lag_results = {}
        
        # Convert DataFrames to percentage changes to ensure stationarity
        returns_dict = {}
        for asset, df in data_dict.items():
            returns_dict[asset] = df['close'].pct_change().dropna()
        
        # Perform Granger causality tests
        for related_asset in self.related_assets:
            if related_asset in data_dict:
                # Check if A causes B
                base_to_related = self._granger_causality_test(
                    returns_dict[self.base_asset], 
                    returns_dict[related_asset]
                )
                
                # Check if B causes A
                related_to_base = self._granger_causality_test(
                    returns_dict[related_asset], 
                    returns_dict[self.base_asset]
                )
                
                lead_lag_results[f"{self.base_asset}_{related_asset}"] = {
                    'base_causes_related': base_to_related,
                    'related_causes_base': related_to_base,
                    'strongest_relationship': 'base_causes_related' if base_to_related['min_pvalue'] < related_to_base['min_pvalue'] else 'related_causes_base',
                    'lag': base_to_related['best_lag'] if base_to_related['min_pvalue'] < related_to_base['min_pvalue'] else related_to_base['best_lag']
                }
        
        logger.info(f"Analyzed lead-lag relationships for {len(lead_lag_results)} asset pairs")
        
        return lead_lag_results
    
    def _granger_causality_test(self, x, y, max_lag=GRANGER_MAX_LAG):
        """
        Perform Granger causality test to determine if x causes y
        
        Args:
            x (pd.Series): Independent variable time series
            y (pd.Series): Dependent variable time series
            max_lag (int): Maximum lag to test
            
        Returns:
            dict: Test results with minimum p-value and best lag
        """
        # Align the series
        aligned_data = pd.concat([x, y], axis=1).dropna()
        
        if len(aligned_data) <= max_lag + 1:
            return {'min_pvalue': 1.0, 'best_lag': 0, 'significant': False}
        
        try:
            test_results = {}
            min_pvalue = 1.0
            best_lag = 0
            
            for lag in range(1, max_lag + 1):
                result = grangercausalitytests(aligned_data, maxlag=lag, verbose=False)
                # Extract p-value from F-test
                pvalue = result[lag][0]['ssr_ftest'][1]
                test_results[lag] = pvalue
                
                if pvalue < min_pvalue:
                    min_pvalue = pvalue
                    best_lag = lag
            
            return {
                'min_pvalue': min_pvalue,
                'best_lag': best_lag,
                'significant': min_pvalue < 0.05,
                'all_pvalues': test_results
            }
        except Exception as e:
            logger.error(f"Error in Granger causality test: {e}")
            return {'min_pvalue': 1.0, 'best_lag': 0, 'significant': False}
    
    def _test_cointegration(self, data_dict):
        """
        Test for cointegration between assets
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
            
        Returns:
            dict: Cointegration test results
        """
        cointegration_results = {}
        
        # Extract price data
        prices = {}
        for asset, df in data_dict.items():
            prices[asset] = df['close'].values
        
        # Test for pairwise cointegration
        for related_asset in self.related_assets:
            if related_asset in data_dict:
                # Prepare data
                pair_data = np.column_stack((
                    prices[self.base_asset],
                    prices[related_asset]
                ))
                
                try:
                    # Perform Johansen test
                    johansen_result = coint_johansen(pair_data, det_order=0, k_ar_diff=1)
                    
                    # Extract trace statistics and critical values
                    trace_stat = johansen_result.lr1[0]
                    crit_value = johansen_result.cvt[0, 1]  # 95% critical value
                    is_cointegrated = trace_stat > crit_value
                    
                    cointegration_results[f"{self.base_asset}_{related_asset}"] = {
                        'trace_statistic': float(trace_stat),
                        'critical_value': float(crit_value),
                        'is_cointegrated': bool(is_cointegrated)
                    }
                except Exception as e:
                    logger.error(f"Error in cointegration test for {self.base_asset}_{related_asset}: {e}")
                    cointegration_results[f"{self.base_asset}_{related_asset}"] = {
                        'trace_statistic': None,
                        'critical_value': None,
                        'is_cointegrated': False,
                        'error': str(e)
                    }
        
        logger.info(f"Tested cointegration for {len(cointegration_results)} asset pairs")
        
        return cointegration_results
    
    def _select_predictive_features(self):
        """
        Select the most predictive features based on the analysis
        
        Returns:
            list: Selected features with their assets and lags
        """
        selected_features = []
        
        # 1. Use high correlation assets
        if isinstance(self.correlation_matrix, pd.DataFrame):
            for asset in self.related_assets:
                if asset in self.correlation_matrix.columns:
                    correlation = self.correlation_matrix.loc[self.base_asset, asset]
                    if abs(correlation) >= CORRELATION_THRESHOLD:
                        selected_features.append({
                            'asset': asset,
                            'feature_type': 'price',
                            'lag': 0,
                            'reason': f'High correlation ({correlation:.3f})'
                        })
        
        # 2. Use assets with significant lead-lag relationships
        for pair, result in self.lead_lag_relationships.items():
            if result.get('related_causes_base', {}).get('significant', False):
                asset = pair.split('_')[1]  # Extract the related asset name
                lag = result.get('related_causes_base', {}).get('best_lag', 1)
                
                selected_features.append({
                    'asset': asset,
                    'feature_type': 'price',
                    'lag': lag,
                    'reason': f'Granger causes base asset (p={result["related_causes_base"]["min_pvalue"]:.4f})'
                })
        
        # 3. Use cointegrated assets
        for pair, result in self.cointegration_results.items():
            if result.get('is_cointegrated', False):
                asset = pair.split('_')[1]  # Extract the related asset name
                
                # Check if we've already added this asset
                if not any(f['asset'] == asset for f in selected_features):
                    selected_features.append({
                        'asset': asset,
                        'feature_type': 'price',
                        'lag': 0,
                        'reason': 'Cointegrated with base asset'
                    })
        
        logger.info(f"Selected {len(selected_features)} predictive features")
        
        return selected_features
    
    def calculate_cross_asset_features(self, current_data):
        """
        Calculate cross-asset features based on current market data
        
        Args:
            current_data (dict): Dictionary with asset name keys and current data
                Each entry should have at least 'close' price
                
        Returns:
            dict: Calculated cross-asset features
        """
        if not self.selected_features:
            logger.warning("No selected features available. Run analyze_relationships first.")
            return {}
        
        features = {}
        base_price = current_data.get(self.base_asset, {}).get('close')
        
        if base_price is None:
            logger.error(f"Missing price data for base asset {self.base_asset}")
            return features
        
        # Calculate price ratios and spreads with selected assets
        for feature in self.selected_features:
            asset = feature['asset']
            lag = feature['lag']
            
            if asset in current_data and 'close' in current_data[asset]:
                related_price = current_data[asset]['close']
                
                # Calculate various cross-asset features
                features[f"{asset}_price_ratio"] = related_price / base_price
                features[f"{asset}_log_return_diff"] = np.log(related_price) - np.log(base_price)
                features[f"{asset}_price_spread"] = related_price - base_price
                
                # If we have high-low data, calculate volatility relationships
                if 'high' in current_data[asset] and 'low' in current_data[asset] and \
                   'high' in current_data[self.base_asset] and 'low' in current_data[self.base_asset]:
                    # Calculate relative volatility
                    base_volatility = (current_data[self.base_asset]['high'] - current_data[self.base_asset]['low']) / base_price
                    related_volatility = (current_data[asset]['high'] - current_data[asset]['low']) / related_price
                    features[f"{asset}_relative_volatility"] = related_volatility / base_volatility
        
        logger.debug(f"Calculated {len(features)} cross-asset features")
        
        return features
    
    def calculate_rolling_correlations(self, data_dict, window=CORRELATION_WINDOW):
        """
        Calculate rolling correlations between the base asset and related assets
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
            window (int): Window size for rolling correlation
            
        Returns:
            pd.DataFrame: DataFrame with rolling correlations
        """
        all_assets = [self.base_asset] + self.related_assets
        available_assets = [asset for asset in all_assets if asset in data_dict]
        
        # Create a DataFrame with returns for all assets
        returns_data = {}
        for asset in available_assets:
            returns_data[asset] = data_dict[asset]['close'].pct_change().fillna(0)
        
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate rolling correlations
        rolling_correlations = {}
        for asset in self.related_assets:
            if asset in returns_df.columns:
                # Calculate Pearson correlation in a rolling window
                rolling_corr = returns_df[self.base_asset].rolling(window=window).corr(returns_df[asset])
                rolling_correlations[asset] = rolling_corr
        
        rolling_df = pd.DataFrame(rolling_correlations)
        logger.info(f"Calculated rolling correlations with window={window} for {len(rolling_correlations)} assets")
        
        return rolling_df
    
    def generate_cross_asset_signals(self, current_data, historical_data=None):
        """
        Generate trading signals based on cross-asset relationships
        
        Args:
            current_data (dict): Dictionary with current data for all assets
            historical_data (dict, optional): Dictionary with historical data for calculating trends
            
        Returns:
            dict: Generated signals with reasons
        """
        signals = {
            'long': [],
            'short': [],
            'neutral': []
        }
        
        # 1. Check correlation-based signals
        if isinstance(self.correlation_matrix, pd.DataFrame):
            for asset in self.related_assets:
                if asset in current_data and asset in self.correlation_matrix.columns:
                    correlation = self.correlation_matrix.loc[self.base_asset, asset]
                    
                    # Calculate recent returns
                    if historical_data and asset in historical_data and self.base_asset in historical_data:
                        related_return = (current_data[asset]['close'] / historical_data[asset]['close'].iloc[-2] - 1) * 100
                        
                        # If assets are positively correlated
                        if correlation > CORRELATION_THRESHOLD:
                            if related_return > 1.0:  # Strong positive move in related asset
                                signals['long'].append({
                                    'reason': f"Positive correlation with {asset} ({correlation:.2f}), which moved up {related_return:.2f}%",
                                    'strength': abs(correlation) * abs(related_return) / 10,
                                    'asset': asset
                                })
                            elif related_return < -1.0:  # Strong negative move in related asset
                                signals['short'].append({
                                    'reason': f"Positive correlation with {asset} ({correlation:.2f}), which moved down {related_return:.2f}%",
                                    'strength': abs(correlation) * abs(related_return) / 10,
                                    'asset': asset
                                })
                        
                        # If assets are negatively correlated
                        elif correlation < -CORRELATION_THRESHOLD:
                            if related_return > 1.0:  # Strong positive move in related asset
                                signals['short'].append({
                                    'reason': f"Negative correlation with {asset} ({correlation:.2f}), which moved up {related_return:.2f}%",
                                    'strength': abs(correlation) * abs(related_return) / 10,
                                    'asset': asset
                                })
                            elif related_return < -1.0:  # Strong negative move in related asset
                                signals['long'].append({
                                    'reason': f"Negative correlation with {asset} ({correlation:.2f}), which moved down {related_return:.2f}%",
                                    'strength': abs(correlation) * abs(related_return) / 10,
                                    'asset': asset
                                })
        
        # 2. Check lead-lag relationship signals
        for pair, result in self.lead_lag_relationships.items():
            if result.get('related_causes_base', {}).get('significant', False):
                parts = pair.split('_')
                if len(parts) == 2:
                    related_asset = parts[1]
                    
                    if related_asset in current_data and historical_data and related_asset in historical_data:
                        # Get the best lag for prediction
                        lag = result.get('related_causes_base', {}).get('best_lag', 1)
                        
                        if len(historical_data[related_asset]) > lag + 1:
                            # Calculate lagged return of the related asset
                            lagged_return = (historical_data[related_asset]['close'].iloc[-(lag+1)] / 
                                           historical_data[related_asset]['close'].iloc[-(lag+2)] - 1) * 100
                            
                            # If the lagged return is significant, generate a signal
                            if abs(lagged_return) > 0.5:
                                signal_type = 'long' if lagged_return > 0 else 'short'
                                signals[signal_type].append({
                                    'reason': f"Lead-lag relationship with {related_asset} (lag={lag}), which had {lagged_return:.2f}% return",
                                    'strength': min(abs(lagged_return) / 2, 5),  # Cap at 5
                                    'asset': related_asset
                                })
        
        # 3. Check cointegration-based signals
        for pair, result in self.cointegration_results.items():
            if result.get('is_cointegrated', False):
                parts = pair.split('_')
                if len(parts) == 2:
                    related_asset = parts[1]
                    
                    if related_asset in current_data and self.base_asset in current_data:
                        # Calculate z-score of the spread
                        if historical_data and related_asset in historical_data and self.base_asset in historical_data:
                            # Create spread series
                            base_prices = historical_data[self.base_asset]['close'].values
                            related_prices = historical_data[related_asset]['close'].values
                            
                            # Normalize prices to make them comparable
                            base_norm = base_prices / base_prices[0]
                            related_norm = related_prices / related_prices[0]
                            
                            # Calculate spread
                            spread = base_norm - related_norm
                            
                            # Calculate z-score of current spread
                            mean_spread = np.mean(spread)
                            std_spread = np.std(spread)
                            
                            if std_spread > 0:
                                current_base_norm = current_data[self.base_asset]['close'] / base_prices[0]
                                current_related_norm = current_data[related_asset]['close'] / related_prices[0]
                                current_spread = current_base_norm - current_related_norm
                                z_score = (current_spread - mean_spread) / std_spread
                                
                                # Generate mean-reversion signals based on z-score
                                if z_score > 2.0:  # Base asset is overvalued relative to related
                                    signals['short'].append({
                                        'reason': f"Cointegration with {related_asset}: base asset relatively overvalued (z-score={z_score:.2f})",
                                        'strength': min(abs(z_score) - 1, 5),  # Cap at 5
                                        'asset': related_asset
                                    })
                                elif z_score < -2.0:  # Base asset is undervalued relative to related
                                    signals['long'].append({
                                        'reason': f"Cointegration with {related_asset}: base asset relatively undervalued (z-score={z_score:.2f})",
                                        'strength': min(abs(z_score) - 1, 5),  # Cap at 5
                                        'asset': related_asset
                                    })
        
        # If we have no clear signals, return neutral
        if not signals['long'] and not signals['short']:
            signals['neutral'].append({
                'reason': 'No significant cross-asset signals detected',
                'strength': 5.0
            })
        
        logger.info(f"Generated cross-asset signals: {len(signals['long'])} long, {len(signals['short'])} short, {len(signals['neutral'])} neutral")
        
        return signals
    
    def create_cross_asset_features_for_model(self, data_dict):
        """
        Create features for machine learning models based on cross-asset relationships
        
        Args:
            data_dict (dict): Dictionary with asset name keys and price DataFrame values
            
        Returns:
            pd.DataFrame: Features DataFrame for model input
        """
        if not self.selected_features:
            logger.warning("No selected features available. Run analyze_relationships first.")
            return pd.DataFrame()
        
        # Initialize feature DataFrame
        features_data = []
        timestamps = data_dict[self.base_asset]['timestamp']
        
        # Ensure all assets have the same index
        aligned_data = {}
        for asset, df in data_dict.items():
            if 'timestamp' in df.columns:
                aligned_data[asset] = df.set_index('timestamp')
            else:
                aligned_data[asset] = df
        
        # Create a date range index
        if 'timestamp' in data_dict[self.base_asset].columns:
            all_timestamps = data_dict[self.base_asset]['timestamp']
        else:
            # Use row indices if timestamps not available
            all_timestamps = range(len(data_dict[self.base_asset]))
        
        # For each timestamp, calculate features
        for i, timestamp in enumerate(all_timestamps):
            if i < max([f.get('lag', 0) for f in self.selected_features] + [5]):
                continue  # Skip early timestamps where lagged data isn't available
            
            features = {}
            features['timestamp'] = timestamp
            
            # Add base asset features
            base_data = data_dict[self.base_asset]
            features['base_price'] = base_data.loc[i, 'close'] if 'close' in base_data.columns else None
            features['base_return'] = base_data.loc[i, 'close'] / base_data.loc[i-1, 'close'] - 1 if 'close' in base_data.columns else None
            
            # Add features for each selected asset
            for feature in self.selected_features:
                asset = feature['asset']
                lag = feature['lag']
                
                if asset in data_dict and i >= lag:
                    asset_data = data_dict[asset]
                    asset_price = asset_data.loc[i-lag, 'close'] if 'close' in asset_data.columns else None
                    
                    if asset_price is not None and features['base_price'] is not None:
                        # Price ratio
                        features[f"{asset}_price_ratio_lag{lag}"] = asset_price / features['base_price']
                        
                        # Returns
                        if i-lag-1 >= 0 and 'close' in asset_data.columns:
                            asset_return = asset_price / asset_data.loc[i-lag-1, 'close'] - 1
                            features[f"{asset}_return_lag{lag}"] = asset_return
                            
                            # Return difference
                            if features['base_return'] is not None:
                                features[f"{asset}_return_diff_lag{lag}"] = asset_return - features['base_return']
                        
                        # Volatility if available
                        if 'high' in asset_data.columns and 'low' in asset_data.columns and \
                           'high' in base_data.columns and 'low' in base_data.columns:
                            asset_volatility = (asset_data.loc[i-lag, 'high'] - asset_data.loc[i-lag, 'low']) / asset_price
                            base_volatility = (base_data.loc[i, 'high'] - base_data.loc[i, 'low']) / features['base_price']
                            
                            if base_volatility > 0:
                                features[f"{asset}_rel_volatility_lag{lag}"] = asset_volatility / base_volatility
            
            features_data.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_data)
        
        # Drop rows with missing values
        features_df = features_df.dropna()
        
        logger.info(f"Created {len(features_df)} rows of cross-asset features with {features_df.shape[1]} columns")
        
        return features_df
    
    def extract_principal_components(self, features_df, n_components=5):
        """
        Extract principal components from cross-asset features
        
        Args:
            features_df (pd.DataFrame): DataFrame with cross-asset features
            n_components (int): Number of principal components to extract
            
        Returns:
            pd.DataFrame: DataFrame with principal components
        """
        # Remove non-numeric columns
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Remove timestamp column if present
        if 'timestamp' in numeric_features.columns:
            numeric_features = numeric_features.drop(columns=['timestamp'])
        
        if numeric_features.empty:
            logger.warning("No numeric features available for PCA")
            return pd.DataFrame()
        
        # Standardize the features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        # Apply PCA
        self.pca = PCA(n_components=min(n_components, scaled_features.shape[1]))
        principal_components = self.pca.fit_transform(scaled_features)
        
        # Create DataFrame with principal components
        pc_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
        )
        
        # Add timestamp if available
        if 'timestamp' in features_df.columns:
            pc_df['timestamp'] = features_df['timestamp'].values
        
        logger.info(f"Extracted {principal_components.shape[1]} principal components explaining {sum(self.pca.explained_variance_ratio_)*100:.2f}% of variance")
        
        return pc_df

def main():
    """Test the multi-asset feature fusion module"""
    logger.info("Multi-Asset Feature Fusion module imported")

if __name__ == "__main__":
    main()