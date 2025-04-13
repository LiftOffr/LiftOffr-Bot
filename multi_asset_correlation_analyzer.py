#!/usr/bin/env python3
"""
Multi-Asset Correlation Analyzer for Kraken Trading Bot

This script analyzes correlations between multiple cryptocurrency assets
and provides insights for portfolio diversification and strategy optimization.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
RESULTS_DIR = "analysis_results"
PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
TIMEFRAMES = ["1h", "4h", "1d"]

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_historical_data(pair, timeframe):
    """
    Load historical price data for a specific pair and timeframe
    
    Args:
        pair (str): Trading pair code (e.g., "SOLUSD")
        timeframe (str): Timeframe code (e.g., "1h", "4h", "1d")
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    file_path = os.path.join(DATA_DIR, pair, f"{pair}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.warning(f"No data file found for {pair} on {timeframe} timeframe")
        return None
    
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded {len(df)} rows for {pair} on {timeframe} timeframe")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {pair} on {timeframe} timeframe: {e}")
        return None


def calculate_returns(df, column='close'):
    """
    Calculate returns for a price series
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        column (str): Column to use for price data
        
    Returns:
        pd.Series: Returns series
    """
    if df is None or column not in df.columns:
        return None
    
    returns = df[column].pct_change().dropna()
    return returns


def create_multi_asset_dataframe(pairs, timeframe):
    """
    Create a DataFrame with returns for multiple assets
    
    Args:
        pairs (list): List of trading pairs
        timeframe (str): Timeframe code
        
    Returns:
        pd.DataFrame: DataFrame with returns for all assets
    """
    returns_dict = {}
    
    for pair in pairs:
        df = load_historical_data(pair, timeframe)
        if df is not None:
            returns = calculate_returns(df)
            if returns is not None:
                returns_dict[pair] = returns
    
    if not returns_dict:
        logger.error("No valid return series found")
        return None
    
    # Create DataFrame from dictionary
    returns_df = pd.DataFrame(returns_dict)
    
    # Drop rows with any NaN values
    returns_df.dropna(inplace=True)
    
    logger.info(f"Created multi-asset returns DataFrame with {len(returns_df)} rows")
    return returns_df


def calculate_correlation_matrix(returns_df, method='pearson'):
    """
    Calculate correlation matrix for multiple assets
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        method (str): Correlation method ('pearson' or 'spearman')
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    if returns_df is None or returns_df.empty:
        return None
    
    if method.lower() == 'pearson':
        corr_matrix = returns_df.corr(method='pearson')
    elif method.lower() == 'spearman':
        corr_matrix = returns_df.corr(method='spearman')
    else:
        logger.error(f"Unknown correlation method: {method}")
        return None
    
    return corr_matrix


def calculate_rolling_correlation(returns_df, window=30, method='pearson'):
    """
    Calculate rolling correlation between assets
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        window (int): Rolling window size
        method (str): Correlation method
        
    Returns:
        dict: Dictionary with rolling correlations
    """
    if returns_df is None or returns_df.empty:
        return None
    
    pairs = list(returns_df.columns)
    rolling_corrs = {}
    
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            pair1 = pairs[i]
            pair2 = pairs[j]
            
            if method.lower() == 'pearson':
                rolling_corr = returns_df[pair1].rolling(window=window).corr(returns_df[pair2])
            elif method.lower() == 'spearman':
                # For spearman, we need to calculate it for each window manually
                rolling_corr = returns_df[[pair1, pair2]].rolling(window=window).apply(
                    lambda x: spearmanr(x[pair1], x[pair2])[0] if len(x) >= window else np.nan
                )
            else:
                logger.error(f"Unknown correlation method: {method}")
                return None
            
            key = f"{pair1}_{pair2}"
            rolling_corrs[key] = rolling_corr
    
    return rolling_corrs


def plot_correlation_matrix(corr_matrix, title, save_path=None):
    """
    Plot correlation matrix heatmap
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    if corr_matrix is None:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        linewidths=0.5,
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt='.2f'
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved correlation matrix plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rolling_correlations(returns_df, rolling_corrs, window, save_path=None):
    """
    Plot rolling correlations between assets
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        rolling_corrs (dict): Dictionary with rolling correlations
        window (int): Rolling window size
        save_path (str, optional): Path to save the plot
    """
    if rolling_corrs is None or not rolling_corrs:
        return
    
    plt.figure(figsize=(14, 10))
    
    for pair, corr in rolling_corrs.items():
        # Make sure we're using the correct index length for the correlation data
        date_index = returns_df.index[window-1:len(corr)+window-1]
        if len(date_index) != len(corr):
            logger.warning(f"Dimension mismatch for {pair}: date_index={len(date_index)}, corr={len(corr)}")
            # If sizes don't match, use the smaller size for both arrays
            min_size = min(len(date_index), len(corr))
            date_index = date_index[:min_size]
            corr = corr[:min_size]
        
        plt.plot(date_index, corr, label=pair)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title(f"{window}-Period Rolling Correlation")
    plt.xlabel("Date")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved rolling correlation plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def cluster_assets_by_returns(returns_df, n_clusters=3):
    """
    Cluster assets based on their return patterns
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        n_clusters (int): Number of clusters
        
    Returns:
        tuple: (cluster_labels, cluster_centers)
    """
    if returns_df is None or returns_df.empty:
        return None, None
    
    # Transpose the DataFrame to get assets as rows
    X = returns_df.T
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Create a DataFrame with cluster labels
    assets = returns_df.columns
    clusters_df = pd.DataFrame({'Asset': assets, 'Cluster': cluster_labels})
    
    logger.info(f"Asset clustering results:\n{clusters_df}")
    
    return cluster_labels, cluster_centers


def calculate_strategy_correlations(returns_df, strategy_returns):
    """
    Calculate correlations between strategy returns and asset returns
    
    Args:
        returns_df (pd.DataFrame): DataFrame with asset returns
        strategy_returns (pd.Series): Series with strategy returns
        
    Returns:
        pd.Series: Correlations with each asset
    """
    if returns_df is None or returns_df.empty or strategy_returns is None or strategy_returns.empty:
        return None
    
    correlations = {}
    
    for asset in returns_df.columns:
        # Align the data (matching indices)
        asset_returns = returns_df[asset]
        common_index = asset_returns.index.intersection(strategy_returns.index)
        
        if len(common_index) > 0:
            asset_returns = asset_returns.loc[common_index]
            strat_returns = strategy_returns.loc[common_index]
            
            corr, _ = pearsonr(asset_returns, strat_returns)
            correlations[asset] = corr
    
    return pd.Series(correlations)


def calculate_beta(returns_df, market_index='BTCUSD'):
    """
    Calculate beta (market sensitivity) for each asset
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        market_index (str): Column to use as market index
        
    Returns:
        pd.Series: Beta for each asset
    """
    if returns_df is None or returns_df.empty or market_index not in returns_df.columns:
        return None
    
    market_returns = returns_df[market_index]
    betas = {}
    
    for asset in returns_df.columns:
        if asset == market_index:
            betas[asset] = 1.0
            continue
        
        asset_returns = returns_df[asset]
        
        # Calculate covariance and variance
        covariance = asset_returns.cov(market_returns)
        variance = market_returns.var()
        
        if variance > 0:
            beta = covariance / variance
        else:
            beta = np.nan
        
        betas[asset] = beta
    
    return pd.Series(betas)


def calculate_volatility(returns_df, window=30, annualize=True):
    """
    Calculate rolling volatility for each asset
    
    Args:
        returns_df (pd.DataFrame): DataFrame with returns
        window (int): Rolling window size
        annualize (bool): Whether to annualize the volatility
        
    Returns:
        pd.DataFrame: Rolling volatility for each asset
    """
    if returns_df is None or returns_df.empty:
        return None
    
    volatility = pd.DataFrame(index=returns_df.index)
    
    for asset in returns_df.columns:
        asset_returns = returns_df[asset]
        rolling_std = asset_returns.rolling(window=window).std()
        
        if annualize:
            # Assuming returns are in the same frequency as the data
            if '1h' in asset:
                # Hourly data to annual (24 * 365)
                rolling_std = rolling_std * np.sqrt(24 * 365)
            elif '4h' in asset:
                # 4-hour data to annual (6 * 365)
                rolling_std = rolling_std * np.sqrt(6 * 365)
            elif '1d' in asset:
                # Daily data to annual (365)
                rolling_std = rolling_std * np.sqrt(365)
        
        volatility[asset] = rolling_std
    
    return volatility


def plot_volatility(volatility_df, window, save_path=None):
    """
    Plot rolling volatility for each asset
    
    Args:
        volatility_df (pd.DataFrame): DataFrame with volatility
        window (int): Rolling window size
        save_path (str, optional): Path to save the plot
    """
    if volatility_df is None or volatility_df.empty:
        return
    
    plt.figure(figsize=(14, 10))
    
    for asset in volatility_df.columns:
        plt.plot(volatility_df.index, volatility_df[asset], label=asset)
    
    plt.title(f"{window}-Period Rolling Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved volatility plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_correlation_analysis_report(pairs, timeframe, window=30):
    """
    Generate a comprehensive correlation analysis report
    
    Args:
        pairs (list): List of trading pairs
        timeframe (str): Timeframe code
        window (int): Rolling window size
        
    Returns:
        dict: Analysis results
    """
    logger.info(f"Generating correlation analysis for {len(pairs)} pairs on {timeframe} timeframe")
    
    # Load and prepare data
    returns_df = create_multi_asset_dataframe(pairs, timeframe)
    if returns_df is None or returns_df.empty:
        logger.error("Failed to create returns DataFrame")
        return None
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(returns_df, method='pearson')
    
    # Calculate rolling correlations
    rolling_corrs = calculate_rolling_correlation(returns_df, window=window)
    
    # Calculate asset betas
    betas = calculate_beta(returns_df)
    
    # Calculate volatility
    volatility = calculate_volatility(returns_df, window=window)
    
    # Cluster assets by return patterns
    n_clusters = min(3, len(pairs))
    cluster_labels, _ = cluster_assets_by_returns(returns_df, n_clusters=n_clusters)
    
    # Prepare report data
    report = {
        "timeframe": timeframe,
        "window": window,
        "correlation_matrix": corr_matrix.to_dict() if corr_matrix is not None else None,
        "betas": betas.to_dict() if betas is not None else None,
        "clusters": {pairs[i]: int(cluster_labels[i]) for i in range(len(pairs))} if cluster_labels is not None else None,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save plots
    plot_folder = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    
    # Plot correlation matrix
    corr_plot_path = os.path.join(plot_folder, f"correlation_matrix_{timeframe}.png")
    plot_correlation_matrix(corr_matrix, f"Asset Correlation Matrix ({timeframe})", corr_plot_path)
    
    # Plot rolling correlations
    rolling_plot_path = os.path.join(plot_folder, f"rolling_correlation_{timeframe}_{window}.png")
    plot_rolling_correlations(returns_df, rolling_corrs, window, rolling_plot_path)
    
    # Plot volatility
    vol_plot_path = os.path.join(plot_folder, f"volatility_{timeframe}_{window}.png")
    plot_volatility(volatility, window, vol_plot_path)
    
    # Save report
    report_file = os.path.join(RESULTS_DIR, f"correlation_analysis_{timeframe}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved correlation analysis report to {report_file}")
    
    return report


def main():
    """Main function to run the correlation analysis"""
    for timeframe in TIMEFRAMES:
        generate_correlation_analysis_report(PAIRS, timeframe)


if __name__ == "__main__":
    main()