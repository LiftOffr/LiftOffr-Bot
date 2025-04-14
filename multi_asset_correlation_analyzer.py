#!/usr/bin/env python3
"""
Multi-Asset Correlation Analyzer

This module analyzes correlations between multiple cryptocurrency assets to discover
predictive relationships that can improve trading performance across the entire portfolio.

Key features:
1. Lead-lag relationship detection between assets
2. Correlation stability analysis across different market regimes
3. Cointegration testing for long-term equilibrium relationships
4. Dynamic correlation coefficient calculation
5. Visualization of cross-asset relationships
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CORRELATION_DIR = "correlation_analysis"
os.makedirs(CORRELATION_DIR, exist_ok=True)


class MultiAssetCorrelationAnalyzer:
    """
    Analyzes correlations and relationships between multiple cryptocurrency assets
    to discover predictive signals and improve portfolio-wide trading performance.
    """
    
    def __init__(self, assets=None, timeframes=None):
        """
        Initialize the correlation analyzer
        
        Args:
            assets (list): List of assets to analyze (e.g., ["BTC/USD", "ETH/USD"])
            timeframes (list): List of timeframes to analyze (e.g., ["1d", "4h", "1h"])
        """
        self.assets = assets or ["BTC/USD", "ETH/USD", "SOL/USD", "DOT/USD", "LINK/USD"]
        self.timeframes = timeframes or ["1d", "4h", "1h"]
        self.data_dict = {}
        self.correlation_matrices = {}
        self.lead_lag_relationships = {}
        self.cointegration_results = {}
        
        logger.info(f"Initialized Multi-Asset Correlation Analyzer for {len(self.assets)} assets")
    
    def load_data(self, data_source=None):
        """
        Load historical data for all assets
        
        Args:
            data_source (dict, optional): Data dictionary with asset data
            
        Returns:
            bool: Success indicator
        """
        if data_source:
            self.data_dict = data_source
            logger.info(f"Loaded data from provided source: {len(self.data_dict)} assets")
            return True
        
        # Load from historical data directory if no source provided
        try:
            self.data_dict = {}
            historical_data_dir = "historical_data"
            
            for asset in self.assets:
                formatted_asset = asset.replace("/", "")
                asset_data = {}
                
                for timeframe in self.timeframes:
                    file_path = os.path.join(historical_data_dir, formatted_asset, f"{formatted_asset}_{timeframe}.csv")
                    
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        
                        # Ensure timestamp is datetime
                        if "timestamp" in df.columns:
                            df["timestamp"] = pd.to_datetime(df["timestamp"])
                            df.set_index("timestamp", inplace=True)
                        
                        asset_data[timeframe] = df
                        logger.info(f"Loaded {len(df)} records for {asset} ({timeframe})")
                    else:
                        logger.warning(f"Data file not found: {file_path}")
                
                if asset_data:
                    self.data_dict[asset] = asset_data
            
            logger.info(f"Loaded historical data for {len(self.data_dict)} assets")
            return len(self.data_dict) > 0
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def calculate_correlation_matrices(self, timeframe="1d", window=30, method="pearson"):
        """
        Calculate correlation matrices for all assets
        
        Args:
            timeframe (str): Timeframe to analyze
            window (int): Rolling window size for correlation calculation
            method (str): Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            dict: Correlation matrices
        """
        # Create a combined DataFrame with closing prices
        close_dfs = {}
        
        for asset, data in self.data_dict.items():
            if timeframe in data:
                df = data[timeframe].copy()
                if "close" in df.columns:
                    close_dfs[asset] = df["close"]
        
        if len(close_dfs) < 2:
            logger.warning("Not enough assets with data to calculate correlations")
            return {}
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(close_dfs)
        
        # Calculate static correlation
        static_corr = combined_df.corr(method=method)
        
        # Calculate rolling correlation
        correlation_over_time = {}
        
        for asset1 in self.assets:
            if asset1 not in combined_df.columns:
                continue
                
            for asset2 in self.assets:
                if asset2 not in combined_df.columns or asset1 == asset2:
                    continue
                
                # Calculate rolling correlation
                rolling_corr = combined_df[asset1].rolling(window=window).corr(combined_df[asset2])
                
                # Store in dictionary
                pair = (asset1, asset2)
                correlation_over_time[pair] = rolling_corr
        
        # Store results
        self.correlation_matrices[timeframe] = {
            "static": static_corr,
            "rolling": correlation_over_time,
            "window": window,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Calculated correlation matrices for {timeframe} data")
        return self.correlation_matrices[timeframe]
    
    def detect_lead_lag_relationships(self, timeframe="1d", max_lag=5):
        """
        Detect lead-lag relationships between assets using Granger causality
        
        Args:
            timeframe (str): Timeframe to analyze
            max_lag (int): Maximum lag to test
            
        Returns:
            dict: Lead-lag relationship results
        """
        # Create a combined DataFrame with closing prices
        close_dfs = {}
        
        for asset, data in self.data_dict.items():
            if timeframe in data:
                df = data[timeframe].copy()
                if "close" in df.columns:
                    # Calculate returns to ensure stationarity
                    close_dfs[asset] = df["close"].pct_change().dropna()
        
        if len(close_dfs) < 2:
            logger.warning("Not enough assets with data to detect lead-lag relationships")
            return {}
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(close_dfs)
        
        # Test for Granger causality
        results = {}
        
        for asset1 in combined_df.columns:
            for asset2 in combined_df.columns:
                if asset1 == asset2:
                    continue
                
                # Skip if either series has NaN values
                if combined_df[asset1].isna().any() or combined_df[asset2].isna().any():
                    continue
                
                # Test if asset1 Granger-causes asset2
                try:
                    pair_data = combined_df[[asset1, asset2]].dropna()
                    test_result = grangercausalitytests(pair_data, maxlag=max_lag, verbose=False)
                    
                    # Extract p-values for each lag
                    p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                    
                    # Check if any lag has significant p-value
                    min_p_value = min(p_values)
                    significant_lag = p_values.index(min_p_value) + 1 if min_p_value < 0.05 else None
                    
                    # Store result
                    results[(asset1, asset2)] = {
                        "p_values": p_values,
                        "min_p_value": min_p_value,
                        "significant_lag": significant_lag,
                        "has_causality": min_p_value < 0.05
                    }
                    
                except Exception as e:
                    logger.error(f"Error testing Granger causality for {asset1} -> {asset2}: {e}")
        
        # Store results
        self.lead_lag_relationships[timeframe] = {
            "results": results,
            "max_lag": max_lag,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log significant relationships
        significant_relationships = [(a1, a2) for (a1, a2), res in results.items() if res["has_causality"]]
        logger.info(f"Detected {len(significant_relationships)} significant lead-lag relationships for {timeframe} data")
        
        return self.lead_lag_relationships[timeframe]
    
    def test_cointegration(self, timeframe="1d"):
        """
        Test for cointegration between asset pairs
        
        Args:
            timeframe (str): Timeframe to analyze
            
        Returns:
            dict: Cointegration test results
        """
        # Create a combined DataFrame with closing prices
        close_dfs = {}
        
        for asset, data in self.data_dict.items():
            if timeframe in data:
                df = data[timeframe].copy()
                if "close" in df.columns:
                    close_dfs[asset] = df["close"]
        
        if len(close_dfs) < 2:
            logger.warning("Not enough assets with data to test cointegration")
            return {}
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(close_dfs)
        
        # Test for cointegration
        results = {}
        
        for i, asset1 in enumerate(combined_df.columns):
            for asset2 in list(combined_df.columns)[i+1:]:
                # Skip if either series has NaN values
                if combined_df[asset1].isna().any() or combined_df[asset2].isna().any():
                    continue
                
                # Test for cointegration
                try:
                    pair_data = combined_df[[asset1, asset2]].dropna()
                    score, p_value, _ = coint(pair_data[asset1], pair_data[asset2])
                    
                    # Calculate spread (might be used for pairs trading)
                    spread = pair_data[asset1] - (pair_data[asset2] * (pair_data[asset1].mean() / pair_data[asset2].mean()))
                    
                    # Test spread for stationarity
                    adf_result = adfuller(spread)
                    
                    # Store result
                    results[(asset1, asset2)] = {
                        "score": score,
                        "p_value": p_value,
                        "is_cointegrated": p_value < 0.05,
                        "adf_test": {
                            "statistic": adf_result[0],
                            "p_value": adf_result[1],
                            "is_stationary": adf_result[1] < 0.05
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Error testing cointegration for {asset1} and {asset2}: {e}")
        
        # Store results
        self.cointegration_results[timeframe] = {
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log cointegrated pairs
        cointegrated_pairs = [(a1, a2) for (a1, a2), res in results.items() if res["is_cointegrated"]]
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs for {timeframe} data")
        
        return self.cointegration_results[timeframe]
    
    def calculate_pca_components(self, timeframe="1d", n_components=3):
        """
        Calculate principal components of asset returns
        
        Args:
            timeframe (str): Timeframe to analyze
            n_components (int): Number of principal components to extract
            
        Returns:
            dict: PCA results
        """
        # Create a combined DataFrame with returns
        return_dfs = {}
        
        for asset, data in self.data_dict.items():
            if timeframe in data:
                df = data[timeframe].copy()
                if "close" in df.columns:
                    return_dfs[asset] = df["close"].pct_change().dropna()
        
        if len(return_dfs) < 3:  # Need at least 3 assets for meaningful PCA
            logger.warning("Not enough assets with data to perform PCA")
            return None
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(return_dfs).dropna()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_df)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(combined_df.columns)))
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame with principal components
        pca_df = pd.DataFrame(
            pca_result,
            index=combined_df.index,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        
        # Store results
        pca_results = {
            "pca_df": pca_df,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "components": pca.components_,
            "feature_names": combined_df.columns.tolist()
        }
        
        logger.info(f"Calculated {pca.n_components_} principal components for {timeframe} data")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_}")
        
        return pca_results
    
    def identify_market_leading_assets(self, timeframe="1d"):
        """
        Identify market-leading assets based on lead-lag relationships
        
        Args:
            timeframe (str): Timeframe to analyze
            
        Returns:
            dict: Market leaders with scores
        """
        # Ensure lead-lag relationships are calculated
        if timeframe not in self.lead_lag_relationships:
            self.detect_lead_lag_relationships(timeframe)
        
        if timeframe not in self.lead_lag_relationships:
            logger.warning(f"No lead-lag relationships found for {timeframe}")
            return {}
        
        # Calculate leadership scores
        results = self.lead_lag_relationships[timeframe]["results"]
        leadership_scores = {asset: 0 for asset in self.assets}
        
        for (asset1, asset2), res in results.items():
            if res["has_causality"]:
                # Asset1 leads Asset2
                leadership_scores[asset1] += 1
        
        # Normalize scores
        max_score = max(leadership_scores.values())
        if max_score > 0:
            normalized_scores = {asset: score / max_score for asset, score in leadership_scores.items()}
        else:
            normalized_scores = leadership_scores
        
        # Rank assets
        ranked_assets = sorted([(asset, score) for asset, score in normalized_scores.items()], 
                              key=lambda x: x[1], reverse=True)
        
        logger.info(f"Identified market leaders for {timeframe}: {ranked_assets[:3]}")
        
        return {
            "ranked_assets": ranked_assets,
            "leadership_scores": leadership_scores,
            "normalized_scores": normalized_scores
        }
    
    def get_highly_correlated_pairs(self, timeframe="1d", threshold=0.7):
        """
        Get pairs of assets with correlation above threshold
        
        Args:
            timeframe (str): Timeframe to analyze
            threshold (float): Correlation threshold (0-1)
            
        Returns:
            list: Pairs of highly correlated assets
        """
        # Ensure correlation matrices are calculated
        if timeframe not in self.correlation_matrices:
            self.calculate_correlation_matrices(timeframe)
        
        if timeframe not in self.correlation_matrices:
            logger.warning(f"No correlation matrix found for {timeframe}")
            return []
        
        # Get static correlation matrix
        corr_matrix = self.correlation_matrices[timeframe]["static"]
        
        # Find pairs above threshold
        highly_correlated = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1 = corr_matrix.columns[i]
                asset2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    highly_correlated.append((asset1, asset2, corr_val))
        
        # Sort by correlation (absolute value)
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        logger.info(f"Found {len(highly_correlated)} pairs with correlation above {threshold} for {timeframe}")
        
        return highly_correlated
    
    def generate_correlation_report(self, output_path=None):
        """
        Generate a comprehensive correlation analysis report
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            dict: Report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "assets_analyzed": self.assets,
            "timeframes_analyzed": list(self.correlation_matrices.keys()),
            "correlation_summary": {},
            "lead_lag_summary": {},
            "cointegration_summary": {},
            "market_leaders": {}
        }
        
        # Generate summaries for each timeframe
        for timeframe in self.correlation_matrices.keys():
            # Correlation summary
            highly_correlated = self.get_highly_correlated_pairs(timeframe, threshold=0.7)
            negatively_correlated = [(a1, a2, corr) for a1, a2, corr in 
                                     self.get_highly_correlated_pairs(timeframe, threshold=0.5) 
                                     if corr < 0]
            
            report["correlation_summary"][timeframe] = {
                "highly_correlated_pairs": highly_correlated,
                "negatively_correlated_pairs": negatively_correlated
            }
            
            # Lead-lag summary
            if timeframe in self.lead_lag_relationships:
                lead_lag_results = self.lead_lag_relationships[timeframe]["results"]
                significant_relationships = [(a1, a2) for (a1, a2), res in lead_lag_results.items() 
                                            if res["has_causality"]]
                
                report["lead_lag_summary"][timeframe] = {
                    "significant_relationships": significant_relationships
                }
            
            # Cointegration summary
            if timeframe in self.cointegration_results:
                coint_results = self.cointegration_results[timeframe]["results"]
                cointegrated_pairs = [(a1, a2) for (a1, a2), res in coint_results.items() 
                                     if res["is_cointegrated"]]
                
                report["cointegration_summary"][timeframe] = {
                    "cointegrated_pairs": cointegrated_pairs
                }
            
            # Market leaders
            market_leaders = self.identify_market_leading_assets(timeframe)
            report["market_leaders"][timeframe] = market_leaders
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved correlation report to {output_path}")
        
        return report
    
    def plot_correlation_matrix(self, timeframe="1d", output_path=None):
        """
        Plot correlation matrix as a heatmap
        
        Args:
            timeframe (str): Timeframe to plot
            output_path (str, optional): Path to save the plot
        """
        if timeframe not in self.correlation_matrices:
            logger.warning(f"No correlation matrix found for {timeframe}")
            return
        
        # Get static correlation matrix
        corr_matrix = self.correlation_matrices[timeframe]["static"]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.matshow(corr_matrix, fignum=1, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add labels
        assets = corr_matrix.columns
        plt.xticks(range(len(assets)), assets, rotation=45)
        plt.yticks(range(len(assets)), assets)
        
        # Add colorbar
        plt.colorbar(label='Correlation Coefficient')
        
        # Add title
        plt.title(f'Asset Correlation Matrix ({timeframe})')
        
        # Add text annotations
        for i in range(len(assets)):
            for j in range(len(assets)):
                plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                         ha="center", va="center", 
                         color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved correlation matrix plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rolling_correlation(self, asset1, asset2, timeframe="1d", output_path=None):
        """
        Plot rolling correlation between two assets
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset
            timeframe (str): Timeframe to plot
            output_path (str, optional): Path to save the plot
        """
        if timeframe not in self.correlation_matrices:
            logger.warning(f"No correlation matrix found for {timeframe}")
            return
        
        # Get rolling correlation
        if (asset1, asset2) in self.correlation_matrices[timeframe]["rolling"]:
            rolling_corr = self.correlation_matrices[timeframe]["rolling"][(asset1, asset2)]
        elif (asset2, asset1) in self.correlation_matrices[timeframe]["rolling"]:
            rolling_corr = self.correlation_matrices[timeframe]["rolling"][(asset2, asset1)]
        else:
            logger.warning(f"No rolling correlation found for {asset1} and {asset2}")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot rolling correlation
        rolling_corr.plot()
        
        # Add horizontal lines
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3)
        plt.axhline(y=-0.5, color='r', linestyle='--', alpha=0.3)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Correlation Coefficient')
        plt.title(f'Rolling Correlation: {asset1} vs {asset2} ({timeframe})')
        plt.ylim(-1.1, 1.1)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved rolling correlation plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_full_analysis(self, timeframes=None, output_dir=None):
        """
        Run a full correlation analysis for all assets and timeframes
        
        Args:
            timeframes (list, optional): Timeframes to analyze
            output_dir (str, optional): Directory to save output files
            
        Returns:
            dict: Analysis results
        """
        if not timeframes:
            timeframes = self.timeframes
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = CORRELATION_DIR
        
        # Results container
        results = {}
        
        # Run analysis for each timeframe
        for timeframe in timeframes:
            logger.info(f"Running correlation analysis for {timeframe}")
            
            # Calculate correlation matrices
            self.calculate_correlation_matrices(timeframe)
            
            # Detect lead-lag relationships
            self.detect_lead_lag_relationships(timeframe)
            
            # Test cointegration
            self.test_cointegration(timeframe)
            
            # Identify market leaders
            market_leaders = self.identify_market_leading_assets(timeframe)
            
            # Generate plots
            if output_dir:
                # Correlation matrix
                matrix_path = os.path.join(output_dir, f"correlation_matrix_{timeframe}.png")
                self.plot_correlation_matrix(timeframe, matrix_path)
                
                # Rolling correlations for top pairs
                highly_correlated = self.get_highly_correlated_pairs(timeframe, threshold=0.7)
                for i, (asset1, asset2, _) in enumerate(highly_correlated[:5]):
                    rolling_path = os.path.join(output_dir, f"rolling_corr_{asset1.replace('/', '_')}_{asset2.replace('/', '_')}_{timeframe}.png")
                    self.plot_rolling_correlation(asset1, asset2, timeframe, rolling_path)
            
            # Store results
            results[timeframe] = {
                "correlation_matrices": True,
                "lead_lag_relationships": True,
                "cointegration_results": True,
                "market_leaders": market_leaders["ranked_assets"]
            }
        
        # Generate comprehensive report
        report_path = os.path.join(output_dir, f"correlation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report = self.generate_correlation_report(report_path)
        
        logger.info(f"Completed full correlation analysis for {len(timeframes)} timeframes")
        
        return report

def main():
    """
    Main function to run correlation analysis independently
    """
    # Create and configure analyzer
    analyzer = MultiAssetCorrelationAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        logger.error("Failed to load data. Exiting.")
        return 1
    
    # Run full analysis
    analyzer.run_full_analysis(output_dir=CORRELATION_DIR)
    
    logger.info("Analysis completed successfully")
    
    return 0

if __name__ == "__main__":
    main()