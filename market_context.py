#!/usr/bin/env python3
"""
Market Context Awareness Module for Kraken Trading Bot

This module provides functionality to analyze broader market conditions
and incorporate market context into trading decisions.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('market_context.log'),
        logging.StreamHandler()
    ]
)

# Constants
DEFAULT_CONFIG_FILE = 'config.py'
MARKET_CONTEXT_DIR = 'market_context'
CACHE_DURATION = 3600  # 1 hour cache duration


class MarketContext:
    """Market context analyzer for trading bot"""
    
    def __init__(self, cache_dir=MARKET_CONTEXT_DIR, cache_duration=CACHE_DURATION):
        """
        Initialize the market context analyzer
        
        Args:
            cache_dir (str): Directory for caching market data
            cache_duration (int): Cache duration in seconds
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize cache
        self.cache = {}
    
    def get_market_context(self, trading_pair: str = "SOL/USD") -> Dict:
        """
        Get comprehensive market context
        
        Args:
            trading_pair (str): Trading pair to analyze
            
        Returns:
            dict: Market context data
        """
        try:
            # Extract base asset from trading pair
            base_asset = trading_pair.split('/')[0].lower()
            
            # Get market data for different timeframes
            market_context = {
                'timestamp': datetime.now().isoformat(),
                'trading_pair': trading_pair,
                'base_asset': base_asset,
                'general_market': self.get_general_market_data(),
                'asset_specific': self.get_asset_specific_data(base_asset),
                'correlations': self.get_correlation_data(base_asset),
                'market_sentiment': self.get_market_sentiment(base_asset),
                'volatility_metrics': self.get_volatility_metrics(trading_pair)
            }
            
            return market_context
        
        except Exception as e:
            logging.error(f"Error getting market context: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'trading_pair': trading_pair,
                'error': str(e)
            }
    
    def get_general_market_data(self) -> Dict:
        """
        Get general crypto market data (BTC dominance, total cap, etc.)
        
        Returns:
            dict: General market data
        """
        # Use cached data if available and not expired
        cache_key = 'general_market'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For this demo, we'll create synthetic data
            # In a real implementation, we would fetch data from external APIs
            
            # Example data structure with relevant metrics
            general_data = {
                'btc_dominance': 0.47,  # BTC market dominance percentage
                'total_market_cap': 1_970_000_000_000,  # Total crypto market cap in USD
                'defi_tvl': 75_000_000_000,  # Total Value Locked in DeFi in USD
                'btc_price': 43_500.0,  # Current BTC price in USD
                'eth_price': 2_700.0,  # Current ETH price in USD
                'market_trend': self._calculate_market_trend(),  # Overall market trend
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, general_data)
            
            return general_data
        
        except Exception as e:
            logging.error(f"Error getting general market data: {str(e)}")
            return {}
    
    def get_asset_specific_data(self, asset: str) -> Dict:
        """
        Get specific data for a crypto asset
        
        Args:
            asset (str): Asset symbol (e.g. 'sol')
            
        Returns:
            dict: Asset-specific data
        """
        # Use cached data if available and not expired
        cache_key = f'asset_specific_{asset}'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For this demo, we'll create synthetic data
            # In a real implementation, we would fetch data from external APIs
            
            # Example data structure with relevant metrics
            asset_data = {
                'market_cap': 35_000_000_000,  # Market cap in USD
                'volume_24h': 1_200_000_000,  # 24h trading volume in USD
                'change_24h': -0.025,  # 24h price change percentage
                'change_7d': 0.045,  # 7d price change percentage
                'rank': 8,  # Market cap rank
                'developer_activity': 'high',  # Developer activity level
                'social_volume': 'medium',  # Social media activity level
                'network_health': 0.82,  # Network health score (0-1)
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, asset_data)
            
            return asset_data
        
        except Exception as e:
            logging.error(f"Error getting asset-specific data for {asset}: {str(e)}")
            return {}
    
    def get_correlation_data(self, asset: str) -> Dict:
        """
        Get correlation data for an asset with major crypto assets
        
        Args:
            asset (str): Asset symbol (e.g. 'sol')
            
        Returns:
            dict: Correlation data
        """
        # Use cached data if available and not expired
        cache_key = f'correlation_{asset}'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For this demo, we'll create synthetic data
            # In a real implementation, we would calculate real correlations
            
            # Example data structure with asset correlations
            correlation_data = {
                'btc_correlation': 0.78,  # Correlation with BTC (-1 to 1)
                'eth_correlation': 0.85,  # Correlation with ETH (-1 to 1)
                'total_market_correlation': 0.82,  # Correlation with total market (-1 to 1)
                'sector_correlation': 0.92,  # Correlation with sector (e.g. L1 blockchains) (-1 to 1)
                'correlations': {
                    'btc': 0.78,
                    'eth': 0.85,
                    'bnb': 0.71,
                    'xrp': 0.65,
                    'ada': 0.72,
                    'avax': 0.89
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, correlation_data)
            
            return correlation_data
        
        except Exception as e:
            logging.error(f"Error getting correlation data for {asset}: {str(e)}")
            return {}
    
    def get_market_sentiment(self, asset: str) -> Dict:
        """
        Get market sentiment data for an asset
        
        Args:
            asset (str): Asset symbol (e.g. 'sol')
            
        Returns:
            dict: Market sentiment data
        """
        # Use cached data if available and not expired
        cache_key = f'sentiment_{asset}'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For this demo, we'll create synthetic data
            # In a real implementation, we would calculate real sentiment
            
            # Example data structure with sentiment metrics
            sentiment_data = {
                'fear_greed_index': 65,  # Fear and Greed Index (0-100)
                'social_sentiment': 0.62,  # Social media sentiment (-1 to 1)
                'news_sentiment': 0.45,  # News sentiment (-1 to 1)
                'funding_rate': 0.0012,  # Funding rate on perpetual futures
                'long_short_ratio': 1.25,  # Long/short positions ratio
                'trend_strength': 'medium',  # Trend strength indicator
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, sentiment_data)
            
            return sentiment_data
        
        except Exception as e:
            logging.error(f"Error getting market sentiment for {asset}: {str(e)}")
            return {}
    
    def get_volatility_metrics(self, trading_pair: str) -> Dict:
        """
        Get volatility metrics for a trading pair
        
        Args:
            trading_pair (str): Trading pair (e.g. 'SOL/USD')
            
        Returns:
            dict: Volatility metrics
        """
        # Use cached data if available and not expired
        cache_key = f'volatility_{trading_pair}'
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # For this demo, we'll create synthetic data
            # In a real implementation, we would calculate real volatility metrics
            
            # Example data structure with volatility metrics
            volatility_data = {
                'historical_volatility': 0.035,  # 30-day historical volatility
                'implied_volatility': 0.042,  # Current implied volatility
                'volatility_rank': 0.65,  # Volatility percentile rank (0-1)
                'volatility_trend': 'increasing',  # Volatility trend
                'bollinger_bandwidth': 0.035,  # Bollinger Bands width / price ratio
                'atr_percentage': 0.018,  # ATR as percentage of price
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self._cache_data(cache_key, volatility_data)
            
            return volatility_data
        
        except Exception as e:
            logging.error(f"Error getting volatility metrics for {trading_pair}: {str(e)}")
            return {}
    
    def analyze_market_context(self, trading_pair: str = "SOL/USD") -> Dict:
        """
        Analyze market context and provide trading recommendations
        
        Args:
            trading_pair (str): Trading pair to analyze
            
        Returns:
            dict: Market analysis and recommendations
        """
        try:
            # Get market context
            context = self.get_market_context(trading_pair)
            
            # Extract base asset from trading pair
            base_asset = trading_pair.split('/')[0].lower()
            
            # Analyze market conditions
            market_trend = context['general_market'].get('market_trend', 'neutral')
            asset_trend = 'bullish' if context['asset_specific'].get('change_7d', 0) > 0 else 'bearish'
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(context)
            
            # Determine optimal position size as percentage (0-1)
            optimal_position_size = self._calculate_optimal_position_size(context, risk_score)
            
            # Determine trade direction bias
            direction_bias = self._calculate_direction_bias(context)
            
            # Provide recommendations
            recommendations = self._generate_recommendations(context, risk_score, direction_bias)
            
            # Return analysis
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'trading_pair': trading_pair,
                'market_trend': market_trend,
                'asset_trend': asset_trend,
                'risk_score': risk_score,
                'optimal_position_size': optimal_position_size,
                'direction_bias': direction_bias,
                'recommendations': recommendations
            }
            
            return analysis
        
        except Exception as e:
            logging.error(f"Error analyzing market context: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'trading_pair': trading_pair,
                'error': str(e)
            }
    
    def _calculate_market_trend(self) -> str:
        """
        Calculate overall market trend
        
        Returns:
            str: Market trend ('strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish')
        """
        # In a real implementation, we would calculate trend based on multiple indicators
        # For this demo, we'll return a random trend weighted toward neutral
        trends = ['strongly_bullish', 'bullish', 'neutral', 'bearish', 'strongly_bearish']
        weights = [0.1, 0.25, 0.3, 0.25, 0.1]  # Weighted toward neutral
        
        return np.random.choice(trends, p=weights)
    
    def _calculate_risk_score(self, context: Dict) -> float:
        """
        Calculate risk score based on market context
        
        Args:
            context (dict): Market context data
            
        Returns:
            float: Risk score (0-100, where 100 is highest risk)
        """
        risk_score = 50.0  # Start at neutral
        
        # Adjust based on general market
        general = context.get('general_market', {})
        if general:
            if general.get('market_trend') == 'strongly_bullish':
                risk_score -= 15
            elif general.get('market_trend') == 'bullish':
                risk_score -= 10
            elif general.get('market_trend') == 'bearish':
                risk_score += 10
            elif general.get('market_trend') == 'strongly_bearish':
                risk_score += 15
        
        # Adjust based on asset specifics
        asset = context.get('asset_specific', {})
        if asset:
            # Higher volume is lower risk
            volume_factor = min(1.0, asset.get('volume_24h', 0) / 2_000_000_000)
            risk_score -= volume_factor * 10
            
            # Recent price changes
            if asset.get('change_24h', 0) > 0.1:
                risk_score += 10  # Large positive change increases risk (potential reversal)
            elif asset.get('change_24h', 0) < -0.1:
                risk_score += 15  # Large negative change increases risk
        
        # Adjust based on volatility
        volatility = context.get('volatility_metrics', {})
        if volatility:
            # Higher volatility means higher risk
            vol_value = volatility.get('historical_volatility', 0.03)
            if vol_value > 0.05:
                risk_score += 20 * (vol_value - 0.05) / 0.05
            
            # Increasing volatility means higher risk
            if volatility.get('volatility_trend') == 'increasing':
                risk_score += 10
        
        # Adjust based on sentiment
        sentiment = context.get('market_sentiment', {})
        if sentiment:
            fear_greed = sentiment.get('fear_greed_index', 50)
            if fear_greed < 25:
                risk_score += 15  # Extreme fear increases risk
            elif fear_greed > 75:
                risk_score += 15  # Extreme greed increases risk
        
        # Cap the risk score between 0 and 100
        return max(0, min(100, risk_score))
    
    def _calculate_optimal_position_size(self, context: Dict, risk_score: float) -> float:
        """
        Calculate optimal position size based on market context and risk score
        
        Args:
            context (dict): Market context data
            risk_score (float): Risk score (0-100)
            
        Returns:
            float: Optimal position size as percentage of maximum (0-1)
        """
        # Convert risk score to a position size inversely
        # Higher risk = smaller position
        base_size = 1.0 - (risk_score / 100.0) * 0.8  # Scale to 0.2-1.0
        
        # Adjust based on directional confidence
        direction_bias = self._calculate_direction_bias(context)
        confidence_factor = abs(direction_bias) * 0.4 + 0.6  # Scale to 0.6-1.0
        
        # Apply adjustments
        adjusted_size = base_size * confidence_factor
        
        # Cap the position size between 0.1 and 1.0
        return max(0.1, min(1.0, adjusted_size))
    
    def _calculate_direction_bias(self, context: Dict) -> float:
        """
        Calculate directional bias based on market context
        
        Args:
            context (dict): Market context data
            
        Returns:
            float: Direction bias (-1.0 to 1.0, where -1.0 is strongly bearish, 1.0 is strongly bullish)
        """
        bias = 0.0  # Start at neutral
        
        # Adjust based on general market
        general = context.get('general_market', {})
        if general:
            if general.get('market_trend') == 'strongly_bullish':
                bias += 0.3
            elif general.get('market_trend') == 'bullish':
                bias += 0.2
            elif general.get('market_trend') == 'bearish':
                bias -= 0.2
            elif general.get('market_trend') == 'strongly_bearish':
                bias -= 0.3
        
        # Adjust based on asset specifics
        asset = context.get('asset_specific', {})
        if asset:
            # Recent price changes
            bias += asset.get('change_7d', 0) * 2  # 7-day change has more impact
            bias += asset.get('change_24h', 0)  # Recent change has less impact
        
        # Adjust based on sentiment
        sentiment = context.get('market_sentiment', {})
        if sentiment:
            # Social and news sentiment
            bias += sentiment.get('social_sentiment', 0) * 0.2
            bias += sentiment.get('news_sentiment', 0) * 0.15
            
            # Funding rate (negative funding rate is bullish)
            funding_rate = sentiment.get('funding_rate', 0)
            bias -= funding_rate * 30  # Scale to significant impact
            
            # Long/short ratio
            long_short = sentiment.get('long_short_ratio', 1.0)
            if long_short > 1.5:
                bias += 0.1  # Strong long bias
            elif long_short < 0.7:
                bias -= 0.1  # Strong short bias
        
        # Cap the bias between -1.0 and 1.0
        return max(-1.0, min(1.0, bias))
    
    def _generate_recommendations(self, context: Dict, risk_score: float, direction_bias: float) -> List[str]:
        """
        Generate trading recommendations based on analysis
        
        Args:
            context (dict): Market context data
            risk_score (float): Risk score (0-100)
            direction_bias (float): Direction bias (-1.0 to 1.0)
            
        Returns:
            list: Trading recommendations
        """
        recommendations = []
        
        # Risk level recommendation
        if risk_score > 75:
            recommendations.append("Very high market risk detected. Consider waiting for stabilization.")
        elif risk_score > 60:
            recommendations.append("Elevated market risk. Reduce position sizes by 30-50%.")
        elif risk_score < 25:
            recommendations.append("Low market risk. Favorable conditions for normal position sizing.")
        
        # Direction bias recommendation
        if direction_bias > 0.5:
            recommendations.append("Strong bullish bias. Favor long positions and limit short exposure.")
        elif direction_bias > 0.2:
            recommendations.append("Moderate bullish bias. Slight preference for long positions.")
        elif direction_bias < -0.5:
            recommendations.append("Strong bearish bias. Favor short positions and limit long exposure.")
        elif direction_bias < -0.2:
            recommendations.append("Moderate bearish bias. Slight preference for short positions.")
        else:
            recommendations.append("Neutral market bias. No directional advantage detected.")
        
        # Volatility-based recommendation
        volatility = context.get('volatility_metrics', {})
        if volatility:
            if volatility.get('volatility_trend') == 'increasing':
                recommendations.append("Increasing volatility detected. Widen stop loss levels and consider reducing leverage.")
            if volatility.get('historical_volatility', 0) > 0.05:
                recommendations.append("High volatility environment. Use wider stop losses and reduced position sizes.")
            elif volatility.get('historical_volatility', 0) < 0.02:
                recommendations.append("Low volatility environment. Consider strategies that benefit from volatility expansion.")
        
        # Correlation-based recommendation
        correlation = context.get('correlations', {})
        if correlation:
            btc_corr = correlation.get('btc_correlation', 0)
            if btc_corr > 0.8:
                recommendations.append("High correlation with BTC. Monitor BTC for potential direction changes.")
        
        # Sentiment-based recommendation
        sentiment = context.get('market_sentiment', {})
        if sentiment:
            fear_greed = sentiment.get('fear_greed_index', 50)
            if fear_greed < 20:
                recommendations.append("Extreme fear detected. Potential contrarian buying opportunity if other factors align.")
            elif fear_greed > 80:
                recommendations.append("Extreme greed detected. Consider caution with new long positions.")
        
        return recommendations
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """
        Get data from cache if available and not expired
        
        Args:
            key (str): Cache key
            
        Returns:
            dict or None: Cached data or None if not available
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_path):
            return None
        
        # Check if cache is expired
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime > self.cache_duration:
            return None
        
        # Read cache file
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _cache_data(self, key: str, data: Dict) -> None:
        """
        Cache data to file
        
        Args:
            key (str): Cache key
            data (dict): Data to cache
        """
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error caching data: {str(e)}")


def adjust_signal_strength(signal_strength: float, direction: str, market_context: Dict) -> float:
    """
    Adjust signal strength based on market context
    
    Args:
        signal_strength (float): Original signal strength (0-1)
        direction (str): Signal direction ('BUY' or 'SELL')
        market_context (dict): Market context data
        
    Returns:
        float: Adjusted signal strength (0-1)
    """
    if not market_context:
        return signal_strength
    
    direction_bias = market_context.get('direction_bias', 0)
    risk_score = market_context.get('risk_score', 50)
    
    # Normalize risk score to 0-1 range
    normalized_risk = risk_score / 100.0
    
    # Adjust signal strength based on direction and market bias
    if direction == 'BUY':
        # Boost buy signals when bullish, reduce when bearish
        adjustment = direction_bias * 0.2
    else:  # SELL
        # Boost sell signals when bearish, reduce when bullish
        adjustment = -direction_bias * 0.2
    
    # Reduce signal strength in high-risk environments
    risk_adjustment = -0.15 * (normalized_risk - 0.5)
    
    # Apply adjustments
    adjusted_strength = signal_strength + adjustment + risk_adjustment
    
    # Cap the result between 0 and 1
    return max(0.0, min(1.0, adjusted_strength))


def main():
    parser = argparse.ArgumentParser(description='Market Context Analyzer')
    parser.add_argument('--trading-pair', type=str, default='SOL/USD', help='Trading pair to analyze')
    parser.add_argument('--cache-dir', type=str, default=MARKET_CONTEXT_DIR, help='Cache directory')
    parser.add_argument('--cache-duration', type=int, default=CACHE_DURATION, help='Cache duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize market context analyzer
    context_analyzer = MarketContext(
        cache_dir=args.cache_dir,
        cache_duration=args.cache_duration
    )
    
    # Get and print market context
    analysis = context_analyzer.analyze_market_context(args.trading_pair)
    
    # Print results in a readable format
    print("\n" + "=" * 80)
    print(f"MARKET CONTEXT ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print(f"\nTrading Pair: {analysis['trading_pair']}")
    print(f"Market Trend: {analysis['market_trend']}")
    print(f"Asset Trend: {analysis['asset_trend']}")
    print(f"Risk Score: {analysis['risk_score']:.1f}/100")
    print(f"Direction Bias: {analysis['direction_bias']:.2f} (-1.0 to 1.0)")
    print(f"Optimal Position Size: {analysis['optimal_position_size']:.2f} (0.0 to 1.0)")
    
    print("\nRECOMMENDATIONS:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())