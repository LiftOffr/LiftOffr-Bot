#!/usr/bin/env python3
"""
Market Analyzer

This module analyzes market conditions to identify regimes, trends, and volatility
patterns. It provides insights for risk management and parameter optimization.

Features:
1. Market regime detection (trending, ranging, volatile)
2. Trend strength assessment
3. Volatility analysis
4. Support and resistance detection
5. Correlation analysis between assets
"""

import logging
import math
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enumeration of market regime types"""
    TRENDING_UP = auto()      # Strong uptrend
    TRENDING_DOWN = auto()    # Strong downtrend
    RANGING = auto()          # Sideways market
    VOLATILE = auto()         # High volatility, unpredictable
    BREAKOUT = auto()         # Breaking out from range
    REVERSAL = auto()         # Trend reversal
    UNKNOWN = auto()          # Insufficient data to determine

class MarketAnalyzer:
    """
    Analyzes market conditions to identify regimes, trends, and patterns.
    """
    
    def __init__(self):
        """Initialize the market analyzer"""
        # Configuration parameters
        self.trend_lookback = 50        # Periods for trend analysis
        self.volatility_lookback = 20   # Periods for volatility calculation
        self.range_threshold = 0.05     # 5% range threshold
        self.volatility_threshold = 0.03  # 3% volatility threshold
        self.trend_threshold = 0.7      # 70% directional trend required
        
        logger.info("Market analyzer initialized")
    
    def analyze_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market regimes and conditions.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with market analysis results
        """
        if data.empty:
            return {"current_regime": "UNKNOWN", "confidence": 0.0}
        
        # Calculate key metrics
        trend_direction, trend_strength = self._analyze_trend(data)
        volatility = self._calculate_volatility(data)
        is_ranging = self._is_ranging_market(data)
        
        # Determine market regime
        regime, confidence = self._determine_regime(trend_direction, trend_strength, volatility, is_ranging)
        
        # Calculate regime duration
        regime_duration = self._calculate_regime_duration(data, regime)
        
        # Compile results
        results = {
            "current_regime": regime.name,
            "confidence": confidence,
            "trend": {
                "direction": trend_direction,
                "strength": trend_strength
            },
            "volatility": {
                "value": volatility,
                "percentile": self._calculate_volatility_percentile(data, volatility),
                "is_high": volatility > self.volatility_threshold
            },
            "ranging": {
                "is_ranging": is_ranging,
                "range_width": self._calculate_range_width(data) if is_ranging else None
            },
            "regime_duration": regime_duration
        }
        
        logger.info(f"Market regime analysis: {regime.name} with {confidence:.2f} confidence")
        
        return results
    
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Analyze trend direction and strength.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (direction, strength)
        """
        if len(data) < self.trend_lookback:
            return "neutral", 0.0
        
        # Use EMA slopes to determine trend
        if "ema50" not in data.columns or "ema200" not in data.columns:
            # Calculate EMAs if not present
            data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
            data["ema200"] = data["close"].ewm(span=200, adjust=False).mean()
        
        # Calculate EMA slopes
        ema50_slope = (data["ema50"].iloc[-1] / data["ema50"].iloc[-self.trend_lookback] - 1) * 100
        ema200_slope = (data["ema200"].iloc[-1] / data["ema200"].iloc[-self.trend_lookback] - 1) * 100
        
        # Determine trend direction
        if ema50_slope > 0 and data["ema50"].iloc[-1] > data["ema200"].iloc[-1]:
            trend_direction = "bullish"
        elif ema50_slope < 0 and data["ema50"].iloc[-1] < data["ema200"].iloc[-1]:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
        
        # Calculate trending bars percentage
        bars_in_trend = 0
        for i in range(1, min(self.trend_lookback, len(data))):
            if trend_direction == "bullish" and data["close"].iloc[-i] > data["close"].iloc[-(i+1)]:
                bars_in_trend += 1
            elif trend_direction == "bearish" and data["close"].iloc[-i] < data["close"].iloc[-(i+1)]:
                bars_in_trend += 1
        
        trend_strength = bars_in_trend / min(self.trend_lookback, len(data) - 1)
        
        return trend_direction, trend_strength
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate recent market volatility.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Volatility as a decimal (e.g., 0.02 for 2%)
        """
        if len(data) < self.volatility_lookback:
            return 0.0
        
        # Calculate returns
        returns = data["close"].pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        recent_returns = returns.iloc[-self.volatility_lookback:]
        volatility = recent_returns.std() * math.sqrt(len(recent_returns))
        
        return volatility
    
    def _is_ranging_market(self, data: pd.DataFrame) -> bool:
        """
        Determine if the market is in a ranging (sideways) pattern.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            True if market is ranging, False otherwise
        """
        if len(data) < self.trend_lookback:
            return False
        
        # Calculate price range as percentage
        recent_data = data.iloc[-self.trend_lookback:]
        price_range = (recent_data["high"].max() - recent_data["low"].min()) / recent_data["close"].mean()
        
        # Check if price is moving sideways (range is small)
        if price_range < self.range_threshold:
            # Additional check for flat EMAs
            if "ema50" in data.columns and "ema200" in data.columns:
                ema50_slope = abs(data["ema50"].iloc[-1] / data["ema50"].iloc[-self.trend_lookback] - 1)
                ema200_slope = abs(data["ema200"].iloc[-1] / data["ema200"].iloc[-self.trend_lookback] - 1)
                
                # If EMAs are relatively flat, market is likely ranging
                if ema50_slope < 0.02 and ema200_slope < 0.01:
                    return True
            
            # If EMAs not available, just use price range
            return True
        
        return False
    
    def _determine_regime(self, trend_direction: str, trend_strength: float, 
                        volatility: float, is_ranging: bool) -> Tuple[MarketRegime, float]:
        """
        Determine the current market regime.
        
        Args:
            trend_direction: Trend direction ("bullish", "bearish", "neutral")
            trend_strength: Trend strength (0-1)
            volatility: Volatility value
            is_ranging: Whether market is in a range
            
        Returns:
            Tuple of (MarketRegime, confidence)
        """
        # High volatility overrides other considerations
        if volatility > self.volatility_threshold * 1.5:
            return MarketRegime.VOLATILE, min(1.0, volatility / (self.volatility_threshold * 2))
        
        # Strong trends
        if trend_strength > self.trend_threshold:
            if trend_direction == "bullish":
                return MarketRegime.TRENDING_UP, trend_strength
            elif trend_direction == "bearish":
                return MarketRegime.TRENDING_DOWN, trend_strength
        
        # Ranging market
        if is_ranging:
            return MarketRegime.RANGING, 1.0 - volatility / self.volatility_threshold
        
        # Mixed signals or weaker trends
        if trend_direction == "bullish":
            return MarketRegime.TRENDING_UP, trend_strength
        elif trend_direction == "bearish":
            return MarketRegime.TRENDING_DOWN, trend_strength
        else:
            # Default to ranging with low confidence
            return MarketRegime.RANGING, 0.5
    
    def _calculate_regime_duration(self, data: pd.DataFrame, 
                                 current_regime: MarketRegime) -> int:
        """
        Calculate how long the current regime has been in effect.
        
        Args:
            data: DataFrame with price data
            current_regime: Current market regime
            
        Returns:
            Number of periods the regime has been active
        """
        # Simple approximation based on trend consistency
        if current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            consistent_periods = 0
            direction = 1 if current_regime == MarketRegime.TRENDING_UP else -1
            
            for i in range(1, min(len(data), 100)):
                if direction * (data["close"].iloc[-i] - data["close"].iloc[-(i+1)]) > 0:
                    consistent_periods += 1
                else:
                    break
            
            return consistent_periods
        
        # For ranging or volatile regimes, look for consistent lack of trend
        elif current_regime in [MarketRegime.RANGING, MarketRegime.VOLATILE]:
            # Calculate rolling volatility
            returns = data["close"].pct_change().dropna()
            rolling_vol = returns.rolling(self.volatility_lookback).std()
            
            # Find how long volatility has been similar to current
            current_vol = rolling_vol.iloc[-1]
            similar_periods = 0
            
            for i in range(1, min(len(rolling_vol), 50)):
                past_vol = rolling_vol.iloc[-i]
                if abs(past_vol / current_vol - 1) < 0.3:  # Within 30% of current volatility
                    similar_periods += 1
                else:
                    break
            
            return similar_periods
        
        # Default
        return 0
    
    def _calculate_range_width(self, data: pd.DataFrame) -> float:
        """
        Calculate the width of the current trading range.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Range width as percentage of average price
        """
        if len(data) < 20:
            return 0.0
        
        recent_data = data.iloc[-20:]
        range_width = (recent_data["high"].max() - recent_data["low"].min()) / recent_data["close"].mean()
        
        return range_width
    
    def _calculate_volatility_percentile(self, data: pd.DataFrame, current_volatility: float) -> float:
        """
        Calculate where current volatility ranks in historical context.
        
        Args:
            data: DataFrame with price data
            current_volatility: Current volatility value
            
        Returns:
            Percentile (0-1) of current volatility
        """
        if len(data) < 100:
            return 0.5  # Not enough data for meaningful percentile
            
        # Calculate historical volatilities
        returns = data["close"].pct_change().dropna()
        
        # Calculate rolling volatilities
        rolling_vol = []
        for i in range(len(returns) - self.volatility_lookback + 1):
            window = returns.iloc[i:i+self.volatility_lookback]
            vol = window.std() * math.sqrt(len(window))
            rolling_vol.append(vol)
        
        # Calculate percentile
        lower_count = sum(1 for vol in rolling_vol if vol < current_volatility)
        percentile = lower_count / len(rolling_vol) if rolling_vol else 0.5
        
        return percentile
    
    def analyze_trend_strength(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend strength using multiple indicators.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(data) < 50:
            return {
                "trend_direction": "neutral",
                "trend_strength": "weak",
                "adx": 0,
                "consistent_bars": 0
            }
        
        # Use ADX for trend strength if available
        if "adx" in data.columns:
            adx = data["adx"].iloc[-1]
        else:
            # Simple ADX approximation
            plus_dm = pd.Series(np.where(data["high"].diff() > 0, data["high"].diff(), 0))
            minus_dm = pd.Series(np.where(data["low"].diff() < 0, -data["low"].diff(), 0))
            tr = pd.Series(np.maximum(data["high"] - data["low"], 
                                      np.maximum(abs(data["high"] - data["close"].shift(1)), 
                                                abs(data["low"] - data["close"].shift(1)))))
            
            plus_di = 100 * (plus_dm.rolling(14).sum() / tr.rolling(14).sum())
            minus_di = 100 * (minus_dm.rolling(14).sum() / tr.rolling(14).sum())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
        
        # Determine trend direction
        trend_direction, trend_strength_value = self._analyze_trend(data)
        
        # Categorize trend strength
        if adx < 20:
            trend_strength = "weak"
        elif adx < 30:
            trend_strength = "moderate"
        elif adx < 50:
            trend_strength = "strong"
        else:
            trend_strength = "very strong"
        
        # Count consistent trending bars
        direction = 1 if trend_direction == "bullish" else (-1 if trend_direction == "bearish" else 0)
        consistent_bars = 0
        
        for i in range(1, min(len(data), 20)):
            current_direction = 1 if data["close"].iloc[-i] > data["close"].iloc[-(i+1)] else -1
            if current_direction == direction:
                consistent_bars += 1
            else:
                break
        
        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "adx": adx,
            "consistent_bars": consistent_bars,
            "trend_strength_value": trend_strength_value
        }
    
    def detect_support_resistance(self, data: pd.DataFrame, 
                                levels: int = 3) -> Dict[str, List[float]]:
        """
        Detect key support and resistance levels.
        
        Args:
            data: DataFrame with price data
            levels: Number of levels to detect
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(data) < 100:
            return {"support": [], "resistance": []}
        
        # Find swing highs and lows
        support_levels = []
        resistance_levels = []
        
        # Simple method - use recent lows for support, highs for resistance
        window = min(200, len(data))
        recent_data = data.iloc[-window:]
        
        # Sort prices for clustering
        prices = np.concatenate([recent_data["high"].values, recent_data["low"].values])
        prices.sort()
        
        # Find clusters of prices (potential S/R levels)
        clusters = []
        current_cluster = [prices[0]]
        
        for i in range(1, len(prices)):
            # If price is within 0.5% of the current cluster average, add to cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(prices[i] / cluster_avg - 1) < 0.005:
                current_cluster.append(prices[i])
            else:
                # Start a new cluster
                if len(current_cluster) > 3:  # Only consider clusters with multiple points
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [prices[i]]
        
        # Add the last cluster
        if len(current_cluster) > 3:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        # Sort clusters by price
        clusters.sort()
        
        # Current price
        current_price = data["close"].iloc[-1]
        
        # Separate into support and resistance based on current price
        for cluster in clusters:
            if cluster < current_price:
                support_levels.append(cluster)
            else:
                resistance_levels.append(cluster)
        
        # Limit to requested number of levels
        support_levels = sorted(support_levels, reverse=True)[:levels]
        resistance_levels = sorted(resistance_levels)[:levels]
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def calculate_price_targets(self, data: pd.DataFrame, 
                              confidence: float = 0.7) -> Dict[str, float]:
        """
        Calculate potential price targets based on trend and volatility.
        
        Args:
            data: DataFrame with price data
            confidence: Confidence level (0-1)
            
        Returns:
            Dictionary with price targets
        """
        if len(data) < 50:
            return {"up_target": None, "down_target": None}
            
        current_price = data["close"].iloc[-1]
        
        # Calculate ATR if available
        if "atr" in data.columns:
            atr = data["atr"].iloc[-1]
        else:
            # Calculate ATR manually
            high = data["high"]
            low = data["low"]
            close = data["close"].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
            atr = tr.iloc[-14:].mean()
        
        # Analyze trend
        trend_analysis = self.analyze_trend_strength(data)
        trend_direction = trend_analysis["trend_direction"]
        trend_strength = trend_analysis["trend_strength_value"]
        
        # Calculate targets based on ATR and trend
        atr_multiplier = 3.0 * confidence * (0.5 + trend_strength)
        
        up_target = current_price + (atr * atr_multiplier)
        down_target = current_price - (atr * atr_multiplier)
        
        # Adjust based on trend direction
        if trend_direction == "bullish":
            up_target = current_price + (atr * atr_multiplier * 1.2)
            down_target = current_price - (atr * atr_multiplier * 0.8)
        elif trend_direction == "bearish":
            up_target = current_price + (atr * atr_multiplier * 0.8)
            down_target = current_price - (atr * atr_multiplier * 1.2)
        
        return {
            "up_target": up_target,
            "down_target": down_target,
            "next_target": up_target if trend_direction == "bullish" else 
                           down_target if trend_direction == "bearish" else None
        }
    
    def calculate_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame, 
                            window: int = 50) -> float:
        """
        Calculate correlation between two assets.
        
        Args:
            data1: DataFrame with price data for first asset
            data2: DataFrame with price data for second asset
            window: Number of periods for correlation calculation
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(data1) < window or len(data2) < window:
            return 0.0
        
        # Calculate returns
        returns1 = data1["close"].pct_change().iloc[-window:].values
        returns2 = data2["close"].pct_change().iloc[-window:].values
        
        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        return correlation