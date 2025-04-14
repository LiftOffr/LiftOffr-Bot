#!/usr/bin/env python3
"""
Market Context Analyzer

This module analyzes market conditions to detect the current market regime,
which helps the trading bot adapt its strategies and risk parameters.

Market regimes include:
1. Volatile trending up - Rapid upward price movement with high volatility
2. Volatile trending down - Rapid downward price movement with high volatility
3. Normal trending up - Steady upward price movement with normal volatility
4. Normal trending down - Steady downward price movement with normal volatility
5. Ranging/neutral - Sideways price movement with low volatility

The detected market regime is used to adjust:
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
- Position sizing and leverage
- Entry and exit criteria
- Stop loss and take profit levels
- Trade frequency
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Market regime definitions
class MarketRegime(Enum):
    VOLATILE_TRENDING_UP = "volatile_trending_up"
    VOLATILE_TRENDING_DOWN = "volatile_trending_down"
    NORMAL_TRENDING_UP = "normal_trending_up"
    NORMAL_TRENDING_DOWN = "normal_trending_down"
    NEUTRAL = "neutral"

# Default thresholds
DEFAULT_VOLATILITY_THRESHOLD = 0.015  # 1.5% daily volatility is considered high
DEFAULT_TREND_THRESHOLD = 0.005  # 0.5% change over period for trend detection
DEFAULT_ADX_THRESHOLD = 25.0  # ADX above 25 indicates a trending market
DEFAULT_RSI_OVERBOUGHT = 70.0
DEFAULT_RSI_OVERSOLD = 30.0

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate price volatility using rolling standard deviation of returns
    
    Args:
        df: DataFrame with price data
        window: Rolling window size
        
    Returns:
        Series: Volatility as a percentage
    """
    # Calculate daily returns
    if 'close' in df.columns:
        returns = df['close'].pct_change()
    else:
        returns = df.pct_change()
    
    # Calculate rolling volatility (standard deviation of returns)
    volatility = returns.rolling(window=window).std()
    
    return volatility

def calculate_trend(df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> pd.Series:
    """
    Calculate trend using exponential moving averages
    
    Args:
        df: DataFrame with price data
        short_window: Short EMA window
        long_window: Long EMA window
        
    Returns:
        Series: Trend indicator (positive values for uptrend, negative for downtrend)
    """
    # Get close prices
    if 'close' in df.columns:
        prices = df['close']
    else:
        prices = df
    
    # Calculate short and long EMAs
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    
    # Calculate trend indicator (EMA difference normalized by price)
    trend = (short_ema - long_ema) / prices
    
    return trend

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) to measure trend strength
    
    Args:
        df: DataFrame with OHLC price data
        period: ADX calculation period
        
    Returns:
        Series: ADX values
    """
    # Ensure we have required columns
    required_columns = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        logger.warning("Missing required columns for ADX calculation")
        return pd.Series(index=df.index, data=np.nan)
    
    # Calculate +DM and -DM
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    # Calculate +DM
    condition1 = (high_diff > 0) & (high_diff > low_diff.abs())
    plus_dm[condition1] = high_diff[condition1]
    
    # Calculate -DM
    condition2 = (low_diff < 0) & (low_diff.abs() > high_diff)
    minus_dm[condition2] = low_diff.abs()[condition2]
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth +DM, -DM, and TR using Wilder's smoothing technique
    smoothed_plus_dm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    smoothed_tr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df: DataFrame with price data
        period: RSI calculation period
        
    Returns:
        Series: RSI values
    """
    # Get close prices
    if 'close' in df.columns:
        prices = df['close']
    else:
        prices = df
    
    # Calculate price changes
    delta = prices.diff()
    
    # Split gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def detect_market_regime(
    df: pd.DataFrame,
    volatility_threshold: float = DEFAULT_VOLATILITY_THRESHOLD,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
    rsi_overbought: float = DEFAULT_RSI_OVERBOUGHT,
    rsi_oversold: float = DEFAULT_RSI_OVERSOLD
) -> str:
    """
    Detect the current market regime based on price data
    
    Args:
        df: DataFrame with OHLCV price data
        volatility_threshold: Threshold for high volatility
        trend_threshold: Threshold for trend detection
        adx_threshold: Threshold for ADX trend strength
        rsi_overbought: RSI threshold for overbought condition
        rsi_oversold: RSI threshold for oversold condition
        
    Returns:
        str: Market regime (from MarketRegime enum)
    """
    # Calculate indicators
    volatility = calculate_volatility(df)
    trend = calculate_trend(df)
    adx = calculate_adx(df)
    rsi = calculate_rsi(df)
    
    # Get latest values
    latest_volatility = volatility.iloc[-1] if not volatility.empty else np.nan
    latest_trend = trend.iloc[-1] if not trend.empty else np.nan
    latest_adx = adx.iloc[-1] if not adx.empty else np.nan
    latest_rsi = rsi.iloc[-1] if not rsi.empty else np.nan
    
    # Check if data is valid
    if np.isnan(latest_volatility) or np.isnan(latest_trend) or np.isnan(latest_adx):
        logger.warning("Missing data for market regime detection, defaulting to NEUTRAL")
        return MarketRegime.NEUTRAL.value
    
    # Detect high volatility
    is_volatile = latest_volatility > volatility_threshold
    
    # Detect trend direction
    is_uptrend = latest_trend > trend_threshold
    is_downtrend = latest_trend < -trend_threshold
    
    # Detect trend strength
    is_trending = latest_adx > adx_threshold
    
    # Detect extreme RSI conditions
    is_overbought = not np.isnan(latest_rsi) and latest_rsi > rsi_overbought
    is_oversold = not np.isnan(latest_rsi) and latest_rsi < rsi_oversold
    
    # Determine market regime
    if is_volatile and is_uptrend:
        # Consider RSI for potential trend reversal
        if is_overbought:
            logger.info(f"Detected VOLATILE_TRENDING_UP with overbought RSI ({latest_rsi:.2f})")
        else:
            logger.info(f"Detected VOLATILE_TRENDING_UP (Volatility: {latest_volatility:.4f}, Trend: {latest_trend:.4f}, ADX: {latest_adx:.2f})")
        return MarketRegime.VOLATILE_TRENDING_UP.value
    
    elif is_volatile and is_downtrend:
        # Consider RSI for potential trend reversal
        if is_oversold:
            logger.info(f"Detected VOLATILE_TRENDING_DOWN with oversold RSI ({latest_rsi:.2f})")
        else:
            logger.info(f"Detected VOLATILE_TRENDING_DOWN (Volatility: {latest_volatility:.4f}, Trend: {latest_trend:.4f}, ADX: {latest_adx:.2f})")
        return MarketRegime.VOLATILE_TRENDING_DOWN.value
    
    elif is_trending and is_uptrend:
        logger.info(f"Detected NORMAL_TRENDING_UP (Volatility: {latest_volatility:.4f}, Trend: {latest_trend:.4f}, ADX: {latest_adx:.2f})")
        return MarketRegime.NORMAL_TRENDING_UP.value
    
    elif is_trending and is_downtrend:
        logger.info(f"Detected NORMAL_TRENDING_DOWN (Volatility: {latest_volatility:.4f}, Trend: {latest_trend:.4f}, ADX: {latest_adx:.2f})")
        return MarketRegime.NORMAL_TRENDING_DOWN.value
    
    else:
        logger.info(f"Detected NEUTRAL market (Volatility: {latest_volatility:.4f}, Trend: {latest_trend:.4f}, ADX: {latest_adx:.2f})")
        return MarketRegime.NEUTRAL.value

def get_regime_leverage_factor(regime: str, aggressive: bool = True) -> float:
    """
    Get leverage multiplier based on market regime
    
    Args:
        regime: Market regime
        aggressive: Whether to use aggressive or conservative factors
        
    Returns:
        float: Leverage multiplier
    """
    # Aggressive factors (maximize profit potential)
    aggressive_factors = {
        MarketRegime.VOLATILE_TRENDING_UP.value: 1.2,    # High volatility can be profitable in uptrends
        MarketRegime.VOLATILE_TRENDING_DOWN.value: 0.9,  # Slightly reduced for volatile downtrends
        MarketRegime.NORMAL_TRENDING_UP.value: 1.8,      # Maximize leverage in stable uptrends
        MarketRegime.NORMAL_TRENDING_DOWN.value: 1.2,    # Moderate leverage in stable downtrends
        MarketRegime.NEUTRAL.value: 1.4                  # Default for ranging markets
    }
    
    # Conservative factors (prioritize risk management)
    conservative_factors = {
        MarketRegime.VOLATILE_TRENDING_UP.value: 0.8,    # Reduce leverage in volatile conditions
        MarketRegime.VOLATILE_TRENDING_DOWN.value: 0.7,  # Further reduce for volatile downtrends
        MarketRegime.NORMAL_TRENDING_UP.value: 1.2,      # Moderate increase for stable uptrends
        MarketRegime.NORMAL_TRENDING_DOWN.value: 0.9,    # Slight decrease for stable downtrends
        MarketRegime.NEUTRAL.value: 1.0                  # Default for ranging markets
    }
    
    factors = aggressive_factors if aggressive else conservative_factors
    return factors.get(regime, 1.0)

def get_optimal_trade_parameters(df: pd.DataFrame, regime: str, 
                               direction: str, aggressive: bool = True) -> Dict[str, float]:
    """
    Calculate optimal trade parameters for the current market conditions
    
    Args:
        df: DataFrame with OHLCV price data
        regime: Market regime
        direction: Trade direction ('long' or 'short')
        aggressive: Whether to use aggressive or conservative settings
        
    Returns:
        Dict: Trade parameters
    """
    # Calculate ATR for dynamic stop loss and take profit
    close = df['close'].iloc[-1] if 'close' in df.columns else df.iloc[-1]
    
    # Calculate ATR
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]
    else:
        # Use price volatility as a proxy for ATR
        atr = df.iloc[-20:].std().iloc[-1] * 2.0
    
    # ATR as percentage of price
    atr_pct = atr / close
    
    # Leverage factor based on market regime
    leverage_factor = get_regime_leverage_factor(regime, aggressive)
    
    # Base parameters
    base_stop_loss_pct = 0.04 if aggressive else 0.03  # Default 4% stop loss
    base_take_profit_pct = 0.12 if aggressive else 0.09  # Default 12% take profit
    
    # Adjust parameters based on market regime
    if regime == MarketRegime.VOLATILE_TRENDING_UP.value:
        if direction == 'long':
            stop_loss_pct = base_stop_loss_pct * 1.2  # Wider stop for volatility
            take_profit_pct = base_take_profit_pct * 1.5  # Higher profit target
        else:  # short
            stop_loss_pct = base_stop_loss_pct * 1.4  # Even wider stop for counter-trend
            take_profit_pct = base_take_profit_pct * 0.8  # Lower profit target
    
    elif regime == MarketRegime.VOLATILE_TRENDING_DOWN.value:
        if direction == 'short':
            stop_loss_pct = base_stop_loss_pct * 1.2  # Wider stop for volatility
            take_profit_pct = base_take_profit_pct * 1.5  # Higher profit target
        else:  # long
            stop_loss_pct = base_stop_loss_pct * 1.4  # Even wider stop for counter-trend
            take_profit_pct = base_take_profit_pct * 0.8  # Lower profit target
    
    elif regime == MarketRegime.NORMAL_TRENDING_UP.value:
        if direction == 'long':
            stop_loss_pct = base_stop_loss_pct * 0.9  # Tighter stop in clear trend
            take_profit_pct = base_take_profit_pct * 1.3  # Higher profit target
        else:  # short
            stop_loss_pct = base_stop_loss_pct * 1.5  # Wider stop for counter-trend
            take_profit_pct = base_take_profit_pct * 0.7  # Lower profit target
    
    elif regime == MarketRegime.NORMAL_TRENDING_DOWN.value:
        if direction == 'short':
            stop_loss_pct = base_stop_loss_pct * 0.9  # Tighter stop in clear trend
            take_profit_pct = base_take_profit_pct * 1.3  # Higher profit target
        else:  # long
            stop_loss_pct = base_stop_loss_pct * 1.5  # Wider stop for counter-trend
            take_profit_pct = base_take_profit_pct * 0.7  # Lower profit target
    
    else:  # NEUTRAL
        stop_loss_pct = base_stop_loss_pct  # Default stop loss
        take_profit_pct = base_take_profit_pct  # Default take profit
    
    # Fine-tune using ATR
    atr_factor = min(3.0, max(1.0, atr_pct * 100 / 2.0))  # Scale ATR to reasonable range
    stop_loss_pct = max(0.01, stop_loss_pct * atr_factor)  # Minimum 1% stop loss
    take_profit_pct = max(0.02, take_profit_pct * atr_factor)  # Minimum 2% take profit
    
    # Ensure take profit is at least 2x stop loss
    take_profit_pct = max(take_profit_pct, stop_loss_pct * 2.0)
    
    return {
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'leverage_factor': leverage_factor,
        'atr': atr,
        'atr_pct': atr_pct
    }

def analyze_market_context(df: pd.DataFrame, aggressive: bool = True) -> Dict[str, Any]:
    """
    Comprehensive market analysis for trading decisions
    
    Args:
        df: DataFrame with OHLCV price data
        aggressive: Whether to use aggressive or conservative settings
        
    Returns:
        Dict: Market context analysis
    """
    # Detect market regime
    regime = detect_market_regime(df)
    
    # Get latest price
    close = df['close'].iloc[-1] if 'close' in df.columns else df.iloc[-1]
    
    # Calculate indicators
    volatility = calculate_volatility(df).iloc[-1]
    trend = calculate_trend(df).iloc[-1]
    adx = calculate_adx(df).iloc[-1]
    rsi = calculate_rsi(df).iloc[-1]
    
    # Determine optimal trade direction
    if trend > DEFAULT_TREND_THRESHOLD and rsi < DEFAULT_RSI_OVERBOUGHT:
        optimal_direction = 'long'
    elif trend < -DEFAULT_TREND_THRESHOLD and rsi > DEFAULT_RSI_OVERSOLD:
        optimal_direction = 'short'
    else:
        optimal_direction = 'neutral'
    
    # Get optimal parameters for both directions
    long_params = get_optimal_trade_parameters(df, regime, 'long', aggressive)
    short_params = get_optimal_trade_parameters(df, regime, 'short', aggressive)
    
    # Determine trade confidence (0-1 scale)
    trade_confidence = min(1.0, abs(trend) * 10 + (adx / 100))
    
    # Calculate trade viability score (0-100 scale)
    if optimal_direction == 'long':
        trade_score = min(100, 50 + (trend * 100) + (adx / 2) - (max(0, rsi - 50) / 2))
    elif optimal_direction == 'short':
        trade_score = min(100, 50 - (trend * 100) + (adx / 2) - (max(0, 50 - rsi) / 2))
    else:
        trade_score = 0
    
    return {
        'regime': regime,
        'optimal_direction': optimal_direction,
        'long_params': long_params,
        'short_params': short_params,
        'price': close,
        'volatility': volatility,
        'trend': trend,
        'adx': adx,
        'rsi': rsi,
        'confidence': trade_confidence,
        'trade_score': trade_score
    }

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download some data
    ticker = "SOL-USD"
    data = yf.download(ticker, period="1mo", interval="1h")
    
    # Analyze market context
    context = analyze_market_context(data)
    
    # Print results
    print(f"Market Analysis for {ticker}")
    print(f"Current Price: ${context['price']:.2f}")
    print(f"Market Regime: {context['regime']}")
    print(f"Optimal Direction: {context['optimal_direction']}")
    print(f"Trade Confidence: {context['confidence']:.2f}")
    print(f"Trade Score: {context['trade_score']:.1f}/100")
    print("\nLong Trade Parameters:")
    print(f"  Stop Loss: {context['long_params']['stop_loss_pct']:.2f}%")
    print(f"  Take Profit: {context['long_params']['take_profit_pct']:.2f}%")
    print(f"  Leverage Factor: {context['long_params']['leverage_factor']:.2f}x")
    print("\nShort Trade Parameters:")
    print(f"  Stop Loss: {context['short_params']['stop_loss_pct']:.2f}%")
    print(f"  Take Profit: {context['short_params']['take_profit_pct']:.2f}%")
    print(f"  Leverage Factor: {context['short_params']['leverage_factor']:.2f}x")
    print("\nKey Indicators:")
    print(f"  Volatility: {context['volatility']:.4f}")
    print(f"  Trend: {context['trend']:.4f}")
    print(f"  ADX: {context['adx']:.1f}")
    print(f"  RSI: {context['rsi']:.1f}")