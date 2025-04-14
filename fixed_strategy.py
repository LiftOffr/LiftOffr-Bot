#!/usr/bin/env python3
"""
Fixed Strategy Implementation

This module implements both ARIMA and Adaptive trading strategies with 
enhanced predictive capabilities and risk management.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy.log')
    ]
)
logger = logging.getLogger(__name__)

class ARIMAStrategy:
    """
    ARIMA-based trading strategy with enhanced features for optimal performance
    """
    def __init__(
        self,
        trading_pair: str,
        timeframe: str = "1h",
        max_leverage: int = 125,
        config_path: str = "ml_enhanced_config.json"
    ):
        """
        Initialize the ARIMA strategy
        
        Args:
            trading_pair: Trading pair to analyze
            timeframe: Timeframe for analysis
            max_leverage: Maximum allowed leverage
            config_path: Path to configuration file
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.max_leverage = max_leverage
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize state
        self.last_forecast = None
        self.forecast_history = []
        self.accuracy_history = []
        
        # ARIMA parameters (configurable)
        self.p = 5
        self.d = 1
        self.q = 0
        self.forecast_periods = 5
        
        # Performance tracking
        self.correct_forecasts = 0
        self.total_forecasts = 0
        
        logger.info(f"ARIMA Strategy initialized for {trading_pair} on {timeframe}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return config.get("arima_settings", {})
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate ARIMA forecast
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Ensure enough data
            if len(data) < 60:
                return {"signal": "neutral", "strength": 0, "message": "Not enough data"}
            
            # Get closing prices
            prices = data["close"].values
            
            # Generate ARIMA forecast
            forecast, confidence = self._generate_arima_forecast(prices)
            
            # Determine signal
            current_price = prices[-1]
            forecast_price = forecast[0]
            
            # Calculate percentage change
            pct_change = (forecast_price - current_price) / current_price
            
            # Threshold for generating signals
            threshold = 0.001  # 0.1% threshold
            
            # Determine signal direction
            if pct_change > threshold:
                signal = "buy"
            elif pct_change < -threshold:
                signal = "sell"
            else:
                signal = "neutral"
            
            # Calculate signal strength (0.5-1.0)
            strength = min(1.0, max(0.5, 0.5 + abs(pct_change) * 200))
            
            # Store forecast for tracking
            self.last_forecast = {
                "price": forecast_price,
                "timestamp": datetime.now(),
                "signal": signal,
                "strength": strength
            }
            
            # Add to history
            self.forecast_history.append(self.last_forecast)
            if len(self.forecast_history) > 100:
                self.forecast_history.pop(0)
            
            # Log the forecast
            logger.info(f"ARIMA Forecast: {signal.upper()} | Current: ${current_price:.2f} → Target: ${forecast_price:.2f}")
            
            # Check market trend
            ema50 = self._calculate_ema(prices, 50)
            ema100 = self._calculate_ema(prices, 100)
            
            trend = "BULLISH" if ema50 > ema100 else "BEARISH" if ema50 < ema100 else "NEUTRAL"
            logger.info(f"Market Trend: {trend} | EMA50 vs EMA100: {ema50:.2f} vs {ema100:.2f}")
            
            # Don't short in strong uptrends
            if signal == "sell" and trend == "BULLISH":
                # Look for confirmation
                if pct_change < -0.003:  # Stronger threshold for going against trend
                    logger.info("Taking SHORT despite bullish trend (strong signal)")
                else:
                    logger.info("Avoiding SHORT in bullish market")
                    signal = "neutral"
                    strength = 0.3
            
            return {
                "signal": signal,
                "strength": strength,
                "forecast": forecast_price,
                "confidence": confidence,
                "trend": trend,
                "ema50": ema50,
                "ema100": ema100,
                "current_price": current_price,
                "pct_change": pct_change
            }
        
        except Exception as e:
            logger.error(f"Error in ARIMA analysis: {e}")
            return {"signal": "neutral", "strength": 0, "message": f"Error: {str(e)}"}
    
    def _generate_arima_forecast(self, prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generate forecast using ARIMA model
        
        Args:
            prices: Array of price data
            
        Returns:
            Tuple of (forecast, confidence)
        """
        try:
            # Import statsmodels lazily to avoid startup overhead
            from statsmodels.tsa.arima.model import ARIMA
            
            # Update ARIMA parameters from config if available
            if self.config:
                self.p = self.config.get("p", self.p)
                self.d = self.config.get("d", self.d)
                self.q = self.config.get("q", self.q)
            
            # Fit ARIMA model
            model = ARIMA(prices, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=self.forecast_periods)
            
            # Calculate confidence based on model fit
            confidence = 0.7  # Default confidence
            
            # Try to get AIC for confidence calculation
            try:
                aic = model_fit.aic
                # Scale AIC to confidence (lower AIC is better)
                # Simple scaling formula, could be improved
                confidence = min(0.9, max(0.5, 0.9 - abs(aic) / 10000))
            except:
                pass
            
            return forecast, confidence
            
        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            # Return a neutral forecast (current price)
            if len(prices) > 0:
                current_price = prices[-1]
                return np.array([current_price] * self.forecast_periods), 0.5
            else:
                return np.array([0] * self.forecast_periods), 0.5
    
    def _calculate_ema(self, data: np.ndarray, window: int) -> float:
        """Calculate EMA for the given window"""
        if len(data) < window:
            return data[-1] if len(data) > 0 else 0
        
        alpha = 2 / (window + 1)
        ema = [data[0]]
        
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
        
        return ema[-1]
    
    def update_performance(self, was_correct: bool) -> None:
        """
        Update performance tracking
        
        Args:
            was_correct: Whether the prediction was correct
        """
        self.total_forecasts += 1
        if was_correct:
            self.correct_forecasts += 1
        
        # Add to accuracy history
        self.accuracy_history.append(1 if was_correct else 0)
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)
    
    def get_win_rate(self) -> float:
        """
        Get the strategy's win rate
        
        Returns:
            Win rate (0-1)
        """
        if self.total_forecasts == 0:
            return 0
        
        # Calculate recent win rate
        if len(self.accuracy_history) > 10:
            recent_rate = sum(self.accuracy_history[-10:]) / 10
            overall_rate = self.correct_forecasts / self.total_forecasts
            
            # Weighted average favoring recent performance
            return recent_rate * 0.7 + overall_rate * 0.3
        
        return self.correct_forecasts / self.total_forecasts
        
    def generate_signal_for_candle(self, candle_data: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Generate a trading signal for a single candle
        
        Args:
            candle_data: DataFrame containing a single candle's OHLCV data
            
        Returns:
            Tuple containing (signal, forecast_price, strength)
        """
        result = self.analyze(candle_data)
        signal = result.get("signal", "neutral")
        forecast_price = result.get("forecast", candle_data["close"].iloc[-1])
        strength = result.get("strength", 0.0)
        
        return signal, forecast_price, strength


class AdaptiveStrategy:
    """
    Advanced adaptive trading strategy with technical indicators and dynamic controls
    """
    def __init__(
        self,
        trading_pair: str,
        timeframe: str = "1h",
        max_leverage: int = 125,
        config_path: str = "ml_enhanced_config.json"
    ):
        """
        Initialize the Adaptive strategy
        
        Args:
            trading_pair: Trading pair to analyze
            timeframe: Timeframe for analysis
            max_leverage: Maximum allowed leverage
            config_path: Path to configuration file
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.max_leverage = max_leverage
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize state
        self.last_signal = None
        self.signal_history = []
        self.accuracy_history = []
        
        # Performance tracking
        self.correct_signals = 0
        self.total_signals = 0
        
        # Strategy parameters (configurable)
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volatility_threshold = 0.006
        self.atr_period = 14
        self.ema_short = 9
        self.ema_long = 21
        self.adx_period = 14
        self.adx_threshold = 20
        
        # Apply config if available
        self._apply_config()
        
        logger.info(f"Adaptive Strategy initialized for {trading_pair} on {timeframe}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return config.get("adaptive_settings", {})
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _apply_config(self) -> None:
        """Apply configuration to strategy parameters"""
        if self.config:
            self.rsi_period = self.config.get("rsi_period", self.rsi_period)
            self.rsi_overbought = self.config.get("rsi_overbought", self.rsi_overbought)
            self.rsi_oversold = self.config.get("rsi_oversold", self.rsi_oversold)
            self.volatility_threshold = self.config.get("volatility_threshold", self.volatility_threshold)
            self.atr_period = self.config.get("atr_period", self.atr_period)
            self.ema_short = self.config.get("ema_short", self.ema_short)
            self.ema_long = self.config.get("ema_long", self.ema_long)
            self.adx_period = self.config.get("adx_period", self.adx_period)
            self.adx_threshold = self.config.get("adx_threshold", self.adx_threshold)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signal
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Ensure enough data
            if len(data) < 50:
                return {"signal": "neutral", "strength": 0, "message": "Not enough data"}
            
            # Calculate indicators
            indicators = self._calculate_indicators(data)
            
            # Analyze the indicators
            analysis = self._analyze_indicators(indicators)
            
            # Get current price
            current_price = data["close"].iloc[-1]
            
            # Calculate volatility
            returns = data["close"].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate ATR
            atr = self._calculate_atr(data)
            
            # Determine signal based on analysis
            signal, strength, forecast = self._determine_signal(analysis, indicators, volatility)
            
            # Store signal for tracking
            self.last_signal = {
                "signal": signal,
                "strength": strength,
                "timestamp": datetime.now(),
                "price": current_price,
                "forecast": forecast
            }
            
            # Add to history
            self.signal_history.append(self.last_signal)
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
            
            # Calculate volatility stop level
            volatility_stop = self._calculate_volatility_stop(data, signal)
            
            # Log the analysis
            logger.info(f"============================================================")
            logger.info(f"【ANALYSIS】 Forecast: {analysis['forecast'].upper()} | Current: ${current_price:.2f} → Target: ${forecast:.2f}")
            logger.info(f"【INDICATORS】 EMA{self.ema_short} {'<' if indicators['ema_signal'] == -1 else '>='} EMA{self.ema_long} | RSI = {indicators['rsi']:.2f} {'✓' if analysis['rsi_valid'] else '✗'} | MACD {'>' if indicators['macd_signal'] == 1 else '<'} Signal | ADX = {indicators['adx']:.2f} {'✓' if indicators['adx'] >= self.adx_threshold else '✗'}")
            logger.info(f"【VOLATILITY】 Volatility = {volatility:.4f} {'✓' if volatility <= self.volatility_threshold else '✗'} (threshold: {self.volatility_threshold})")
            
            # Detailed bands information
            logger.info(f"【BANDS】 EMA{self.ema_short} {'<' if indicators['ema_signal'] == -1 else '>='} EMA{self.ema_long} ({indicators['ema_short']:.2f} vs {indicators['ema_long']:.2f}) | RSI = {indicators['rsi']:.2f} {'✓' if analysis['rsi_valid'] else '✗'} | MACD {'>' if indicators['macd_signal'] == 1 else '<'} Signal ({indicators['macd']:.4f} vs {indicators['macd_signal_line']:.4f}) | ADX = {indicators['adx']:.2f} {'✓' if indicators['adx'] >= self.adx_threshold else '✗'} | Volatility = {volatility:.4f} {'✓' if volatility <= self.volatility_threshold else '✗'} (threshold: {self.volatility_threshold}) | Price {'>' if current_price > indicators['upper_bb'] else '<'} Upper BB ({current_price:.2f} vs {indicators['upper_bb']:.2f}) | Price {'>' if current_price > indicators['lower_bb'] else '<'} Lower BB ({current_price:.2f} vs {indicators['lower_bb']:.2f}) | Price vs KC Middle: {current_price:.2f} vs {indicators['kc_middle']:.2f}")
            
            logger.info(f"【SIGNAL】 {'🟢 BUY' if signal == 'buy' else '🔴 SELL' if signal == 'sell' else '⚪ NEUTRAL'} - {analysis['reasoning']}")
            logger.info(f"============================================================")
            
            return {
                "signal": signal,
                "strength": strength,
                "forecast": forecast,
                "volatility": volatility,
                "atr": atr,
                "indicators": indicators,
                "analysis": analysis,
                "current_price": current_price,
                "volatility_stop": volatility_stop
            }
        
        except Exception as e:
            logger.error(f"Error in Adaptive analysis: {e}")
            return {"signal": "neutral", "strength": 0, "message": f"Error: {str(e)}"}
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with calculated indicators
        """
        # Calculate RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate EMAs
        ema_short = data["close"].ewm(span=self.ema_short, adjust=False).mean()
        ema_long = data["close"].ewm(span=self.ema_long, adjust=False).mean()
        
        # EMA crossover signal (-1: bearish, 0: neutral, 1: bullish)
        ema_signal = 1 if ema_short.iloc[-1] > ema_long.iloc[-1] else -1
        
        # Calculate MACD
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # MACD signal (-1: bearish, 0: neutral, 1: bullish)
        macd_signal = 1 if macd.iloc[-1] > macd_signal_line.iloc[-1] else -1
        
        # Calculate ADX
        adx = self._calculate_adx(data)
        
        # Calculate Bollinger Bands
        sma20 = data["close"].rolling(window=20).mean()
        std20 = data["close"].rolling(window=20).std()
        upper_bb = sma20 + (std20 * 2)
        lower_bb = sma20 - (std20 * 2)
        
        # Calculate Keltner Channels
        atr = self._calculate_atr(data)
        kc_middle = ema_long.iloc[-1]
        kc_upper = kc_middle + (atr * 2)
        kc_lower = kc_middle - (atr * 2)
        
        return {
            "rsi": rsi.iloc[-1],
            "ema_short": ema_short.iloc[-1],
            "ema_long": ema_long.iloc[-1],
            "ema_signal": ema_signal,
            "macd": macd.iloc[-1],
            "macd_signal_line": macd_signal_line.iloc[-1],
            "macd_signal": macd_signal,
            "adx": adx,
            "upper_bb": upper_bb.iloc[-1],
            "lower_bb": lower_bb.iloc[-1],
            "sma20": sma20.iloc[-1],
            "kc_middle": kc_middle,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower
        }
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """
        Calculate Average True Range
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ATR value
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr.iloc[-1]
    
    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """
        Calculate ADX (Average Directional Index)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ADX value
        """
        # Simplified ADX calculation (can be improved)
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().multiply(-1)
        
        # When +DM > -DM and +DM > 0, +DM = +DM, else +DM = 0
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        
        # When -DM > +DM and -DM > 0, -DM = -DM, else -DM = 0
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate smoothed values
        smoothing = self.adx_period
        
        # First value is a simple average
        plus_di = plus_dm.rolling(window=smoothing).mean() / tr.rolling(window=smoothing).mean() * 100
        minus_di = minus_dm.rolling(window=smoothing).mean() / tr.rolling(window=smoothing).mean() * 100
        
        # Calculate DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.rolling(window=smoothing).mean()
        
        return adx.iloc[-1]
    
    def _analyze_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze indicators to determine trading signal
        
        Args:
            indicators: Dictionary with calculated indicators
            
        Returns:
            Dictionary with analysis results
        """
        # Extract key indicators
        rsi = indicators["rsi"]
        ema_signal = indicators["ema_signal"]
        macd_signal = indicators["macd_signal"]
        adx = indicators["adx"]
        
        # Check RSI conditions
        rsi_buy = rsi < self.rsi_oversold
        rsi_sell = rsi > self.rsi_overbought
        rsi_valid = rsi_buy or rsi_sell or (rsi > 40 and rsi < 60)
        
        # Check ADX (trend strength)
        trend_strong = adx >= self.adx_threshold
        
        # Combine signals
        buy_signals = []
        sell_signals = []
        
        # EMA crossover
        if ema_signal == 1:
            buy_signals.append("EMA crossover bullish")
        elif ema_signal == -1:
            sell_signals.append("EMA crossover bearish")
        
        # RSI conditions
        if rsi_buy:
            buy_signals.append(f"RSI oversold ({rsi:.2f})")
        elif rsi_sell:
            sell_signals.append(f"RSI overbought ({rsi:.2f})")
        
        # MACD
        if macd_signal == 1:
            buy_signals.append("MACD bullish crossover")
        elif macd_signal == -1:
            sell_signals.append("MACD bearish crossover")
        
        # Determine overall direction
        if len(buy_signals) > len(sell_signals) and trend_strong:
            direction = "buy"
            forecast = "BULLISH"
            reasoning = "Multiple bullish signals in strong trend"
        elif len(sell_signals) > len(buy_signals) and trend_strong:
            direction = "sell"
            forecast = "BEARISH"
            reasoning = "Multiple bearish signals in strong trend"
        else:
            direction = "neutral"
            forecast = "NEUTRAL"
            reasoning = "No clear trade signal detected"
        
        return {
            "direction": direction,
            "forecast": forecast,
            "reasoning": reasoning,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "rsi_valid": rsi_valid,
            "trend_strong": trend_strong
        }
    
    def _determine_signal(self, analysis: Dict[str, Any], indicators: Dict[str, Any], volatility: float) -> Tuple[str, float, float]:
        """
        Determine final trading signal, strength, and price forecast
        
        Args:
            analysis: Dictionary with indicator analysis
            indicators: Dictionary with calculated indicators
            volatility: Current market volatility
            
        Returns:
            Tuple of (signal, strength, forecast_price)
        """
        # Get base signal from analysis
        signal = analysis["direction"]
        
        # Calculate signal strength (0.5-1.0)
        base_strength = 0.5
        
        # Add strength for strong trend
        if analysis["trend_strong"]:
            base_strength += 0.2
        
        # Add strength for number of confirming signals
        if signal == "buy":
            base_strength += min(0.3, len(analysis["buy_signals"]) * 0.1)
        elif signal == "sell":
            base_strength += min(0.3, len(analysis["sell_signals"]) * 0.1)
        
        # Reduce strength in high volatility
        if volatility > self.volatility_threshold:
            base_strength *= 0.7
        
        # Cap strength between 0.3 and 1.0
        strength = min(1.0, max(0.3, base_strength))
        
        # If neutral, cap strength at 0.5
        if signal == "neutral":
            strength = min(strength, 0.3)
        
        # Calculate price forecast
        current_price = indicators["kc_middle"]  # Using EMA as current reference
        
        if signal == "buy":
            # Bullish: forecast price above current
            forecast = current_price * (1 + (strength * 0.005))  # Up to 0.5% increase
        elif signal == "sell":
            # Bearish: forecast price below current
            forecast = current_price * (1 - (strength * 0.005))  # Up to 0.5% decrease
        else:
            # Neutral: minimal change
            forecast = current_price * (1 + (np.random.random() - 0.5) * 0.001)  # ±0.05% random change
        
        return signal, strength, forecast
    
    def _calculate_volatility_stop(self, data: pd.DataFrame, signal: str) -> float:
        """
        Calculate volatility-based stop loss level
        
        Args:
            data: DataFrame with OHLCV data
            signal: Trading signal (buy, sell, neutral)
            
        Returns:
            Volatility stop level
        """
        # Calculate ATR
        atr = self._calculate_atr(data)
        
        # Get current price
        current_price = data["close"].iloc[-1]
        
        # Set multiplier based on signal
        if signal == "buy":
            # For buy signals, stop is below current price
            return current_price - (atr * 1.5)
        elif signal == "sell":
            # For sell signals, stop is above current price
            return current_price + (atr * 1.5)
        else:
            # For neutral, return the current price
            return current_price
    
    def update_performance(self, was_correct: bool) -> None:
        """
        Update performance tracking
        
        Args:
            was_correct: Whether the prediction was correct
        """
        self.total_signals += 1
        if was_correct:
            self.correct_signals += 1
        
        # Add to accuracy history
        self.accuracy_history.append(1 if was_correct else 0)
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)
    
    def get_win_rate(self) -> float:
        """
        Get the strategy's win rate
        
        Returns:
            Win rate (0-1)
        """
        if self.total_signals == 0:
            return 0
        
        # Calculate recent win rate
        if len(self.accuracy_history) > 10:
            recent_rate = sum(self.accuracy_history[-10:]) / 10
            overall_rate = self.correct_signals / self.total_signals
            
            # Weighted average favoring recent performance
            return recent_rate * 0.7 + overall_rate * 0.3
        
        return self.correct_signals / self.total_signals
        
    def generate_signal_for_candle(self, candle_data: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Generate a trading signal for a single candle
        
        Args:
            candle_data: DataFrame containing a single candle's OHLCV data
            
        Returns:
            Tuple containing (signal, strength, volatility)
        """
        result = self.analyze(candle_data)
        signal = result.get("signal", "neutral")
        strength = result.get("strength", 0.0)
        volatility = result.get("volatility", 0.0)
        
        return signal, strength, volatility