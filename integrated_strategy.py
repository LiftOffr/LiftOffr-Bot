"""
Integrated Trading Strategy - "him all along"

This file implements an integrated strategy that combines volatile market indicators 
with an ARIMA-style component, working as a single portfolio strategy where multiple
sub-strategies compete to determine the market view.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from decimal import Decimal
import statsmodels.api as sm

from trading_strategy import TradingStrategy
from config import (
    SIGNAL_STRENGTH_ADVANTAGE, STRONGER_SIGNAL_DOMINANCE,
    MIN_SIGNAL_STRENGTH, ENTRY_ATR_MULTIPLIER
)

logger = logging.getLogger(__name__)

class IntegratedStrategy(TradingStrategy):
    """
    Integrated strategy combining volatile market indicators and ARIMA-style forecasting.
    
    This strategy implements a single-portfolio approach where multiple
    sub-strategies (signal generators) compete to determine the market view.
    It combines:
    1. Volatile Market Indicators: EMA, RSI, MACD, DMI/ADX, Bollinger Bands, Keltner Channels
    2. ARIMA Component: Uses linear regression as an approximation of ARIMA forecasting
    """
    
    def __init__(self, symbol: str,
                 leverage: float = 25.0,
                 liq_exit_offset_percent: float = 0.01,
                 atr_period: int = 14,
                 fixed_stop_multiplier: float = 2.0,
                 entry_offset_multiplier: float = 0.01,
                 trail_atr_multiplier: float = 2.0,
                 vol_threshold: float = 0.006,
                 min_strength: float = 0.1,
                 arima_lookback: int = 20,
                 arima_entry_offset_multiplier: float = 0.01):
        """
        Initialize the integrated strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'SOLUSD')
            leverage (float): Leverage used for position sizing and risk management
            liq_exit_offset_percent (float): Small percentage offset from liquidation price for exits
            atr_period (int): Period for calculating ATR
            fixed_stop_multiplier (float): Multiplier for fixed stop loss (as ATR multiple)
            entry_offset_multiplier (float): Offset for entry prices (as ATR multiple)
            trail_atr_multiplier (float): Multiplier for trailing stop (as ATR multiple)
            vol_threshold (float): Threshold for volatility filter
            min_strength (float): Minimum threshold for signal strength
            arima_lookback (int): Number of periods for ARIMA component regression
            arima_entry_offset_multiplier (float): Entry offset for ARIMA-based entries
        """
        super().__init__(symbol)
        
        # Set strategy category
        self.category = "him all along"
        
        # Initialize parameters
        self.leverage = leverage
        self.liq_exit_offset_percent = liq_exit_offset_percent
        self.atr_period = atr_period
        self.fixed_stop_multiplier = fixed_stop_multiplier
        self.entry_offset_multiplier = entry_offset_multiplier
        self.trail_atr_multiplier = trail_atr_multiplier
        self.vol_threshold = vol_threshold
        self.min_strength = min_strength if min_strength > 0 else MIN_SIGNAL_STRENGTH
        self.arima_lookback = arima_lookback
        self.arima_entry_offset_multiplier = arima_entry_offset_multiplier
        
        # Trading state
        self.current_signal = None  # 'long', 'short', or None
        self.signal_strength = 0.0  # Between 0 and 1
        
        # OHLC data storage
        self.ohlc_data = {
            'current_tf': {
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'time': []
            },
            'm45': {  # 45-minute timeframe data for ARIMA-style forecasting
                'close': [],
                'high': [],
                'low': [],
                'time': []
            }
        }
        
        # Technical indicators
        self.indicators = {
            'ema9': None,
            'ema21': None,
            'ema50': None,
            'ema100': None,
            'rsi': None,
            'macd': None,
            'macd_signal': None,
            'adx': None,
            'plus_di': None,
            'minus_di': None,
            'bb_upper': None,
            'bb_lower': None,
            'bb_middle': None,
            'kc_upper': None,
            'kc_lower': None,
            'kc_middle': None,
            'atr': None,
            'volatility': None,
            'arima_forecast': None
        }
        
        # Signal strengths for each component
        self.signal_strengths = {
            'ema': 0.0,
            'rsi': 0.0,
            'macd': 0.0,
            'adx': 0.0,
            'bb': 0.0,
            'kc': 0.0,
            'arima': 0.0,
            'volatility': 0.0
        }
        
        # Final signal
        self.bullish_strength = 0.0
        self.bearish_strength = 0.0
        self.strongest_bullish = None  # Name of strongest bullish indicator
        self.strongest_bearish = None  # Name of strongest bearish indicator
        
        # Entry prices
        self.long_entry_price = None
        self.short_entry_price = None
        
        # Initialize with 100 candles minimum
        for _ in range(100):
            self.ohlc_data['current_tf']['open'].append(None)
            self.ohlc_data['current_tf']['high'].append(None)
            self.ohlc_data['current_tf']['low'].append(None)
            self.ohlc_data['current_tf']['close'].append(None)
            self.ohlc_data['current_tf']['time'].append(time.time())
        
        # Initialize 45-minute data
        for _ in range(30):  # 30 periods for 45-minute timeframe
            self.ohlc_data['m45']['close'].append(None)
            self.ohlc_data['m45']['high'].append(None)
            self.ohlc_data['m45']['low'].append(None)
            self.ohlc_data['m45']['time'].append(time.time())
        
        logger.info(f"Initialized IntegratedStrategy for {symbol}")
    
    def add_current_tf_data(self, open_price, high_price, low_price, close_price):
        """
        Add data point to current timeframe data
        
        Args:
            open_price (float): Open price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
        """
        self.ohlc_data['current_tf']['open'].append(open_price)
        self.ohlc_data['current_tf']['high'].append(high_price)
        self.ohlc_data['current_tf']['low'].append(low_price)
        self.ohlc_data['current_tf']['close'].append(close_price)
        self.ohlc_data['current_tf']['time'].append(time.time())
        
        # Keep only the last 300 data points
        if len(self.ohlc_data['current_tf']['close']) > 300:
            self.ohlc_data['current_tf']['open'] = self.ohlc_data['current_tf']['open'][-300:]
            self.ohlc_data['current_tf']['high'] = self.ohlc_data['current_tf']['high'][-300:]
            self.ohlc_data['current_tf']['low'] = self.ohlc_data['current_tf']['low'][-300:]
            self.ohlc_data['current_tf']['close'] = self.ohlc_data['current_tf']['close'][-300:]
            self.ohlc_data['current_tf']['time'] = self.ohlc_data['current_tf']['time'][-300:]
    
    def add_m45_data(self, close_price, high_price, low_price):
        """
        Add data point to 45-minute timeframe data
        
        Args:
            close_price (float): Close price
            high_price (float): High price
            low_price (float): Low price
        """
        self.ohlc_data['m45']['close'].append(close_price)
        self.ohlc_data['m45']['high'].append(high_price)
        self.ohlc_data['m45']['low'].append(low_price)
        self.ohlc_data['m45']['time'].append(time.time())
        
        # Keep only the last 100 data points
        if len(self.ohlc_data['m45']['close']) > 100:
            self.ohlc_data['m45']['close'] = self.ohlc_data['m45']['close'][-100:]
            self.ohlc_data['m45']['high'] = self.ohlc_data['m45']['high'][-100:]
            self.ohlc_data['m45']['low'] = self.ohlc_data['m45']['low'][-100:]
            self.ohlc_data['m45']['time'] = self.ohlc_data['m45']['time'][-100:]
    
    def update_ohlc(self, open_price, high_price, low_price, close_price):
        """
        Update OHLC data and indicators
        
        Args:
            open_price (float): Open price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
        """
        # Update parent class
        super().update_ohlc(open_price, high_price, low_price, close_price)
        
        # Add to current timeframe data
        self.add_current_tf_data(open_price, high_price, low_price, close_price)
        
        # Update 45-minute data (simplified approach - in production would need timestamp-based logic)
        # Here we just add a new 45-minute candle every 45 regular candles
        if len(self.ohlc_data['current_tf']['close']) % 45 == 0:
            # Use the last 45 candles to create a 45-minute candle
            last_45_close = self.ohlc_data['current_tf']['close'][-45:]
            last_45_high = self.ohlc_data['current_tf']['high'][-45:]
            last_45_low = self.ohlc_data['current_tf']['low'][-45:]
            
            # Calculate OHLC for the 45-minute candle
            m45_close = last_45_close[-1]
            m45_high = max([x for x in last_45_high if x is not None], default=close_price)
            m45_low = min([x for x in last_45_low if x is not None], default=close_price)
            
            # Add 45-minute data
            self.add_m45_data(m45_close, m45_high, m45_low)
        
        # Update all indicators
        self._update_indicators()
        
        # Calculate ARIMA forecast
        self._calculate_arima_forecast()
        
        # Calculate signal strengths for each indicator
        self._calculate_signal_strengths()
        
        # Determine final signal
        self._determine_final_signal()
        
        # Calculate order prices
        self._calculate_order_prices()
    
    def _update_indicators(self):
        """Update technical indicators"""
        # Filter out None values
        close_data = [x for x in self.ohlc_data['current_tf']['close'] if x is not None]
        high_data = [x for x in self.ohlc_data['current_tf']['high'] if x is not None]
        low_data = [x for x in self.ohlc_data['current_tf']['low'] if x is not None]
        
        if len(close_data) < 100 or len(high_data) < 100 or len(low_data) < 100:
            logger.warning("Not enough data for indicators")
            return
        
        # Convert to numpy arrays
        close_np = np.array(close_data)
        high_np = np.array(high_data)
        low_np = np.array(low_data)
        
        # Calculate EMAs
        self.indicators['ema9'] = self._calculate_ema(close_np, 9)
        self.indicators['ema21'] = self._calculate_ema(close_np, 21)
        self.indicators['ema50'] = self._calculate_ema(close_np, 50)
        self.indicators['ema100'] = self._calculate_ema(close_np, 100)
        
        # Calculate RSI
        self.indicators['rsi'] = self._calculate_rsi(close_np, 14)
        
        # Calculate MACD
        macd, macd_signal = self._calculate_macd(close_np, 12, 26, 9)
        self.indicators['macd'] = macd
        self.indicators['macd_signal'] = macd_signal
        
        # Calculate ADX, +DI, -DI
        adx, plus_di, minus_di = self._calculate_adx(high_np, low_np, close_np, 14)
        self.indicators['adx'] = adx
        self.indicators['plus_di'] = plus_di
        self.indicators['minus_di'] = minus_di
        
        # Calculate ATR
        self.indicators['atr'] = self._calculate_atr(high_np, low_np, close_np, self.atr_period)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_np, 20, 2)
        self.indicators['bb_upper'] = bb_upper
        self.indicators['bb_middle'] = bb_middle
        self.indicators['bb_lower'] = bb_lower
        
        # Calculate Keltner Channels
        kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(close_np, high_np, low_np, 20, 2)
        self.indicators['kc_upper'] = kc_upper
        self.indicators['kc_middle'] = kc_middle
        self.indicators['kc_lower'] = kc_lower
        
        # Calculate volatility (simple percent change)
        self.indicators['volatility'] = abs(close_np[-1] - close_np[-2]) / close_np[-2]
    
    def _calculate_arima_forecast(self):
        """Calculate linear regression forecast as ARIMA approximation"""
        # Get 45-minute data
        m45_close = [x for x in self.ohlc_data['m45']['close'] if x is not None]
        
        if len(m45_close) < self.arima_lookback:
            logger.warning(f"Not enough 45-minute data for ARIMA forecast (have {len(m45_close)}, need {self.arima_lookback})")
            return
        
        # Use the last arima_lookback periods
        data = m45_close[-self.arima_lookback:]
        
        try:
            # Simple linear regression as ARIMA approximation
            X = np.arange(len(data)).reshape(-1, 1)
            y = np.array(data)
            
            # Add a constant to the model
            X = sm.add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            
            # Predict the next value
            next_x = np.array([[1, len(data)]])
            forecast = model.predict(next_x)[0]
            
            self.indicators['arima_forecast'] = forecast
            
            current_close = self.ohlc_data['current_tf']['close'][-1]
            if current_close is not None and forecast is not None:
                forecast_diff = ((forecast - current_close) / current_close) * 100
                logger.info(f"【ANALYSIS】 Integrated ARIMA Forecast: {'BULLISH' if forecast > current_close else 'BEARISH'} | Current: ${current_close:.2f} → Target: ${forecast:.2f} | Diff: {forecast_diff:.2f}%")
        
        except Exception as e:
            logger.error(f"Error in ARIMA forecast calculation: {e}")
    
    def _calculate_signal_strengths(self):
        """
        Calculate strength values for each indicator/sub-strategy
        Strength is normalized to a scale of 0.0 to 1.0
        """
        close_price = self.ohlc_data['current_tf']['close'][-1]
        if close_price is None:
            return
        
        # EMA strength - based on distance and alignment
        if (self.indicators['ema9'] is not None and 
            self.indicators['ema21'] is not None and 
            self.indicators['ema50'] is not None and 
            self.indicators['ema100'] is not None):
            
            # EMA alignment score
            ema_alignment = 0
            ema_count = 0
            
            if self.indicators['ema9'] > self.indicators['ema21']:
                ema_alignment += 1
                ema_count += 1
            elif self.indicators['ema9'] < self.indicators['ema21']:
                ema_alignment -= 1
                ema_count += 1
            
            if self.indicators['ema21'] > self.indicators['ema50']:
                ema_alignment += 1
                ema_count += 1
            elif self.indicators['ema21'] < self.indicators['ema50']:
                ema_alignment -= 1
                ema_count += 1
            
            if self.indicators['ema50'] > self.indicators['ema100']:
                ema_alignment += 1
                ema_count += 1
            elif self.indicators['ema50'] < self.indicators['ema100']:
                ema_alignment -= 1
                ema_count += 1
            
            # Normalize to -1.0 to 1.0 range
            if ema_count > 0:
                ema_strength = ema_alignment / ema_count
                # Convert to 0.0 to 1.0 range
                if ema_strength > 0:
                    self.signal_strengths['ema'] = ema_strength
                else:
                    self.signal_strengths['ema'] = -ema_strength
            else:
                self.signal_strengths['ema'] = 0.0
        
        # RSI strength
        if self.indicators['rsi'] is not None:
            rsi = self.indicators['rsi']
            if rsi > 70:
                # Overbought - strength for bearish signal
                self.signal_strengths['rsi'] = min(1.0, (rsi - 70) / 30)
            elif rsi < 30:
                # Oversold - strength for bullish signal
                self.signal_strengths['rsi'] = min(1.0, (30 - rsi) / 30)
            else:
                self.signal_strengths['rsi'] = 0.0
        
        # MACD strength
        if (self.indicators['macd'] is not None and 
            self.indicators['macd_signal'] is not None):
            
            macd = self.indicators['macd']
            macd_signal = self.indicators['macd_signal']
            
            if macd > macd_signal:
                # Bullish MACD crossover
                diff = macd - macd_signal
                # Normalize diff against price
                norm_diff = diff / close_price * 100
                self.signal_strengths['macd'] = min(1.0, norm_diff * 20)  # Scale to 0-1
            elif macd < macd_signal:
                # Bearish MACD crossover
                diff = macd_signal - macd
                # Normalize diff against price
                norm_diff = diff / close_price * 100
                self.signal_strengths['macd'] = min(1.0, norm_diff * 20)
            else:
                self.signal_strengths['macd'] = 0.0
        
        # ADX strength
        if (self.indicators['adx'] is not None and 
            self.indicators['plus_di'] is not None and 
            self.indicators['minus_di'] is not None):
            
            adx = self.indicators['adx']
            plus_di = self.indicators['plus_di']
            minus_di = self.indicators['minus_di']
            
            trend_strength = min(1.0, adx / 50)  # ADX above 25 is strong trend
            
            if plus_di > minus_di:
                # Bullish trend
                self.signal_strengths['adx'] = trend_strength
            elif minus_di > plus_di:
                # Bearish trend
                self.signal_strengths['adx'] = trend_strength
            else:
                self.signal_strengths['adx'] = 0.0
        
        # Bollinger Bands strength
        if (self.indicators['bb_upper'] is not None and 
            self.indicators['bb_lower'] is not None and 
            self.indicators['bb_middle'] is not None):
            
            bb_upper = self.indicators['bb_upper']
            bb_lower = self.indicators['bb_lower']
            bb_middle = self.indicators['bb_middle']
            
            # Calculate percent from middle to bands
            band_range = bb_upper - bb_lower
            if band_range > 0:
                # Position relative to bands
                if close_price > bb_upper:
                    # Overbought - bearish signal
                    self.signal_strengths['bb'] = min(1.0, (close_price - bb_upper) / (band_range * 0.1))
                elif close_price < bb_lower:
                    # Oversold - bullish signal
                    self.signal_strengths['bb'] = min(1.0, (bb_lower - close_price) / (band_range * 0.1))
                else:
                    self.signal_strengths['bb'] = 0.0
            else:
                self.signal_strengths['bb'] = 0.0
        
        # Keltner Channels strength
        if (self.indicators['kc_upper'] is not None and 
            self.indicators['kc_lower'] is not None and 
            self.indicators['kc_middle'] is not None):
            
            kc_upper = self.indicators['kc_upper']
            kc_lower = self.indicators['kc_lower']
            kc_middle = self.indicators['kc_middle']
            
            # Calculate percent from middle to channels
            channel_range = kc_upper - kc_lower
            if channel_range > 0:
                # Position relative to channels
                if close_price > kc_upper:
                    # Strong bullish - price breaking out above KC
                    self.signal_strengths['kc'] = min(1.0, (close_price - kc_upper) / (channel_range * 0.1))
                elif close_price < kc_lower:
                    # Strong bearish - price breaking down below KC
                    self.signal_strengths['kc'] = min(1.0, (kc_lower - close_price) / (channel_range * 0.1))
                else:
                    # Inside channels - no significant signal
                    self.signal_strengths['kc'] = 0.0
            else:
                self.signal_strengths['kc'] = 0.0
        
        # ARIMA strength - based on forecast direction and magnitude
        if self.indicators['arima_forecast'] is not None:
            forecast = self.indicators['arima_forecast']
            percent_diff = abs((forecast - close_price) / close_price) * 100
            
            # Scale percent_diff to 0-1 range, capped at 1.0
            arima_strength = min(1.0, percent_diff * 20)  # 0.05% diff = 1.0 strength
            
            self.signal_strengths['arima'] = arima_strength
        
        # Volatility filter strength
        if self.indicators['volatility'] is not None:
            volatility = self.indicators['volatility']
            vol_strength = min(1.0, volatility / self.vol_threshold)
            self.signal_strengths['volatility'] = vol_strength
    
    def _determine_final_signal(self):
        """
        Compare strengths from all sub-strategies to find
        the strongest bullish and bearish signals
        """
        # Reset values
        self.bullish_strength = 0.0
        self.bearish_strength = 0.0
        self.strongest_bullish = None
        self.strongest_bearish = None
        
        # Get current close price
        close_price = self.ohlc_data['current_tf']['close'][-1]
        if close_price is None:
            return
        
        # Log indicator values for debugging
        logger.info(f"【INTEGRATED】 Current price: ${close_price:.2f}")
        
        if self.indicators['ema9'] is not None and self.indicators['ema21'] is not None:
            logger.info(f"【INTEGRATED】 EMA9 vs EMA21: {self.indicators['ema9']:.2f} vs {self.indicators['ema21']:.2f} | {'Bullish' if self.indicators['ema9'] > self.indicators['ema21'] else 'Bearish'}")
        
        if self.indicators['rsi'] is not None:
            rsi_state = "Oversold" if self.indicators['rsi'] < 30 else "Overbought" if self.indicators['rsi'] > 70 else "Neutral"
            logger.info(f"【INTEGRATED】 RSI: {self.indicators['rsi']:.2f} | State: {rsi_state}")
        
        if self.indicators['macd'] is not None and self.indicators['macd_signal'] is not None:
            logger.info(f"【INTEGRATED】 MACD vs Signal: {self.indicators['macd']:.4f} vs {self.indicators['macd_signal']:.4f} | {'Bullish' if self.indicators['macd'] > self.indicators['macd_signal'] else 'Bearish'}")
        
        if self.indicators['adx'] is not None and self.indicators['plus_di'] is not None and self.indicators['minus_di'] is not None:
            adx_state = "Strong Trend" if self.indicators['adx'] > 25 else "Weak Trend"
            di_state = "Bullish" if self.indicators['plus_di'] > self.indicators['minus_di'] else "Bearish"
            logger.info(f"【INTEGRATED】 ADX: {self.indicators['adx']:.2f} | State: {adx_state} | +DI vs -DI: {self.indicators['plus_di']:.2f} vs {self.indicators['minus_di']:.2f} | {di_state}")
        
        if self.indicators['bb_upper'] is not None and self.indicators['bb_lower'] is not None:
            bb_position = "Above Upper" if close_price > self.indicators['bb_upper'] else "Below Lower" if close_price < self.indicators['bb_lower'] else "Inside"
            logger.info(f"【INTEGRATED】 Bollinger Bands: Upper={self.indicators['bb_upper']:.2f}, Lower={self.indicators['bb_lower']:.2f} | Position: {bb_position}")
        
        if self.indicators['kc_upper'] is not None and self.indicators['kc_lower'] is not None:
            kc_position = "Above Upper" if close_price > self.indicators['kc_upper'] else "Below Lower" if close_price < self.indicators['kc_lower'] else "Inside"
            logger.info(f"【INTEGRATED】 Keltner Channels: Upper={self.indicators['kc_upper']:.2f}, Lower={self.indicators['kc_lower']:.2f} | Position: {kc_position}")
        
        if self.indicators['arima_forecast'] is not None:
            arima_direction = "Bullish" if self.indicators['arima_forecast'] > close_price else "Bearish" if self.indicators['arima_forecast'] < close_price else "Neutral"
            forecast_pct = ((self.indicators['arima_forecast'] / close_price) - 1) * 100
            logger.info(f"【INTEGRATED】 ARIMA Forecast: ${self.indicators['arima_forecast']:.2f} | Direction: {arima_direction} | Change: {forecast_pct:+.2f}%")
        
        if self.indicators['volatility'] is not None:
            vol_state = "Above Threshold" if self.indicators['volatility'] >= self.vol_threshold else "Below Threshold"
            logger.info(f"【INTEGRATED】 Volatility: {self.indicators['volatility']:.4f} | State: {vol_state} (threshold: {self.vol_threshold})")
        
        # Log all signal strengths before final determination
        logger.info(f"【INTEGRATED】 Signal Strengths: EMA={self.signal_strengths['ema']:.2f}, RSI={self.signal_strengths['rsi']:.2f}, MACD={self.signal_strengths['macd']:.2f}, ADX={self.signal_strengths['adx']:.2f}, BB={self.signal_strengths['bb']:.2f}, KC={self.signal_strengths['kc']:.2f}, ARIMA={self.signal_strengths['arima']:.2f}")
        
        # --- Determine Bullish Signals ---
        bullish_signals = []
        
        # EMA signal
        if (self.indicators['ema9'] is not None and 
            self.indicators['ema21'] is not None and 
            self.indicators['ema9'] > self.indicators['ema21']):
            
            bullish_signals.append(("EMA", self.signal_strengths['ema']))
            if self.signal_strengths['ema'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['ema']
                self.strongest_bullish = 'ema'
        
        # RSI signal
        if self.indicators['rsi'] is not None and self.indicators['rsi'] < 40:
            bullish_signals.append(("RSI", self.signal_strengths['rsi']))
            if self.signal_strengths['rsi'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['rsi']
                self.strongest_bullish = 'rsi'
        
        # MACD signal
        if (self.indicators['macd'] is not None and 
            self.indicators['macd_signal'] is not None and 
            self.indicators['macd'] > self.indicators['macd_signal']):
            
            bullish_signals.append(("MACD", self.signal_strengths['macd']))
            if self.signal_strengths['macd'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['macd']
                self.strongest_bullish = 'macd'
        
        # ADX signal
        if (self.indicators['adx'] is not None and 
            self.indicators['plus_di'] is not None and 
            self.indicators['minus_di'] is not None and 
            self.indicators['plus_di'] > self.indicators['minus_di']):
            
            bullish_signals.append(("ADX", self.signal_strengths['adx']))
            if self.signal_strengths['adx'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['adx']
                self.strongest_bullish = 'adx'
        
        # Bollinger Bands signal
        if (self.indicators['bb_lower'] is not None and 
            close_price < self.indicators['bb_lower']):
            
            bullish_signals.append(("BB", self.signal_strengths['bb']))
            if self.signal_strengths['bb'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['bb']
                self.strongest_bullish = 'bb'
        
        # Keltner Channels signal
        if (self.indicators['kc_upper'] is not None and 
            close_price > self.indicators['kc_upper']):
            
            bullish_signals.append(("KC", self.signal_strengths['kc']))
            if self.signal_strengths['kc'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['kc']
                self.strongest_bullish = 'kc'
        
        # ARIMA signal
        if (self.indicators['arima_forecast'] is not None and 
            self.indicators['arima_forecast'] > close_price):
            
            bullish_signals.append(("ARIMA", self.signal_strengths['arima']))
            if self.signal_strengths['arima'] > self.bullish_strength:
                self.bullish_strength = self.signal_strengths['arima']
                self.strongest_bullish = 'arima'
        
        # --- Determine Bearish Signals ---
        bearish_signals = []
        
        # EMA signal
        if (self.indicators['ema9'] is not None and 
            self.indicators['ema21'] is not None and 
            self.indicators['ema9'] < self.indicators['ema21']):
            
            bearish_signals.append(("EMA", self.signal_strengths['ema']))
            if self.signal_strengths['ema'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['ema']
                self.strongest_bearish = 'ema'
        
        # RSI signal
        if self.indicators['rsi'] is not None and self.indicators['rsi'] > 60:
            bearish_signals.append(("RSI", self.signal_strengths['rsi']))
            if self.signal_strengths['rsi'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['rsi']
                self.strongest_bearish = 'rsi'
        
        # MACD signal
        if (self.indicators['macd'] is not None and 
            self.indicators['macd_signal'] is not None and 
            self.indicators['macd'] < self.indicators['macd_signal']):
            
            bearish_signals.append(("MACD", self.signal_strengths['macd']))
            if self.signal_strengths['macd'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['macd']
                self.strongest_bearish = 'macd'
        
        # ADX signal
        if (self.indicators['adx'] is not None and 
            self.indicators['plus_di'] is not None and 
            self.indicators['minus_di'] is not None and 
            self.indicators['plus_di'] < self.indicators['minus_di']):
            
            bearish_signals.append(("ADX", self.signal_strengths['adx']))
            if self.signal_strengths['adx'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['adx']
                self.strongest_bearish = 'adx'
        
        # Bollinger Bands signal
        if (self.indicators['bb_upper'] is not None and 
            close_price > self.indicators['bb_upper']):
            
            bearish_signals.append(("BB", self.signal_strengths['bb']))
            if self.signal_strengths['bb'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['bb']
                self.strongest_bearish = 'bb'
        
        # Keltner Channels signal
        if (self.indicators['kc_lower'] is not None and 
            close_price < self.indicators['kc_lower']):
            
            bearish_signals.append(("KC", self.signal_strengths['kc']))
            if self.signal_strengths['kc'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['kc']
                self.strongest_bearish = 'kc'
        
        # ARIMA signal
        if (self.indicators['arima_forecast'] is not None and 
            self.indicators['arima_forecast'] < close_price):
            
            bearish_signals.append(("ARIMA", self.signal_strengths['arima']))
            if self.signal_strengths['arima'] > self.bearish_strength:
                self.bearish_strength = self.signal_strengths['arima']
                self.strongest_bearish = 'arima'
        
        # Log all bullish and bearish signals before filtering
        if bullish_signals:
            bullish_signals.sort(key=lambda x: x[1], reverse=True)
            bullish_list = ", ".join([f"{sig[0]}:{sig[1]:.2f}" for sig in bullish_signals])
            logger.info(f"【INTEGRATED】 Bullish Signals: {bullish_list}")
        
        if bearish_signals:
            bearish_signals.sort(key=lambda x: x[1], reverse=True)
            bearish_list = ", ".join([f"{sig[0]}:{sig[1]:.2f}" for sig in bearish_signals])
            logger.info(f"【INTEGRATED】 Bearish Signals: {bearish_list}")
        
        # Apply volatility filter
        if self.indicators['volatility'] is not None:
            volatility = self.indicators['volatility']
            if volatility < self.vol_threshold:
                # Log before applying volatility filter
                logger.info(f"【INTEGRATED】 Before volatility filter - Bullish: {self.bullish_strength:.2f}, Bearish: {self.bearish_strength:.2f}")
                
                # Reduce both signal strengths based on low volatility
                vol_factor = volatility / self.vol_threshold
                self.bullish_strength *= vol_factor
                self.bearish_strength *= vol_factor
                
                # Log after applying volatility filter
                logger.info(f"【INTEGRATED】 After volatility filter (factor: {vol_factor:.2f}) - Bullish: {self.bullish_strength:.2f}, Bearish: {self.bearish_strength:.2f}")
        
        # Determine the final signal
        if self.bullish_strength >= self.min_strength and self.bullish_strength > self.bearish_strength:
            self.current_signal = 'long'
            self.signal_strength = self.bullish_strength
            logger.info(f"【SIGNAL】 🟢 BULLISH - {self.strongest_bullish.upper()} signal with strength {self.bullish_strength:.2f}")
            logger.info(f"【INTEGRATED】 Final Decision: LONG (strength: {self.bullish_strength:.2f}) driven by {self.strongest_bullish.upper()}")
        elif self.bearish_strength >= self.min_strength and self.bearish_strength > self.bullish_strength:
            self.current_signal = 'short'
            self.signal_strength = self.bearish_strength
            logger.info(f"【SIGNAL】 🔴 BEARISH - {self.strongest_bearish.upper()} signal with strength {self.bearish_strength:.2f}")
            logger.info(f"【INTEGRATED】 Final Decision: SHORT (strength: {self.bearish_strength:.2f}) driven by {self.strongest_bearish.upper()}")
        else:
            self.current_signal = None
            self.signal_strength = 0.0
            logger.info(f"【SIGNAL】 ⚪ NEUTRAL - No strong signal detected (bullish: {self.bullish_strength:.2f}, bearish: {self.bearish_strength:.2f})")
            logger.info(f"【INTEGRATED】 Final Decision: NEUTRAL - Insufficient signal strength (min threshold: {self.min_strength})")
    
    def _calculate_order_prices(self):
        """Calculate optimal entry prices for long and short positions"""
        close_price = self.ohlc_data['current_tf']['close'][-1]
        if close_price is None or self.indicators['atr'] is None:
            return
        
        atr = self.indicators['atr']
        
        # Calculate base entry offset
        entry_offset = atr * self.entry_offset_multiplier
        
        # Adjust offset based on the source of the signal
        if self.strongest_bullish == 'arima' or self.strongest_bearish == 'arima':
            # ARIMA-based entries use a different offset multiplier
            entry_offset = atr * self.arima_entry_offset_multiplier
        
        # Calculate entry prices with offset
        self.long_entry_price = close_price - entry_offset
        self.short_entry_price = close_price + entry_offset
    
    def should_buy(self) -> bool:
        """
        Determine if a buy signal is present
        
        Returns:
            bool: True if should buy, False otherwise
        """
        if self.current_signal == 'long' and self.signal_strength >= self.min_strength:
            # Check if there's already a position and if the signal is strong enough to override
            if self.position == 'short' and STRONGER_SIGNAL_DOMINANCE:
                # Need a significantly stronger signal to override
                current_strength = self.get_stored_strength()
                if (self.signal_strength - current_strength) >= SIGNAL_STRENGTH_ADVANTAGE:
                    logger.info(f"【OVERRIDE】 Long signal ({self.signal_strength:.2f}) overriding short ({current_strength:.2f})")
                    return True
                else:
                    return False
            elif self.position == 'long':
                # Already long, no need to buy again
                return False
            else:
                # No position, buy if signal is strong enough
                return True
        return False
    
    def should_sell(self) -> bool:
        """
        Determine if a sell signal is present
        
        Returns:
            bool: True if should sell, False otherwise
        """
        if self.current_signal == 'short' and self.signal_strength >= self.min_strength:
            # Check if there's already a position and if the signal is strong enough to override
            if self.position == 'long' and STRONGER_SIGNAL_DOMINANCE:
                # Need a significantly stronger signal to override
                current_strength = self.get_stored_strength()
                if (self.signal_strength - current_strength) >= SIGNAL_STRENGTH_ADVANTAGE:
                    logger.info(f"【OVERRIDE】 Short signal ({self.signal_strength:.2f}) overriding long ({current_strength:.2f})")
                    return True
                else:
                    return False
            elif self.position == 'short':
                # Already short, no need to sell again
                return False
            else:
                # No position, sell if signal is strong enough
                return True
        return False
    
    def should_exit_short(self) -> bool:
        """
        Determine if a short position should be exited
        
        Returns:
            bool: True if should exit short, False otherwise
        """
        # Exit if there's a strong opposing signal 
        return self.current_signal == 'long' and self.signal_strength >= self.min_strength
    
    def get_stored_strength(self) -> float:
        """
        Get the strength of the current position when it was opened
        
        Returns:
            float: Strength value at position entry
        """
        return self.signal_strength
    
    def get_entry_price(self, is_long: bool) -> float:
        """
        Calculate the optimal entry price
        
        Args:
            is_long (bool): True for long entry, False for short entry
            
        Returns:
            float: Optimal entry price
        """
        if is_long and self.long_entry_price is not None:
            return self.long_entry_price
        elif not is_long and self.short_entry_price is not None:
            return self.short_entry_price
        else:
            # Default to current price if entry prices not calculated
            return self.ohlc_data['current_tf']['close'][-1]
    
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate trading signals from DataFrame with indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # This implementation is simplified since we use internal state
        # to track signals, but we need to implement this for the interface
        return self.should_buy(), self.should_sell(), self.indicators['atr'] if self.indicators['atr'] is not None else 0.0
    
    def get_status(self) -> Dict:
        """
        Get current strategy status and metrics
        
        Returns:
            dict: Strategy status information
        """
        status = super().get_status()
        
        # Add integrated strategy specific status
        status.update({
            "category": self.category,
            "signal_strength": self.signal_strength,
            "bullish_strength": self.bullish_strength,
            "bearish_strength": self.bearish_strength,
            "strongest_bullish_signal": self.strongest_bullish,
            "strongest_bearish_signal": self.strongest_bearish,
            "indicators": {name: value for name, value in self.indicators.items() if value is not None},
            "signal_strengths": self.signal_strengths,
        })
        
        return status
    
    # Helper methods for indicator calculations
    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data[-period:].mean() if len(data) >= period else None
    
    def _calculate_rsi(self, data, period):
        """Calculate Relative Strength Index"""
        if len(data) < period + 1:
            return None
        
        # Calculate price changes
        delta = np.diff(data)
        
        # Separate gains and losses
        gains = np.copy(delta)
        losses = np.copy(delta)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gain and loss
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        # Calculate RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, data, fast_period, slow_period, signal_period):
        """Calculate MACD and Signal line"""
        if len(data) < slow_period + signal_period:
            return None, None
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(data, fast_period)
        slow_ema = self._calculate_ema(data, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None, None
        
        # Calculate MACD
        macd = fast_ema - slow_ema
        
        # Calculate Signal line (EMA of MACD)
        # Simplified for this example
        signal = macd * 0.2 + (macd * 0.8)  # Approximation of 9-period EMA
        
        return macd, signal
    
    def _calculate_adx(self, high, low, close, period):
        """Calculate ADX, +DI, -DI"""
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return None, None, None
        
        # Simplified calculation
        # In a real implementation, we would calculate True Range, +DM, -DM, etc.
        return 25.0, 20.0, 15.0  # Placeholder values
    
    def _calculate_atr(self, high, low, close, period):
        """Calculate Average True Range"""
        if len(high) < period or len(low) < period or len(close) < period:
            return None
        
        # Calculate True Range
        tr1 = np.max(np.column_stack((high[1:] - low[1:], 
                                      abs(high[1:] - close[:-1]), 
                                      abs(low[1:] - close[:-1]))), axis=1)
        
        # Calculate ATR
        atr = np.mean(tr1[-period:])
        
        return atr
    
    def _calculate_bollinger_bands(self, data, period, stdev_multiplier):
        """Calculate Bollinger Bands"""
        if len(data) < period:
            return None, None, None
        
        # Calculate SMA
        sma = np.mean(data[-period:])
        
        # Calculate Standard Deviation
        stdev = np.std(data[-period:])
        
        # Calculate Bands
        upper_band = sma + (stdev * stdev_multiplier)
        lower_band = sma - (stdev * stdev_multiplier)
        
        return upper_band, sma, lower_band
    
    def _calculate_keltner_channels(self, close, high, low, period, multiplier):
        """Calculate Keltner Channels"""
        if len(close) < period or len(high) < period or len(low) < period:
            return None, None, None
        
        # Calculate EMA
        ema = self._calculate_ema(close, period)
        
        # Calculate ATR
        atr = self._calculate_atr(high, low, close, period)
        
        if ema is None or atr is None:
            return None, None, None
        
        # Calculate Keltner Channels
        upper_channel = ema + (atr * multiplier)
        lower_channel = ema - (atr * multiplier)
        
        return upper_channel, ema, lower_channel