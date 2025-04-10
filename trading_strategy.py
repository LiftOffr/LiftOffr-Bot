import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from utils import calculate_sma, calculate_rsi, calculate_atr, calculate_ema
from config import (
    SMA_SHORT_PERIOD, SMA_LONG_PERIOD, 
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    VOL_THRESHOLD, ENTRY_ATR_MULTIPLIER, 
    BREAKEVEN_PROFIT_TARGET, ARIMA_LOOKBACK
)

logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies
    """
    def __init__(self, symbol: str):
        """
        Initialize the trading strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'XBTUSD')
        """
        self.symbol = symbol
        self.prices = []
        self.highs = []
        self.lows = []
        self.position = None  # None, "long", or "short"
        self.entry_price = None
        
        # For trailing stops and exits
        self.trailing_max_price = None  # For long positions
        self.trailing_min_price = None  # For short positions
        
        # ATR value for position sizing and stop placement
        self.current_atr = None
    
    def update_ohlc(self, open_price: float, high_price: float, low_price: float, close_price: float):
        """
        Update OHLC data
        
        Args:
            open_price (float): Open price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
        """
        self.prices.append(close_price)
        self.highs.append(high_price)
        self.lows.append(low_price)
        
        # Calculate ATR if we have enough data
        if len(self.highs) > 14:
            self.current_atr = calculate_atr(self.highs, self.lows, self.prices, 14)
    
    def update_position(self, position: Optional[str], entry_price: Optional[float] = None):
        """
        Update current position status
        
        Args:
            position (str): Current position ("long", "short", or None)
            entry_price (float, optional): Price at which position was entered
        """
        self.position = position
        self.entry_price = entry_price
        
        # Reset trailing prices on new positions
        if position == "long":
            self.trailing_max_price = entry_price
            self.trailing_min_price = None
        elif position == "short":
            self.trailing_min_price = entry_price
            self.trailing_max_price = None
        else:
            self.trailing_max_price = None
            self.trailing_min_price = None
    
    def update_trailing_prices(self, current_price: float):
        """
        Update trailing price levels for exit strategies
        
        Args:
            current_price (float): Current market price
        """
        if self.position == "long" and self.trailing_max_price is not None:
            self.trailing_max_price = max(self.trailing_max_price, current_price)
        elif self.position == "short" and self.trailing_min_price is not None:
            self.trailing_min_price = min(self.trailing_min_price, current_price)
    
    @abstractmethod
    def should_buy(self) -> bool:
        """
        Determine if a buy signal is present
        
        Returns:
            bool: True if should buy, False otherwise
        """
        pass
    
    @abstractmethod
    def should_sell(self) -> bool:
        """
        Determine if a sell signal is present
        
        Returns:
            bool: True if should sell, False otherwise
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate trading signals from DataFrame with indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        pass

class SimpleMovingAverageStrategy(TradingStrategy):
    """
    Simple Moving Average (SMA) crossover strategy
    """
    def __init__(self, symbol: str, short_period: int = SMA_SHORT_PERIOD, 
                long_period: int = SMA_LONG_PERIOD):
        """
        Initialize the SMA strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'XBTUSD')
            short_period (int, optional): Short period for SMA
            long_period (int, optional): Long period for SMA
        """
        super().__init__(symbol)
        self.short_period = short_period
        self.long_period = long_period
        
        # Store previous SMAs for crossover detection
        self.prev_short_sma = None
        self.prev_long_sma = None
    
    def should_buy(self) -> bool:
        """
        Buy when short SMA crosses above long SMA
        
        Returns:
            bool: True if should buy, False otherwise
        """
        if len(self.prices) < self.long_period:
            return False
        
        short_sma = calculate_sma(self.prices, self.short_period)
        long_sma = calculate_sma(self.prices, self.long_period)
        
        # Check for crossover
        is_crossover = (self.prev_short_sma is not None and 
                       self.prev_long_sma is not None and 
                       self.prev_short_sma <= self.prev_long_sma and 
                       short_sma > long_sma)
        
        # Update previous SMAs
        self.prev_short_sma = short_sma
        self.prev_long_sma = long_sma
        
        return is_crossover and self.position is None
    
    def should_sell(self) -> bool:
        """
        Sell when short SMA crosses below long SMA
        
        Returns:
            bool: True if should sell, False otherwise
        """
        if len(self.prices) < self.long_period:
            return False
        
        short_sma = calculate_sma(self.prices, self.short_period)
        long_sma = calculate_sma(self.prices, self.long_period)
        
        # Check for crossover
        is_crossover = (self.prev_short_sma is not None and 
                       self.prev_long_sma is not None and 
                       self.prev_short_sma >= self.prev_long_sma and 
                       short_sma < long_sma)
        
        # Update previous SMAs
        self.prev_short_sma = short_sma
        self.prev_long_sma = long_sma
        
        return is_crossover and self.position == "long"
    
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate SMA signals from DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # Calculate indicators if they don't exist
        if 'SMA_short' not in df.columns:
            df['SMA_short'] = df['close'].rolling(window=self.short_period).mean()
        if 'SMA_long' not in df.columns:
            df['SMA_long'] = df['close'].rolling(window=self.long_period).mean()
        if 'ATR' not in df.columns:
            df['ATR'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # Get last row for signal calculation
        last = df.iloc[-1]
        second_last = df.iloc[-2] if len(df) > 1 else None
        
        # Buy signal: Short SMA crosses above long SMA
        buy_signal = (second_last is not None and 
                     second_last['SMA_short'] <= second_last['SMA_long'] and 
                     last['SMA_short'] > last['SMA_long'])
        
        # Sell signal: Short SMA crosses below long SMA
        sell_signal = (second_last is not None and 
                      second_last['SMA_short'] >= second_last['SMA_long'] and 
                      last['SMA_short'] < last['SMA_long'])
        
        return buy_signal, sell_signal, last['ATR']

class RSIStrategy(TradingStrategy):
    """
    Relative Strength Index (RSI) strategy
    """
    def __init__(self, symbol: str, period: int = RSI_PERIOD, 
                overbought: int = RSI_OVERBOUGHT, oversold: int = RSI_OVERSOLD):
        """
        Initialize the RSI strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'XBTUSD')
            period (int, optional): Period for RSI calculation
            overbought (int, optional): Overbought threshold
            oversold (int, optional): Oversold threshold
        """
        super().__init__(symbol)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
        # Store previous RSI for change detection
        self.prev_rsi = None
    
    def should_buy(self) -> bool:
        """
        Buy when RSI crosses above oversold threshold
        
        Returns:
            bool: True if should buy, False otherwise
        """
        if len(self.prices) <= self.period:
            return False
        
        rsi = calculate_rsi(self.prices, self.period)
        
        # Check for crossover of oversold threshold
        is_crossover = (self.prev_rsi is not None and 
                       self.prev_rsi <= self.oversold and 
                       rsi > self.oversold)
        
        # Update previous RSI
        self.prev_rsi = rsi
        
        return is_crossover and self.position is None
    
    def should_sell(self) -> bool:
        """
        Sell when RSI crosses below overbought threshold
        
        Returns:
            bool: True if should sell, False otherwise
        """
        if len(self.prices) <= self.period:
            return False
        
        rsi = calculate_rsi(self.prices, self.period)
        
        # Check for crossover of overbought threshold
        is_crossover = (self.prev_rsi is not None and 
                       self.prev_rsi >= self.overbought and 
                       rsi < self.overbought)
        
        # Update previous RSI
        self.prev_rsi = rsi
        
        return is_crossover and self.position == "long"
    
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate RSI signals from DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # Calculate indicators if they don't exist
        if 'RSI' not in df.columns:
            df['RSI'] = calculate_rsi(df['close'].values, self.period)
        if 'ATR' not in df.columns:
            df['ATR'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # Get last row for signal calculation
        last = df.iloc[-1]
        second_last = df.iloc[-2] if len(df) > 1 else None
        
        # Buy signal: RSI crosses above oversold threshold
        buy_signal = (second_last is not None and 
                     second_last['RSI'] <= self.oversold and 
                     last['RSI'] > self.oversold)
        
        # Sell signal: RSI crosses below overbought threshold
        sell_signal = (second_last is not None and 
                      second_last['RSI'] >= self.overbought and 
                      last['RSI'] < self.overbought)
        
        return buy_signal, sell_signal, last['ATR']

class AdaptiveStrategy(TradingStrategy):
    """
    Advanced adaptive strategy (based on user's original code)
    Using EMA, RSI, MACD, ADX, ATR, Bollinger Bands, and Keltner Channels
    """
    def __init__(self, symbol: str):
        """
        Initialize the adaptive strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'XBTUSD')
        """
        super().__init__(symbol)
        # Store the latest indicators
        self.latest_indicators = None
        # For linear regression forecast
        self.arima_lookback = ARIMA_LOOKBACK
    
    def compute_arima_forecast(self, series):
        """
        Compute simple linear regression forecast
        
        Args:
            series: Price series to forecast
            
        Returns:
            float: Forecasted value
        """
        if len(series) < self.arima_lookback:
            return series.iloc[-1]
        data = series.iloc[-self.arima_lookback:]
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope, intercept = coeffs[0], coeffs[1]
        # Forecast one period ahead
        forecast = slope * len(data) + intercept
        return forecast
    
    def should_buy(self) -> bool:
        """
        Buy signal for adaptive strategy (simple implementation)
        
        Returns:
            bool: True if should buy, False otherwise
        """
        # This is a simplified version, the full version uses the calculate_signals method
        if len(self.prices) < 30 or self.current_atr is None:
            return False
        
        # Removed volatility threshold check
        last_price = self.prices[-1] if self.prices else 0
        if last_price == 0:
            return False
        
        # Modified: Remove volatility threshold condition entirely
        # Just check if we're not in a position
        return self.position is None
    
    def should_sell(self) -> bool:
        """
        Sell signal for adaptive strategy (simple implementation)
        
        Returns:
            bool: True if should sell, False otherwise
        """
        # This is a simplified version, the full version uses the calculate_signals method
        if len(self.prices) < 30 or self.current_atr is None:
            return False
        
        # Check if in long position and we've hit trailing stop
        if self.position == "long" and self.trailing_max_price is not None:
            last_price = self.prices[-1]
            trailing_stop = self.trailing_max_price - (2.0 * self.current_atr)
            if last_price <= trailing_stop:
                return True
        
        return False
    
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate complex adaptive signals from DataFrame with indicators
        Implements the original strategy logic from user's code
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicators
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # Ensure all required indicators are calculated
        if 'EMA9' not in df.columns:
            df['EMA9'] = calculate_ema(df['close'].values, 9)
        if 'EMA21' not in df.columns:
            df['EMA21'] = calculate_ema(df['close'].values, 21)
        if 'RSI14' not in df.columns:
            df['RSI14'] = calculate_rsi(df['close'].values, 14)
        if 'ATR14' not in df.columns:
            df['ATR14'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # MACD calculation
        if 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=12, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ADX calculation
        if 'ADX' not in df.columns:
            # Simplified ADX since we don't have the full implementation
            df['ADX'] = 25  # Default to 25 for now
        
        # Bollinger Bands
        if 'bb_upper' not in df.columns:
            sma20 = df['close'].rolling(window=20).mean()
            stddev = df['close'].rolling(window=20).std()
            df['bb_middle'] = sma20
            df['bb_upper'] = sma20 + (stddev * 2)
            df['bb_lower'] = sma20 - (stddev * 2)
        
        # Keltner Channels
        if 'kc_middle' not in df.columns:
            df['kc_middle'] = df['close'].ewm(span=20, adjust=False).mean()
            df['kc_upper'] = df['kc_middle'] + (df['ATR14'] * 2)
            df['kc_lower'] = df['kc_middle'] - (df['ATR14'] * 2)
        
        # Get the last row for signal calculation
        last = df.iloc[-1]
        
        # Store current ATR
        atr_value = last['ATR14']
        
        # Normalized ATR for adaptive filter
        normalized_atr = atr_value / last['close'] if last['close'] != 0 else 0
        
        # Modified: Setting adaptive_filter to True to remove volatility threshold condition
        adaptive_filter = True
        
        # Calculate bullish and bearish conditions (from original code)
        ema_condition = last['EMA9'] > last['EMA21']
        rsi_bullish_condition = 45 < last['RSI14'] < 75
        rsi_bearish_condition = 25 < last['RSI14'] < 55
        macd_condition = last['MACD'] > last['MACD_signal']
        adx_condition = last['ADX'] > 20
        volatility_condition = True  # Modified: Always True to remove this condition
        bollinger_upper_condition = last['close'] < last['bb_upper']
        bollinger_lower_condition = last['close'] > last['bb_lower']
        keltner_middle_condition_bull = last['close'] > last['kc_middle']
        keltner_middle_condition_bear = last['close'] < last['kc_middle']
        
        # Log detailed conditions
        condition_log = [
            f"EMA9 {'>=' if ema_condition else '<'} EMA21 ({last['EMA9']:.2f} vs {last['EMA21']:.2f})",
            f"RSI = {last['RSI14']:.2f} {'✓' if rsi_bullish_condition or rsi_bearish_condition else '✗'}",
            f"MACD {'>=' if macd_condition else '<'} Signal ({last['MACD']:.4f} vs {last['MACD_signal']:.4f})",
            f"ADX = {last['ADX']:.2f} {'✓' if adx_condition else '✗'}",
            f"Volatility = {normalized_atr:.4f} {'✓' if volatility_condition else '✗'} (threshold: {VOL_THRESHOLD})",
            f"Price {'<' if bollinger_upper_condition else '>='} Upper BB ({last['close']:.2f} vs {last['bb_upper']:.2f})",
            f"Price {'>' if bollinger_lower_condition else '<='} Lower BB ({last['close']:.2f} vs {last['bb_lower']:.2f})",
            f"Price vs KC Middle: {last['close']:.2f} vs {last['kc_middle']:.2f}"
        ]
        
        # Combine all conditions
        bullish = ema_condition and rsi_bullish_condition and macd_condition and adx_condition and \
                  volatility_condition and bollinger_upper_condition and keltner_middle_condition_bull
        
        bearish = (not ema_condition) and rsi_bearish_condition and (not macd_condition) and adx_condition and \
                  volatility_condition and bollinger_lower_condition and keltner_middle_condition_bear
        
        # Add ARIMA forecast influence
        arima_forecast = self.compute_arima_forecast(df['close'])
        forecast_direction = "BULLISH" if arima_forecast > last['close'] else "BEARISH" if arima_forecast < last['close'] else "NEUTRAL"
        
        # Create more formatted logs with consistent spacing and clear sections
        logger.info("============================================================")
        logger.info(f"【ANALYSIS】 Forecast: {forecast_direction} | Current: ${last['close']:.2f} → Target: ${arima_forecast:.2f}")
        
        # Format technical indicators in a more readable grouped format
        ema_status = "EMA9 >= EMA21" if ema_condition else "EMA9 < EMA21"
        rsi_value = f"RSI = {last['RSI14']:.2f} {'✓' if (rsi_bullish_condition or rsi_bearish_condition) else '✗'}"
        macd_status = "MACD >= Signal" if macd_condition else "MACD < Signal"
        adx_info = f"ADX = {last['ADX']:.2f} {'✓' if adx_condition else '✗'}"
        vol_info = f"Volatility = {normalized_atr:.4f} {'✓' if volatility_condition else '✗'} (threshold: {VOL_THRESHOLD})"
        
        logger.info(f"【INDICATORS】 {ema_status} | {rsi_value} | {macd_status} | {adx_info}")
        logger.info(f"【VOLATILITY】 {vol_info}")
        logger.info(f"【BANDS】 {' | '.join(condition_log)}")
        
        agg_signal = (1.0 if bullish else (-1.0 if bearish else 0.0))
        agg_signal += 0.5 if (arima_forecast > last['close']) else (-0.5 if arima_forecast < last['close'] else 0.0)
        
        final_bullish = agg_signal >= 1.0
        final_bearish = agg_signal <= -1.0
        
        # Format signal with emoji indicators and clear action
        signal_emoji = "🟢" if final_bullish else "🔴" if final_bearish else "⚪"
        if final_bullish:
            logger.info(f"【SIGNAL】 {signal_emoji} BULLISH - Trade conditions met for LONG position")
        elif final_bearish:
            logger.info(f"【SIGNAL】 {signal_emoji} BEARISH - Trade conditions met for SHORT position")
        else:
            logger.info(f"【SIGNAL】 {signal_emoji} NEUTRAL - No clear trade signal detected")
        logger.info("============================================================")
        
        # Store the latest indicators
        self.latest_indicators = last
        
        return final_bullish, final_bearish, atr_value

def get_strategy(strategy_type: str, symbol: str) -> TradingStrategy:
    """
    Factory function to get a trading strategy
    
    Args:
        strategy_type (str): Type of strategy ('simple_moving_average', 'rsi', 'adaptive', 'combined', 'arima')
        symbol (str): Trading symbol
    
    Returns:
        TradingStrategy: Trading strategy instance
    """
    if strategy_type.lower() == 'arima':
        from arima_strategy import ARIMAStrategy
        return ARIMAStrategy(symbol)
    elif strategy_type.lower() == 'simple_moving_average':
        return SimpleMovingAverageStrategy(symbol)
    elif strategy_type.lower() == 'rsi':
        return RSIStrategy(symbol)
    elif strategy_type.lower() == 'adaptive':
        return AdaptiveStrategy(symbol)
    else:
        logger.warning(f"Unknown strategy type '{strategy_type}', using adaptive strategy")
        return AdaptiveStrategy(symbol)
