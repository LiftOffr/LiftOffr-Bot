import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import traceback
from typing import Optional, Tuple, List, Dict, Any
from trading_strategy import TradingStrategy

# Try to import ARIMA, but provide fallback if not available
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    # Simple fallback implementation will be used

# Configure logging
logger = logging.getLogger(__name__)

def calculate_atr(high_values, low_values, close_values, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        high_values (numpy.ndarray): Array of high prices
        low_values (numpy.ndarray): Array of low prices
        close_values (numpy.ndarray): Array of close prices
        period (int): Period for ATR calculation
        
    Returns:
        numpy.ndarray: ATR values
    """
    if len(high_values) < period + 1:
        # Not enough data for ATR calculation
        return np.zeros(len(high_values))
        
    # Calculate true range
    tr1 = np.abs(high_values[1:] - low_values[1:])
    tr2 = np.abs(high_values[1:] - close_values[:-1])
    tr3 = np.abs(low_values[1:] - close_values[:-1])
    
    # True range is the max of the three
    tr = np.vstack((tr1, tr2, tr3)).max(axis=0)
    
    # Calculate ATR using simple moving average
    atr = np.zeros(len(high_values))
    atr[period] = np.mean(tr[:period])
    
    # Calculate smoothed ATR
    for i in range(period + 1, len(atr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
        
    return atr

class ARIMAStrategy(TradingStrategy):
    """
    Trading strategy based on ARIMA forecasting
    
    This strategy implements the approach described in the TradingView Pine Script:
    - Uses ARIMA forecast to predict future price movements
    - Generates buy signals when the forecast is above current price
    - Generates sell signals when the forecast is below current price
    - Applies risk management checks including ATR buffer
    """
    
    def __init__(self, 
                 symbol: str = "SOLUSD",
                 lookback_period=32,
                 atr_trailing_multiplier=2.0,
                 entry_atr_multiplier=0.01,
                 leverage=30,
                 risk_buffer_multiplier=1.25,
                 arima_order=(1,1,0),
                 max_loss_percent=4.0):
        """
        Initialize the ARIMA strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'SOLUSD')
            lookback_period (int): Number of periods for lookback (approx 1 day of 45-min candles)
            atr_trailing_multiplier (float): Multiplier for ATR trailing stop
            entry_atr_multiplier (float): Multiplier for ATR entry price offset
            leverage (int): Leverage for trading
            risk_buffer_multiplier (float): ATR buffer multiplier for risk management
            arima_order (tuple): ARIMA model parameters (p,d,q)
            max_loss_percent (float): Maximum allowed loss as percentage (4.0 = 4%)
        """
        # Call the parent class constructor to initialize common fields
        super().__init__(symbol)
        
        # ARIMA strategy specific parameters
        self.lookback_period = lookback_period
        self.atr_trailing_multiplier = atr_trailing_multiplier
        self.entry_atr_multiplier = entry_atr_multiplier
        self.leverage = leverage
        self.risk_buffer_multiplier = risk_buffer_multiplier
        self.arima_order = arima_order
        self.max_loss_percent = max_loss_percent
        
        # Additional data storage specific to ARIMA strategy
        self.ohlc_data = []
        
        # Forecast values
        self.arima_forecast = None
        self.forecast_direction = None
        
        # Technical indicators (some are already in the parent class)
        self.ema9 = None
        self.ema21 = None
        self.ema50 = None  # Added for trend detection
        self.ema100 = None # Added for trend detection
        self.rsi = None
        self.macd = None
        self.macd_signal = None
        self.adx = None
        self.market_trend = "neutral"  # Overall market trend
        
        logger.info(f"ARIMA Strategy initialized with lookback={lookback_period}, trailing_mult={atr_trailing_multiplier}, symbol={symbol}")
    
    def update_ohlc(self, open_price, high_price, low_price, close_price):
        """
        Update OHLC data and indicators
        
        Args:
            open_price (float): Open price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
        """
        # Add new price data
        self.ohlc_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        
        # Keep only the most recent lookback_period + extra buffer
        max_data_points = self.lookback_period * 2
        if len(self.ohlc_data) > max_data_points:
            self.ohlc_data = self.ohlc_data[-max_data_points:]
        
        # Extract price series
        self.prices = [d['close'] for d in self.ohlc_data]
        
        # Update indicators if we have enough data
        if len(self.prices) >= self.lookback_period:
            self._update_indicators()
            self._calculate_arima_forecast()
    
    def update_position(self, position: Optional[str], entry_price: Optional[float] = None):
        """
        Update current position status
        
        Args:
            position (str, optional): Current position ("long", "short", or None)
            entry_price (float, optional): Price at which position was entered
        """
        # Call parent method to maintain common functionality
        super().update_position(position, entry_price)
        
        # Additional ARIMA-specific updates
        # Reset price tracking for trailing stop
        if position == 'long':
            self.highest_price = entry_price
            self.lowest_price = None
        elif position == 'short':
            self.lowest_price = entry_price
            self.highest_price = None
        else:
            self.highest_price = None
            self.lowest_price = None
    
    def _update_indicators(self):
        """
        Update technical indicators based on current price data
        """
        df = pd.DataFrame(self.ohlc_data)
        
        # Calculate EMA indicators
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()  # Added for trend detection
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()  # Added for trend detection
        
        # Calculate RSI (14 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (12, 26, 9)
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate ADX (14 period)
        # Simplified calculation - in production would use TA-Lib
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr14'] = df['tr'].rolling(window=14).mean()
        
        # Simplified ADX calculation
        df['adx'] = 25.0  # Fixed value for simplicity
        
        # Store the most recent indicator values
        if len(df) > 0:
            latest = df.iloc[-1]
            self.ema9 = latest['ema9']
            self.ema21 = latest['ema21']
            self.ema50 = latest['ema50'] if not pd.isna(latest['ema50']) else None
            self.ema100 = latest['ema100'] if not pd.isna(latest['ema100']) else None
            self.rsi = latest['rsi']
            self.macd = latest['macd']
            self.macd_signal = latest['macd_signal']
            self.adx = latest['adx']
            self.atr = latest['atr14']
            
            # Detect market trend using EMAs
            if self.ema50 is not None and self.ema100 is not None:
                if self.ema50 > self.ema100 and self.ema9 > self.ema21:
                    self.market_trend = "bullish"
                elif self.ema50 < self.ema100 and self.ema9 < self.ema21:
                    self.market_trend = "bearish"
                else:
                    self.market_trend = "neutral"
                
                logger.info(f"Market Trend: {self.market_trend.upper()} | EMA50 vs EMA100: {self.ema50:.2f} vs {self.ema100:.2f}")
            else:
                # Default to short-term trend if long-term EMAs not available
                if self.ema9 > self.ema21:
                    self.market_trend = "bullish"
                elif self.ema9 < self.ema21:
                    self.market_trend = "bearish"
                else:
                    self.market_trend = "neutral"
    
    def _calculate_arima_forecast(self):
        """
        Calculate ARIMA forecast based on price history
        """
        try:
            if len(self.prices) < self.lookback_period:
                logger.warning(f"Not enough data for ARIMA forecast. Need {self.lookback_period}, have {len(self.prices)}")
                return
            
            # Use recent price data for forecast
            recent_prices = self.prices[-self.lookback_period:]
            
            # Calculate simple linear regression forecast as an alternative to ARIMA
            # This approximates the TradingView Pine Script approach
            x = np.array(range(len(recent_prices)))
            y = np.array(recent_prices)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Forecast next value
            next_x = len(recent_prices)
            linear_forecast = slope * next_x + intercept
            
            # Try ARIMA model if we want more sophisticated forecasting
            try:
                model = ARIMA(recent_prices, order=self.arima_order)
                model_fit = model.fit()
                arima_forecast = model_fit.forecast(steps=1)[0]
                
                # Use ARIMA forecast if available
                self.arima_forecast = arima_forecast
            except Exception as e:
                # Fallback to linear regression if ARIMA fails
                logger.warning(f"ARIMA model failed, using linear regression instead: {e}")
                self.arima_forecast = linear_forecast
            
            # Determine forecast direction
            current_price = self.prices[-1]
            if self.arima_forecast > current_price:
                self.forecast_direction = "bullish"
            elif self.arima_forecast < current_price:
                self.forecast_direction = "bearish"
            else:
                self.forecast_direction = "neutral"
                
            logger.info(f"ARIMA Forecast: {self.forecast_direction.upper()} | Current: ${current_price:.2f} → Target: ${self.arima_forecast:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating ARIMA forecast: {e}")
            self.arima_forecast = None
            self.forecast_direction = "neutral"
    
    def should_buy(self) -> bool:
        """
        Determine if a buy signal is present based on ARIMA forecast
        
        Returns:
            bool: True if should buy, False otherwise
        """
        if not self.prices or self.arima_forecast is None or self.atr is None:
            return False
            
        current_price = self.prices[-1]
        
        # Generate buy signal when forecast is bullish (forecast > current price)
        if self.forecast_direction == "bullish" and self.position is None:
            logger.info(f"ARIMA BUY SIGNAL - Forecast: ${self.arima_forecast:.2f} > Current: ${current_price:.2f}")
            return True
            
        return False
    
    def should_sell(self) -> bool:
        """
        Determine if a sell signal is present based on ARIMA forecast
        
        Returns:
            bool: True if should sell, False otherwise
        """
        if not self.prices or self.arima_forecast is None or self.atr is None:
            return False
            
        current_price = self.prices[-1]
        
        # Generate sell signal when forecast is bearish (forecast < current price) 
        if self.forecast_direction == "bearish" and self.position is None:
            logger.info(f"ARIMA SELL SIGNAL - Forecast: ${self.arima_forecast:.2f} < Current: ${current_price:.2f}")
            return True
            
        # Check if we need to exit an existing position
        if self.check_exit_signal(current_price):
            return True
            
        return False
            
    def check_entry_signal(self, current_price):
        """
        Check if there's an entry signal based on ARIMA forecast and market trend
        
        Args:
            current_price (float): Current market price
            
        Returns:
            tuple: (buy_signal, sell_signal, entry_price)
        """
        # Default values
        buy_signal = False
        sell_signal = False
        entry_price = None
        
        # Skip if we don't have forecast or ATR
        if self.arima_forecast is None or self.atr is None:
            return buy_signal, sell_signal, entry_price
        
        # Calculate entry prices with ATR offset
        long_entry_price = current_price - (self.entry_atr_multiplier * self.atr)
        short_entry_price = current_price + (self.entry_atr_multiplier * self.atr)
        
        # Calculate hypothetical liquidation levels
        long_liquidation = long_entry_price * (1 - (1 / self.leverage))
        short_liquidation = short_entry_price * (1 + (1 / self.leverage))
        
        # Check if there's enough buffer to liquidation level
        long_buffer = long_entry_price - long_liquidation
        short_buffer = short_liquidation - short_entry_price
        min_buffer = self.risk_buffer_multiplier * self.atr
        
        # Generate signals based on ARIMA forecast, market trend and risk checks
        if self.forecast_direction == "bullish" and self.position is None and long_buffer >= min_buffer:
            # Only go long if we're not in a strongly bearish market
            if self.market_trend != "bearish":
                buy_signal = True
                entry_price = long_entry_price
                logger.info(f"ARIMA BUY SIGNAL - Forecast: ${self.arima_forecast:.2f} > Current: ${current_price:.2f}")
                logger.info(f"Entry price: ${entry_price:.2f} | ATR offset: ${self.entry_atr_multiplier * self.atr:.2f} | Market Trend: {self.market_trend.upper()}")
            else:
                logger.info(f"ARIMA BUY SIGNAL SUPPRESSED - Market trend is BEARISH - avoiding counter-trend trade")
        
        elif self.forecast_direction == "bearish" and self.position is None and short_buffer >= min_buffer:
            # Only go short if we're not in a strongly bullish market
            if self.market_trend != "bullish":
                sell_signal = True
                entry_price = short_entry_price
                logger.info(f"ARIMA SELL SIGNAL - Forecast: ${self.arima_forecast:.2f} < Current: ${current_price:.2f}")
                logger.info(f"Entry price: ${entry_price:.2f} | ATR offset: ${self.entry_atr_multiplier * self.atr:.2f} | Market Trend: {self.market_trend.upper()}")
            else:
                logger.info(f"ARIMA SELL SIGNAL SUPPRESSED - Market trend is BULLISH - avoiding counter-trend trade")
        
        return buy_signal, sell_signal, entry_price
    
    def check_exit_signal(self, current_price):
        """
        Check if there's an exit signal based on trailing stop, forecast reversal, or max loss
        
        Args:
            current_price (float): Current market price
            
        Returns:
            bool: Whether to exit position
        """
        if self.position is None or self.atr is None or self.entry_price is None:
            return False
        
        exit_signal = False
        
        # Update highest/lowest price for trailing stop
        if self.position == "long" and (self.highest_price is None or current_price > self.highest_price):
            self.highest_price = current_price
        elif self.position == "short" and (self.lowest_price is None or current_price < self.lowest_price):
            self.lowest_price = current_price
        
        # Check trailing stop
        if self.position == "long" and self.highest_price is not None:
            trailing_stop = self.highest_price - (self.atr_trailing_multiplier * self.atr)
            if current_price <= trailing_stop:
                logger.info(f"TRAILING STOP EXIT - Price: ${current_price:.2f} <= Stop: ${trailing_stop:.2f}")
                exit_signal = True
        
        elif self.position == "short" and self.lowest_price is not None:
            trailing_stop = self.lowest_price + (self.atr_trailing_multiplier * self.atr)
            if current_price >= trailing_stop:
                logger.info(f"TRAILING STOP EXIT - Price: ${current_price:.2f} >= Stop: ${trailing_stop:.2f}")
                exit_signal = True
        
        # Check for forecast reversal
        if self.position == "long" and self.forecast_direction == "bearish":
            logger.info(f"FORECAST REVERSAL EXIT - Direction: {self.forecast_direction.upper()}")
            exit_signal = True
        elif self.position == "short" and self.forecast_direction == "bullish":
            logger.info(f"FORECAST REVERSAL EXIT - Direction: {self.forecast_direction.upper()}")
            exit_signal = True
        
        # Check for maximum percentage loss
        if self.position == "long":
            current_loss_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            if current_loss_pct <= -self.max_loss_percent:
                logger.info(f"MAXIMUM LOSS EXIT - Loss: {current_loss_pct:.2f}% exceeds limit of {self.max_loss_percent:.2f}%")
                exit_signal = True
        elif self.position == "short":
            current_loss_pct = ((self.entry_price - current_price) / self.entry_price) * 100
            if current_loss_pct <= -self.max_loss_percent:
                logger.info(f"MAXIMUM LOSS EXIT - Loss: {current_loss_pct:.2f}% exceeds limit of {self.max_loss_percent:.2f}%")
                exit_signal = True
        
        return exit_signal
    
    def get_trailing_stop_price(self):
        """
        Get current trailing stop price
        
        Returns:
            float: Trailing stop price
        """
        if self.atr is None:
            return None
        
        if self.position == "long" and self.highest_price is not None:
            return self.highest_price - (self.atr_trailing_multiplier * self.atr)
        
        elif self.position == "short" and self.lowest_price is not None:
            return self.lowest_price + (self.atr_trailing_multiplier * self.atr)
        
        return None
    
    def calculate_signals(self, df: pd.DataFrame) -> tuple:
        """
        Calculate trading signals from DataFrame with indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # Make sure all required columns exist
        if 'close' not in df.columns:
            logger.error("DataFrame missing 'close' column")
            return False, False, 0.0
            
        # Calculate ATR if not already present
        if 'ATR' not in df.columns and len(df) > 14:
            df['ATR'] = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        
        # Get last price and ATR
        current_price = df['close'].iloc[-1]
        atr_value = df['ATR'].iloc[-1] if 'ATR' in df.columns and len(df) > 14 else 0.0
        
        # Calculate ARIMA forecast using the same logic as in _calculate_arima_forecast
        try:
            if len(df) < self.lookback_period:
                return False, False, atr_value
                
            # Use recent price data for forecast
            recent_prices = df['close'].iloc[-self.lookback_period:].values
            
            # Linear regression forecast
            x = np.arange(len(recent_prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
            next_x = len(recent_prices)
            arima_forecast = slope * next_x + intercept
            
            # Determine forecast direction and signals
            if arima_forecast > current_price:
                forecast_direction = "bullish"
                buy_signal = True
                sell_signal = False
            elif arima_forecast < current_price:
                forecast_direction = "bearish"
                buy_signal = False
                sell_signal = True
            else:
                forecast_direction = "neutral"
                buy_signal = False
                sell_signal = False
                
            logger.info(f"ARIMA Forecast: {forecast_direction.upper()} | Current: ${current_price:.2f} → Target: ${arima_forecast:.2f}")
            
            return buy_signal, sell_signal, atr_value
            
        except Exception as e:
            logger.error(f"Error in calculate_signals: {e}")
            return False, False, atr_value
    
    def get_status(self):
        """
        Get current strategy status and metrics
        
        Returns:
            dict: Strategy status information
        """
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "forecast_direction": self.forecast_direction,
            "arima_forecast": self.arima_forecast,
            "current_price": self.prices[-1] if self.prices else None,
            "ema9": self.ema9,
            "ema21": self.ema21,
            "rsi": self.rsi,
            "atr": self.atr,
            "trailing_stop": self.get_trailing_stop_price()
        }