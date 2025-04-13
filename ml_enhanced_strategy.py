#!/usr/bin/env python3
"""
ML-Enhanced Trading Strategy for Kraken Trading Bot

This module extends existing trading strategies with ML predictions
from the ensemble model, creating enhanced versions that combine
traditional technical analysis with machine learning insights.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import pandas as pd

# Import trading strategy base
from trading_strategy import TradingStrategy

# Import ML strategy integrator
from ml_strategy_integrator import MLStrategyIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MLEnhancedStrategy(TradingStrategy):
    """
    ML-Enhanced Trading Strategy
    
    This strategy wraps an existing trading strategy and enhances it with
    ML predictions from the ensemble model, allowing for better decision making
    and risk management.
    """
    
    def __init__(self, trading_pair="SOL/USD", base_strategy=None, 
                timeframe="1h", ml_influence=0.5, confidence_threshold=0.6, **kwargs):
        """
        Initialize the ML-enhanced strategy
        
        Args:
            trading_pair (str): Trading pair to analyze
            base_strategy (TradingStrategy, optional): Base trading strategy to enhance
            timeframe (str): Timeframe for analysis
            ml_influence (float): Weight of ML predictions in final decision (0.0-1.0)
            confidence_threshold (float): Minimum confidence required for ML to influence decision
            **kwargs: Additional parameters passed to base strategy
        """
        # Extract symbol from trading pair
        self.symbol = trading_pair.split('/')[0] if '/' in trading_pair else trading_pair
        
        # Initialize with Trading Strategy properties
        super().__init__(symbol=self.symbol)
        
        # If base strategy is not provided, create a default strategy
        if base_strategy is None:
            from arima_strategy import ARIMAStrategy
            base_strategy = ARIMAStrategy(self.symbol, **kwargs)
            
        # Set up all required properties
        self.trading_pair = trading_pair
        
        # Initialize prices array for TradingStrategy compatibility
        self.prices = []
        
        # Initialize OHLC data structure
        self.ohlc_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        # Store lookback period for compatibility
        self.lookback_period = getattr(base_strategy, 'lookback_period', 30)
        
        # Store base strategy
        self.base_strategy = base_strategy
        
        # Initialize ML integrator
        self.ml_integrator = MLStrategyIntegrator(
            trading_pair=trading_pair,
            timeframe=timeframe,
            influence_weight=ml_influence,
            confidence_threshold=confidence_threshold
        )
        
        # Set strategy name and description
        self.name = f"ML-Enhanced {getattr(self.base_strategy, 'name', 'Strategy')}"
        self.description = f"Machine Learning Enhanced {getattr(self.base_strategy, 'description', 'Trading Strategy')}"
        
        # Initialize position and trailing stop attributes
        self.position = None
        self.entry_price = None
        self.trailing_stop = None
        
        # Track ML influence on decisions
        self.ml_influenced_decisions = 0
        self.total_decisions = 0
        
        # Last ML prediction details
        self.last_ml_prediction = None
        self.last_market_regime = None
        
        logger.info(f"Initialized ML-Enhanced strategy with base strategy: {getattr(self.base_strategy, 'name', 'Unknown')}")
        logger.info(f"ML influence: {ml_influence}, confidence threshold: {confidence_threshold}")
    
    def update_ohlc(self, open_price, high_price, low_price, close_price):
        """
        Update OHLC data for the strategy
        
        Args:
            open_price (float): Open price
            high_price (float): High price
            low_price (float): Low price
            close_price (float): Close price
        """
        # Update base strategy
        self.base_strategy.update_ohlc(open_price, high_price, low_price, close_price)
        
        # Copy relevant data
        self.ohlc_data = self.base_strategy.ohlc_data
        
        # Update prices array for TradingStrategy compatibility
        self.prices.append(close_price)
        # Keep only the most recent lookback_period prices
        if len(self.prices) > self.lookback_period:
            self.prices = self.prices[-self.lookback_period:]
    
    def update_position(self, position: Optional[str], entry_price: Optional[float] = None):
        """
        Update current position status
        
        Args:
            position (str, optional): Current position ("long", "short", or None)
            entry_price (float, optional): Price at which position was entered
        """
        # Update base strategy
        self.base_strategy.update_position(position, entry_price)
        
        # Copy position data
        self.position = self.base_strategy.position
        self.entry_price = self.base_strategy.entry_price
        self.trailing_stop = self.base_strategy.trailing_stop
    
    def prepare_market_data(self) -> pd.DataFrame:
        """
        Prepare market data for ML prediction
        
        Returns:
            pd.DataFrame: Market data with indicators
        """
        try:
            # Check if OHLC data is available
            if not self.ohlc_data or 'close' not in self.ohlc_data or not self.ohlc_data['close']:
                logger.warning("OHLC data is missing or incomplete")
                return pd.DataFrame()  # Return empty DataFrame
                
            # Get the length of available data
            data_length = len(self.ohlc_data['close'])
            
            # Ensure all required keys exist with proper lengths
            required_keys = ['open', 'high', 'low', 'close']
            for key in required_keys:
                if key not in self.ohlc_data or len(self.ohlc_data[key]) != data_length:
                    logger.warning(f"OHLC data for {key} is missing or has wrong length")
                    return pd.DataFrame()  # Return empty DataFrame
            
            # Create a DataFrame from OHLC data
            data = {
                'timestamp': [datetime.now() for _ in range(data_length)],
                'open': self.ohlc_data['open'],
                'high': self.ohlc_data['high'],
                'low': self.ohlc_data['low'],
                'close': self.ohlc_data['close'],
                'volume': self.ohlc_data.get('volume', [0] * data_length)
            }
            
            df = pd.DataFrame(data)
            
            # Add indicators that are available in the base strategy
            for key, values in self.ohlc_data.items():
                if key not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']:
                    # Make sure the length matches
                    if len(values) == data_length:
                        df[key] = values
                    else:
                        logger.warning(f"Indicator {key} has inconsistent length, skipping")
            
            # Calculate additional indicators if needed
            if data_length >= 20:
                # Add SMA20 if not already present
                if 'sma20' not in df.columns:
                    df['sma20'] = df['close'].rolling(window=20).mean()
                
                # Add volatility if not already present
                if 'volatility' not in df.columns:
                    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            logger.info(f"Prepared market data with shape {df.shape} and columns {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()  # Return empty DataFrame in case of error
    
    def check_entry_signal(self, current_price):
        """
        Check for entry signals combining base strategy and ML predictions
        
        Args:
            current_price (float): Current market price
            
        Returns:
            tuple: (buy_signal, sell_signal, entry_price)
        """
        # Get base strategy signal
        buy_signal, sell_signal, entry_price = self.base_strategy.check_entry_signal(current_price)
        
        # Convert to unified signal format
        if buy_signal:
            base_signal = "buy"
            base_strength = 0.7  # Default strength for base strategy
        elif sell_signal:
            base_signal = "sell"
            base_strength = 0.7
        else:
            base_signal = "neutral"
            base_strength = 0.3
        
        # Prepare market data for ML prediction
        market_data = self.prepare_market_data()
        
        # Get enhanced signal from ML integrator
        integrated_signal, integrated_strength, details = self.ml_integrator.integrate_with_strategy_signal(
            strategy_signal=base_signal,
            strategy_strength=base_strength,
            market_data=market_data
        )
        
        # Store last ML prediction details
        self.last_ml_prediction = details
        
        # Check if ML influenced the decision
        ml_influenced = details['ml_influence'] > 0.0
        if ml_influenced:
            self.ml_influenced_decisions += 1
        self.total_decisions += 1
        
        # Convert integrated signal back to strategy format
        enhanced_buy_signal = integrated_signal == "buy"
        enhanced_sell_signal = integrated_signal == "sell"
        
        # Adjust entry price if needed
        enhanced_entry_price = entry_price
        
        # Log the decision
        if enhanced_buy_signal:
            logger.info(f"ML-Enhanced BUY signal (base: {base_signal}, strength: {integrated_strength:.2f})")
        elif enhanced_sell_signal:
            logger.info(f"ML-Enhanced SELL signal (base: {base_signal}, strength: {integrated_strength:.2f})")
        else:
            logger.info(f"ML-Enhanced NEUTRAL signal (base: {base_signal}, strength: {integrated_strength:.2f})")
        
        return enhanced_buy_signal, enhanced_sell_signal, enhanced_entry_price
    
    def check_exit_signal(self, current_price):
        """
        Check for exit signals combining base strategy and ML predictions
        
        Args:
            current_price (float): Current market price
            
        Returns:
            bool: Whether to exit position
        """
        # Get base strategy exit signal
        base_exit = self.base_strategy.check_exit_signal(current_price)
        
        # If no position, no need to check
        if not self.position:
            return base_exit
        
        # Prepare market data for ML prediction
        market_data = self.prepare_market_data()
        
        # Analyze market regime
        regime, confidence, regime_details = self.ml_integrator.analyze_market_regime(market_data)
        self.last_market_regime = {
            'regime': regime,
            'confidence': confidence,
            'details': regime_details
        }
        
        # Get ML prediction
        ml_prediction, ml_confidence, ml_details = self.ml_integrator.get_ml_prediction(market_data)
        
        # ML can accelerate exit but not prevent it
        # If base strategy says exit, we exit
        if base_exit:
            return True
        
        # Enhanced exit logic based on ML prediction
        # If ML strongly contradicts current position with high confidence,
        # consider early exit
        if ml_confidence >= self.ml_integrator.confidence_threshold:
            if self.position == "long" and ml_prediction < -0.5:
                logger.info(f"ML-Enhanced early exit from LONG position (prediction: {ml_prediction:.2f}, confidence: {ml_confidence:.2f})")
                return True
            elif self.position == "short" and ml_prediction > 0.5:
                logger.info(f"ML-Enhanced early exit from SHORT position (prediction: {ml_prediction:.2f}, confidence: {ml_confidence:.2f})")
                return True
        
        return False
    
    def get_trailing_stop_price(self):
        """
        Get current trailing stop price, potentially enhanced by ML
        
        Returns:
            float: Enhanced trailing stop price
        """
        # Get base trailing stop
        base_stop = self.base_strategy.get_trailing_stop_price()
        
        # If no position, use base stop
        if not self.position or not self.entry_price:
            return base_stop
        
        # Prepare market data for ML recommendation
        market_data = self.prepare_market_data()
        
        # Get ML-enhanced stop loss
        enhanced_stop, adjustment_factor, details = self.ml_integrator.recommend_stop_loss(
            base_stop_loss=base_stop,
            entry_price=self.entry_price,
            position_type=self.position,
            market_data=market_data
        )
        
        logger.info(f"ML-Enhanced trailing stop: {enhanced_stop:.2f} (base: {base_stop:.2f}, factor: {adjustment_factor:.2f})")
        
        return enhanced_stop
    
    def calculate_position_size(self, available_funds, current_price):
        """
        Calculate position size potentially enhanced by ML
        
        Args:
            available_funds (float): Available funds for position
            current_price (float): Current market price
            
        Returns:
            float: Calculated position size
        """
        # Get base position size
        base_size = self.base_strategy.calculate_position_size(available_funds, current_price)
        
        # Prepare market data for ML adjustment
        market_data = self.prepare_market_data()
        
        # Get ML-adjusted position size
        adjusted_size, adjustment_factor = self.ml_integrator.adjust_position_size(
            base_position_size=base_size,
            market_data=market_data
        )
        
        logger.info(f"ML-Enhanced position size: {adjusted_size:.2f} (base: {base_size:.2f}, factor: {adjustment_factor:.2f})")
        
        return adjusted_size
    
    def should_buy(self) -> bool:
        """
        Determine if a buy signal is present, enhanced by ML predictions
        
        Returns:
            bool: True if should buy, False otherwise
        """
        # Get base strategy decision
        base_buy = self.base_strategy.should_buy()
        
        # If not enough data for ML, use base strategy
        if len(self.prices) < 30:
            return base_buy
            
        # Prepare market data
        market_data = self.prepare_market_data()
        
        # Get ML prediction
        ml_prediction, ml_confidence, _ = self.ml_integrator.get_ml_prediction(market_data)
        
        # If ML confidence is high and it suggests buying
        if ml_confidence >= self.ml_integrator.confidence_threshold and ml_prediction > 0.5:
            # ML strongly suggests buying
            enhancement_factor = self.ml_integrator.influence_weight
            # If base is already positive or ML is very confident, buy
            if base_buy or ml_prediction > 0.8:
                logger.info(f"ML-enhanced BUY signal (base: {'BUY' if base_buy else 'NEUTRAL'}, ML prediction: {ml_prediction:.2f}, confidence: {ml_confidence:.2f})")
                return True
        
        # Default to base strategy if no ML enhancement
        return base_buy
    
    def should_sell(self) -> bool:
        """
        Determine if a sell signal is present, enhanced by ML predictions
        
        Returns:
            bool: True if should sell, False otherwise
        """
        # Get base strategy decision
        base_sell = self.base_strategy.should_sell()
        
        # If not enough data for ML, use base strategy
        if len(self.prices) < 30:
            return base_sell
            
        # Prepare market data
        market_data = self.prepare_market_data()
        
        # Get ML prediction
        ml_prediction, ml_confidence, _ = self.ml_integrator.get_ml_prediction(market_data)
        
        # If ML confidence is high and it suggests selling
        if ml_confidence >= self.ml_integrator.confidence_threshold and ml_prediction < -0.5:
            # ML strongly suggests selling
            enhancement_factor = self.ml_integrator.influence_weight
            # If base is already negative or ML is very confident, sell
            if base_sell or ml_prediction < -0.8:
                logger.info(f"ML-enhanced SELL signal (base: {'SELL' if base_sell else 'NEUTRAL'}, ML prediction: {ml_prediction:.2f}, confidence: {ml_confidence:.2f})")
                return True
        
        # Default to base strategy if no ML enhancement
        return base_sell
    
    def calculate_signals(self, df: pd.DataFrame) -> Tuple[bool, bool, float]:
        """
        Calculate trading signals from DataFrame with indicators, enhanced with ML
        
        Args:
            df (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            Tuple[bool, bool, float]: (buy_signal, sell_signal, atr_value)
        """
        # Get base strategy signals
        base_buy, base_sell, atr_value = self.base_strategy.calculate_signals(df)
        
        # Get ML prediction if data is sufficient
        if len(df) >= 30:
            try:
                # Get ML prediction
                ml_prediction, ml_confidence, _ = self.ml_integrator.get_ml_prediction(df)
                
                # Enhanced buy signal: Base says buy or ML strongly suggests buy with high confidence
                buy_signal = base_buy or (ml_prediction > 0.5 and ml_confidence >= self.ml_integrator.confidence_threshold)
                
                # Enhanced sell signal: Base says sell or ML strongly suggests sell with high confidence
                sell_signal = base_sell or (ml_prediction < -0.5 and ml_confidence >= self.ml_integrator.confidence_threshold)
                
                # Don't allow conflicting signals
                if buy_signal and sell_signal:
                    # In case of conflict, use ML confidence to decide
                    if ml_confidence >= 0.8:
                        buy_signal = ml_prediction > 0
                        sell_signal = ml_prediction < 0
                    else:
                        # Default to base strategy in case of conflict with low ML confidence
                        buy_signal = base_buy
                        sell_signal = base_sell
                
                return buy_signal, sell_signal, atr_value
                
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
        
        # Fallback to base signals
        return base_buy, base_sell, atr_value
    
    def get_status(self) -> Dict:
        """
        Get current status of the ML-enhanced strategy
        
        Returns:
            dict: Status information
        """
        # Get base strategy status
        base_status = self.base_strategy.get_status()
        
        # Add ML-specific information
        ml_info = {
            "ml_influence": self.ml_integrator.influence_weight,
            "confidence_threshold": self.ml_integrator.confidence_threshold,
            "ml_influenced_decisions": self.ml_influenced_decisions,
            "total_decisions": self.total_decisions,
            "ml_influence_rate": self.ml_influenced_decisions / max(1, self.total_decisions),
            "last_prediction": self.last_ml_prediction,
            "last_market_regime": self.last_market_regime
        }
        
        # Get ML integrator status
        ml_integrator_status = self.ml_integrator.get_status()
        
        # Combine statuses
        enhanced_status = {
            **base_status,
            "ml_enhanced": True,
            "ml_info": ml_info,
            "ml_integrator": ml_integrator_status
        }
        
        return enhanced_status

def enhance_strategy(strategy: TradingStrategy, trading_pair="SOL/USD", 
                    timeframe="1h", ml_influence=0.5, confidence_threshold=0.6) -> MLEnhancedStrategy:
    """
    Enhance an existing trading strategy with ML capabilities
    
    Args:
        strategy (TradingStrategy): Strategy to enhance
        trading_pair (str): Trading pair
        timeframe (str): Timeframe
        ml_influence (float): Weight of ML in decision making
        confidence_threshold (float): Confidence threshold
        
    Returns:
        MLEnhancedStrategy: Enhanced strategy
    """
    enhanced = MLEnhancedStrategy(
        base_strategy=strategy,
        trading_pair=trading_pair,
        timeframe=timeframe,
        ml_influence=ml_influence,
        confidence_threshold=confidence_threshold
    )
    
    return enhanced

def main():
    """Test ML-enhanced strategy"""
    # Import strategies for testing
    from arima_strategy import ARIMAStrategy
    
    # Create base strategy
    base_strategy = ARIMAStrategy(symbol="SOLUSD")
    
    # Enhance strategy with ML
    enhanced_strategy = enhance_strategy(
        strategy=base_strategy,
        trading_pair="SOL/USD",
        ml_influence=0.5,
        confidence_threshold=0.6
    )
    
    # Print information
    logger.info(f"Created ML-Enhanced strategy: {enhanced_strategy.name}")
    
    # Create some dummy OHLC data
    ohlc_data = {
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    
    # Update strategy
    for i in range(len(ohlc_data['close'])):
        enhanced_strategy.update_ohlc(
            ohlc_data['open'][i],
            ohlc_data['high'][i],
            ohlc_data['low'][i],
            ohlc_data['close'][i]
        )
    
    # Check entry signal
    buy_signal, sell_signal, entry_price = enhanced_strategy.check_entry_signal(106)
    
    logger.info(f"Entry signal: buy={buy_signal}, sell={sell_signal}, price={entry_price}")
    
    # Check status
    status = enhanced_strategy.get_status()
    logger.info(f"Status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    main()