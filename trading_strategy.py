import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from utils import calculate_sma, calculate_rsi
from config import (
    SMA_SHORT_PERIOD, SMA_LONG_PERIOD, 
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD
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
        self.in_position = False
        self.position_price = None
    
    def update_price(self, price: float):
        """
        Update price data
        
        Args:
            price (float): Latest price
        """
        self.prices.append(price)
    
    def update_position(self, in_position: bool, position_price: Optional[float] = None):
        """
        Update current position status
        
        Args:
            in_position (bool): Whether currently in position
            position_price (float, optional): Price at which position was entered
        """
        self.in_position = in_position
        self.position_price = position_price
    
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
        
        return is_crossover and not self.in_position
    
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
        
        return is_crossover and self.in_position

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
        
        return is_crossover and not self.in_position
    
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
        
        return is_crossover and self.in_position

class CombinedStrategy(TradingStrategy):
    """
    Combined strategy using both SMA and RSI
    """
    def __init__(self, symbol: str):
        """
        Initialize the combined strategy
        
        Args:
            symbol (str): Trading symbol (e.g. 'XBTUSD')
        """
        super().__init__(symbol)
        self.sma_strategy = SimpleMovingAverageStrategy(symbol)
        self.rsi_strategy = RSIStrategy(symbol)
    
    def update_price(self, price: float):
        """
        Update price data for all strategies
        
        Args:
            price (float): Latest price
        """
        super().update_price(price)
        self.sma_strategy.update_price(price)
        self.rsi_strategy.update_price(price)
    
    def update_position(self, in_position: bool, position_price: Optional[float] = None):
        """
        Update position status for all strategies
        
        Args:
            in_position (bool): Whether currently in position
            position_price (float, optional): Price at which position was entered
        """
        super().update_position(in_position, position_price)
        self.sma_strategy.update_position(in_position, position_price)
        self.rsi_strategy.update_position(in_position, position_price)
    
    def should_buy(self) -> bool:
        """
        Buy when both strategies give buy signals
        
        Returns:
            bool: True if should buy, False otherwise
        """
        return (self.sma_strategy.should_buy() and self.rsi_strategy.should_buy() and 
                not self.in_position)
    
    def should_sell(self) -> bool:
        """
        Sell when either strategy gives sell signal
        
        Returns:
            bool: True if should sell, False otherwise
        """
        return (self.sma_strategy.should_sell() or self.rsi_strategy.should_sell()) and self.in_position

def get_strategy(strategy_type: str, symbol: str) -> TradingStrategy:
    """
    Factory function to get a trading strategy
    
    Args:
        strategy_type (str): Type of strategy ('simple_moving_average', 'rsi', 'combined')
        symbol (str): Trading symbol
    
    Returns:
        TradingStrategy: Trading strategy instance
    """
    if strategy_type.lower() == 'simple_moving_average':
        return SimpleMovingAverageStrategy(symbol)
    elif strategy_type.lower() == 'rsi':
        return RSIStrategy(symbol)
    elif strategy_type.lower() == 'combined':
        return CombinedStrategy(symbol)
    else:
        logger.warning(f"Unknown strategy type '{strategy_type}', using simple_moving_average")
        return SimpleMovingAverageStrategy(symbol)
