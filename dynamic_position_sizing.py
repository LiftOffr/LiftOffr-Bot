"""
Dynamic position sizing implementation for Kraken Trading Bot

This file provides the necessary functions and implementation details for 
adding dynamic position sizing based on signal strength to the trading bot.
"""

import logging
from config import (
    ENABLE_DYNAMIC_POSITION_SIZING,
    BASE_MARGIN_PERCENT,
    MAX_MARGIN_PERCENT,
    STRONG_SIGNAL_THRESHOLD,
    VERY_STRONG_SIGNAL_THRESHOLD
)

logger = logging.getLogger(__name__)

def calculate_dynamic_margin_percent(signal_strength=0.0, base_margin_percent=0.2):
    """
    Calculate dynamic margin percentage based on signal strength
    
    Args:
        signal_strength (float): Strength of the signal (0.0 to 1.0)
        base_margin_percent (float): The base margin percentage to use (default from config)
        
    Returns:
        float: Calculated margin percentage
    """
    if not ENABLE_DYNAMIC_POSITION_SIZING or signal_strength <= 0.0:
        return base_margin_percent
        
    # Use base margin percentage as the minimum
    margin_percent = BASE_MARGIN_PERCENT
    
    # For strong signals, increase position size
    if signal_strength >= VERY_STRONG_SIGNAL_THRESHOLD:
        # For very strong signals (>=0.90), use the maximum margin percentage
        margin_percent = MAX_MARGIN_PERCENT
        logger.info(f"Using maximum margin ({MAX_MARGIN_PERCENT*100:.1f}%) for very strong signal (strength: {signal_strength:.2f})")
    elif signal_strength >= STRONG_SIGNAL_THRESHOLD:
        # For strong signals (>=0.80), scale between base and max based on strength
        # Calculate how far above the strong threshold we are
        strength_above_threshold = signal_strength - STRONG_SIGNAL_THRESHOLD
        # Scale from base to max based on where we are between STRONG and VERY_STRONG thresholds
        scale_factor = strength_above_threshold / (VERY_STRONG_SIGNAL_THRESHOLD - STRONG_SIGNAL_THRESHOLD)
        # Interpolate between base and max margin percentages
        margin_percent = BASE_MARGIN_PERCENT + scale_factor * (MAX_MARGIN_PERCENT - BASE_MARGIN_PERCENT)
        logger.info(f"Using scaled margin ({margin_percent*100:.1f}%) for strong signal (strength: {signal_strength:.2f})")
        
    return margin_percent

def update_bot_manager_implementation():
    """
    Instructions for updating the BotManager implementation for dynamic position sizing
    
    1. Update track_position_change method in BotManager class to accept signal_strength parameter:
       
       def track_position_change(self, bot_id: str, new_position: str, previous_position: str, 
                                margin_percent: float, funds_to_allocate: float = 0.0, 
                                signal_strength: float = 0.0):
       
    2. Update all track_position_change calls in kraken_trading_bot.py to include signal_strength
       
    3. Update execute_buy and execute_sell to use dynamic margin percentage
    """
    pass