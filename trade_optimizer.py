#!/usr/bin/env python3
"""
Trade Optimizer for ML Trading System

This module integrates with the ML trading system to optimize 
trade entries/exits and maximize profitability.
"""
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
OPTIMIZATION_FILE = f"{DATA_DIR}/trade_optimization.json"

# Market state definitions
MARKET_STATES = ['trending_up', 'trending_down', 'ranging', 'volatile', 'normal']

class TradeOptimizer:
    """
    Trade Optimizer that analyzes historical trade data, market conditions, and volatility patterns
    to determine optimal entry/exit times, adjust position sizing, implement dynamic trailing stops,
    and avoid trades during suboptimal market conditions.
    """
    def __init__(self, trading_pairs: List[str], sandbox: bool = True):
        """
        Initialize the trade optimizer
        
        Args:
            trading_pairs: List of trading pairs to optimize
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs
        self.sandbox = sandbox
        
        # Create data directory if needed
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize optimization data
        self.optimization_data = self._load_optimization_data()
        
        # Track current prices
        self.current_prices = {}
        
        # Load historical trades
        self.trades = self._load_trades()
        
        # Initialize current market states
        self.market_states = {}
        for pair in trading_pairs:
            self.market_states[pair] = self.optimization_data.get('market_states', {}).get(pair, 'normal')
        
        # Track volatility regimes
        self.volatility_data = {}
        for pair in trading_pairs:
            self.volatility_data[pair] = self.optimization_data.get('volatility_data', {}).get(pair, {
                'current': 0.0,
                'historical': [],
                'regime': 'normal'
            })
        
        logger.info(f"Initialized trade optimizer for {len(trading_pairs)} pairs")
    
    def _load_optimization_data(self) -> Dict[str, Any]:
        """
        Load optimization data from file
        
        Returns:
            Optimization data dictionary
        """
        default_data = {
            'market_states': {},
            'optimal_hours': {},
            'volatility_data': {},
            'pair_specific_params': {},
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            if os.path.exists(OPTIMIZATION_FILE):
                with open(OPTIMIZATION_FILE, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded optimization data from {OPTIMIZATION_FILE}")
                return data
        except Exception as e:
            logger.error(f"Error loading optimization data: {e}")
        
        return default_data
    
    def _save_optimization_data(self):
        """Save optimization data to file"""
        try:
            # Update last updated timestamp
            self.optimization_data['last_updated'] = datetime.now().isoformat()
            
            # Save optimization data
            with open(OPTIMIZATION_FILE, 'w') as f:
                json.dump(self.optimization_data, f, indent=2)
            
            logger.info(f"Saved optimization data to {OPTIMIZATION_FILE}")
        except Exception as e:
            logger.error(f"Error saving optimization data: {e}")
    
    def _load_trades(self) -> List[Dict[str, Any]]:
        """
        Load trades from file
        
        Returns:
            List of trades
        """
        try:
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
                logger.info(f"Loaded {len(trades)} trades from {TRADES_FILE}")
                return trades
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
        
        return []
    
    def _load_positions(self) -> Dict[str, Any]:
        """
        Load positions from file
        
        Returns:
            Dictionary of positions
        """
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    positions_data = json.load(f)
                
                # Check if positions is a list or dictionary
                if isinstance(positions_data, list):
                    # Convert list of positions to dictionary by pair
                    positions_dict = {}
                    for position in positions_data:
                        pair = position.get('pair')
                        if pair:
                            positions_dict[pair] = position
                    return positions_dict
                elif isinstance(positions_data, dict):
                    return positions_data
                else:
                    logger.error(f"Unexpected positions data type: {type(positions_data)}")
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
        
        return {}
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """
        Load portfolio from file
        
        Returns:
            Portfolio dictionary
        """
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    portfolio = json.load(f)
                return portfolio
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
        
        return {'initial_capital': 20000.0, 'current_capital': 20000.0}
    
    def update_market_state(self, pair: str, current_price: float):
        """
        Update market state for a pair
        
        Args:
            pair: Trading pair
            current_price: Current price
        """
        # Update current price
        self.current_prices[pair] = current_price
        
        # Get pair-specific trades
        pair_trades = [t for t in self.trades if t.get('pair') == pair]
        
        # Need several trades to determine market state
        if len(pair_trades) < 5:
            self.market_states[pair] = 'normal'
            return
        
        # Sort trades by timestamp (most recent first)
        pair_trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)
        
        # Get last 5 trades
        recent_trades = pair_trades[:5]
        
        # Check market state from trades
        market_states = [t.get('market_state', 'normal') for t in recent_trades]
        
        # Get most common market state
        market_state_counts = {}
        for state in market_states:
            market_state_counts[state] = market_state_counts.get(state, 0) + 1
        
        most_common_state = max(market_state_counts.items(), key=lambda x: x[1])[0]
        
        # Update market state
        self.market_states[pair] = most_common_state
        
        # Update optimization data
        self.optimization_data['market_states'][pair] = most_common_state
        
        # Calculate volatility
        price_changes = []
        for i in range(1, len(recent_trades)):
            if (recent_trades[i].get('exit_price') and recent_trades[i-1].get('entry_price')):
                try:
                    price_change = abs(
                        (float(recent_trades[i].get('exit_price')) / float(recent_trades[i-1].get('entry_price'))) - 1
                    ) * 100
                    price_changes.append(price_change)
                except (ValueError, ZeroDivisionError):
                    continue
        
        if price_changes:
            current_volatility = sum(price_changes) / len(price_changes)
            
            # Update volatility data
            if pair not in self.volatility_data:
                self.volatility_data[pair] = {
                    'current': current_volatility,
                    'historical': [current_volatility],
                    'regime': 'normal'
                }
            else:
                self.volatility_data[pair]['current'] = current_volatility
                self.volatility_data[pair]['historical'].append(current_volatility)
                
                # Limit historical volatility data
                if len(self.volatility_data[pair]['historical']) > 100:
                    self.volatility_data[pair]['historical'] = self.volatility_data[pair]['historical'][-100:]
                
                # Determine volatility regime
                avg_volatility = sum(self.volatility_data[pair]['historical']) / len(self.volatility_data[pair]['historical'])
                
                if current_volatility > 2 * avg_volatility:
                    volatility_regime = 'high'
                elif current_volatility < 0.5 * avg_volatility:
                    volatility_regime = 'low'
                else:
                    volatility_regime = 'normal'
                
                self.volatility_data[pair]['regime'] = volatility_regime
            
            # Update optimization data
            self.optimization_data['volatility_data'][pair] = self.volatility_data[pair]
        
        logger.debug(f"Updated market state for {pair}: {most_common_state}")
    
    def get_best_entry_time(self, pair: str) -> bool:
        """
        Check if current time is optimal for entry
        
        Args:
            pair: Trading pair
        
        Returns:
            True if current time is optimal for entry, False otherwise
        """
        optimal_hours = self.optimization_data.get('optimal_hours', {}).get(pair, {}).get('entry', list(range(24)))
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Check if current hour is in optimal hours
        return current_hour in optimal_hours
    
    def get_best_exit_time(self, pair: str) -> bool:
        """
        Check if current time is optimal for exit
        
        Args:
            pair: Trading pair
        
        Returns:
            True if current time is optimal for exit, False otherwise
        """
        optimal_hours = self.optimization_data.get('optimal_hours', {}).get(pair, {}).get('exit', list(range(24)))
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Check if current hour is in optimal hours
        return current_hour in optimal_hours
    
    def get_optimal_position_size(self, pair: str, confidence: float, direction: str) -> float:
        """
        Get optimal position size based on portfolio, market state, and confidence
        
        Args:
            pair: Trading pair
            confidence: Confidence level (0-1)
            direction: Trade direction ('long' or 'short')
        
        Returns:
            Optimal position size in USD
        """
        # Load portfolio
        portfolio = self._load_portfolio()
        
        # Determine market state adjustment
        market_state = self.market_states.get(pair, 'normal')
        
        state_adjustments = {
            'trending_up': 1.2 if direction == 'long' else 0.7,
            'trending_down': 0.7 if direction == 'long' else 1.2,
            'ranging': 0.9,
            'volatile': 0.8,
            'normal': 1.0
        }
        
        state_adjustment = state_adjustments.get(market_state, 1.0)
        
        # Determine volatility adjustment
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        
        volatility_adjustments = {
            'high': 0.7,
            'normal': 1.0,
            'low': 1.1
        }
        
        volatility_adjustment = volatility_adjustments.get(volatility_regime, 1.0)
        
        # Determine confidence adjustment (higher confidence = larger position)
        confidence_adjustment = 0.5 + (confidence / 2)  # Range: 0.5 - 1.0
        
        # Calculate base position size (20% of available capital)
        available_capital = portfolio.get('current_capital', 20000.0)
        base_position_size = available_capital * 0.2
        
        # Apply adjustments
        optimal_position_size = base_position_size * state_adjustment * volatility_adjustment * confidence_adjustment
        
        # Ensure position doesn't exceed available capital
        optimal_position_size = min(optimal_position_size, available_capital * 0.5)
        
        logger.debug(f"Optimal position size for {pair}: ${optimal_position_size:.2f}")
        
        return optimal_position_size
    
    def get_optimal_leverage(self, pair: str, confidence: float, direction: str) -> float:
        """
        Get optimal leverage based on market state, volatility, and confidence
        
        Args:
            pair: Trading pair
            confidence: Confidence level (0-1)
            direction: Trade direction ('long' or 'short')
        
        Returns:
            Optimal leverage (5-125)
        """
        # Determine market state adjustment
        market_state = self.market_states.get(pair, 'normal')
        
        state_adjustments = {
            'trending_up': 1.2 if direction == 'long' else 0.7,
            'trending_down': 0.7 if direction == 'long' else 1.2,
            'ranging': 0.8,
            'volatile': 0.6,
            'normal': 1.0
        }
        
        state_adjustment = state_adjustments.get(market_state, 1.0)
        
        # Determine volatility adjustment
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        current_volatility = self.volatility_data.get(pair, {}).get('current', 0.0)
        historical_volatility = self.volatility_data.get(pair, {}).get('historical', [])
        avg_volatility = sum(historical_volatility) / len(historical_volatility) if historical_volatility else 0.0
        
        # More sophisticated volatility adjustment based on current vs historical volatility
        if current_volatility > 0 and avg_volatility > 0:
            volatility_ratio = current_volatility / avg_volatility
            if volatility_ratio > 2.0:
                # High volatility, reduce leverage significantly
                volatility_adjustment = 0.5
            elif volatility_ratio > 1.5:
                # Above average volatility, reduce leverage moderately
                volatility_adjustment = 0.7
            elif volatility_ratio < 0.5:
                # Very low volatility, can increase leverage
                volatility_adjustment = 1.3
            elif volatility_ratio < 0.8:
                # Below average volatility, increase leverage slightly
                volatility_adjustment = 1.1
            else:
                # Normal volatility range
                volatility_adjustment = 1.0
        else:
            # Default adjustments if we don't have enough data
            volatility_adjustments = {
                'high': 0.6,
                'normal': 1.0,
                'low': 1.2
            }
            volatility_adjustment = volatility_adjustments.get(volatility_regime, 1.0)
        
        # Enhanced confidence-based leverage calculation
        # More granular steps for higher precision
        if confidence >= 0.95:  # Extremely high confidence
            base_leverage = 125.0
        elif confidence >= 0.9:  # Very high confidence
            base_leverage = 110.0
        elif confidence >= 0.85:  # High confidence
            base_leverage = 95.0
        elif confidence >= 0.8:  # Good confidence
            base_leverage = 80.0
        elif confidence >= 0.75:  # Moderate confidence
            base_leverage = 65.0
        elif confidence >= 0.7:  # Reasonable confidence
            base_leverage = 50.0
        elif confidence >= 0.65:  # Low confidence
            base_leverage = 35.0
        else:  # Very low confidence
            base_leverage = 20.0
        
        # Apply adjustments
        optimal_leverage = base_leverage * state_adjustment * volatility_adjustment
        
        # Apply non-linear scaling for extreme confidence values
        # This creates a more exponential curve for very high confidence
        if confidence > 0.9:
            confidence_boost = (confidence - 0.9) * 2.0  # Scale the boost
            optimal_leverage *= (1.0 + confidence_boost)
        
        # Ensure leverage is within bounds (5-125)
        optimal_leverage = max(5.0, min(125.0, optimal_leverage))
        
        logger.debug(f"Optimal leverage for {pair}: {optimal_leverage:.2f}x (confidence: {confidence:.2f}, " +
                    f"market: {market_state}, volatility: {volatility_regime})")
        
        return optimal_leverage
    
    def get_optimal_stop_loss(self, pair: str, direction: str, leverage: float) -> float:
        """
        Get optimal stop loss percentage based on market state and volatility
        
        Args:
            pair: Trading pair
            direction: Trade direction ('long' or 'short')
            leverage: Leverage
        
        Returns:
            Optimal stop loss percentage
        """
        # Get pair-specific parameters
        pair_params = self.optimization_data.get('pair_specific_params', {}).get(pair, {})
        base_stop_loss = pair_params.get('optimal_stop_loss', 4.0)
        
        # Determine market state adjustment
        market_state = self.market_states.get(pair, 'normal')
        
        state_adjustments = {
            'trending_up': 1.1 if direction == 'long' else 0.9,
            'trending_down': 0.9 if direction == 'long' else 1.1,
            'ranging': 1.0,
            'volatile': 0.8,
            'normal': 1.0
        }
        
        state_adjustment = state_adjustments.get(market_state, 1.0)
        
        # Determine volatility adjustment
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        
        volatility_adjustments = {
            'high': 0.8,
            'normal': 1.0,
            'low': 1.2
        }
        
        volatility_adjustment = volatility_adjustments.get(volatility_regime, 1.0)
        
        # Calculate optimal stop loss
        optimal_stop_loss = base_stop_loss * state_adjustment * volatility_adjustment
        
        # Adjust for leverage (higher leverage = tighter stop)
        leverage_factor = 1.0 - (leverage / 250.0)  # 125x leverage => 0.5 factor
        optimal_stop_loss *= leverage_factor
        
        # Ensure stop loss doesn't exceed liquidation point
        max_stop_loss = 95.0 / leverage  # 95% of liquidation level
        optimal_stop_loss = min(optimal_stop_loss, max_stop_loss)
        
        # Ensure minimum stop loss to prevent immediate liquidation
        optimal_stop_loss = max(1.0, optimal_stop_loss)
        
        logger.debug(f"Optimal stop loss for {pair}: {optimal_stop_loss:.2f}%")
        
        return optimal_stop_loss
    
    def get_optimal_take_profit(self, pair: str, direction: str) -> float:
        """
        Get optimal take profit percentage based on market state and volatility
        
        Args:
            pair: Trading pair
            direction: Trade direction ('long' or 'short')
        
        Returns:
            Optimal take profit percentage
        """
        # Get pair-specific parameters
        pair_params = self.optimization_data.get('pair_specific_params', {}).get(pair, {})
        base_take_profit = pair_params.get('optimal_take_profit', 15.0)
        
        # Determine market state adjustment
        market_state = self.market_states.get(pair, 'normal')
        
        state_adjustments = {
            'trending_up': 1.2 if direction == 'long' else 0.9,
            'trending_down': 0.9 if direction == 'long' else 1.2,
            'ranging': 0.8,
            'volatile': 1.2,
            'normal': 1.0
        }
        
        state_adjustment = state_adjustments.get(market_state, 1.0)
        
        # Determine volatility adjustment
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        
        volatility_adjustments = {
            'high': 1.3,
            'normal': 1.0,
            'low': 0.8
        }
        
        volatility_adjustment = volatility_adjustments.get(volatility_regime, 1.0)
        
        # Calculate optimal take profit
        optimal_take_profit = base_take_profit * state_adjustment * volatility_adjustment
        
        logger.debug(f"Optimal take profit for {pair}: {optimal_take_profit:.2f}%")
        
        return optimal_take_profit
    
    def calculate_optimal_entry_price(self, pair: str, current_price: float, direction: str, confidence: float) -> float:
        """
        Calculate optimal entry price based on market conditions and confidence
        
        Args:
            pair: Trading pair
            current_price: Current market price
            direction: Trade direction ('long' or 'short')
            confidence: Confidence level (0-1)
            
        Returns:
            Optimal entry price
        """
        # Get market state and volatility information
        market_state = self.market_states.get(pair, 'normal')
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        current_volatility = self.volatility_data.get(pair, {}).get('current', 0.0)
        
        # Default adjustment: no change
        price_adjustment_pct = 0.0
        
        # For low confidence trades, we want to be more conservative with entry
        if confidence < 0.7:
            # For low confidence long, wait for a slight dip (1-3%)
            # For low confidence short, wait for a slight rise (1-3%)
            base_adjustment = 1.5 * (0.7 - confidence) * 10  # Up to 3% for 0.5 confidence
            price_adjustment_pct = base_adjustment if direction == 'long' else -base_adjustment
            
        # For high confidence trades, we can be more aggressive
        elif confidence > 0.85:
            # For high confidence long, we can enter at or slightly above market (0-1%)
            # For high confidence short, we can enter at or slightly below market (0-1%)
            confidence_boost = (confidence - 0.85) * 2.0  # Up to 0.3 for 1.0 confidence
            price_adjustment_pct = -confidence_boost if direction == 'long' else confidence_boost
        
        # Adjust based on market state
        if market_state == 'trending_up':
            # In uptrend, more aggressive on longs, more conservative on shorts
            trend_adj = 0.5 if direction == 'long' else 1.0
            price_adjustment_pct += trend_adj if direction == 'short' else -trend_adj
        elif market_state == 'trending_down':
            # In downtrend, more aggressive on shorts, more conservative on longs
            trend_adj = 0.5 if direction == 'short' else 1.0
            price_adjustment_pct += trend_adj if direction == 'long' else -trend_adj
        elif market_state == 'volatile':
            # In volatile market, more conservative for both directions (wait for better entry)
            vol_adj = 2.0 * (1.0 - confidence)  # More conservative for lower confidence
            price_adjustment_pct += vol_adj if direction == 'long' else -vol_adj
            
        # Adjust based on volatility regime
        if volatility_regime == 'high':
            # In high volatility, be more conservative (expect larger price swings)
            vol_adj = 1.5 * (1.0 - confidence)  # Up to 1.5% for 0.0 confidence
            price_adjustment_pct += vol_adj if direction == 'long' else -vol_adj
        elif volatility_regime == 'low':
            # In low volatility, be less conservative (smaller price swings expected)
            vol_adj = 0.5 * (1.0 - confidence)  # Up to 0.5% for 0.0 confidence
            price_adjustment_pct -= vol_adj if direction == 'long' else -vol_adj
            
        # Limit maximum adjustment to +/-5%
        price_adjustment_pct = max(-5.0, min(5.0, price_adjustment_pct))
        
        # Calculate adjusted price
        price_multiplier = 1.0 + (price_adjustment_pct / 100.0)
        optimal_entry = current_price * price_multiplier
        
        logger.debug(
            f"Optimal entry for {pair} {direction} (confidence: {confidence:.2f}): "
            f"${optimal_entry:.2f} (adj: {price_adjustment_pct:.2f}%, current: ${current_price:.2f})"
        )
        
        return optimal_entry
        
    def get_trailing_stop_params(self, pair: str) -> Tuple[float, float]:
        """
        Get trailing stop parameters (activation level and trailing distance)
        
        Args:
            pair: Trading pair
        
        Returns:
            Tuple of (activation_level, trailing_distance) in percentages
        """
        # Get pair-specific parameters
        pair_params = self.optimization_data.get('pair_specific_params', {}).get(pair, {})
        activation_level = pair_params.get('trailing_stop_activation', 5.0)
        trailing_distance = pair_params.get('trailing_stop_distance', 2.5)
        
        # Determine volatility adjustment
        volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
        
        volatility_adjustments = {
            'high': (0.9, 1.3),  # (activation, distance)
            'normal': (1.0, 1.0),
            'low': (1.1, 0.8)
        }
        
        v_adj = volatility_adjustments.get(volatility_regime, (1.0, 1.0))
        
        # Apply adjustments
        activation_level *= v_adj[0]
        trailing_distance *= v_adj[1]
        
        return (activation_level, trailing_distance)
    
    def analyze_existing_positions(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze existing positions for potential adjustments or exits
        
        Args:
            current_prices: Dictionary of current prices by pair
        
        Returns:
            Dictionary with position adjustments and exit recommendations
        """
        # Load positions
        positions = self._load_positions()
        
        # Initialize result
        result = {
            'adjustments': [],
            'exits': []
        }
        
        # Iterate through positions
        for pair, position in positions.items():
            if pair not in current_prices:
                continue
            
            try:
                direction = position.get('direction', 'long')
                entry_price = float(position.get('entry_price', 0))
                current_price = current_prices[pair]
                leverage = float(position.get('leverage', 50.0))
                
                # Calculate current PnL
                if direction == 'long':
                    pnl_pct = ((current_price / entry_price) - 1) * 100 * leverage
                else:
                    pnl_pct = ((entry_price / current_price) - 1) * 100 * leverage
                
                # Check for exit signals
                market_state = self.market_states.get(pair, 'normal')
                is_good_exit_time = self.get_best_exit_time(pair)
                
                # Criteria for recommending exit
                exit_signals = []
                
                # Check for trend change
                if (direction == 'long' and market_state == 'trending_down') or \
                   (direction == 'short' and market_state == 'trending_up'):
                    exit_signals.append('TREND_CHANGE')
                
                # Check for approaching take profit
                optimal_tp = self.get_optimal_take_profit(pair, direction)
                if pnl_pct >= optimal_tp * 0.9:
                    exit_signals.append('APPROACHING_TAKE_PROFIT')
                
                # Check for optimal exit time with decent profit
                if is_good_exit_time and pnl_pct > 5.0:
                    exit_signals.append('OPTIMAL_EXIT_TIME')
                
                # Check for high volatility at profit
                volatility_regime = self.volatility_data.get(pair, {}).get('regime', 'normal')
                if volatility_regime == 'high' and pnl_pct > 8.0:
                    exit_signals.append('HIGH_VOLATILITY_PROFIT_TAKING')
                
                # Check for stagnation
                entry_time = position.get('entry_time')
                if entry_time:
                    try:
                        entry_dt = datetime.fromisoformat(entry_time)
                        days_held = (datetime.now() - entry_dt).days
                        
                        # If position held for more than 3 days with minimal profit, consider exit
                        if days_held >= 3 and -2.0 <= pnl_pct <= 5.0:
                            exit_signals.append('STAGNANT_POSITION')
                    except (ValueError, TypeError):
                        pass
                
                # Add to exit recommendations if any signals
                if exit_signals:
                    result['exits'].append({
                        'pair': pair,
                        'current_price': current_price,
                        'current_pnl_pct': pnl_pct,
                        'signals': exit_signals,
                        'confidence': len(exit_signals) / 5.0  # Normalize confidence 0-1
                    })
                
                # Check for position adjustments
                adjustment_signals = []
                
                # Check for stop loss adjustment
                optimal_sl = self.get_optimal_stop_loss(pair, direction, leverage)
                current_sl = float(position.get('stop_loss_pct', 4.0))
                
                if abs(optimal_sl - current_sl) / current_sl > 0.25:  # >25% difference
                    adjustment_signals.append(('ADJUST_STOP_LOSS', optimal_sl))
                
                # Check for take profit adjustment
                optimal_tp = self.get_optimal_take_profit(pair, direction)
                current_tp = float(position.get('take_profit_pct', 15.0))
                
                if abs(optimal_tp - current_tp) / current_tp > 0.25:  # >25% difference
                    adjustment_signals.append(('ADJUST_TAKE_PROFIT', optimal_tp))
                
                # Check for trailing stop adjustment
                activation_level, trailing_distance = self.get_trailing_stop_params(pair)
                current_activation = float(position.get('trailing_stop_activation', 5.0))
                current_distance = float(position.get('trailing_stop_distance', 2.5))
                
                if abs(activation_level - current_activation) / current_activation > 0.25 or \
                   abs(trailing_distance - current_distance) / current_distance > 0.25:
                    adjustment_signals.append((
                        'ADJUST_TRAILING_STOP',
                        {'activation': activation_level, 'distance': trailing_distance}
                    ))
                
                # Add to adjustment recommendations if any signals
                if adjustment_signals:
                    result['adjustments'].append({
                        'pair': pair,
                        'current_price': current_price,
                        'current_pnl_pct': pnl_pct,
                        'signals': adjustment_signals
                    })
            
            except Exception as e:
                logger.error(f"Error analyzing position for {pair}: {e}")
        
        return result
    
    def calculate_portfolio_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal portfolio allocations for trading pairs
        
        Returns:
            Dictionary of allocation percentages by pair (0-1)
        """
        # Get pair performance metrics
        pair_metrics = {}
        
        for pair in self.trading_pairs:
            # Get pair-specific trades
            pair_trades = [t for t in self.trades if t.get('pair') == pair]
            
            # Skip if not enough trades
            if len(pair_trades) < 5:
                pair_metrics[pair] = {
                    'win_rate': 0.5,
                    'avg_pnl': 0.0,
                    'volatility': 0.0,
                    'recent_performance': 0.0
                }
                continue
            
            # Calculate win rate
            wins = sum(1 for t in pair_trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(pair_trades)
            
            # Calculate average PnL
            pnls = [t.get('pnl', 0) for t in pair_trades]
            avg_pnl = sum(pnls) / len(pnls)
            
            # Calculate volatility
            pnls_std = np.std(pnls) if len(pnls) > 1 else 0
            
            # Calculate recent performance (last 5 trades)
            recent_trades = sorted(pair_trades, key=lambda x: x.get('entry_time', ''), reverse=True)[:5]
            recent_pnls = [t.get('pnl', 0) for t in recent_trades]
            recent_performance = sum(recent_pnls) / len(recent_pnls) if recent_pnls else 0
            
            # Store metrics
            pair_metrics[pair] = {
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'volatility': pnls_std,
                'recent_performance': recent_performance
            }
        
        # Calculate allocation scores
        allocation_scores = {}
        total_score = 0
        
        for pair, metrics in pair_metrics.items():
            # Score formula: (2*win_rate + avg_pnl + recent_performance) / volatility
            # Higher win rate, higher PnL, higher recent performance, lower volatility = higher score
            
            # Avoid division by zero
            volatility = max(0.1, metrics.get('volatility', 0.1))
            
            # Calculate score
            score = (
                2.0 * metrics.get('win_rate', 0.5) + 
                0.0001 * metrics.get('avg_pnl', 0) + 
                0.0001 * metrics.get('recent_performance', 0)
            ) / volatility
            
            # Store score
            allocation_scores[pair] = max(0.1, score)  # Ensure minimum score
            total_score += allocation_scores[pair]
        
        # Calculate allocations
        allocations = {}
        
        for pair, score in allocation_scores.items():
            # Allocate based on normalized score
            allocations[pair] = score / total_score
        
        # Rebalance to ensure total <= 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 1.0:
            for pair in allocations:
                allocations[pair] /= total_allocation
        
        return allocations
    
    def run_optimization(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Run optimization for all trading pairs
        
        Args:
            current_prices: Dictionary of current prices by pair
        
        Returns:
            Dictionary with optimization results
        """
        # Update current prices
        self.current_prices.update(current_prices)
        
        # Initialize result
        result = {
            'market_states': {},
            'volatility_data': {},
            'position_optimizations': {},
            'optimal_hours': {},
            'allocations': {}
        }
        
        # Update market states
        for pair, price in current_prices.items():
            if pair in self.trading_pairs:
                self.update_market_state(pair, price)
                result['market_states'][pair] = self.market_states.get(pair, 'normal')
        
        # Analyze existing positions
        position_optimizations = self.analyze_existing_positions(current_prices)
        result['position_optimizations'] = position_optimizations
        
        # Calculate portfolio allocations
        allocations = self.calculate_portfolio_allocations()
        result['allocations'] = allocations
        
        # Add volatility data
        result['volatility_data'] = self.volatility_data
        
        # Add optimal hours
        result['optimal_hours'] = self.optimization_data.get('optimal_hours', {})
        
        # Save optimization data
        self._save_optimization_data()
        
        logger.info(f"Optimization complete for {len(current_prices)} pairs")
        return result