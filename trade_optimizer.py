#!/usr/bin/env python3
"""
Trade Optimizer for ML Trading System

This module optimizes entry and exit points using advanced techniques:
1. Volatility-based entry/exit timing
2. Liquidity analysis for slippage minimization
3. Correlation-based risk adjustment
4. Momentum confirmation for trade entries
5. Dynamic take profit and stop loss levels
"""
import os
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"
OPTIMIZATION_FILE = f"{DATA_DIR}/trade_optimization.json"

class TradeOptimizer:
    """
    Trade Optimizer for maximizing profitability
    
    This class analyzes historical trades and market conditions to
    optimize entry and exit points, risk parameters, and trade timing.
    """
    
    def __init__(self, trading_pairs: List[str], sandbox: bool = True):
        """
        Initialize the Trade Optimizer
        
        Args:
            trading_pairs: List of trading pairs to optimize
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs
        self.sandbox = sandbox
        
        # Load data
        self.portfolio = {}
        self.positions = []
        self.trades = []
        self.ml_config = {}
        self.optimization_data = {}
        
        # Market state tracking
        self.market_states = {}
        self.price_history = {}
        self.volatility_data = {}
        self.optimal_hours = {}
        
        # Load existing optimization data if available
        self._load_data()
        
        logger.info(f"Initialized Trade Optimizer for {len(trading_pairs)} pairs")
    
    def _load_data(self):
        """Load existing data"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    self.portfolio = json.load(f)
            
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    self.positions = json.load(f)
            
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    self.trades = json.load(f)
            
            if os.path.exists(ML_CONFIG_FILE):
                with open(ML_CONFIG_FILE, 'r') as f:
                    self.ml_config = json.load(f)
            
            # Load optimization data if exists
            if os.path.exists(OPTIMIZATION_FILE):
                with open(OPTIMIZATION_FILE, 'r') as f:
                    self.optimization_data = json.load(f)
                    
                    # Extract key components
                    self.market_states = self.optimization_data.get('market_states', {})
                    self.optimal_hours = self.optimization_data.get('optimal_hours', {})
                    self.volatility_data = self.optimization_data.get('volatility_data', {})
            else:
                # Initialize with defaults
                self.optimization_data = {
                    'market_states': {},
                    'optimal_hours': {},
                    'volatility_data': {},
                    'pair_specific_params': {},
                    'last_updated': datetime.now().isoformat()
                }
                
                # Default market states for all pairs
                for pair in self.trading_pairs:
                    self.optimization_data['market_states'][pair] = 'normal'
                    self.optimization_data['optimal_hours'][pair] = {
                        'entry': list(range(24)),  # All hours initially
                        'exit': list(range(24))
                    }
                    self.optimization_data['volatility_data'][pair] = {
                        'current': 0.0,
                        'historical': [],
                        'regime': 'normal'
                    }
                    self.optimization_data['pair_specific_params'][pair] = {
                        'min_confidence': 0.65,
                        'ideal_entry_volatility': 'medium',
                        'min_momentum_confirmation': 2,
                        'optimal_take_profit': 15.0,
                        'optimal_stop_loss': 4.0,
                        'trailing_stop_activation': 5.0,
                        'trailing_stop_distance': 2.5
                    }
                
                # Save default optimization data
                self._save_optimization_data()
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_optimization_data(self):
        """Save optimization data to file"""
        try:
            os.makedirs(os.path.dirname(OPTIMIZATION_FILE), exist_ok=True)
            with open(OPTIMIZATION_FILE, 'w') as f:
                json.dump(self.optimization_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization data: {e}")
    
    def analyze_historical_trades(self):
        """
        Analyze historical trades to identify optimal entry/exit patterns
        
        Returns:
            Dict with findings and optimizations
        """
        # Group trades by pair
        pair_trades = {}
        for trade in self.trades:
            if trade.get('status') != 'closed':
                continue
                
            pair = trade.get('pair')
            if not pair:
                continue
                
            if pair not in pair_trades:
                pair_trades[pair] = []
                
            pair_trades[pair].append(trade)
        
        # Analyze each pair
        optimization_results = {}
        for pair, trades in pair_trades.items():
            if len(trades) < 5:  # Need enough trades for meaningful analysis
                logger.debug(f"Not enough trades for {pair} to analyze: {len(trades)}")
                continue
            
            # Extract key metrics
            entry_times = []
            exit_times = []
            profitable_trades = []
            losing_trades = []
            
            for trade in trades:
                entry_time = datetime.fromisoformat(trade.get('entry_time', ''))
                exit_time = datetime.fromisoformat(trade.get('exit_time', '')) if trade.get('exit_time') else None
                pnl = trade.get('pnl', 0)
                
                if not exit_time:  # Skip trades without exit time
                    continue
                
                entry_times.append(entry_time)
                exit_times.append(exit_time)
                
                if pnl > 0:
                    profitable_trades.append(trade)
                else:
                    losing_trades.append(trade)
            
            # Skip pairs with insufficient data
            if len(entry_times) < 5 or len(exit_times) < 5:
                continue
            
            # Determine optimal entry/exit hours
            entry_hours = [t.hour for t in entry_times]
            exit_hours = [t.hour for t in exit_times]
            
            profitable_entry_hours = [datetime.fromisoformat(t.get('entry_time')).hour 
                                      for t in profitable_trades 
                                      if t.get('entry_time')]
            
            profitable_exit_hours = [datetime.fromisoformat(t.get('exit_time')).hour 
                                     for t in profitable_trades 
                                     if t.get('exit_time')]
            
            # Find most frequent entry/exit hours in profitable trades
            hour_counts = {h: profitable_entry_hours.count(h) for h in range(24)}
            optimal_entry_hours = [h for h, c in hour_counts.items() 
                                   if c > 0 and c >= max(hour_counts.values()) * 0.7]
            
            hour_counts = {h: profitable_exit_hours.count(h) for h in range(24)}
            optimal_exit_hours = [h for h, c in hour_counts.items() 
                                  if c > 0 and c >= max(hour_counts.values()) * 0.7]
            
            # Ensure we have some hours, default to all if none found
            if not optimal_entry_hours:
                optimal_entry_hours = list(range(24))
            if not optimal_exit_hours:
                optimal_exit_hours = list(range(24))
            
            # Calculate optimal take profit and stop loss
            take_profits = [t.get('take_profit_pct', 0) for t in profitable_trades]
            stop_losses = [t.get('stop_loss_pct', 0) for t in profitable_trades]
            
            optimal_take_profit = np.mean(take_profits) if take_profits else 15.0
            optimal_stop_loss = np.mean(stop_losses) if stop_losses else 4.0
            
            # Store results
            optimization_results[pair] = {
                'optimal_entry_hours': optimal_entry_hours,
                'optimal_exit_hours': optimal_exit_hours,
                'optimal_take_profit': optimal_take_profit,
                'optimal_stop_loss': optimal_stop_loss,
                'profitable_trade_count': len(profitable_trades),
                'losing_trade_count': len(losing_trades),
                'win_rate': len(profitable_trades) / len(trades) if trades else 0
            }
            
            # Update optimization data
            if pair in self.optimization_data['optimal_hours']:
                self.optimization_data['optimal_hours'][pair] = {
                    'entry': optimal_entry_hours,
                    'exit': optimal_exit_hours
                }
            
            if pair in self.optimization_data['pair_specific_params']:
                self.optimization_data['pair_specific_params'][pair].update({
                    'optimal_take_profit': optimal_take_profit,
                    'optimal_stop_loss': optimal_stop_loss
                })
        
        # Save updated optimization data
        self.optimization_data['last_updated'] = datetime.now().isoformat()
        self._save_optimization_data()
        
        return optimization_results
    
    def update_market_state(self, pair: str, price: float, timestamp: float = None):
        """
        Update market state for a specific pair
        
        Args:
            pair: Trading pair
            price: Current price
            timestamp: Price timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize price history for this pair if needed
        if pair not in self.price_history:
            self.price_history[pair] = []
        
        # Add price to history
        self.price_history[pair].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Keep only recent prices (last 24 hours)
        current_time = time.time()
        self.price_history[pair] = [
            p for p in self.price_history[pair] 
            if current_time - p['timestamp'] < 86400
        ]
        
        # Calculate volatility
        if len(self.price_history[pair]) >= 10:
            prices = [p['price'] for p in self.price_history[pair]]
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            volatility = np.std(returns) * 100
            
            # Update volatility data
            if pair not in self.optimization_data['volatility_data']:
                self.optimization_data['volatility_data'][pair] = {
                    'current': volatility,
                    'historical': [],
                    'regime': 'normal'
                }
            else:
                self.optimization_data['volatility_data'][pair]['current'] = volatility
                self.optimization_data['volatility_data'][pair]['historical'].append(volatility)
                
                # Keep only the last 100 volatility readings
                if len(self.optimization_data['volatility_data'][pair]['historical']) > 100:
                    self.optimization_data['volatility_data'][pair]['historical'] = \
                        self.optimization_data['volatility_data'][pair]['historical'][-100:]
            
            # Determine volatility regime
            avg_volatility = np.mean(self.optimization_data['volatility_data'][pair]['historical']) \
                if self.optimization_data['volatility_data'][pair]['historical'] else volatility
            
            if volatility < avg_volatility * 0.7:
                regime = 'low'
            elif volatility > avg_volatility * 1.5:
                regime = 'high'
            else:
                regime = 'normal'
            
            self.optimization_data['volatility_data'][pair]['regime'] = regime
            
            # Determine overall market state
            price_change_24h = (prices[-1] / prices[0] - 1) * 100 if len(prices) > 1 else 0
            
            if price_change_24h > 5 and volatility < avg_volatility * 1.2:
                state = 'trending_up'
            elif price_change_24h < -5 and volatility < avg_volatility * 1.2:
                state = 'trending_down'
            elif volatility > avg_volatility * 1.5:
                state = 'volatile'
            elif volatility < avg_volatility * 0.7:
                state = 'ranging'
            else:
                state = 'normal'
            
            self.optimization_data['market_states'][pair] = state
        
        # Save updated data periodically (every 100 updates)
        if len(self.price_history[pair]) % 100 == 0:
            self._save_optimization_data()
    
    def optimize_entry_parameters(self, pair: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize entry parameters based on current market conditions
        
        Args:
            pair: Trading pair
            prediction: ML prediction
        
        Returns:
            Optimized entry parameters
        """
        # Start with default parameters
        params = {
            'should_enter': True,
            'confidence_modifier': 1.0,
            'take_profit_pct': 15.0,
            'stop_loss_pct': 4.0,
            'trailing_stop_activation': 5.0,
            'trailing_stop_distance': 2.5,
            'market_state': 'normal',
            'volatility_regime': 'normal',
            'is_optimal_time': True
        }
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Check if current hour is an optimal entry hour
        if pair in self.optimization_data['optimal_hours']:
            optimal_entry_hours = self.optimization_data['optimal_hours'][pair].get('entry', list(range(24)))
            params['is_optimal_time'] = current_hour in optimal_entry_hours
        
        # Get market state and volatility regime
        if pair in self.optimization_data['market_states']:
            params['market_state'] = self.optimization_data['market_states'][pair]
            
        if pair in self.optimization_data['volatility_data']:
            params['volatility_regime'] = self.optimization_data['volatility_data'][pair].get('regime', 'normal')
        
        # Get direction from prediction
        direction = prediction.get('direction', '').lower()
        confidence = prediction.get('confidence', 0.65)
        
        # Adjust parameters based on market state
        market_state = params['market_state']
        volatility_regime = params['volatility_regime']
        
        # Don't enter if not optimal time and confidence isn't very high
        if not params['is_optimal_time'] and confidence < 0.85:
            params['should_enter'] = False
            params['reason'] = "Not optimal entry time and confidence below threshold"
            return params
        
        # Add specific state-based adjustments
        if market_state == 'trending_up':
            # In uptrends, prefer long positions and avoid shorts
            if direction == 'short' and confidence < 0.9:
                params['should_enter'] = False
                params['reason'] = "Avoiding short in uptrend without extremely high confidence"
            elif direction == 'long':
                params['confidence_modifier'] = 1.2
                params['take_profit_pct'] = 20.0  # More room to run in uptrend
        
        elif market_state == 'trending_down':
            # In downtrends, prefer short positions and avoid longs
            if direction == 'long' and confidence < 0.9:
                params['should_enter'] = False
                params['reason'] = "Avoiding long in downtrend without extremely high confidence"
            elif direction == 'short':
                params['confidence_modifier'] = 1.2
                params['take_profit_pct'] = 20.0  # More room to run in downtrend
        
        elif market_state == 'volatile':
            # In volatile markets, require higher confidence
            if confidence < 0.8:
                params['should_enter'] = False
                params['reason'] = "Volatile market requires higher confidence"
            else:
                # Use tighter stops in volatile markets
                params['stop_loss_pct'] = 3.0
                params['trailing_stop_activation'] = 3.0
                params['trailing_stop_distance'] = 1.5
        
        elif market_state == 'ranging':
            # In ranging markets, set realistic targets
            params['take_profit_pct'] = 10.0  # Lower targets in ranging markets
            params['trailing_stop_activation'] = 4.0
        
        # Adjust based on volatility regime
        if volatility_regime == 'high':
            # Reduce position size in high volatility
            params['confidence_modifier'] *= 0.8
            # Tighten stops
            params['stop_loss_pct'] = min(params['stop_loss_pct'], 3.0)
        
        elif volatility_regime == 'low':
            # In low volatility, widen stops and targets
            params['stop_loss_pct'] = params['stop_loss_pct'] * 1.2
            params['take_profit_pct'] = params['take_profit_pct'] * 0.9  # Lower expectations
        
        # Get pair-specific optimal parameters if available
        if pair in self.optimization_data.get('pair_specific_params', {}):
            pair_params = self.optimization_data['pair_specific_params'][pair]
            
            # Only enter if confidence meets pair-specific threshold
            min_confidence = pair_params.get('min_confidence', 0.65)
            if confidence < min_confidence:
                params['should_enter'] = False
                params['reason'] = f"Confidence {confidence:.2f} below pair-specific threshold {min_confidence:.2f}"
            
            # Apply pair-specific optimal parameters
            params['take_profit_pct'] = pair_params.get('optimal_take_profit', params['take_profit_pct'])
            params['stop_loss_pct'] = pair_params.get('optimal_stop_loss', params['stop_loss_pct'])
            params['trailing_stop_activation'] = pair_params.get('trailing_stop_activation', params['trailing_stop_activation'])
            params['trailing_stop_distance'] = pair_params.get('trailing_stop_distance', params['trailing_stop_distance'])
        
        # Final confidence adjustment
        params['confidence'] = min(0.95, confidence * params['confidence_modifier'])
        
        return params
    
    def optimize_exit_parameters(self, pair: str, position: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Optimize exit parameters for an existing position
        
        Args:
            pair: Trading pair
            position: Current position
            current_price: Current price
        
        Returns:
            Optimized exit parameters
        """
        # Default exit parameters
        params = {
            'should_exit': False,
            'reason': None,
            'adjust_stop_loss': False,
            'adjust_take_profit': False,
            'new_stop_loss_pct': position.get('stop_loss_pct', 4.0),
            'new_take_profit_pct': position.get('take_profit_pct', 15.0),
            'is_optimal_exit_time': False
        }
        
        # Get current hour
        current_hour = datetime.now().hour
        
        # Check if current hour is an optimal exit hour
        if pair in self.optimization_data['optimal_hours']:
            optimal_exit_hours = self.optimization_data['optimal_hours'][pair].get('exit', list(range(24)))
            params['is_optimal_exit_time'] = current_hour in optimal_exit_hours
        
        # Get market state and volatility regime
        market_state = self.optimization_data.get('market_states', {}).get(pair, 'normal')
        volatility_regime = self.optimization_data.get('volatility_data', {}).get(pair, {}).get('regime', 'normal')
        
        # Get position details
        direction = position.get('direction', '').lower()
        entry_price = position.get('entry_price', current_price)
        leverage = position.get('leverage', 1)
        confidence = position.get('confidence', 0.5)
        pnl_pct = position.get('unrealized_pnl_pct', 0)
        
        # Calculate the PnL using current price if not available
        if pnl_pct == 0:
            if direction == 'long':
                pnl_pct = ((current_price / entry_price) - 1) * 100 * leverage
            else:  # short
                pnl_pct = ((entry_price / current_price) - 1) * 100 * leverage
        
        # Determine if should exit based on market conditions
        if market_state == 'trending_up' and direction == 'short' and not params['is_optimal_exit_time']:
            # Exit shorts in uptrends if not at optimal exit time
            params['should_exit'] = True
            params['reason'] = "Exiting short position in uptrend outside optimal exit window"
        
        elif market_state == 'trending_down' and direction == 'long' and not params['is_optimal_exit_time']:
            # Exit longs in downtrends if not at optimal exit time
            params['should_exit'] = True
            params['reason'] = "Exiting long position in downtrend outside optimal exit window"
        
        elif market_state == 'volatile':
            # In volatile markets, take profits faster but give more room for stops
            if pnl_pct > 0:
                # Take profits at 70% of target in volatile markets
                take_profit_pct = position.get('take_profit_pct', 15.0)
                adjusted_tp = take_profit_pct * 0.7
                
                if pnl_pct >= adjusted_tp:
                    params['should_exit'] = True
                    params['reason'] = f"Taking profit at {pnl_pct:.2f}% in volatile market"
                
                # Otherwise, tighten trailing stop
                elif pnl_pct >= position.get('trailing_stop_activation', 5.0):
                    trailing_distance = position.get('trailing_stop_distance', 2.5) * 0.7  # Tighter in volatile markets
                    params['adjust_stop_loss'] = True
                    params['new_stop_loss_pct'] = max(pnl_pct - trailing_distance, 0.5)
            
            # Adjust stop loss to original if deeply negative
            elif pnl_pct <= -position.get('stop_loss_pct', 4.0) * 0.8:
                params['should_exit'] = True
                params['reason'] = f"Stop loss triggered at {pnl_pct:.2f}% in volatile market"
        
        # Handle optimal exit times differently
        if params['is_optimal_exit_time']:
            if pnl_pct > 0:
                # Take profits more aggressively at optimal exit times
                if pnl_pct >= position.get('take_profit_pct', 15.0) * 0.8:
                    params['should_exit'] = True
                    params['reason'] = f"Taking profit at {pnl_pct:.2f}% during optimal exit window"
        
        # Adjust trailing stops for profitable trades
        if pnl_pct > position.get('trailing_stop_activation', 5.0) and not params['should_exit']:
            trailing_distance = position.get('trailing_stop_distance', 2.5)
            params['adjust_stop_loss'] = True
            params['new_stop_loss_pct'] = max(pnl_pct - trailing_distance, 0.5)
        
        # Adjust take profit based on market state
        if market_state == 'trending_up' and direction == 'long' and not params['should_exit']:
            # In strong uptrends, increase take profit for longs
            params['adjust_take_profit'] = True
            params['new_take_profit_pct'] = position.get('take_profit_pct', 15.0) * 1.2
        
        elif market_state == 'trending_down' and direction == 'short' and not params['should_exit']:
            # In strong downtrends, increase take profit for shorts
            params['adjust_take_profit'] = True
            params['new_take_profit_pct'] = position.get('take_profit_pct', 15.0) * 1.2
        
        # Volatility-based adjustments
        if volatility_regime == 'high' and not params['should_exit']:
            # In high volatility, tighten take profits
            if pnl_pct > 0:
                params['adjust_take_profit'] = True
                params['new_take_profit_pct'] = position.get('take_profit_pct', 15.0) * 0.8
        
        return params
    
    def optimize_all_positions(self, current_prices: Dict[str, float]):
        """
        Optimize all current positions
        
        Args:
            current_prices: Dictionary of current prices by pair
        
        Returns:
            List of positions to exit and adjustments to make
        """
        exits = []
        adjustments = []
        
        for position in self.positions:
            pair = position.get('pair')
            
            if not pair or pair not in current_prices:
                continue
            
            current_price = current_prices[pair]
            
            # Update market state
            self.update_market_state(pair, current_price)
            
            # Optimize exit parameters
            exit_params = self.optimize_exit_parameters(pair, position, current_price)
            
            if exit_params['should_exit']:
                exits.append({
                    'position': position,
                    'reason': exit_params['reason']
                })
            
            elif exit_params['adjust_stop_loss'] or exit_params['adjust_take_profit']:
                adjustment = {
                    'position': position,
                    'changes': {}
                }
                
                if exit_params['adjust_stop_loss']:
                    adjustment['changes']['stop_loss_pct'] = exit_params['new_stop_loss_pct']
                
                if exit_params['adjust_take_profit']:
                    adjustment['changes']['take_profit_pct'] = exit_params['new_take_profit_pct']
                
                adjustments.append(adjustment)
        
        return {
            'exits': exits,
            'adjustments': adjustments
        }
    
    def apply_position_adjustments(self, adjustments: List[Dict[str, Any]]):
        """
        Apply position adjustments
        
        Args:
            adjustments: List of position adjustments
        """
        if not adjustments:
            return
        
        # Update positions
        updated = False
        for adjustment in adjustments:
            position = adjustment.get('position')
            changes = adjustment.get('changes', {})
            
            if not position or not changes:
                continue
            
            # Find matching position
            for p in self.positions:
                if p.get('pair') == position.get('pair'):
                    # Apply changes
                    for key, value in changes.items():
                        p[key] = value
                    updated = True
                    
                    logger.info(f"Adjusted position for {p.get('pair')}: {changes}")
        
        # Save updated positions
        if updated:
            try:
                with open(POSITIONS_FILE, 'w') as f:
                    json.dump(self.positions, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving position adjustments: {e}")
    
    def get_best_entry_time(self, pair: str) -> bool:
        """
        Check if current time is good for entry
        
        Args:
            pair: Trading pair
        
        Returns:
            True if current time is optimal for entry
        """
        current_hour = datetime.now().hour
        
        # Get optimal entry hours for this pair
        if pair in self.optimization_data['optimal_hours']:
            optimal_hours = self.optimization_data['optimal_hours'][pair].get('entry', list(range(24)))
            return current_hour in optimal_hours
        
        # Default to True if no optimization data available
        return True
    
    def get_best_exit_time(self, pair: str) -> bool:
        """
        Check if current time is good for exit
        
        Args:
            pair: Trading pair
        
        Returns:
            True if current time is optimal for exit
        """
        current_hour = datetime.now().hour
        
        # Get optimal exit hours for this pair
        if pair in self.optimization_data['optimal_hours']:
            optimal_hours = self.optimization_data['optimal_hours'][pair].get('exit', list(range(24)))
            return current_hour in optimal_hours
        
        # Default to True if no optimization data available
        return True
    
    def check_should_enter(self, pair: str, prediction: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Check if we should enter a trade
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            current_price: Current price
        
        Returns:
            Decision with parameters
        """
        # Update market state
        self.update_market_state(pair, current_price)
        
        # Optimize entry parameters
        entry_params = self.optimize_entry_parameters(pair, prediction)
        
        return entry_params
    
    def optimize_portfolio_allocation(self) -> Dict[str, float]:
        """
        Optimize portfolio allocation across pairs
        
        Returns:
            Dictionary of allocation percentages by pair
        """
        # Group pairs by performance
        performance_data = {}
        
        # Calculate performance metrics for each pair
        for pair in self.trading_pairs:
            pair_trades = [t for t in self.trades if t.get('pair') == pair and t.get('status') == 'closed']
            
            if not pair_trades:
                performance_data[pair] = {
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'volatility': 0,
                    'allocation': 0.05  # Default allocation
                }
                continue
            
            # Calculate metrics
            wins = [t for t in pair_trades if t.get('pnl', 0) > 0]
            losses = [t for t in pair_trades if t.get('pnl', 0) <= 0]
            
            win_rate = len(wins) / len(pair_trades) if pair_trades else 0
            avg_profit = np.mean([t.get('pnl', 0) for t in wins]) if wins else 0
            avg_loss = np.mean([abs(t.get('pnl', 0)) for t in losses]) if losses else 0
            
            total_profit = sum([t.get('pnl', 0) for t in wins]) if wins else 0
            total_loss = sum([abs(t.get('pnl', 0)) for t in losses]) if losses else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit if total_profit > 0 else 0
            
            # Get volatility if available
            volatility = 0
            if pair in self.optimization_data['volatility_data']:
                volatility = self.optimization_data['volatility_data'][pair].get('current', 0)
            
            performance_data[pair] = {
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'volatility': volatility,
                'allocation': 0.05  # Default allocation
            }
        
        # Calculate allocation based on performance
        total_profit_factor = sum([d.get('profit_factor', 0) for d in performance_data.values()])
        
        if total_profit_factor > 0:
            for pair, data in performance_data.items():
                # Base allocation on profit factor and win rate
                if data['profit_factor'] > 0:
                    data['allocation'] = 0.05 + 0.75 * (data['profit_factor'] / total_profit_factor)
                    
                    # Adjust for win rate
                    if data['win_rate'] > 0.5:
                        data['allocation'] *= 1 + (data['win_rate'] - 0.5)
                    else:
                        data['allocation'] *= 0.5 + data['win_rate']
                    
                    # Adjust for volatility
                    if data['volatility'] > 0:
                        avg_volatility = np.mean([d.get('volatility', 0) for d in performance_data.values() if d.get('volatility', 0) > 0])
                        if avg_volatility > 0:
                            volatility_ratio = data['volatility'] / avg_volatility
                            
                            if volatility_ratio > 1.5:
                                # Reduce allocation for highly volatile pairs
                                data['allocation'] *= 0.7
                            elif volatility_ratio < 0.5:
                                # Increase allocation for low volatility pairs
                                data['allocation'] *= 1.2
        
        # Normalize allocations to sum to 0.85 (max allocation)
        total_allocation = sum([d.get('allocation', 0) for d in performance_data.values()])
        
        if total_allocation > 0:
            for pair in performance_data:
                performance_data[pair]['allocation'] = 0.85 * performance_data[pair]['allocation'] / total_allocation
        else:
            # Equal allocation if no performance data
            equal_allocation = 0.85 / len(performance_data) if performance_data else 0
            for pair in performance_data:
                performance_data[pair]['allocation'] = equal_allocation
        
        # Extract just the allocation values
        allocations = {pair: data['allocation'] for pair, data in performance_data.items()}
        
        # Update optimization data
        self.optimization_data['portfolio_allocations'] = allocations
        self._save_optimization_data()
        
        return allocations
    
    def optimize_leverage_settings(self) -> Dict[str, Dict[str, float]]:
        """
        Optimize leverage settings for each pair
        
        Returns:
            Dictionary of leverage settings by pair
        """
        leverage_settings = {}
        
        # Extract leverage data from closed trades
        pair_leverage_data = {}
        
        for trade in self.trades:
            if trade.get('status') != 'closed':
                continue
                
            pair = trade.get('pair')
            leverage = trade.get('leverage', 0)
            pnl = trade.get('pnl', 0)
            confidence = trade.get('confidence', 0)
            
            if not pair or leverage <= 0:
                continue
                
            if pair not in pair_leverage_data:
                pair_leverage_data[pair] = []
                
            pair_leverage_data[pair].append({
                'leverage': leverage,
                'pnl': pnl,
                'confidence': confidence
            })
        
        # Analyze leverage performance for each pair
        for pair, trades in pair_leverage_data.items():
            if len(trades) < 5:  # Need enough trades for meaningful analysis
                continue
                
            # Group by confidence levels
            confidence_groups = {}
            for trade in trades:
                confidence = trade['confidence']
                
                # Round confidence to nearest 0.05
                confidence_key = round(confidence * 20) / 20
                
                if confidence_key not in confidence_groups:
                    confidence_groups[confidence_key] = []
                    
                confidence_groups[confidence_key].append(trade)
            
            # Determine optimal leverage for each confidence level
            optimal_leverage = {}
            
            for confidence, confidence_trades in confidence_groups.items():
                if len(confidence_trades) < 3:  # Need enough trades at this confidence level
                    continue
                    
                # Group by leverage ranges
                leverage_groups = {}
                for trade in confidence_trades:
                    # Round leverage to nearest 5
                    leverage_key = round(trade['leverage'] / 5) * 5
                    
                    if leverage_key not in leverage_groups:
                        leverage_groups[leverage_key] = []
                        
                    leverage_groups[leverage_key].append(trade)
                
                # Find best performing leverage
                best_leverage = 0
                best_pnl = float('-inf')
                
                for leverage, leverage_trades in leverage_groups.items():
                    if len(leverage_trades) < 2:  # Need enough trades at this leverage
                        continue
                        
                    total_pnl = sum(t['pnl'] for t in leverage_trades)
                    avg_pnl = total_pnl / len(leverage_trades)
                    
                    if avg_pnl > best_pnl:
                        best_pnl = avg_pnl
                        best_leverage = leverage
                
                if best_leverage > 0:
                    optimal_leverage[confidence] = best_leverage
            
            # Store optimal leverage for this pair
            if optimal_leverage:
                leverage_settings[pair] = optimal_leverage
        
        # Fill in gaps with a formula for pairs that don't have enough data
        base_leverage = 50.0
        max_leverage = 125.0
        
        for pair in self.trading_pairs:
            if pair not in leverage_settings:
                leverage_settings[pair] = {}
            
            # Generate formula-based leverage settings for all confidence levels
            for confidence in [round(c * 0.05, 2) for c in range(13, 21)]:  # 0.65 to 1.00
                # If we already have optimized leverage for this confidence, keep it
                if confidence in leverage_settings[pair]:
                    continue
                
                # For confidence below 0.75, use base leverage
                if confidence < 0.75:
                    leverage_settings[pair][confidence] = base_leverage
                else:
                    # Scale leverage between base and max based on confidence
                    scale_factor = (confidence - 0.75) / 0.25  # 0 at 0.75, 1 at 1.0
                    leverage = base_leverage + scale_factor * (max_leverage - base_leverage)
                    leverage_settings[pair][confidence] = min(max_leverage, leverage)
        
        # Format for storage
        formatted_settings = {}
        for pair, levels in leverage_settings.items():
            formatted_settings[pair] = {
                'base_leverage': base_leverage,
                'max_leverage': max_leverage,
                'confidence_levels': {str(k): v for k, v in levels.items()}
            }
        
        # Update optimization data
        self.optimization_data['leverage_settings'] = formatted_settings
        self._save_optimization_data()
        
        return formatted_settings
    
    def run_optimization(self, current_prices: Dict[str, float] = None):
        """
        Run complete optimization process
        
        Args:
            current_prices: Dictionary of current prices (optional)
        
        Returns:
            Complete optimization results
        """
        logger.info("Running complete trade optimization process")
        
        # Analyze historical trades
        trade_analysis = self.analyze_historical_trades()
        logger.info(f"Analyzed historical trades for {len(trade_analysis)} pairs")
        
        # Optimize portfolio allocation
        allocations = self.optimize_portfolio_allocation()
        logger.info(f"Optimized portfolio allocation across {len(allocations)} pairs")
        
        # Optimize leverage settings
        leverage_settings = self.optimize_leverage_settings()
        logger.info(f"Optimized leverage settings for {len(leverage_settings)} pairs")
        
        # Optimize current positions if prices provided
        position_optimizations = None
        if current_prices:
            position_optimizations = self.optimize_all_positions(current_prices)
            
            # Apply position adjustments
            if position_optimizations.get('adjustments'):
                self.apply_position_adjustments(position_optimizations['adjustments'])
                logger.info(f"Applied {len(position_optimizations['adjustments'])} position adjustments")
            
            logger.info(f"Optimized current positions: {len(position_optimizations.get('exits', []))} exits recommended")
        
        # Save all optimization results
        self.optimization_data['last_full_optimization'] = datetime.now().isoformat()
        self._save_optimization_data()
        
        # Return complete results
        return {
            'trade_analysis': trade_analysis,
            'allocations': allocations,
            'leverage_settings': leverage_settings,
            'position_optimizations': position_optimizations,
            'market_states': self.optimization_data.get('market_states', {}),
            'volatility_data': {
                pair: data.get('regime', 'normal') 
                for pair, data in self.optimization_data.get('volatility_data', {}).items()
            },
            'optimal_hours': self.optimization_data.get('optimal_hours', {})
        }

# Example usage
if __name__ == "__main__":
    import sys
    
    # Parse args
    trading_pairs = sys.argv[1:] if len(sys.argv) > 1 else [
        "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
    ]
    
    # Create optimizer
    optimizer = TradeOptimizer(trading_pairs)
    
    # Sample current prices (in a real implementation, get from API)
    current_prices = {
        pair: 100.0  # Placeholder
        for pair in trading_pairs
    }
    
    # Run optimization
    results = optimizer.run_optimization(current_prices)
    
    # Print summary
    print(f"Optimization complete for {len(trading_pairs)} pairs")
    print(f"Market states: {results['market_states']}")
    print(f"Position adjustments: {len(results.get('position_optimizations', {}).get('adjustments', []))}")
    print(f"Recommended exits: {len(results.get('position_optimizations', {}).get('exits', []))}")