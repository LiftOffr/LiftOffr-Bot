#!/usr/bin/env python3
"""
Strategy Parameter Optimization System

This module implements an advanced optimization system for trading strategies
that can find optimal parameter values for different market conditions.

Key features:
1. Walk-forward optimization for realistic parameter selection
2. Bayesian optimization for efficient parameter search
3. Multi-objective optimization for balancing risk and reward
4. Market regime-specific parameter sets
5. Performance stability analysis

Target: Supporting the 90%+ backtesting accuracy goal
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import TimeSeriesSplit

# Try to import bayesian optimization
try:
    import skopt
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    bayesian_optimization_available = True
except ImportError:
    logger.warning("scikit-optimize not available, Bayesian optimization disabled")
    bayesian_optimization_available = False

# Import local modules
from enhanced_backtesting import (
    EnhancedBacktester, 
    load_historical_data,
    optimize_strategy_parameters
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOLS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
RESULTS_DIR = "optimization_results"
DEFAULT_CV_SPLITS = 5
DEFAULT_INITIAL_CAPITAL = 20000.0

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

class StrategyOptimizer:
    """
    Advanced strategy optimization system that can find optimal parameter values
    for different market conditions using sophisticated optimization techniques.
    """
    
    def __init__(self, strategy_class, symbol, timeframe, 
                 initial_capital=DEFAULT_INITIAL_CAPITAL,
                 cv_splits=DEFAULT_CV_SPLITS, 
                 use_bayesian=True, 
                 n_calls=50,
                 random_state=42):
        """
        Initialize the strategy optimizer
        
        Args:
            strategy_class: Strategy class to optimize
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            initial_capital (float): Initial capital
            cv_splits (int): Number of cross-validation splits
            use_bayesian (bool): Whether to use Bayesian optimization
            n_calls (int): Number of calls for Bayesian optimization
            random_state (int): Random state for reproducibility
        """
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.cv_splits = cv_splits
        self.use_bayesian = use_bayesian and bayesian_optimization_available
        self.n_calls = n_calls
        self.random_state = random_state
        
        # Load data
        self.data = load_historical_data(symbol, timeframe)
        
        if self.data is None:
            raise ValueError(f"No data available for {symbol} on {timeframe} timeframe")
        
        logger.info(f"Loaded {len(self.data)} rows of data for {symbol} on {timeframe} timeframe")
        
        # Create backtester
        self.backtester = EnhancedBacktester(
            data=self.data,
            trading_pair=symbol,
            initial_capital=initial_capital,
            timeframe=timeframe
        )
        
        # Optimization results
        self.optimization_results = None
        self.best_params = None
        self.param_importance = None
        
        logger.info(f"Strategy optimizer initialized for {symbol} on {timeframe} timeframe")
        logger.info(f"Using Bayesian optimization: {self.use_bayesian}")
    
    def create_parameter_grid(self, param_ranges):
        """
        Create parameter grid for optimization
        
        Args:
            param_ranges (dict): Parameter ranges
                Each key is a parameter name, each value is a tuple (min, max, step)
                for numerical parameters or a list of values for categorical parameters
            
        Returns:
            dict: Parameter grid for grid search
            list: Parameter space for Bayesian optimization
        """
        # Grid for grid search
        param_grid = {}
        
        # Space for Bayesian optimization
        param_space = []
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, (list, tuple)):
                if len(param_range) == 3 and all(isinstance(x, (int, float)) for x in param_range):
                    # Numerical parameter with (min, max, step)
                    min_val, max_val, step = param_range
                    
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        param_grid[param_name] = list(range(min_val, max_val + 1, step))
                        
                        # For Bayesian optimization
                        param_space.append(Integer(min_val, max_val, name=param_name))
                    else:
                        # Float parameter
                        values = np.arange(min_val, max_val + step, step)
                        param_grid[param_name] = values.tolist()
                        
                        # For Bayesian optimization
                        param_space.append(Real(min_val, max_val, name=param_name))
                else:
                    # Categorical parameter
                    param_grid[param_name] = list(param_range)
                    
                    # For Bayesian optimization
                    param_space.append(Categorical(param_range, name=param_name))
            else:
                # Single value
                param_grid[param_name] = [param_range]
                
                # For Bayesian optimization (as categorical with one value)
                param_space.append(Categorical([param_range], name=param_name))
        
        return param_grid, param_space
    
    def objective_function(self, params):
        """
        Objective function for optimization
        
        Args:
            params (dict): Strategy parameters
            
        Returns:
            float: Negative score (for minimization)
        """
        # Create strategy with parameters
        strategy = self.strategy_class(self.symbol, **params)
        
        # Run backtest
        result = self.backtester.run_backtest({"strategy": strategy})
        
        # Extract score (higher is better)
        score = result.get("sharpe_ratio", 0.0)
        
        # Return negative score for minimization
        return -score
    
    @use_named_args
    def bayesian_objective(self, **params):
        """
        Objective function for Bayesian optimization
        
        Args:
            **params: Strategy parameters
            
        Returns:
            float: Negative score (for minimization)
        """
        return self.objective_function(params)
    
    def run_grid_search(self, param_grid, scoring="sharpe_ratio"):
        """
        Run grid search optimization
        
        Args:
            param_grid (dict): Parameter grid
            scoring (str): Scoring metric
            
        Returns:
            tuple: (best_params, cv_results)
        """
        logger.info(f"Running grid search with {scoring} scoring")
        
        # Run walk-forward optimization
        best_params, cv_results = self.backtester.run_walk_forward_optimization(
            strategy_class=self.strategy_class,
            param_grid=param_grid,
            n_splits=self.cv_splits,
            scoring=scoring
        )
        
        return best_params, cv_results
    
    def run_bayesian_optimization(self, param_space, scoring="sharpe_ratio"):
        """
        Run Bayesian optimization
        
        Args:
            param_space (list): Parameter space
            scoring (str): Scoring metric
            
        Returns:
            tuple: (best_params, results)
        """
        if not bayesian_optimization_available:
            logger.error("Bayesian optimization not available")
            return None, None
        
        logger.info(f"Running Bayesian optimization with {scoring} scoring")
        logger.info(f"Parameter space: {param_space}")
        
        # Run Bayesian optimization
        result = gp_minimize(
            self.bayesian_objective,
            param_space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=True
        )
        
        # Extract best parameters
        best_params = {
            dim.name: value for dim, value in zip(param_space, result.x)
        }
        
        # Log best parameters
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {-result.fun}")
        
        return best_params, result
    
    def calculate_parameter_importance(self, result):
        """
        Calculate parameter importance from Bayesian optimization result
        
        Args:
            result: Bayesian optimization result
            
        Returns:
            dict: Parameter importance
        """
        if not bayesian_optimization_available:
            logger.error("Bayesian optimization not available")
            return None
        
        try:
            # Get parameter importance
            importances = skopt.utils.extract_params_from_result(result)
            
            # Convert to dictionary
            importance_dict = {
                name: importance for name, importance in importances.items()
            }
            
            # Log parameter importance
            for param, importance in importance_dict.items():
                logger.info(f"Importance of {param}: {importance:.4f}")
            
            return importance_dict
        
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return None
    
    def optimize(self, param_ranges, scoring="sharpe_ratio"):
        """
        Optimize strategy parameters
        
        Args:
            param_ranges (dict): Parameter ranges
            scoring (str): Scoring metric
            
        Returns:
            dict: Optimization results
        """
        # Start time
        start_time = time.time()
        
        # Create parameter grid and space
        param_grid, param_space = self.create_parameter_grid(param_ranges)
        
        # Choose optimization method
        if self.use_bayesian:
            # Run Bayesian optimization
            best_params, result = self.run_bayesian_optimization(param_space, scoring)
            
            # Calculate parameter importance
            if result is not None:
                param_importance = self.calculate_parameter_importance(result)
            else:
                param_importance = None
            
            # Store optimization method
            optimization_method = "bayesian"
        else:
            # Run grid search
            best_params, result = self.run_grid_search(param_grid, scoring)
            
            # No parameter importance for grid search
            param_importance = None
            
            # Store optimization method
            optimization_method = "grid_search"
        
        # End time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store results
        self.best_params = best_params
        self.optimization_results = result
        self.param_importance = param_importance
        
        # Create results dictionary
        results = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy": self.strategy_class.__name__,
            "scoring": scoring,
            "best_params": best_params,
            "param_importance": param_importance,
            "optimization_method": optimization_method,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Optimization completed in {execution_time:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def save_results(self, results=None, filename=None):
        """
        Save optimization results
        
        Args:
            results (dict, optional): Results to save
            filename (str, optional): Filename
            
        Returns:
            str: Path to saved results
        """
        if results is None:
            if self.best_params is None:
                logger.error("No optimization results to save")
                return None
            
            # Create results dictionary
            results = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "strategy": self.strategy_class.__name__,
                "best_params": self.best_params,
                "param_importance": self.param_importance,
                "timestamp": datetime.now().isoformat()
            }
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy_class.__name__
            filename = f"{self.symbol}_{self.timeframe}_{strategy_name}_{timestamp}.json"
        
        # Create path
        result_path = os.path.join(RESULTS_DIR, filename)
        
        # Save results
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {result_path}")
        
        return result_path
    
    def backtest_with_optimal_params(self, plot=False, plot_file=None):
        """
        Run backtest with optimal parameters
        
        Args:
            plot (bool): Whether to plot results
            plot_file (str, optional): Path to save plot
            
        Returns:
            dict: Backtest results
        """
        if self.best_params is None:
            logger.error("No optimal parameters available")
            return None
        
        # Create strategy with optimal parameters
        strategy = self.strategy_class(self.symbol, **self.best_params)
        
        # Run backtest
        result = self.backtester.run_backtest({"strategy": strategy})
        
        # Plot if requested
        if plot:
            self.backtester.plot_results(plot_file)
        
        return result
    
    def plot_parameter_importance(self, param_importance=None, filename=None):
        """
        Plot parameter importance
        
        Args:
            param_importance (dict, optional): Parameter importance
            filename (str, optional): Path to save plot
        """
        if param_importance is None:
            param_importance = self.param_importance
        
        if param_importance is None:
            logger.error("No parameter importance available")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Sort parameters by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Extract names and values
        names = [p[0] for p in sorted_params]
        values = [p[1] for p in sorted_params]
        
        # Plot
        plt.bar(names, values)
        plt.title("Parameter Importance")
        plt.xlabel("Parameter")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if requested
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter importance plot saved to {filename}")
        
        plt.show()


class MarketRegimeOptimizer:
    """
    Optimizer that finds optimal parameters for different market regimes.
    """
    
    def __init__(self, strategy_class, symbol, timeframe, 
                 initial_capital=DEFAULT_INITIAL_CAPITAL,
                 cv_splits=DEFAULT_CV_SPLITS, 
                 use_bayesian=True, 
                 n_calls=50,
                 random_state=42):
        """
        Initialize the market regime optimizer
        
        Args:
            strategy_class: Strategy class to optimize
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            initial_capital (float): Initial capital
            cv_splits (int): Number of cross-validation splits
            use_bayesian (bool): Whether to use Bayesian optimization
            n_calls (int): Number of calls for Bayesian optimization
            random_state (int): Random state for reproducibility
        """
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.cv_splits = cv_splits
        self.use_bayesian = use_bayesian and bayesian_optimization_available
        self.n_calls = n_calls
        self.random_state = random_state
        
        # Load data
        self.data = load_historical_data(symbol, timeframe)
        
        if self.data is None:
            raise ValueError(f"No data available for {symbol} on {timeframe} timeframe")
        
        logger.info(f"Loaded {len(self.data)} rows of data for {symbol} on {timeframe} timeframe")
        
        # Identify market regimes
        self.regime_data = self.identify_market_regimes(self.data)
        
        # Optimization results by regime
        self.regime_optimization_results = {}
        
        logger.info(f"Market regime optimizer initialized for {symbol} on {timeframe} timeframe")
    
    def identify_market_regimes(self, data):
        """
        Identify market regimes in the data
        
        Args:
            data (pd.DataFrame): Historical data
            
        Returns:
            pd.DataFrame: Data with market regime column
        """
        # Copy data
        regime_data = data.copy()
        
        # Calculate returns
        regime_data['returns'] = regime_data['close'].pct_change()
        
        # Calculate volatility (20-day rolling standard deviation of returns)
        regime_data['volatility'] = regime_data['returns'].rolling(window=20).std()
        
        # Calculate trend (200-day moving average direction)
        regime_data['ma200'] = regime_data['close'].rolling(window=200).mean()
        regime_data['trend'] = (regime_data['ma200'].diff() > 0).astype(int)
        
        # Identify regimes
        # 0: Low volatility, uptrend (normal)
        # 1: High volatility, uptrend (volatile_trending_up)
        # 2: Low volatility, downtrend (normal_trending_down)
        # 3: High volatility, downtrend (volatile_trending_down)
        
        # Define volatility threshold (75th percentile)
        volatility_threshold = regime_data['volatility'].quantile(0.75)
        
        # Create regime column
        conditions = [
            (regime_data['volatility'] <= volatility_threshold) & (regime_data['trend'] == 1),
            (regime_data['volatility'] > volatility_threshold) & (regime_data['trend'] == 1),
            (regime_data['volatility'] <= volatility_threshold) & (regime_data['trend'] == 0),
            (regime_data['volatility'] > volatility_threshold) & (regime_data['trend'] == 0)
        ]
        
        regimes = ['normal', 'volatile_trending_up', 'normal_trending_down', 'volatile_trending_down']
        
        regime_data['market_regime'] = np.select(conditions, regimes, default='normal')
        
        # Count regimes
        regime_counts = regime_data['market_regime'].value_counts()
        
        logger.info(f"Market regimes identified: {regime_counts.to_dict()}")
        
        return regime_data
    
    def optimize_for_regime(self, regime, param_ranges, scoring="sharpe_ratio"):
        """
        Optimize parameters for a specific market regime
        
        Args:
            regime (str): Market regime
            param_ranges (dict): Parameter ranges
            scoring (str): Scoring metric
            
        Returns:
            dict: Optimization results
        """
        logger.info(f"Optimizing for {regime} regime")
        
        # Filter data for regime
        regime_filter = self.regime_data['market_regime'] == regime
        regime_indices = self.regime_data.index[regime_filter]
        
        if len(regime_indices) < 100:
            logger.warning(f"Not enough data for {regime} regime ({len(regime_indices)} samples)")
            return None
        
        # Create data subset
        regime_subset = self.data.loc[regime_indices]
        
        logger.info(f"Using {len(regime_subset)} samples for {regime} regime")
        
        # Create backtester for regime
        backtester = EnhancedBacktester(
            data=regime_subset,
            trading_pair=self.symbol,
            initial_capital=self.initial_capital,
            timeframe=self.timeframe
        )
        
        # Create optimizer for regime
        optimizer = StrategyOptimizer(
            strategy_class=self.strategy_class,
            symbol=self.symbol,
            timeframe=self.timeframe,
            initial_capital=self.initial_capital,
            cv_splits=self.cv_splits,
            use_bayesian=self.use_bayesian,
            n_calls=self.n_calls,
            random_state=self.random_state
        )
        
        # Override optimizer's backtester and data
        optimizer.backtester = backtester
        optimizer.data = regime_subset
        
        # Run optimization
        results = optimizer.optimize(param_ranges, scoring)
        
        # Add regime to results
        results['regime'] = regime
        
        # Store results
        self.regime_optimization_results[regime] = results
        
        return results
    
    def optimize_all_regimes(self, param_ranges, scoring="sharpe_ratio"):
        """
        Optimize parameters for all market regimes
        
        Args:
            param_ranges (dict): Parameter ranges
            scoring (str): Scoring metric
            
        Returns:
            dict: Optimization results by regime
        """
        # Get unique regimes
        regimes = self.regime_data['market_regime'].unique()
        
        logger.info(f"Optimizing for {len(regimes)} regimes: {regimes}")
        
        # Optimize for each regime
        results = {}
        
        for regime in regimes:
            regime_results = self.optimize_for_regime(regime, param_ranges, scoring)
            
            if regime_results is not None:
                results[regime] = regime_results
        
        return results
    
    def save_results(self, filename=None):
        """
        Save optimization results
        
        Args:
            filename (str, optional): Filename
            
        Returns:
            str: Path to saved results
        """
        if not self.regime_optimization_results:
            logger.error("No optimization results to save")
            return None
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy_class.__name__
            filename = f"{self.symbol}_{self.timeframe}_{strategy_name}_regime_optimization_{timestamp}.json"
        
        # Create path
        result_path = os.path.join(RESULTS_DIR, filename)
        
        # Save results
        with open(result_path, 'w') as f:
            json.dump(self.regime_optimization_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {result_path}")
        
        return result_path
    
    def plot_parameter_comparison(self, filename=None):
        """
        Plot comparison of optimal parameters across regimes
        
        Args:
            filename (str, optional): Path to save plot
        """
        if not self.regime_optimization_results:
            logger.error("No optimization results to plot")
            return
        
        # Get regimes
        regimes = list(self.regime_optimization_results.keys())
        
        # Get parameters
        all_params = set()
        for regime_results in self.regime_optimization_results.values():
            if 'best_params' in regime_results:
                all_params.update(regime_results['best_params'].keys())
        
        # Create figure
        fig, axes = plt.subplots(len(all_params), 1, figsize=(10, 4 * len(all_params)))
        
        # Ensure axes is iterable
        if len(all_params) == 1:
            axes = [axes]
        
        # Plot each parameter
        for i, param in enumerate(all_params):
            param_values = []
            
            for regime in regimes:
                regime_results = self.regime_optimization_results.get(regime, {})
                best_params = regime_results.get('best_params', {})
                param_values.append(best_params.get(param, None))
            
            # Plot
            axes[i].bar(regimes, param_values)
            axes[i].set_title(f"Optimal {param} by Market Regime")
            axes[i].set_xlabel("Market Regime")
            axes[i].set_ylabel(param)
            axes[i].grid(True)
        
        plt.tight_layout()
        
        # Save if requested
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter comparison plot saved to {filename}")
        
        plt.show()


def create_param_ranges_for_strategy(strategy_name):
    """
    Create parameter ranges for a specific strategy
    
    Args:
        strategy_name (str): Strategy name
        
    Returns:
        dict: Parameter ranges
    """
    if strategy_name.lower() == "arima":
        # ARIMA strategy parameters
        param_ranges = {
            "lookback_period": (10, 50, 5),  # 10, 15, 20, ..., 50
            "signal_threshold": (0.05, 0.5, 0.05),  # 0.05, 0.10, ..., 0.50
            "atr_period": (10, 30, 5),  # 10, 15, 20, ..., 30
            "atr_multiplier": (1.0, 3.0, 0.5)  # 1.0, 1.5, 2.0, 2.5, 3.0
        }
    elif strategy_name.lower() == "integrated":
        # Integrated strategy parameters
        param_ranges = {
            "signal_smoothing": (2, 10, 1),  # 2, 3, ..., 10
            "trend_strength_threshold": (0.1, 0.5, 0.1),  # 0.1, 0.2, ..., 0.5
            "volatility_filter_threshold": (0.005, 0.02, 0.005),  # 0.005, 0.01, 0.015, 0.02
            "min_adx_threshold": (15, 35, 5)  # 15, 20, 25, 30, 35
        }
    elif strategy_name.lower() == "ml_enhanced":
        # ML-enhanced strategy parameters
        param_ranges = {
            "ml_influence": (0.2, 0.8, 0.1),  # 0.2, 0.3, ..., 0.8
            "confidence_threshold": (0.5, 0.8, 0.1)  # 0.5, 0.6, 0.7, 0.8
        }
    else:
        # Default parameters
        param_ranges = {}
    
    return param_ranges


def optimize_strategy(strategy_class, symbol, timeframe, 
                    param_ranges=None, 
                    scoring="sharpe_ratio",
                    use_bayesian=True, 
                    n_calls=50,
                    use_market_regimes=False,
                    plot=True,
                    save_results=True):
    """
    Optimize strategy parameters
    
    Args:
        strategy_class: Strategy class to optimize
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        param_ranges (dict, optional): Parameter ranges
        scoring (str): Scoring metric
        use_bayesian (bool): Whether to use Bayesian optimization
        n_calls (int): Number of calls for Bayesian optimization
        use_market_regimes (bool): Whether to optimize for market regimes
        plot (bool): Whether to plot results
        save_results (bool): Whether to save results
        
    Returns:
        dict: Optimization results
    """
    # Create default parameter ranges if not provided
    if param_ranges is None:
        param_ranges = create_param_ranges_for_strategy(strategy_class.__name__)
    
    # Create optimizer
    if use_market_regimes:
        optimizer = MarketRegimeOptimizer(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=timeframe,
            use_bayesian=use_bayesian,
            n_calls=n_calls
        )
        
        # Optimize for all regimes
        results = optimizer.optimize_all_regimes(param_ranges, scoring)
        
        # Plot parameter comparison
        if plot:
            plot_file = os.path.join(
                RESULTS_DIR, 
                f"{symbol}_{timeframe}_{strategy_class.__name__}_regime_params.png"
            ) if save_results else None
            
            optimizer.plot_parameter_comparison(plot_file)
        
        # Save results
        if save_results:
            optimizer.save_results()
        
    else:
        optimizer = StrategyOptimizer(
            strategy_class=strategy_class,
            symbol=symbol,
            timeframe=timeframe,
            use_bayesian=use_bayesian,
            n_calls=n_calls
        )
        
        # Optimize parameters
        results = optimizer.optimize(param_ranges, scoring)
        
        # Plot parameter importance
        if plot and optimizer.param_importance is not None:
            plot_file = os.path.join(
                RESULTS_DIR, 
                f"{symbol}_{timeframe}_{strategy_class.__name__}_param_importance.png"
            ) if save_results else None
            
            optimizer.plot_parameter_importance(filename=plot_file)
        
        # Run backtest with optimal parameters
        if optimizer.best_params is not None:
            plot_file = os.path.join(
                RESULTS_DIR, 
                f"{symbol}_{timeframe}_{strategy_class.__name__}_optimal_backtest.png"
            ) if save_results and plot else None
            
            backtest_results = optimizer.backtest_with_optimal_params(plot=plot, plot_file=plot_file)
            
            # Add backtest results to optimization results
            results['backtest_results'] = backtest_results
        
        # Save results
        if save_results:
            optimizer.save_results(results)
    
    return results


def main():
    """Main function for strategy parameter optimization"""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimization System")
    
    parser.add_argument("--strategy", type=str, default="arima",
                       help="Strategy to optimize (arima, integrated, ml_enhanced)")
    parser.add_argument("--symbol", type=str, default="SOLUSD",
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                       help="Timeframe (default: 1h)")
    parser.add_argument("--scoring", type=str, default="sharpe_ratio",
                       help="Scoring metric (default: sharpe_ratio)")
    parser.add_argument("--bayesian", action="store_true",
                       help="Use Bayesian optimization")
    parser.add_argument("--n-calls", type=int, default=50,
                       help="Number of calls for Bayesian optimization (default: 50)")
    parser.add_argument("--market-regimes", action="store_true",
                       help="Optimize for market regimes")
    parser.add_argument("--no-plot", action="store_true",
                       help="Don't plot results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results")
    parser.add_argument("--multi-asset", action="store_true",
                       help="Optimize for multiple assets")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="List of symbols to optimize")
    parser.add_argument("--multi-timeframe", action="store_true",
                       help="Optimize for multiple timeframes")
    parser.add_argument("--timeframes", nargs="+", default=None,
                       help="List of timeframes to optimize")
    
    args = parser.parse_args()
    
    # Import strategies
    if args.strategy.lower() == "arima":
        from arima_strategy import ARIMAStrategy as StrategyClass
    elif args.strategy.lower() == "integrated":
        from integrated_strategy import IntegratedStrategy as StrategyClass
    elif args.strategy.lower() == "ml_enhanced":
        try:
            from ml_enhanced_strategy import MLEnhancedStrategy as StrategyClass
        except ImportError:
            logger.error("ML-enhanced strategy not available")
            return
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Multi-asset optimization
    if args.multi_asset:
        symbols = args.symbols or DEFAULT_SYMBOLS
        
        logger.info(f"Optimizing for multiple assets: {symbols}")
        
        # Optimize for each symbol
        for symbol in symbols:
            optimize_strategy(
                strategy_class=StrategyClass,
                symbol=symbol,
                timeframe=args.timeframe,
                scoring=args.scoring,
                use_bayesian=args.bayesian,
                n_calls=args.n_calls,
                use_market_regimes=args.market_regimes,
                plot=not args.no_plot,
                save_results=not args.no_save
            )
    
    # Multi-timeframe optimization
    elif args.multi_timeframe:
        timeframes = args.timeframes or DEFAULT_TIMEFRAMES
        
        logger.info(f"Optimizing for multiple timeframes: {timeframes}")
        
        # Optimize for each timeframe
        for timeframe in timeframes:
            optimize_strategy(
                strategy_class=StrategyClass,
                symbol=args.symbol,
                timeframe=timeframe,
                scoring=args.scoring,
                use_bayesian=args.bayesian,
                n_calls=args.n_calls,
                use_market_regimes=args.market_regimes,
                plot=not args.no_plot,
                save_results=not args.no_save
            )
    
    # Single optimization
    else:
        optimize_strategy(
            strategy_class=StrategyClass,
            symbol=args.symbol,
            timeframe=args.timeframe,
            scoring=args.scoring,
            use_bayesian=args.bayesian,
            n_calls=args.n_calls,
            use_market_regimes=args.market_regimes,
            plot=not args.no_plot,
            save_results=not args.no_save
        )


if __name__ == "__main__":
    main()