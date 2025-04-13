#!/usr/bin/env python3
"""
Strategy Optimizer for Kraken Trading Bot

This module analyzes the performance of trading strategies, identifies improvements,
and recommends parameter adjustments or strategy removal based on profitability metrics.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import traceback
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TRADES_FILE = "trades.csv"
LOGS_DIR = "logs"
RESULTS_DIR = "optimization_results"
MIN_TRADES_FOR_ANALYSIS = 10
PROFITABLE_THRESHOLD = 0.0  # Break-even or better
PERFORMANCE_METRICS_WEIGHT = {
    'win_rate': 0.3,
    'avg_profit_loss': 0.3,
    'sharpe_ratio': 0.2,
    'max_drawdown': 0.1,
    'trade_frequency': 0.1,
}

# Create output directory
os.makedirs(RESULTS_DIR, exist_ok=True)


class StrategyOptimizer:
    """
    Analyzes trading strategy performance and recommends improvements
    based on historical trades and market data.
    """
    
    def __init__(self, trades_file=TRADES_FILE, lookback_days=30):
        """
        Initialize the strategy optimizer
        
        Args:
            trades_file (str): Path to trades CSV file
            lookback_days (int): Number of days to look back for analysis
        """
        self.trades_file = trades_file
        self.lookback_days = lookback_days
        self.trades_df = None
        self.strategies = {}
        self.parameter_ranges = {
            'ARIMAStrategy': {
                'lookback_period': range(20, 60, 4),
                'atr_trailing_multiplier': np.arange(1.5, 3.1, 0.3),
                'entry_atr_multiplier': np.arange(0.005, 0.03, 0.005),
                'risk_buffer_multiplier': np.arange(1.0, 2.1, 0.25),
                'max_loss_percent': np.arange(3.0, 5.1, 0.5),
            },
            'AdaptiveStrategy': {
                'rsi_period': range(10, 22, 2),
                'ema_short': range(7, 13, 1),
                'ema_long': range(18, 26, 2),
                'atr_period': range(10, 20, 2),
                'volatility_threshold': np.arange(0.004, 0.01, 0.001),
            },
            'IntegratedStrategy': {
                'signal_smoothing': range(2, 6),
                'trend_strength_threshold': np.arange(0.3, 0.7, 0.1),
                'volatility_filter_threshold': np.arange(0.005, 0.015, 0.002),
                'min_adx_threshold': range(20, 30, 2),
            }
        }
        
    def load_trades(self):
        """
        Load trades from CSV file
        
        Returns:
            pd.DataFrame: DataFrame with trade data
        """
        if not os.path.exists(self.trades_file):
            logger.error(f"Trades file {self.trades_file} not found.")
            return None
            
        try:
            df = pd.read_csv(self.trades_file)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            
            # Filter by lookback period
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            df = df[df['timestamp'] >= cutoff_date]
            
            # Ensure required columns exist
            required_columns = ['strategy', 'symbol', 'type', 'entry_price', 'exit_price', 'profit_loss']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Required column {col} not found in trades file.")
            
            self.trades_df = df
            logger.info(f"Loaded {len(df)} trades within the last {self.lookback_days} days.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            traceback.print_exc()
            return None
    
    def analyze_strategy_performance(self):
        """
        Analyze performance of each trading strategy
        
        Returns:
            dict: Performance metrics for each strategy
        """
        if self.trades_df is None:
            logger.error("No trades data loaded.")
            return {}
            
        # Group trades by strategy
        grouped = self.trades_df.groupby('strategy')
        
        results = {}
        for strategy_name, trades in grouped:
            # Skip if too few trades for meaningful analysis
            if len(trades) < MIN_TRADES_FOR_ANALYSIS:
                logger.warning(f"Strategy {strategy_name} has too few trades ({len(trades)}) for analysis.")
                continue
                
            # Calculate performance metrics
            metrics = self._calculate_metrics(trades)
            
            # Store results
            results[strategy_name] = metrics
            logger.info(f"Strategy {strategy_name}: {metrics}")
            
        self.strategies = results
        return results
    
    def _calculate_metrics(self, trades_df):
        """
        Calculate performance metrics for a set of trades
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trades
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit_loss = trades_df['profit_loss'].sum()
        avg_profit_loss = trades_df['profit_loss'].mean()
        max_profit = trades_df['profit_loss'].max()
        max_loss = trades_df['profit_loss'].min()
        
        # Calculate daily returns
        trades_df = trades_df.sort_values('timestamp')
        trades_df['date'] = trades_df['timestamp'].dt.date
        daily_returns = trades_df.groupby('date')['profit_loss'].sum()
        
        # Risk metrics
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Trade frequency
        days_with_trades = len(daily_returns)
        trade_frequency = days_with_trades / self.lookback_days
        
        # Combine metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'avg_profit_loss': avg_profit_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_frequency': trade_frequency,
            'is_profitable': total_profit_loss > 0
        }
        
        return metrics
    
    def evaluate_strategy_quality(self, metrics):
        """
        Evaluate overall quality of a strategy based on metrics
        
        Args:
            metrics (dict): Performance metrics
            
        Returns:
            float: Quality score (0-1)
        """
        # Normalize metrics
        normalized = {}
        
        # Win rate (higher is better)
        normalized['win_rate'] = metrics['win_rate']
        
        # Average profit/loss (normalize to -1 to 1 range)
        avg_pl = metrics['avg_profit_loss']
        normalized['avg_profit_loss'] = min(1, max(-1, avg_pl * 10))  # Scale by 10 for better differentiation
        
        # Sharpe ratio (higher is better, cap at 3)
        normalized['sharpe_ratio'] = min(1, max(0, metrics['sharpe_ratio'] / 3))
        
        # Max drawdown (0 to -1, lower is worse)
        normalized['max_drawdown'] = 1 + max(-1, min(0, metrics['max_drawdown']))
        
        # Trade frequency (0 to 1, higher is better up to a point)
        normalized['trade_frequency'] = min(1, metrics['trade_frequency'] * 5)  # Scale by 5
        
        # Calculate weighted score
        score = sum(normalized[metric] * weight 
                    for metric, weight in PERFORMANCE_METRICS_WEIGHT.items())
        
        return score
    
    def identify_strategies_to_improve(self):
        """
        Identify strategies that need improvement or removal
        
        Returns:
            tuple: (strategies_to_improve, strategies_to_remove)
        """
        if not self.strategies:
            logger.error("No strategies analyzed.")
            return [], []
            
        strategies_to_improve = []
        strategies_to_remove = []
        
        for strategy_name, metrics in self.strategies.items():
            # Calculate quality score
            quality_score = self.evaluate_strategy_quality(metrics)
            logger.info(f"Strategy {strategy_name} quality score: {quality_score:.4f}")
            
            # Check if strategy is profitable
            is_profitable = metrics['total_profit_loss'] >= PROFITABLE_THRESHOLD
            
            # Evaluate if strategy should be improved or removed
            if quality_score < 0.4 and not is_profitable:
                strategies_to_remove.append(strategy_name)
            elif quality_score < 0.7:
                strategies_to_improve.append(strategy_name)
        
        logger.info(f"Strategies to improve: {strategies_to_improve}")
        logger.info(f"Strategies to remove: {strategies_to_remove}")
        
        return strategies_to_improve, strategies_to_remove
    
    def generate_parameter_improvements(self, strategy_name):
        """
        Generate improved parameters for a strategy
        
        Args:
            strategy_name (str): Name of the strategy
            
        Returns:
            dict: Recommended parameter changes
        """
        if strategy_name not in self.parameter_ranges:
            logger.warning(f"No parameter ranges defined for {strategy_name}.")
            return {}
            
        if strategy_name not in self.strategies:
            logger.warning(f"No performance data for {strategy_name}.")
            return {}
            
        # Get current metrics
        current_metrics = self.strategies[strategy_name]
        
        # Determine parameter focus areas based on metrics
        focus_areas = self._determine_focus_areas(strategy_name, current_metrics)
        
        # Generate recommendations
        recommendations = {}
        
        # Custom logic for each strategy type
        if strategy_name == "ARIMAStrategy":
            self._recommend_arima_parameters(focus_areas, recommendations)
        elif strategy_name == "AdaptiveStrategy":
            self._recommend_adaptive_parameters(focus_areas, recommendations)
        elif strategy_name == "IntegratedStrategy":
            self._recommend_integrated_parameters(focus_areas, recommendations)
        
        return recommendations
    
    def _determine_focus_areas(self, strategy_name, metrics):
        """
        Determine which aspects of the strategy need improvement
        
        Args:
            strategy_name (str): Strategy name
            metrics (dict): Performance metrics
            
        Returns:
            dict: Focus areas for improvement
        """
        focus = {}
        
        # Check win rate
        if metrics['win_rate'] < 0.5:
            focus['signal_quality'] = True
        else:
            focus['signal_quality'] = False
        
        # Check risk management
        if metrics['max_drawdown'] < -0.15 or metrics['max_loss'] < -0.02:
            focus['risk_management'] = True
        else:
            focus['risk_management'] = False
        
        # Check trade frequency
        if metrics['trade_frequency'] < 0.2:
            focus['sensitivity'] = True
        else:
            focus['sensitivity'] = False
        
        return focus
    
    def _recommend_arima_parameters(self, focus_areas, recommendations):
        """
        Generate ARIMA strategy parameter recommendations
        
        Args:
            focus_areas (dict): Areas to focus on
            recommendations (dict): Recommendations to update
        """
        if focus_areas.get('signal_quality', False):
            recommendations['lookback_period'] = {"change": "increase", "range": [32, 48]}
            recommendations['arima_order'] = {"change": "adjust", "value": "(2,1,1)"}
        
        if focus_areas.get('risk_management', False):
            recommendations['atr_trailing_multiplier'] = {"change": "increase", "range": [2.5, 3.0]}
            recommendations['max_loss_percent'] = {"change": "decrease", "range": [3.0, 3.5]}
        
        if focus_areas.get('sensitivity', False):
            recommendations['entry_atr_multiplier'] = {"change": "decrease", "range": [0.005, 0.015]}
    
    def _recommend_adaptive_parameters(self, focus_areas, recommendations):
        """
        Generate Adaptive strategy parameter recommendations
        
        Args:
            focus_areas (dict): Areas to focus on
            recommendations (dict): Recommendations to update
        """
        if focus_areas.get('signal_quality', False):
            recommendations['rsi_period'] = {"change": "adjust", "range": [14, 18]}
            recommendations['ema_short'] = {"change": "decrease", "range": [7, 9]}
            recommendations['ema_long'] = {"change": "increase", "range": [21, 25]}
        
        if focus_areas.get('risk_management', False):
            recommendations['atr_period'] = {"change": "increase", "range": [16, 20]}
        
        if focus_areas.get('sensitivity', False):
            recommendations['volatility_threshold'] = {"change": "adjust", "range": [0.005, 0.008]}
    
    def _recommend_integrated_parameters(self, focus_areas, recommendations):
        """
        Generate Integrated strategy parameter recommendations
        
        Args:
            focus_areas (dict): Areas to focus on
            recommendations (dict): Recommendations to update
        """
        if focus_areas.get('signal_quality', False):
            recommendations['signal_smoothing'] = {"change": "increase", "range": [3, 5]}
            recommendations['trend_strength_threshold'] = {"change": "increase", "range": [0.4, 0.6]}
        
        if focus_areas.get('risk_management', False):
            recommendations['min_adx_threshold'] = {"change": "increase", "range": [25, 30]}
        
        if focus_areas.get('sensitivity', False):
            recommendations['volatility_filter_threshold'] = {"change": "adjust", "range": [0.006, 0.01]}
    
    def generate_strategy_improvements(self):
        """
        Generate improvements for all strategies that need enhancement
        
        Returns:
            dict: Improvements for each strategy
        """
        strategies_to_improve, strategies_to_remove = self.identify_strategies_to_improve()
        
        improvements = {
            "to_improve": {},
            "to_remove": strategies_to_remove
        }
        
        for strategy_name in strategies_to_improve:
            recommendations = self.generate_parameter_improvements(strategy_name)
            if recommendations:
                improvements["to_improve"][strategy_name] = recommendations
        
        return improvements
    
    def save_recommendations(self, improvements, filename="strategy_improvements.json"):
        """
        Save strategy improvement recommendations to file
        
        Args:
            improvements (dict): Strategy improvements
            filename (str): Output filename
            
        Returns:
            bool: Success status
        """
        filepath = os.path.join(RESULTS_DIR, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(improvements, f, indent=2)
            logger.info(f"Saved recommendations to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            return False
    
    def plot_strategy_performance(self, output_dir=RESULTS_DIR):
        """
        Plot performance metrics for all strategies
        
        Args:
            output_dir (str): Output directory for plots
            
        Returns:
            bool: Success status
        """
        if not self.strategies:
            logger.error("No strategies analyzed.")
            return False
            
        try:
            # Create plots directory
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot win rates
            plt.figure(figsize=(10, 6))
            strategies = list(self.strategies.keys())
            win_rates = [self.strategies[s]['win_rate'] for s in strategies]
            
            plt.bar(strategies, win_rates, color='lightblue')
            plt.title('Win Rate by Strategy')
            plt.ylabel('Win Rate')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(plots_dir, 'win_rates.png'))
            plt.close()
            
            # Plot profit/loss
            plt.figure(figsize=(10, 6))
            total_pnl = [self.strategies[s]['total_profit_loss'] for s in strategies]
            
            plt.bar(strategies, total_pnl, color='lightgreen')
            plt.title('Total Profit/Loss by Strategy')
            plt.ylabel('Profit/Loss')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(plots_dir, 'total_profit_loss.png'))
            plt.close()
            
            # Plot quality scores
            plt.figure(figsize=(10, 6))
            quality_scores = [self.evaluate_strategy_quality(self.strategies[s]) for s in strategies]
            
            plt.bar(strategies, quality_scores, color='purple')
            plt.title('Strategy Quality Scores')
            plt.ylabel('Quality Score (0-1)')
            plt.ylim(0, 1)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(plots_dir, 'quality_scores.png'))
            plt.close()
            
            logger.info(f"Strategy performance plots saved to {plots_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error plotting strategy performance: {e}")
            traceback.print_exc()
            return False
    
    def run_optimization(self):
        """
        Run the complete optimization process
        
        Returns:
            dict: Optimization results
        """
        # Load trades data
        self.load_trades()
        
        # Analyze strategy performance
        self.analyze_strategy_performance()
        
        # Generate improvement recommendations
        improvements = self.generate_strategy_improvements()
        
        # Plot performance
        self.plot_strategy_performance()
        
        # Save recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_recommendations(improvements, f"strategy_improvements_{timestamp}.json")
        
        return improvements


def main():
    """
    Main function to run the strategy optimizer
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Optimize trading strategies")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback period in days")
    parser.add_argument("--trades-file", type=str, default=TRADES_FILE, help="Path to trades CSV file")
    args = parser.parse_args()
    
    # Run optimization
    optimizer = StrategyOptimizer(trades_file=args.trades_file, lookback_days=args.lookback)
    improvements = optimizer.run_optimization()
    
    # Print summary
    print("\nStrategy Optimization Summary:")
    print("=============================")
    
    print("\nStrategies to Improve:")
    for strategy, params in improvements["to_improve"].items():
        print(f"  - {strategy}:")
        for param, value in params.items():
            print(f"    * {param}: {value}")
    
    print("\nStrategies to Remove:")
    for strategy in improvements["to_remove"]:
        print(f"  - {strategy}")
    
    print("\nSee 'optimization_results' directory for detailed information.")


if __name__ == "__main__":
    main()