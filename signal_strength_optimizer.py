#!/usr/bin/env python3
"""
Signal Strength Optimizer for Trading Bot

This module provides functionality to analyze and optimize signal strength
parameters for the trading strategies based on historical performance.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('signal_strength_optimizer.log'),
        logging.StreamHandler()
    ]
)

# Constants
DEFAULT_CONFIG_FILE = 'config.py'
SIGNAL_MARKERS = ["【SIGNAL】", "【ACTION】", "【INTEGRATED】"]
RESULTS_DIR = 'optimization_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


class SignalStrengthOptimizer:
    """Optimizer for signal strength parameters in trading strategies"""
    
    def __init__(self, trades_file='trades.csv', log_file='integrated_strategy_log.txt'):
        """
        Initialize the signal strength optimizer
        
        Args:
            trades_file (str): Path to trades CSV file
            log_file (str): Path to strategy log file
        """
        self.trades_file = trades_file
        self.log_file = log_file
        self.trades_df = None
        self.signal_data = None
        self.current_params = {}
        self.best_params = {}
        self.optimization_results = []
    
    def load_data(self):
        """
        Load trades and signal data
        
        Returns:
            bool: Success indicator
        """
        success = True
        
        # Load trades data
        try:
            if os.path.exists(self.trades_file):
                self.trades_df = pd.read_csv(self.trades_file)
                
                # Convert timestamp to datetime
                self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
                
                # Sort by timestamp
                self.trades_df = self.trades_df.sort_values('timestamp')
                
                logging.info(f"Loaded {len(self.trades_df)} trades from {self.trades_file}")
            else:
                logging.warning(f"Trades file not found: {self.trades_file}")
                success = False
        except Exception as e:
            logging.error(f"Error loading trades data: {str(e)}")
            success = False
        
        # Extract signal data from logs
        try:
            if os.path.exists(self.log_file):
                self.signal_data = self._extract_signal_data()
                logging.info(f"Extracted {len(self.signal_data)} signal records from {self.log_file}")
            else:
                logging.warning(f"Log file not found: {self.log_file}")
                success = False
        except Exception as e:
            logging.error(f"Error extracting signal data: {str(e)}")
            success = False
        
        return success
    
    def _extract_signal_data(self):
        """
        Extract signal data from log file
        
        Returns:
            list: List of signal data records
        """
        signals = []
        current_signal = {}
        
        with open(self.log_file, 'r') as f:
            for line in f:
                # Check if line contains signal information
                if any(marker in line for marker in SIGNAL_MARKERS):
                    # Extract timestamp
                    try:
                        timestamp_str = line.split('[INFO]')[0].strip()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        timestamp = None
                    
                    # Extract signal direction and strength
                    if "【SIGNAL】" in line:
                        if "BULLISH" in line:
                            current_signal = {
                                'timestamp': timestamp,
                                'direction': 'BUY',
                                'strength': None
                            }
                        elif "BEARISH" in line:
                            current_signal = {
                                'timestamp': timestamp,
                                'direction': 'SELL',
                                'strength': None
                            }
                        elif "NEUTRAL" in line:
                            current_signal = {
                                'timestamp': timestamp,
                                'direction': 'NEUTRAL',
                                'strength': None
                            }
                    
                    # Extract action taken
                    elif "【ACTION】" in line:
                        if timestamp and current_signal and 'timestamp' in current_signal:
                            if abs((timestamp - current_signal['timestamp']).total_seconds()) < 10:
                                if "BUY" in line:
                                    current_signal['action'] = 'BUY'
                                elif "SELL" in line:
                                    current_signal['action'] = 'SELL'
                    
                    # Extract signal strength
                    elif "【INTEGRATED】" in line and "Final Signal Strength" in line:
                        if current_signal:
                            try:
                                # Extract strength value
                                strength_text = line.split("Final Signal Strength:")[1].split("(")[0].strip()
                                current_signal['strength'] = float(strength_text)
                                
                                # Add completed signal to list
                                if 'action' in current_signal:
                                    signals.append(current_signal.copy())
                                    current_signal = {}
                            except:
                                pass
        
        return signals
    
    def load_current_parameters(self, config_file=DEFAULT_CONFIG_FILE):
        """
        Load current signal strength parameters from config file
        
        Args:
            config_file (str): Path to config file
            
        Returns:
            dict: Current parameters
        """
        # Default parameters
        params = {
            'MIN_SIGNAL_STRENGTH': 0.65,
            'SIGNAL_STRENGTH_ADVANTAGE': 0.25,
            'EMA_WEIGHT': 1.0,
            'RSI_WEIGHT': 1.2,
            'MACD_WEIGHT': 1.0,
            'ADX_WEIGHT': 0.8,
            'ARIMA_WEIGHT': 1.5,
            'VOLATILITY_PENALTY': 0.5
        }
        
        # Try to extract from config file
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Extract parameters using regex or simple string parsing
                for key in params.keys():
                    if key in content:
                        try:
                            value_str = content.split(f"{key} = ")[1].split("\n")[0].strip()
                            # Handle comments
                            if '#' in value_str:
                                value_str = value_str.split('#')[0].strip()
                            # Remove trailing commas
                            if value_str.endswith(','):
                                value_str = value_str[:-1].strip()
                            
                            # Convert to float
                            params[key] = float(value_str)
                        except:
                            pass
                
                logging.info(f"Loaded current parameters from {config_file}")
            else:
                logging.warning(f"Config file not found: {config_file}")
        except Exception as e:
            logging.error(f"Error loading parameters: {str(e)}")
        
        self.current_params = params
        return params
    
    def evaluate_parameters(self, params, min_trades=10):
        """
        Evaluate a set of signal strength parameters
        
        Args:
            params (dict): Signal strength parameters
            min_trades (int): Minimum number of trades required for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        if self.trades_df is None or self.signal_data is None:
            if not self.load_data():
                return {'error': 'Failed to load data'}
        
        # Apply parameters to filter signals
        filtered_signals = []
        for signal in self.signal_data:
            if signal.get('strength') is not None:
                # Check if signal strength exceeds threshold
                if signal['strength'] >= params['MIN_SIGNAL_STRENGTH']:
                    filtered_signals.append(signal)
        
        # Check if we have enough filtered signals
        if len(filtered_signals) < min_trades:
            return {
                'success': False,
                'error': f'Not enough filtered signals: {len(filtered_signals)} < {min_trades}',
                'num_signals': len(filtered_signals),
                'params': params
            }
        
        # Calculate metrics based on trades that would have been executed
        # This is a simplified evaluation - in a real implementation, we would
        # simulate the exact trading logic with the given parameters
        
        # Match signals to trades using timestamps
        matched_trades = []
        for signal in filtered_signals:
            if 'timestamp' in signal:
                # Find trades within a reasonable time window
                signal_time = signal['timestamp']
                
                # Look for trades within 5 minutes of the signal
                window_start = signal_time - timedelta(minutes=5)
                window_end = signal_time + timedelta(minutes=5)
                
                matching_trades = self.trades_df[
                    (self.trades_df['timestamp'] >= window_start) & 
                    (self.trades_df['timestamp'] <= window_end)
                ]
                
                if len(matching_trades) > 0:
                    # Use the closest trade
                    matching_trades['time_diff'] = (matching_trades['timestamp'] - signal_time).abs()
                    closest_trade = matching_trades.sort_values('time_diff').iloc[0]
                    
                    # Match direction
                    if (signal['direction'] == 'BUY' and closest_trade['side'] == 'buy') or \
                       (signal['direction'] == 'SELL' and closest_trade['side'] == 'sell'):
                        matched_trades.append({
                            'signal': signal,
                            'trade': closest_trade.to_dict()
                        })
        
        # Calculate metrics
        if len(matched_trades) < min_trades:
            return {
                'success': False,
                'error': f'Not enough matched trades: {len(matched_trades)} < {min_trades}',
                'num_trades': len(matched_trades),
                'params': params
            }
        
        # Calculate metrics based on matched trades
        total_pnl = sum(trade['trade'].get('pnl', 0) for trade in matched_trades if 'pnl' in trade['trade'])
        winning_trades = sum(1 for trade in matched_trades if trade['trade'].get('pnl', 0) > 0)
        win_rate = winning_trades / len(matched_trades)
        
        # Calculate average PnL per trade
        if 'pnl' in matched_trades[0]['trade']:
            avg_pnl = total_pnl / len(matched_trades)
        else:
            avg_pnl = None
        
        # Calculate signal strength correlation with trade success
        if 'pnl' in matched_trades[0]['trade']:
            strengths = [trade['signal']['strength'] for trade in matched_trades]
            pnls = [trade['trade']['pnl'] for trade in matched_trades]
            
            correlation = np.corrcoef(strengths, pnls)[0, 1] if len(strengths) > 1 else 0
        else:
            correlation = None
        
        metrics = {
            'success': True,
            'num_trades': len(matched_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl if avg_pnl is not None else None,
            'avg_pnl': avg_pnl,
            'strength_correlation': correlation,
            'params': params,
            'objective': win_rate if avg_pnl is None else (win_rate * 0.4 + total_pnl * 0.6)
        }
        
        return metrics
    
    def optimize_parameters(self, num_iterations=100, min_trades=10):
        """
        Optimize signal strength parameters
        
        Args:
            num_iterations (int): Number of optimization iterations
            min_trades (int): Minimum number of trades required for evaluation
            
        Returns:
            dict: Best parameters
        """
        # Load current parameters
        self.load_current_parameters()
        
        # Define parameter bounds
        param_bounds = {
            'MIN_SIGNAL_STRENGTH': (0.5, 0.9),
            'SIGNAL_STRENGTH_ADVANTAGE': (0.1, 0.5),
            'EMA_WEIGHT': (0.5, 2.0),
            'RSI_WEIGHT': (0.5, 2.0),
            'MACD_WEIGHT': (0.5, 2.0),
            'ADX_WEIGHT': (0.5, 2.0),
            'ARIMA_WEIGHT': (0.5, 2.0),
            'VOLATILITY_PENALTY': (0.1, 1.0)
        }
        
        # Store optimization results
        self.optimization_results = []
        
        # Evaluate current parameters
        current_metrics = self.evaluate_parameters(self.current_params, min_trades)
        if current_metrics.get('success', False):
            self.optimization_results.append(current_metrics)
            self.best_params = self.current_params.copy()
            best_objective = current_metrics['objective']
            logging.info(f"Current parameters objective: {best_objective:.4f}")
        else:
            logging.warning("Current parameters could not be evaluated")
            self.best_params = self.current_params.copy()
            best_objective = float('-inf')
        
        # Run optimization iterations
        for i in range(num_iterations):
            # Generate new parameters
            new_params = self._generate_parameters(param_bounds)
            
            # Evaluate new parameters
            metrics = self.evaluate_parameters(new_params, min_trades)
            
            # Store results
            if metrics.get('success', False):
                self.optimization_results.append(metrics)
                
                # Update best parameters
                if metrics['objective'] > best_objective:
                    best_objective = metrics['objective']
                    self.best_params = new_params.copy()
                    logging.info(f"Iteration {i+1}/{num_iterations}: New best objective: {best_objective:.4f}")
            
            # Log progress
            if (i+1) % 10 == 0:
                logging.info(f"Completed {i+1}/{num_iterations} iterations")
        
        # Sort results by objective
        self.optimization_results.sort(key=lambda x: x.get('objective', float('-inf')), reverse=True)
        
        logging.info("Optimization completed")
        logging.info(f"Best parameters: {json.dumps(self.best_params, indent=2)}")
        logging.info(f"Best objective: {best_objective:.4f}")
        
        return self.best_params
    
    def _generate_parameters(self, bounds):
        """
        Generate new parameters within bounds
        
        Args:
            bounds (dict): Parameter bounds
            
        Returns:
            dict: New parameters
        """
        # Either perform random sampling or use more sophisticated methods like
        # Bayesian optimization. For simplicity, we'll use random sampling here.
        
        new_params = {}
        for param, (min_val, max_val) in bounds.items():
            # Generate random value within bounds
            value = min_val + np.random.random() * (max_val - min_val)
            new_params[param] = value
        
        return new_params
    
    def plot_optimization_results(self, save_path=None):
        """
        Plot optimization results
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Figure object
        """
        if not self.optimization_results:
            logging.warning("No optimization results to plot")
            return None
        
        # Extract data
        objectives = [result['objective'] for result in self.optimization_results 
                    if 'objective' in result and result.get('success', False)]
        iterations = range(1, len(objectives) + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot objectives
        ax.plot(iterations, objectives, 'b-', linewidth=1)
        ax.plot(iterations, objectives, 'ro', markersize=4)
        
        # Plot best objective
        best_idx = np.argmax(objectives)
        ax.plot(best_idx + 1, objectives[best_idx], 'go', markersize=10, label='Best Result')
        
        ax.set_title('Signal Strength Parameter Optimization')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_parameter_importance(self, save_path=None):
        """
        Plot parameter importance
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Figure object
        """
        if not self.optimization_results:
            logging.warning("No optimization results to plot")
            return None
        
        # Extract successful results
        successful_results = [r for r in self.optimization_results if r.get('success', False)]
        if not successful_results:
            logging.warning("No successful optimization results to plot")
            return None
        
        # Create correlation dataframe
        params_data = []
        for result in successful_results:
            row = result['params'].copy()
            row['objective'] = result['objective']
            params_data.append(row)
        
        params_df = pd.DataFrame(params_data)
        
        # Calculate correlations
        correlations = {}
        for param in self.best_params.keys():
            if param in params_df.columns:
                correlations[param] = np.corrcoef(params_df[param], params_df['objective'])[0, 1]
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot correlations
        params = [item[0] for item in sorted_correlations]
        corr_values = [item[1] for item in sorted_correlations]
        
        colors = ['g' if c > 0 else 'r' for c in corr_values]
        
        ax.barh(params, corr_values, color=colors)
        ax.set_title('Parameter Importance (Correlation with Objective)')
        ax.set_xlabel('Correlation Coefficient')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def generate_parameter_report(self, output_file=None):
        """
        Generate a report of the optimization results
        
        Args:
            output_file (str): Path to output file
            
        Returns:
            str: Report text
        """
        if not self.optimization_results:
            return "No optimization results available."
        
        # Generate report text
        report = []
        report.append("=" * 80)
        report.append(f"SIGNAL STRENGTH OPTIMIZATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Current parameters
        report.append("CURRENT PARAMETERS:")
        for param, value in self.current_params.items():
            report.append(f"  {param}: {value:.4f}")
        
        # Evaluation of current parameters
        current_eval = next((r for r in self.optimization_results 
                             if all(r['params'][k] == self.current_params[k] 
                                    for k in self.current_params)), None)
        
        if current_eval and current_eval.get('success', False):
            report.append("\nCURRENT PERFORMANCE:")
            report.append(f"  Win Rate: {current_eval['win_rate']:.2%}")
            if current_eval.get('total_pnl') is not None:
                report.append(f"  Total PnL: {current_eval['total_pnl']:.2f}")
            if current_eval.get('avg_pnl') is not None:
                report.append(f"  Average PnL: {current_eval['avg_pnl']:.2f}")
            report.append(f"  Objective: {current_eval['objective']:.4f}")
        
        report.append("\nOPTIMIZED PARAMETERS:")
        for param, value in self.best_params.items():
            report.append(f"  {param}: {value:.4f}")
        
        # Evaluation of best parameters
        best_eval = next((r for r in self.optimization_results 
                          if all(r['params'][k] == self.best_params[k] 
                                 for k in self.best_params)), None)
        
        if best_eval and best_eval.get('success', False):
            report.append("\nOPTIMIZED PERFORMANCE:")
            report.append(f"  Win Rate: {best_eval['win_rate']:.2%}")
            if best_eval.get('total_pnl') is not None:
                report.append(f"  Total PnL: {best_eval['total_pnl']:.2f}")
            if best_eval.get('avg_pnl') is not None:
                report.append(f"  Average PnL: {best_eval['avg_pnl']:.2f}")
            report.append(f"  Objective: {best_eval['objective']:.4f}")
        
        # Improvement
        if current_eval and best_eval and current_eval.get('success', False) and best_eval.get('success', False):
            improvement = best_eval['objective'] - current_eval['objective']
            report.append(f"\nIMPROVEMENT: {improvement:.4f} ({improvement/current_eval['objective']*100:.2f}%)")
        
        # Parameter importance
        report.append("\nPARAMETER IMPORTANCE:")
        
        # Extract successful results
        successful_results = [r for r in self.optimization_results if r.get('success', False)]
        if successful_results:
            # Create correlation dataframe
            params_data = []
            for result in successful_results:
                row = result['params'].copy()
                row['objective'] = result['objective']
                params_data.append(row)
            
            params_df = pd.DataFrame(params_data)
            
            # Calculate correlations
            correlations = {}
            for param in self.best_params.keys():
                if param in params_df.columns:
                    correlations[param] = np.corrcoef(params_df[param], params_df['objective'])[0, 1]
            
            # Sort by absolute correlation
            sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for param, corr in sorted_correlations:
                report.append(f"  {param}: {corr:.4f}")
        
        # Implementation instructions
        report.append("\nIMPLEMENTATION INSTRUCTIONS:")
        report.append("To implement the optimized parameters, update the following values in config.py:")
        report.append("")
        for param, value in self.best_params.items():
            report.append(f"{param} = {value:.4f}")
        
        report.append("\nPLOTS:")
        
        # Generate and save plots
        plots_dir = os.path.join(RESULTS_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        optimization_plot_path = os.path.join(plots_dir, 'optimization_results.png')
        importance_plot_path = os.path.join(plots_dir, 'parameter_importance.png')
        
        self.plot_optimization_results(save_path=optimization_plot_path)
        self.plot_parameter_importance(save_path=importance_plot_path)
        
        report.append(f"  Optimization Results: {optimization_plot_path}")
        report.append(f"  Parameter Importance: {importance_plot_path}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report))
        
        return '\n'.join(report)
    

def main():
    parser = argparse.ArgumentParser(description='Optimize signal strength parameters for trading bot')
    parser.add_argument('--trades', type=str, default='trades.csv', help='Path to trades CSV file')
    parser.add_argument('--log', type=str, default='integrated_strategy_log.txt', help='Path to strategy log file')
    parser.add_argument('--config', type=str, default='config.py', help='Path to config file')
    parser.add_argument('--iterations', type=int, default=100, help='Number of optimization iterations')
    parser.add_argument('--min-trades', type=int, default=10, help='Minimum number of trades required for evaluation')
    parser.add_argument('--output', type=str, help='Path to output report file')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = SignalStrengthOptimizer(trades_file=args.trades, log_file=args.log)
    
    # Load current parameters
    optimizer.load_current_parameters(args.config)
    
    # Optimize parameters
    optimizer.optimize_parameters(num_iterations=args.iterations, min_trades=args.min_trades)
    
    # Generate report
    output_file = args.output or os.path.join(RESULTS_DIR, f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    report = optimizer.generate_parameter_report(output_file=output_file)
    
    # Print report
    print(report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())