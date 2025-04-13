#!/usr/bin/env python3
"""
Advanced Visualization Dashboard for ML Ensemble Backtests

This script generates comprehensive visualizations for the ML ensemble backtests,
including performance metrics, model comparisons, and market regime analysis.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/visualization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

class BacktestVisualizationDashboard:
    """
    Advanced visualization dashboard for ML ensemble backtests
    
    This class provides comprehensive visualizations of backtest results,
    including portfolio performance, model accuracy, and market regime analysis.
    """
    
    def __init__(self, results_dir='backtest_results/ml_ensemble', 
                output_dir='backtest_results/visualizations'):
        """
        Initialize the visualization dashboard
        
        Args:
            results_dir: Directory containing backtest result files
            output_dir: Directory to save visualization outputs
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set Seaborn style
        sns.set(style="darkgrid")
        
        # Custom color palette
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'neutral': '#95a5a6',
            'dark': '#2c3e50',
            'model_types': {
                'tcn': '#1abc9c',
                'cnn': '#3498db',
                'lstm': '#9b59b6',
                'gru': '#e74c3c',
                'bilstm': '#f39c12',
                'attention': '#2ecc71',
                'transformer': '#8e44ad',
                'hybrid': '#16a085'
            },
            'regimes': {
                'normal_trending_up': '#2ecc71',
                'normal_trending_down': '#95a5a6',
                'volatile_trending_up': '#f39c12',
                'volatile_trending_down': '#e74c3c'
            }
        }
    
    def _load_backtest_results(self):
        """
        Load all backtest result files from the results directory
        
        Returns:
            dict: Dictionary of backtest results by pair and timeframe
        """
        logger.info(f"Loading backtest results from {self.results_dir}")
        
        results = {}
        
        # List all result files
        result_files = [f for f in os.listdir(self.results_dir) 
                       if f.endswith('.txt') and 'comparative_summary' not in f]
        
        for filename in result_files:
            # Parse pair and timeframe from filename
            parts = filename.replace('.txt', '').split('_')
            if len(parts) >= 2:
                pair = parts[0]
                timeframe = parts[-1]
                
                # Format pair name with '/'
                if pair not in results:
                    results[pair] = {}
                
                # Load and parse the result file
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Parse metrics
                metrics = {}
                
                # Extract numerical metrics
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Try to convert value to float if it contains a number
                        if '%' in value:
                            # Handle percentage values
                            try:
                                value = float(value.split('%')[0].strip()) / 100
                                metrics[key] = value
                            except ValueError:
                                metrics[key] = value
                        elif '$' in value:
                            # Handle dollar values
                            try:
                                value = float(value.replace('$', '').strip())
                                metrics[key] = value
                            except ValueError:
                                metrics[key] = value
                        else:
                            # Handle other numeric values
                            try:
                                value = float(value)
                                metrics[key] = value
                            except ValueError:
                                metrics[key] = value
                
                # Extract model accuracies
                model_accuracies = {}
                in_model_section = False
                
                for line in content.split('\n'):
                    if line.strip() == "Individual Model Accuracies:":
                        in_model_section = True
                        continue
                    
                    if in_model_section and line.strip() and line.strip()[0] == ' ':
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            model_type = parts[0].strip()
                            accuracy_str = parts[1].strip()
                            
                            try:
                                # Extract accuracy percentage
                                accuracy = float(accuracy_str.split('%')[0].strip()) / 100
                                model_accuracies[model_type] = accuracy
                            except (ValueError, IndexError):
                                continue
                
                # Store parsed results
                results[pair][timeframe] = {
                    'metrics': metrics,
                    'model_accuracies': model_accuracies,
                    'filepath': filepath
                }
        
        logger.info(f"Loaded results for {len(results)} trading pairs")
        return results
    
    def create_performance_overview(self, results=None):
        """
        Create a performance overview visualization comparing different pairs and timeframes
        
        Args:
            results: Dictionary of backtest results (if None, will load from files)
        """
        if results is None:
            results = self._load_backtest_results()
        
        if not results:
            logger.warning("No backtest results found to visualize")
            return
        
        logger.info("Creating performance overview visualization")
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Extract data for visualization
        pairs = []
        timeframes = []
        returns = []
        sharpes = []
        win_rates = []
        drawdowns = []
        
        for pair, timeframe_results in results.items():
            for timeframe, result_data in timeframe_results.items():
                metrics = result_data.get('metrics', {})
                
                # Extract key metrics
                total_return = metrics.get('Total Return', 0)
                sharpe = metrics.get('Sharpe Ratio', 0)
                win_rate = metrics.get('Win Rate', 0)
                drawdown = metrics.get('Max Drawdown', 0)
                
                pairs.append(pair)
                timeframes.append(timeframe)
                returns.append(total_return)
                sharpes.append(sharpe)
                win_rates.append(win_rate)
                drawdowns.append(drawdown)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Pair': pairs,
            'Timeframe': timeframes,
            'Return': returns,
            'Sharpe': sharpes,
            'Win Rate': win_rates,
            'Drawdown': drawdowns
        })
        
        # Create pair/timeframe labels
        df['Label'] = df['Pair'] + ' (' + df['Timeframe'] + ')'
        
        # Sort by return
        df = df.sort_values('Return', ascending=False)
        
        # Plot 1: Total Returns
        plt.subplot(2, 2, 1)
        bars = plt.bar(df['Label'], df['Return'] * 100, color=self.colors['primary'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Total Return (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Return (%)')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Sharpe Ratios
        plt.subplot(2, 2, 2)
        bars = plt.bar(df['Label'], df['Sharpe'], color=self.colors['secondary'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Good (1.0)')
        plt.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Very Good (2.0)')
        plt.title('Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Sharpe Ratio')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Plot 3: Win Rates
        plt.subplot(2, 2, 3)
        bars = plt.bar(df['Label'], df['Win Rate'] * 100, color=self.colors['warning'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Break-even (50%)')
        plt.title('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Plot 4: Max Drawdowns
        plt.subplot(2, 2, 4)
        bars = plt.bar(df['Label'], df['Drawdown'] * 100, color=self.colors['danger'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Maximum Drawdown (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Drawdown (%)')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_overview.png'))
        logger.info(f"Saved performance overview to {os.path.join(self.output_dir, 'performance_overview.png')}")
        plt.close()
    
    def create_model_accuracy_comparison(self, results=None):
        """
        Create a visualization comparing accuracy across different model types
        
        Args:
            results: Dictionary of backtest results (if None, will load from files)
        """
        if results is None:
            results = self._load_backtest_results()
        
        if not results:
            logger.warning("No backtest results found to visualize")
            return
        
        logger.info("Creating model accuracy comparison visualization")
        
        # Collect model accuracies across all tests
        model_data = {}
        
        for pair, timeframe_results in results.items():
            for timeframe, result_data in timeframe_results.items():
                model_accuracies = result_data.get('model_accuracies', {})
                
                # Add each model's accuracy to the collected data
                for model_type, accuracy in model_accuracies.items():
                    if model_type not in model_data:
                        model_data[model_type] = []
                    
                    model_data[model_type].append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'accuracy': accuracy
                    })
        
        if not model_data:
            logger.warning("No model accuracy data found to visualize")
            return
        
        # Set up the figure
        plt.figure(figsize=(15, 12))
        
        # Create a GridSpec to organize subplots
        gs = gridspec.GridSpec(2, 2)
        
        # Plot 1: Average Accuracy by Model Type
        ax1 = plt.subplot(gs[0, 0])
        
        model_types = []
        avg_accuracies = []
        std_accuracies = []
        model_colors = []
        
        for model_type, accuracies in model_data.items():
            accuracy_values = [entry['accuracy'] for entry in accuracies]
            model_types.append(model_type)
            avg_accuracies.append(np.mean(accuracy_values))
            std_accuracies.append(np.std(accuracy_values))
            model_colors.append(self.colors['model_types'].get(model_type.lower(), '#333333'))
        
        # Sort by average accuracy
        sort_idx = np.argsort(avg_accuracies)[::-1]
        model_types = [model_types[i] for i in sort_idx]
        avg_accuracies = [avg_accuracies[i] for i in sort_idx]
        std_accuracies = [std_accuracies[i] for i in sort_idx]
        model_colors = [model_colors[i] for i in sort_idx]
        
        # Plot bars with error bars
        bars = ax1.bar(model_types, [acc * 100 for acc in avg_accuracies], 
                      yerr=[std * 100 for std in std_accuracies],
                      color=model_colors, alpha=0.7)
        
        # Add values on top of bars
        for bar, acc in zip(bars, avg_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc * 100:.1f}%', ha='center', va='bottom')
        
        ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Minimum (50%)')
        ax1.set_title('Average Model Accuracy by Type')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        ax1.legend()
        
        # Plot 2: Accuracy Distribution (Box Plot)
        ax2 = plt.subplot(gs[0, 1])
        
        # Prepare data for boxplot
        box_data = []
        for model_type in model_types:
            accuracy_values = [entry['accuracy'] * 100 for entry in model_data[model_type]]
            box_data.append(accuracy_values)
        
        # Create box plot
        bp = ax2.boxplot(box_data, patch_artist=True)
        
        # Customize box plot colors
        for i, box in enumerate(bp['boxes']):
            box.set(facecolor=model_colors[i], alpha=0.7)
        
        ax2.set_xticklabels(model_types)
        ax2.set_title('Accuracy Distribution by Model Type')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Best Model by Pair/Timeframe
        ax3 = plt.subplot(gs[1, 0])
        
        # Find best model for each pair/timeframe
        best_models = {}
        
        for pair, timeframe_results in results.items():
            for timeframe, result_data in timeframe_results.items():
                model_accuracies = result_data.get('model_accuracies', {})
                
                if model_accuracies:
                    best_model = max(model_accuracies.items(), key=lambda x: x[1])
                    label = f"{pair} ({timeframe})"
                    best_models[label] = {
                        'model': best_model[0],
                        'accuracy': best_model[1]
                    }
        
        # Sort by accuracy
        sorted_best = sorted(best_models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        labels = [item[0] for item in sorted_best]
        best_accuracies = [item[1]['accuracy'] * 100 for item in sorted_best]
        best_model_names = [item[1]['model'] for item in sorted_best]
        
        # Create colormap based on model type
        bar_colors = [self.colors['model_types'].get(model.lower(), '#333333') 
                     for model in best_model_names]
        
        # Plot horizontal bars
        bars = ax3.barh(labels, best_accuracies, color=bar_colors, alpha=0.7)
        
        # Add model names to the bars
        for i, (bar, model) in enumerate(zip(bars, best_model_names)):
            width = bar.get_width()
            ax3.text(width + 1, i, model, va='center')
        
        ax3.set_title('Best Performing Model by Pair/Timeframe')
        ax3.set_xlabel('Accuracy (%)')
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim(0, 100)
        
        # Plot 4: Heat map of model performance across pairs
        ax4 = plt.subplot(gs[1, 1])
        
        # Prepare data for heatmap
        pairs_timeframes = sorted(set([f"{pair} ({tf})" for pair in results.keys() 
                                    for tf in results[pair].keys()]))
        
        heatmap_data = np.zeros((len(model_types), len(pairs_timeframes)))
        
        for i, model_type in enumerate(model_types):
            for j, pt in enumerate(pairs_timeframes):
                pair, tf = pt.split(' (')
                tf = tf.replace(')', '')
                
                if pair in results and tf in results[pair]:
                    model_accuracies = results[pair][tf].get('model_accuracies', {})
                    if model_type in model_accuracies:
                        heatmap_data[i, j] = model_accuracies[model_type] * 100
        
        # Create heatmap
        im = ax4.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax4, label='Accuracy (%)')
        
        # Add labels
        ax4.set_xticks(np.arange(len(pairs_timeframes)))
        ax4.set_yticks(np.arange(len(model_types)))
        ax4.set_xticklabels(pairs_timeframes, rotation=45, ha='right')
        ax4.set_yticklabels(model_types)
        
        ax4.set_title('Model Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_accuracy_comparison.png'))
        logger.info(f"Saved model accuracy comparison to {os.path.join(self.output_dir, 'model_accuracy_comparison.png')}")
        plt.close()
    
    def create_dashboard(self):
        """Create and save all visualizations for the dashboard"""
        logger.info("Creating visualization dashboard")
        
        # Load results
        results = self._load_backtest_results()
        
        if not results:
            logger.warning("No backtest results found to visualize")
            return
        
        # Create visualizations
        self.create_performance_overview(results)
        self.create_model_accuracy_comparison(results)
        
        logger.info("Visualization dashboard creation complete")
        return os.path.abspath(self.output_dir)

def main():
    """Main function to create the visualization dashboard"""
    dashboard = BacktestVisualizationDashboard()
    output_dir = dashboard.create_dashboard()
    
    if output_dir:
        print(f"\nVisualization dashboard created successfully.")
        print(f"Output directory: {output_dir}")
    else:
        print("\nFailed to create visualization dashboard.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())