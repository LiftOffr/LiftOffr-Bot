#!/usr/bin/env python3
"""
Advanced Performance Metrics for Kraken Trading Bot

This module calculates various performance metrics for the trading strategies,
including risk-adjusted returns, drawdown analysis, and comparative benchmarks.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('performance_metrics.log'),
        logging.StreamHandler()
    ]
)

# Constants
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252
BENCHMARK_SYMBOL = "BTC/USD"  # Default benchmark
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class PerformanceMetrics:
    """Performance metrics calculator for trading strategies"""
    
    def __init__(self, trades_file='trades.csv', initial_capital=10000.0):
        """
        Initialize the performance metrics calculator
        
        Args:
            trades_file (str): Path to trades CSV file
            initial_capital (float): Initial capital amount
        """
        self.trades_file = trades_file
        self.initial_capital = initial_capital
        self.trades_df = None
        self.daily_returns = None
        self.benchmark_returns = None
        self.metrics = {}
    
    def load_data(self):
        """
        Load trades data from CSV file
        
        Returns:
            bool: Success indicator
        """
        try:
            if not os.path.exists(self.trades_file):
                logging.error(f"Trades file not found: {self.trades_file}")
                return False
            
            # Load trades data
            self.trades_df = pd.read_csv(self.trades_file)
            
            # Convert timestamp to datetime
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            
            # Sort by timestamp
            self.trades_df = self.trades_df.sort_values('timestamp')
            
            logging.info(f"Loaded {len(self.trades_df)} trades from {self.trades_file}")
            return True
        
        except Exception as e:
            logging.error(f"Error loading trades data: {str(e)}")
            return False
    
    def calculate_daily_returns(self):
        """
        Calculate daily portfolio returns
        
        Returns:
            pd.Series: Daily returns series
        """
        if self.trades_df is None:
            if not self.load_data():
                return None
        
        # Initialize portfolio value series
        portfolio_values = []
        dates = []
        
        # Start with initial capital
        current_value = self.initial_capital
        
        # Get unique dates
        unique_dates = self.trades_df['timestamp'].dt.date.unique()
        
        # Calculate portfolio value for each day
        for date in unique_dates:
            # Get trades for this day
            day_trades = self.trades_df[self.trades_df['timestamp'].dt.date == date]
            
            # Update portfolio value
            for _, trade in day_trades.iterrows():
                if 'pnl' in trade and not pd.isna(trade['pnl']):
                    current_value += trade['pnl']
            
            portfolio_values.append(current_value)
            dates.append(date)
        
        # Create portfolio value series
        portfolio_series = pd.Series(portfolio_values, index=pd.to_datetime(dates))
        
        # Calculate daily returns
        self.daily_returns = portfolio_series.pct_change().fillna(0)
        
        return self.daily_returns
    
    def calculate_core_metrics(self):
        """
        Calculate core performance metrics
        
        Returns:
            dict: Dictionary of performance metrics
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
            if self.daily_returns is None:
                return {}
        
        # Total return
        total_return = (self.daily_returns + 1).prod() - 1
        
        # Annualized return
        days = len(self.daily_returns)
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / days) - 1
        
        # Volatility (annualized)
        daily_std = self.daily_returns.std()
        annualized_std = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio
        excess_return = annualized_return - RISK_FREE_RATE
        sharpe_ratio = excess_return / annualized_std if annualized_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + self.daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Sortino ratio (downside risk)
        negative_returns = self.daily_returns[self.daily_returns < 0]
        downside_std = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Win rate
        if 'pnl' in self.trades_df.columns:
            winning_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
            total_trades = len(self.trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = None
        
        # Profit factor
        if 'pnl' in self.trades_df.columns:
            gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            profit_factor = None
        
        # Average trade
        if 'pnl' in self.trades_df.columns:
            average_trade = self.trades_df['pnl'].mean()
        else:
            average_trade = None
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_std,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_trade': average_trade,
            'number_of_trades': len(self.trades_df) if self.trades_df is not None else 0
        }
        
        return self.metrics
    
    def calculate_drawdown_metrics(self):
        """
        Calculate detailed drawdown metrics
        
        Returns:
            dict: Dictionary of drawdown metrics
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
            if self.daily_returns is None:
                return {}
        
        # Calculate drawdown series
        cumulative_returns = (1 + self.daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown_series = (cumulative_returns / running_max) - 1
        
        # Maximum drawdown
        max_drawdown = drawdown_series.min()
        max_drawdown_date = drawdown_series.idxmin()
        
        # Find start of drawdown period
        if max_drawdown < 0:
            # Find the last time the drawdown was 0 before the max drawdown
            drawdown_start = drawdown_series[:max_drawdown_date]
            drawdown_start = drawdown_start[drawdown_start >= 0].index[-1] if len(drawdown_start[drawdown_start >= 0]) > 0 else drawdown_series.index[0]
            
            # Find the recovery date (if any)
            drawdown_end = drawdown_series[max_drawdown_date:]
            recovered_points = drawdown_end[drawdown_end >= 0]
            recovery_date = recovered_points.index[0] if len(recovered_points) > 0 else None
            
            # Calculate drawdown duration
            drawdown_duration = (max_drawdown_date - drawdown_start).days
            
            # Calculate recovery duration
            if recovery_date is not None:
                recovery_duration = (recovery_date - max_drawdown_date).days
            else:
                recovery_duration = None
        else:
            drawdown_start = None
            recovery_date = None
            drawdown_duration = 0
            recovery_duration = 0
        
        # Calculate underwater periods (drawdowns > 10%)
        underwater_periods = []
        current_underwater = False
        underwater_start = None
        
        for date, value in drawdown_series.items():
            if not current_underwater and value <= -0.1:
                # Start of underwater period
                current_underwater = True
                underwater_start = date
            elif current_underwater and value > -0.05:
                # End of underwater period (recovery to -5%)
                current_underwater = False
                underwater_periods.append({
                    'start': underwater_start,
                    'end': date,
                    'duration': (date - underwater_start).days,
                    'max_drawdown': drawdown_series[underwater_start:date].min()
                })
                underwater_start = None
        
        # If still underwater at the end
        if current_underwater:
            underwater_periods.append({
                'start': underwater_start,
                'end': drawdown_series.index[-1],
                'duration': (drawdown_series.index[-1] - underwater_start).days,
                'max_drawdown': drawdown_series[underwater_start:].min()
            })
        
        # Store drawdown metrics
        drawdown_metrics = {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'drawdown_start': drawdown_start,
            'recovery_date': recovery_date,
            'drawdown_duration': drawdown_duration,
            'recovery_duration': recovery_duration,
            'underwater_periods': underwater_periods,
            'drawdown_series': drawdown_series
        }
        
        self.metrics.update({'drawdown': drawdown_metrics})
        
        return drawdown_metrics
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot equity curve with drawdowns
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Figure object
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
            if self.daily_returns is None:
                return None
        
        if 'drawdown' not in self.metrics:
            self.calculate_drawdown_metrics()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.daily_returns).cumprod()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=2)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value (normalized)')
        ax1.grid(True)
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot drawdowns
        drawdown_series = self.metrics['drawdown']['drawdown_series']
        ax2.fill_between(drawdown_series.index, 0, drawdown_series.values, facecolor='red', alpha=0.3)
        ax2.set_title('Drawdowns')
        ax2.set_ylabel('Drawdown')
        ax2.set_ylim(min(drawdown_series.min() * 1.1, -0.05), 0.01)
        ax2.grid(True)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_monthly_returns(self, save_path=None):
        """
        Plot monthly returns heatmap
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Figure object
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
            if self.daily_returns is None:
                return None
        
        # Calculate monthly returns
        monthly_returns = self.daily_returns.groupby([
            self.daily_returns.index.year,
            self.daily_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table
        monthly_returns_pivot = pd.DataFrame(monthly_returns)
        monthly_returns_pivot.columns = ['Returns']
        monthly_returns_pivot.index.names = ['Year', 'Month']
        monthly_returns_pivot = monthly_returns_pivot.reset_index()
        monthly_returns_pivot = monthly_returns_pivot.pivot(index='Year', columns='Month', values='Returns')
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot.columns = [month_names[i-1] for i in monthly_returns_pivot.columns]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot heatmap
        cmap = plt.cm.RdYlGn
        im = ax.imshow(monthly_returns_pivot.values, cmap=cmap, vmin=-0.1, vmax=0.1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Returns')
        
        # Configure axes
        ax.set_xticks(np.arange(len(monthly_returns_pivot.columns)))
        ax.set_yticks(np.arange(len(monthly_returns_pivot.index)))
        ax.set_xticklabels(monthly_returns_pivot.columns)
        ax.set_yticklabels(monthly_returns_pivot.index)
        
        # Add values to cells
        for i in range(len(monthly_returns_pivot.index)):
            for j in range(len(monthly_returns_pivot.columns)):
                value = monthly_returns_pivot.iloc[i, j]
                if pd.notna(value):
                    text_color = 'white' if abs(value) > 0.05 else 'black'
                    ax.text(j, i, f'{value:.1%}',
                            ha="center", va="center", color=text_color)
        
        ax.set_title('Monthly Returns')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_strategy_comparison(self, other_returns=None, names=None, save_path=None):
        """
        Plot strategy comparison
        
        Args:
            other_returns (list): List of other return series to compare
            names (list): List of strategy names
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Figure object
        """
        if self.daily_returns is None:
            self.calculate_daily_returns()
            if self.daily_returns is None:
                return None
        
        # Calculate cumulative returns
        cumulative_returns = (1 + self.daily_returns).cumprod()
        
        if other_returns is None:
            other_returns = []
        
        if names is None:
            names = ['Strategy'] + [f'Comparison {i+1}' for i in range(len(other_returns))]
        else:
            names = [names[0]] + names[1:len(other_returns)+1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot main strategy
        ax.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, label=names[0])
        
        # Plot comparison strategies
        for i, returns in enumerate(other_returns):
            if returns is not None:
                comp_cumulative = (1 + returns).cumprod()
                ax.plot(comp_cumulative.index, comp_cumulative.values, 
                       linewidth=2, linestyle='--', label=names[i+1])
        
        ax.set_title('Strategy Comparison')
        ax.set_ylabel('Cumulative Returns')
        ax.grid(True)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def generate_report(self, output_file=None, include_plots=True):
        """
        Generate a comprehensive performance report
        
        Args:
            output_file (str): Path to output file
            include_plots (bool): Whether to include plots in the report
            
        Returns:
            str: Report text
        """
        # Calculate metrics if not already done
        if not self.metrics:
            self.calculate_core_metrics()
            self.calculate_drawdown_metrics()
        
        # Generate report text
        report = []
        report.append("=" * 80)
        report.append(f"PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("OVERVIEW:")
        report.append(f"Initial Capital: ${self.initial_capital:.2f}")
        report.append(f"Total Return: {self.metrics['total_return']:.2%}")
        report.append(f"Annualized Return: {self.metrics['annualized_return']:.2%}")
        report.append(f"Number of Trades: {self.metrics['number_of_trades']}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        report.append(f"Annualized Volatility: {self.metrics['annualized_volatility']:.2%}")
        report.append(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        report.append(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        report.append("")
        
        # Trade Statistics
        if self.metrics['win_rate'] is not None:
            report.append("TRADE STATISTICS:")
            report.append(f"Win Rate: {self.metrics['win_rate']:.2%}")
            report.append(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
            report.append(f"Average Trade: {self.metrics['average_trade']:.2f}")
            report.append("")
        
        # Drawdown Analysis
        report.append("DRAWDOWN ANALYSIS:")
        report.append(f"Maximum Drawdown: {self.metrics['drawdown']['max_drawdown']:.2%}")
        report.append(f"Drawdown Date: {self.metrics['drawdown']['max_drawdown_date'].strftime('%Y-%m-%d')}")
        
        if self.metrics['drawdown']['drawdown_start']:
            report.append(f"Drawdown Start: {self.metrics['drawdown']['drawdown_start'].strftime('%Y-%m-%d')}")
            report.append(f"Drawdown Duration: {self.metrics['drawdown']['drawdown_duration']} days")
        
        if self.metrics['drawdown']['recovery_date']:
            report.append(f"Recovery Date: {self.metrics['drawdown']['recovery_date'].strftime('%Y-%m-%d')}")
            report.append(f"Recovery Duration: {self.metrics['drawdown']['recovery_duration']} days")
        
        # Underwater periods
        if self.metrics['drawdown']['underwater_periods']:
            report.append("")
            report.append("SIGNIFICANT DRAWDOWNS (>10%):")
            for i, period in enumerate(self.metrics['drawdown']['underwater_periods']):
                report.append(f"  {i+1}. {period['start'].strftime('%Y-%m-%d')} to {period['end'].strftime('%Y-%m-%d')}")
                report.append(f"     Duration: {period['duration']} days")
                report.append(f"     Maximum Drawdown: {period['max_drawdown']:.2%}")
        
        report.append("")
        report.append("=" * 80)
        
        # Include plots if requested
        if include_plots:
            report.append("PLOTS:")
            equity_curve_path = os.path.join(PLOTS_DIR, 'equity_curve.png')
            monthly_returns_path = os.path.join(PLOTS_DIR, 'monthly_returns.png')
            
            self.plot_equity_curve(save_path=equity_curve_path)
            self.plot_monthly_returns(save_path=monthly_returns_path)
            
            report.append(f"Equity Curve: {equity_curve_path}")
            report.append(f"Monthly Returns: {monthly_returns_path}")
            report.append("")
            report.append("=" * 80)
        
        # Save report if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    @staticmethod
    def load_benchmark_returns(symbol=BENCHMARK_SYMBOL, start_date=None, end_date=None):
        """
        Load benchmark returns from Kraken API or file
        
        Args:
            symbol (str): Symbol for benchmark
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pd.Series: Benchmark returns
        """
        # Try to load from trades.csv
        try:
            # Implement based on available data sources
            # For now, return dummy data
            # TODO: Implement actual benchmark loading
            logging.warning("Benchmark loading not implemented. Using dummy data.")
            
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create dummy returns (random walk)
            np.random.seed(42)  # For reproducibility
            daily_changes = np.random.normal(0.0005, 0.02, size=len(date_range))
            returns = pd.Series(daily_changes, index=date_range)
            
            return returns
        
        except Exception as e:
            logging.error(f"Error loading benchmark returns: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Calculate performance metrics for trading strategy')
    parser.add_argument('--trades', type=str, default='trades.csv', help='Path to trades CSV file')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital amount')
    parser.add_argument('--output', type=str, help='Path to output report file')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Initialize performance metrics calculator
    perf = PerformanceMetrics(trades_file=args.trades, initial_capital=args.capital)
    
    # Generate report
    report = perf.generate_report(output_file=args.output, include_plots=not args.no_plots)
    
    # Print report
    print(report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())