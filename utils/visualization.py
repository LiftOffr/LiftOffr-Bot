#!/usr/bin/env python3
"""
Visualization Tools for Trading Optimization

This module provides visualization tools for trading optimization results,
including backtest comparisons, parameter optimization, and market regime detection.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('dark_background')

def plot_optimization_results(results: Dict[str, Any], output_dir: str = "optimization_results"):
    """
    Create comprehensive plots for optimization results.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save plots
    """
    logger.info("Creating optimization result plots")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    plot_returns_comparison(results, output_dir)
    plot_parameter_impact(results, output_dir)
    plot_win_rates(results, output_dir)
    
    logger.info(f"Saved optimization plots to {output_dir}")

def plot_returns_comparison(results: Dict[str, Any], output_dir: str):
    """
    Plot comparison of returns for different trading pairs.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save plots
    """
    # Extract returns for each pair
    pairs = []
    returns = []
    
    for pair, pair_results in results.get("detailed_results", {}).items():
        validation = pair_results.get("validation_results", {})
        total_return = validation.get("total_return", 0)
        
        pairs.append(pair)
        returns.append(total_return)
    
    if not pairs:
        logger.warning("No data available for returns comparison plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(pairs, returns, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', color='white')
    
    # Add styling
    plt.title('Total Returns by Trading Pair', fontsize=16)
    plt.xlabel('Trading Pair', fontsize=14)
    plt.ylabel('Total Return', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "returns_comparison.png"), dpi=300)
    plt.close()

def plot_parameter_impact(results: Dict[str, Any], output_dir: str):
    """
    Plot impact of parameters on performance.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save plots
    """
    # Extract parameter values and returns
    risk_percentages = []
    leverage_values = []
    confidence_thresholds = []
    returns = []
    
    for pair, pair_results in results.get("detailed_results", {}).items():
        # Get parameter values
        params = pair_results.get("optimized_parameters", {})
        validation = pair_results.get("validation_results", {})
        
        if params and validation:
            risk_percentages.append(params.get("risk_percentage", 0))
            leverage_values.append(params.get("base_leverage", 0))
            confidence_thresholds.append(params.get("confidence_threshold", 0))
            returns.append(validation.get("total_return", 0))
    
    if not risk_percentages:
        logger.warning("No data available for parameter impact plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot risk percentage vs. return
    axes[0].scatter(risk_percentages, returns, c='skyblue', alpha=0.8, s=100)
    axes[0].set_title('Risk Percentage vs. Return', fontsize=14)
    axes[0].set_xlabel('Risk Percentage', fontsize=12)
    axes[0].set_ylabel('Total Return', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot leverage vs. return
    axes[1].scatter(leverage_values, returns, c='salmon', alpha=0.8, s=100)
    axes[1].set_title('Leverage vs. Return', fontsize=14)
    axes[1].set_xlabel('Base Leverage', fontsize=12)
    axes[1].set_ylabel('Total Return', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Plot confidence threshold vs. return
    axes[2].scatter(confidence_thresholds, returns, c='lightgreen', alpha=0.8, s=100)
    axes[2].set_title('Confidence Threshold vs. Return', fontsize=14)
    axes[2].set_xlabel('Confidence Threshold', fontsize=12)
    axes[2].set_ylabel('Total Return', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Add trend lines
    for i, (x, param_name) in enumerate([(risk_percentages, 'Risk Percentage'),
                                       (leverage_values, 'Base Leverage'),
                                       (confidence_thresholds, 'Confidence Threshold')]):
        if len(x) > 1:  # Need at least 2 points for a trend line
            try:
                z = np.polyfit(x, returns, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(x), max(x), 100)
                axes[i].plot(x_range, p(x_range), '--', color='white', alpha=0.8)
                
                # Add correlation coefficient
                corr = np.corrcoef(x, returns)[0, 1]
                axes[i].annotate(f"Correlation: {corr:.2f}", 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=10, color='white',
                               bbox=dict(boxstyle="round,pad=0.3", fc='#333333', alpha=0.7))
            except Exception as e:
                logger.error(f"Error creating trend line for {param_name}: {e}")
    
    # Add overall title
    plt.suptitle('Parameter Impact on Trading Performance', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "parameter_impact.png"), dpi=300)
    plt.close()

def plot_win_rates(results: Dict[str, Any], output_dir: str):
    """
    Plot win rates for different trading pairs.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save plots
    """
    # Extract win rates for each pair
    pairs = []
    win_rates = []
    profit_factors = []
    
    for pair, pair_results in results.get("detailed_results", {}).items():
        validation = pair_results.get("validation_results", {})
        win_rate = validation.get("win_rate", 0)
        profit_factor = validation.get("profit_factor", 0)
        
        pairs.append(pair)
        win_rates.append(win_rate)
        profit_factors.append(profit_factor)
    
    if not pairs:
        logger.warning("No data available for win rates plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create plot with two metrics
    x = np.arange(len(pairs))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, win_rates, width, label='Win Rate', color='skyblue')
    bars2 = plt.bar(x + width/2, profit_factors, width, label='Profit Factor', color='salmon')
    
    # Add value labels
    for bars, values in [(bars1, win_rates), (bars2, profit_factors)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{value:.2f}', ha='center', va='bottom', color='white', fontsize=9)
    
    # Add styling
    plt.title('Win Rates and Profit Factors by Trading Pair', fontsize=16)
    plt.xlabel('Trading Pair', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(x, pairs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "win_rates.png"), dpi=300)
    plt.close()

def plot_equity_curve(trades: List[Dict[str, Any]], pair: str, output_dir: str):
    """
    Plot equity curve from backtest trades.
    
    Args:
        trades: List of trade dictionaries
        pair: Trading pair symbol
        output_dir: Directory to save plot
    """
    if not trades:
        logger.warning(f"No trades available for equity curve plot for {pair}")
        return
    
    # Extract trade data
    timestamps = []
    equity_values = []
    cumulative_return = 1.0
    drawdowns = []
    max_equity = 1.0
    
    for trade in trades:
        timestamp = trade.get("exit_timestamp")
        profit_percentage = trade.get("profit_percentage", 0)
        
        if timestamp and profit_percentage is not None:
            # Convert timestamp to datetime if necessary
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # Update cumulative return
            cumulative_return *= (1 + profit_percentage)
            
            # Track maximum equity and drawdown
            max_equity = max(max_equity, cumulative_return)
            drawdown = (max_equity - cumulative_return) / max_equity
            
            timestamps.append(timestamp)
            equity_values.append(cumulative_return)
            drawdowns.append(drawdown)
    
    if not timestamps:
        logger.warning(f"No valid trade data for equity curve plot for {pair}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve
    ax1.plot(timestamps, equity_values, 'g-', linewidth=2)
    ax1.set_title(f'Equity Curve - {pair}', fontsize=16)
    ax1.set_ylabel('Equity (starting=1.0)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Highlight winning and losing trades
    for i in range(1, len(timestamps)):
        if equity_values[i] > equity_values[i-1]:
            ax1.plot([timestamps[i-1], timestamps[i]], [equity_values[i-1], equity_values[i]], 'g-', linewidth=2.5)
        else:
            ax1.plot([timestamps[i-1], timestamps[i]], [equity_values[i-1], equity_values[i]], 'r-', linewidth=2.5)
    
    # Plot drawdown
    ax2.fill_between(timestamps, drawdowns, color='red', alpha=0.5)
    ax2.set_title('Drawdown', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown %', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add maximum drawdown line and annotation
    max_drawdown = max(drawdowns) if drawdowns else 0
    ax2.axhline(y=max_drawdown, color='w', linestyle='--', alpha=0.8)
    ax2.annotate(f'Max Drawdown: {max_drawdown:.2%}', 
               xy=(0.02, max_drawdown + 0.02), xycoords=('axes fraction', 'data'),
               fontsize=10, color='white',
               bbox=dict(boxstyle="round,pad=0.3", fc='#333333', alpha=0.7))
    
    # Add final return annotation
    final_return = equity_values[-1] - 1.0 if equity_values else 0
    ax1.annotate(f'Total Return: {final_return:.2%}', 
               xy=(0.02, 0.95), xycoords='axes fraction',
               fontsize=12, color='white',
               bbox=dict(boxstyle="round,pad=0.3", fc='#333333', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    pair_safe = pair.replace('/', '_')
    plt.savefig(os.path.join(output_dir, f"{pair_safe}_equity_curve.png"), dpi=300)
    plt.close()

def plot_market_regimes(historical_data: pd.DataFrame, market_regimes: Dict[str, Any], pair: str, output_dir: str):
    """
    Plot price data with market regime overlay.
    
    Args:
        historical_data: DataFrame with historical price data
        market_regimes: Dictionary with market regime analysis
        pair: Trading pair symbol
        output_dir: Directory to save plot
    """
    if historical_data is None or historical_data.empty:
        logger.warning(f"No historical data available for market regimes plot for {pair}")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot price data
    plt.plot(historical_data.index, historical_data['close'], 'w-', linewidth=1.5, label='Price')
    
    # Add EMA lines
    if 'ema20' in historical_data.columns and 'ema50' in historical_data.columns:
        plt.plot(historical_data.index, historical_data['ema20'], 'b-', linewidth=1, alpha=0.7, label='EMA20')
        plt.plot(historical_data.index, historical_data['ema50'], 'r-', linewidth=1, alpha=0.7, label='EMA50')
    
    # Add Bollinger Bands if available
    if 'bb_upper' in historical_data.columns and 'bb_lower' in historical_data.columns:
        plt.plot(historical_data.index, historical_data['bb_upper'], 'g-', linewidth=1, alpha=0.5)
        plt.plot(historical_data.index, historical_data['bb_lower'], 'g-', linewidth=1, alpha=0.5)
        plt.fill_between(historical_data.index, historical_data['bb_upper'], historical_data['bb_lower'], color='g', alpha=0.1)
    
    # Highlight different market regimes if available
    if market_regimes:
        # Extract regime periods
        regime_periods = market_regimes.get("regime_periods", {})
        
        # Create mappings for regime colors
        regime_colors = {
            "TRENDING_UP": "green",
            "TRENDING_DOWN": "red",
            "RANGING": "yellow",
            "VOLATILE": "purple",
            "UNKNOWN": "gray"
        }
        
        # Add colored backgrounds for different regime periods
        # This would require mapping regime periods to actual date ranges
        # In a real implementation, this would be more sophisticated
    
    # Add styling
    plt.title(f'Price and Market Regimes - {pair}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    pair_safe = pair.replace('/', '_')
    plt.savefig(os.path.join(output_dir, f"{pair_safe}_market_regimes.png"), dpi=300)
    plt.close()

def plot_parameter_optimization_heatmap(optimization_results: List[Dict[str, Any]], 
                                     pair: str, output_dir: str):
    """
    Plot heatmap of parameter optimization results.
    
    Args:
        optimization_results: List of optimization result dictionaries
        pair: Trading pair symbol
        output_dir: Directory to save plot
    """
    if not optimization_results:
        logger.warning(f"No optimization results available for heatmap plot for {pair}")
        return
    
    # Extract parameter combinations and returns
    # This would depend on the exact structure of optimization results
    # In a real implementation, this would be more sophisticated
    
    # Create heatmap
    # In a real implementation, this would create an actual heatmap
    
    logger.info(f"Parameter optimization heatmap for {pair} would be created here in a real implementation")