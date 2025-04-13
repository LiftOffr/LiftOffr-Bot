#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arima_strategy import ARIMAStrategy
from integrated_strategy import IntegratedStrategy
from comprehensive_backtest import load_historical_data, run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_INITIAL_CAPITAL = 10000.0
RESULTS_DIR = 'optimization_results'

def optimize_arima_strategy(pair: str, 
                           timeframe: str, 
                           initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                           max_combinations: int = 16) -> Dict:
    """
    Quickly optimize ARIMA strategy parameters
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe
        initial_capital (float): Initial capital
        max_combinations (int): Maximum parameter combinations to test
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Quick ARIMA strategy optimization for {pair} on {timeframe} timeframe")
    
    # Load historical data
    historical_data = load_historical_data(pair, timeframe)
    if historical_data is None or historical_data.empty:
        logger.error(f"Failed to load historical data for {pair} on {timeframe} timeframe")
        return {}
    
    # Reduced parameter grid for faster testing
    param_grid = {
        "lookback": [24, 36],
        "arima_order_p": [1, 2],
        "arima_order_d": [0, 1],
        "arima_order_q": [0, 1],
        "trailing_stop_type": ["volatility"],
        "trailing_mult": [1.5, 2.0]
    }
    
    # Generate parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    param_combinations = list(product(*param_values))
    
    # Limit the number of combinations if needed
    if len(param_combinations) > max_combinations:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
        param_combinations = [param_combinations[i] for i in indices]
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    results = []
    best_result = None
    best_profit = -float('inf')
    
    # Test each parameter combination
    for i, combo in enumerate(param_combinations):
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Create strategy with these parameters
        strategy = ARIMAStrategy(
            symbol=pair,
            lookback=params["lookback"],
            arima_order_p=params["arima_order_p"],
            arima_order_d=params["arima_order_d"],
            arima_order_q=params["arima_order_q"],
            trailing_stop_type=params["trailing_stop_type"],
            trailing_mult=params["trailing_mult"]
        )
        
        # Run backtest
        backtest_result = run_backtest(historical_data, {pair: strategy}, initial_capital)
        
        # Extract metrics
        metrics = backtest_result.get("metrics", {})
        profit_pct = metrics.get("profit_loss_pct", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        
        result = {
            "params": params,
            "profit_pct": profit_pct,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "total_trades": metrics.get("total_trades", 0)
        }
        
        results.append(result)
        
        # Update best result
        if profit_pct > best_profit:
            best_profit = profit_pct
            best_result = result
        
        logger.info(f"Combination {i+1}/{len(param_combinations)}: Profit={profit_pct:.2%}, Win Rate={win_rate:.2%}, Trades={result['total_trades']}")
    
    # Sort results by profit
    results.sort(key=lambda x: x["profit_pct"], reverse=True)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"arima_optimization_{pair.replace('/', '')}_{timeframe}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "pair": pair,
            "timeframe": timeframe,
            "best_result": best_result,
            "all_results": results[:10]  # Top 10 results
        }, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_file}")
    
    if best_result:
        logger.info(f"Best parameters: {best_result['params']}")
        logger.info(f"Best profit: {best_result['profit_pct']:.2%}")
        logger.info(f"Win rate: {best_result['win_rate']:.2%}")
        logger.info(f"Total trades: {best_result['total_trades']}")
    
    return {
        "best_result": best_result,
        "all_results": results
    }

def optimize_integrated_strategy(pair: str, 
                               timeframe: str, 
                               initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                               max_combinations: int = 16) -> Dict:
    """
    Quickly optimize Integrated strategy parameters
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe
        initial_capital (float): Initial capital
        max_combinations (int): Maximum parameter combinations to test
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Quick Integrated strategy optimization for {pair} on {timeframe} timeframe")
    
    # Load historical data
    historical_data = load_historical_data(pair, timeframe)
    if historical_data is None or historical_data.empty:
        logger.error(f"Failed to load historical data for {pair} on {timeframe} timeframe")
        return {}
    
    # Reduced parameter grid for faster testing
    param_grid = {
        "lookback": [24, 36],
        "rsi_period": [14, 21],
        "atr_period": [14, 21],
        "ema_short": [9, 12],
        "ema_long": [21, 26],
        "volatility_threshold": [0.005, 0.01]
    }
    
    # Generate parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    param_combinations = list(product(*param_values))
    
    # Limit the number of combinations if needed
    if len(param_combinations) > max_combinations:
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
        param_combinations = [param_combinations[i] for i in indices]
    
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    results = []
    best_result = None
    best_profit = -float('inf')
    
    # Test each parameter combination
    for i, combo in enumerate(param_combinations):
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Create strategy with these parameters
        strategy = IntegratedStrategy(
            symbol=pair,
            lookback=params["lookback"],
            rsi_period=params["rsi_period"],
            atr_period=params["atr_period"],
            ema_short=params["ema_short"],
            ema_long=params["ema_long"],
            volatility_threshold=params["volatility_threshold"]
        )
        
        # Run backtest
        backtest_result = run_backtest(historical_data, {pair: strategy}, initial_capital)
        
        # Extract metrics
        metrics = backtest_result.get("metrics", {})
        profit_pct = metrics.get("profit_loss_pct", 0)
        win_rate = metrics.get("win_rate", 0)
        max_drawdown = metrics.get("max_drawdown", 0)
        
        result = {
            "params": params,
            "profit_pct": profit_pct,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "total_trades": metrics.get("total_trades", 0)
        }
        
        results.append(result)
        
        # Update best result
        if profit_pct > best_profit:
            best_profit = profit_pct
            best_result = result
        
        logger.info(f"Combination {i+1}/{len(param_combinations)}: Profit={profit_pct:.2%}, Win Rate={win_rate:.2%}, Trades={result['total_trades']}")
    
    # Sort results by profit
    results.sort(key=lambda x: x["profit_pct"], reverse=True)
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"integrated_optimization_{pair.replace('/', '')}_{timeframe}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "pair": pair,
            "timeframe": timeframe,
            "best_result": best_result,
            "all_results": results[:10]  # Top 10 results
        }, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_file}")
    
    if best_result:
        logger.info(f"Best parameters: {best_result['params']}")
        logger.info(f"Best profit: {best_result['profit_pct']:.2%}")
        logger.info(f"Win rate: {best_result['win_rate']:.2%}")
        logger.info(f"Total trades: {best_result['total_trades']}")
    
    return {
        "best_result": best_result,
        "all_results": results
    }

def optimize_multi_pair(pairs: List[str], 
                      timeframes: List[str], 
                      strategies: List[str],
                      initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                      max_combinations: int = 8,
                      max_workers: int = 2) -> Dict:
    """
    Optimize strategies across multiple pairs and timeframes
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        strategies (list): List of strategies to optimize
        initial_capital (float): Initial capital
        max_combinations (int): Maximum parameter combinations per strategy
        max_workers (int): Maximum number of parallel optimizations
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Multi-pair optimization for {len(pairs)} pairs, {len(timeframes)} timeframes, and {len(strategies)} strategies")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel optimization
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Submit optimization tasks
        for pair in pairs:
            for timeframe in timeframes:
                for strategy in strategies:
                    future_key = f"{strategy}_{pair}_{timeframe}"
                    
                    if strategy.lower() == "arima":
                        future = executor.submit(
                            optimize_arima_strategy,
                            pair, timeframe, initial_capital, max_combinations
                        )
                    elif strategy.lower() == "integrated":
                        future = executor.submit(
                            optimize_integrated_strategy,
                            pair, timeframe, initial_capital, max_combinations
                        )
                    else:
                        logger.warning(f"Unknown strategy: {strategy}")
                        continue
                    
                    futures[future] = future_key
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                results[key] = result
                logger.info(f"Completed optimization for {key}")
            except Exception as e:
                logger.error(f"Error during optimization for {key}: {e}")
    
    # Aggregate best results
    best_configs = {}
    for key, result in results.items():
        if result and "best_result" in result and result["best_result"]:
            best_configs[key] = {
                "params": result["best_result"]["params"],
                "profit_pct": result["best_result"]["profit_pct"],
                "win_rate": result["best_result"]["win_rate"]
            }
    
    # Save aggregated results
    agg_output_file = os.path.join(RESULTS_DIR, "aggregated_optimization_results.json")
    with open(agg_output_file, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    logger.info(f"Aggregated results saved to {agg_output_file}")
    
    return results

def plot_optimization_results(results_file: str, top_n: int = 10):
    """
    Plot optimization results
    
    Args:
        results_file (str): Path to the results JSON file
        top_n (int): Number of top results to plot
    """
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    pair = data["pair"]
    timeframe = data["timeframe"]
    all_results = data["all_results"]
    
    # Sort results by profit
    all_results.sort(key=lambda x: x["profit_pct"], reverse=True)
    top_results = all_results[:top_n]
    
    # Prepare data for plotting
    profits = [r["profit_pct"] for r in top_results]
    win_rates = [r["win_rate"] for r in top_results]
    trades = [r["total_trades"] for r in top_results]
    param_labels = [str(i+1) for i in range(len(top_results))]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot profit
    ax1.bar(param_labels, profits)
    ax1.set_title(f"Top {top_n} Parameter Sets - {pair} ({timeframe})")
    ax1.set_ylabel("Profit (%)")
    ax1.grid(True, alpha=0.3)
    
    # Add profit values
    for i, v in enumerate(profits):
        ax1.text(i, v, f"{v:.2%}", ha='center', va='bottom')
    
    # Plot win rate
    ax2.bar(param_labels, win_rates)
    ax2.set_ylabel("Win Rate")
    ax2.grid(True, alpha=0.3)
    
    # Add win rate values
    for i, v in enumerate(win_rates):
        ax2.text(i, v, f"{v:.2%}", ha='center', va='bottom')
    
    # Plot trade count
    ax3.bar(param_labels, trades)
    ax3.set_xlabel("Parameter Set")
    ax3.set_ylabel("Total Trades")
    ax3.grid(True, alpha=0.3)
    
    # Add trade count values
    for i, v in enumerate(trades):
        ax3.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(RESULTS_DIR, f"optimization_plot_{pair.replace('/', '')}_{timeframe}.png")
    plt.savefig(plot_file)
    logger.info(f"Saved optimization plot to {plot_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Quick Strategy Optimization Tool")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=["SOL/USD"],
                      help="Trading pairs to optimize (default: SOL/USD)")
    parser.add_argument("--timeframes", type=str, nargs="+", default=["1h"],
                      help="Timeframes to optimize (default: 1h)")
    parser.add_argument("--strategies", type=str, nargs="+", default=["arima", "integrated"],
                      help="Strategies to optimize (default: arima, integrated)")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
                      help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})")
    parser.add_argument("--max-combinations", type=int, default=8,
                      help="Maximum parameter combinations to test per strategy (default: 8)")
    parser.add_argument("--max-workers", type=int, default=2,
                      help="Maximum number of parallel optimizations (default: 2)")
    parser.add_argument("--plot", action="store_true",
                      help="Plot optimization results")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run optimization
    results = optimize_multi_pair(
        args.pairs, 
        args.timeframes, 
        args.strategies,
        args.capital,
        args.max_combinations,
        args.max_workers
    )
    
    # Plot results if requested
    if args.plot:
        for pair in args.pairs:
            for timeframe in args.timeframes:
                for strategy in args.strategies:
                    results_file = os.path.join(
                        RESULTS_DIR, 
                        f"{strategy}_optimization_{pair.replace('/', '')}_{timeframe}.json"
                    )
                    if os.path.exists(results_file):
                        plot_optimization_results(results_file)

if __name__ == "__main__":
    main()