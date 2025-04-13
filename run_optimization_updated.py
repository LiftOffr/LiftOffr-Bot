#!/usr/bin/env python3

import os
import argparse
import json
import logging
from itertools import product
import time
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from arima_strategy import ARIMAStrategy
from integrated_strategy import IntegratedStrategy
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Constants
DATA_DIR = "historical_data"
RESULTS_DIR = "optimization_results"
DEFAULT_INITIAL_CAPITAL = 10000.0

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_historical_data(pair: str, timeframe: str = "1h") -> pd.DataFrame:
    """
    Load historical data for backtesting
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe
        
    Returns:
        pd.DataFrame: Historical OHLCV data
    """
    # Format pair for filename
    dir_name = pair.replace("/", "")
    file_path = os.path.join(DATA_DIR, dir_name, f"{dir_name}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Historical data not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # Process timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    
    # Ensure we have all required columns
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in data file")
            return pd.DataFrame()
    
    return df

def run_arima_strategy_backtest(
    data: pd.DataFrame, 
    params: Dict,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> Tuple[float, Dict]:
    """
    Run ARIMA strategy backtest with given parameters
    
    Args:
        data (pd.DataFrame): Historical OHLCV data
        params (dict): Strategy parameters
        initial_capital (float): Initial capital
        
    Returns:
        tuple: (profit_percentage, metrics)
    """
    # Create strategy with parameters
    symbol = "SOL/USD"
    strategy = ARIMAStrategy(
        symbol=symbol,
        lookback_period=params.get("lookback_period", 32),
        atr_trailing_multiplier=params.get("atr_trailing_multiplier", 2.0),
        entry_atr_multiplier=params.get("entry_atr_multiplier", 0.01),
        leverage=params.get("leverage", 30),
        risk_buffer_multiplier=params.get("risk_buffer_multiplier", 1.25),
        arima_order=(
            params.get("arima_order_p", 1), 
            params.get("arima_order_d", 1), 
            params.get("arima_order_q", 0)
        ),
        max_loss_percent=params.get("max_loss_percent", 4.0)
    )
    
    # Initialize metrics
    equity = [initial_capital]
    current_capital = initial_capital
    position = None
    entry_price = 0
    position_size = 0
    
    trades = []
    
    # Simulate trading
    for i in range(params.get("lookback_period", 32), len(data)):
        # Current price data
        current_data = data.iloc[:i+1]
        close_price = current_data.iloc[-1]["close"]
        high_price = current_data.iloc[-1]["high"]
        low_price = current_data.iloc[-1]["low"]
        open_price = current_data.iloc[-1]["open"]
        
        # Process strategy with OHLC data
        strategy.update_ohlc(open_price, high_price, low_price, close_price)
        
        # Check for entry or exit signals
        if strategy.position is None:
            # No position, check for entry signal
            buy_signal, sell_signal, _ = strategy.check_entry_signal(close_price)
            if buy_signal:
                signal = "BUY"
            elif sell_signal:
                signal = "SELL"
            else:
                signal = "NEUTRAL"
        else:
            # Have a position, check for exit signal
            exit_signal, exit_price = strategy.check_exit_signal(close_price)
            if exit_signal:
                if strategy.position == "long":
                    signal = "EXIT_LONG"
                else:
                    signal = "EXIT_SHORT"
            else:
                signal = "HOLD"
        
        # Handle trades
        if position is None:  # No position
            if signal == "BUY":
                # Enter long position
                position = "LONG"
                entry_price = close_price
                position_size = (current_capital * 0.2) / entry_price  # 20% of capital
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "BUY",
                    "price": entry_price,
                    "size": position_size,
                })
                
            elif signal == "SELL":
                # Enter short position
                position = "SHORT"
                entry_price = close_price
                position_size = (current_capital * 0.2) / entry_price  # 20% of capital
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "SELL",
                    "price": entry_price,
                    "size": position_size,
                })
        
        elif position == "LONG":  # Long position
            if signal == "EXIT_LONG" or signal == "SELL":
                # Exit long position
                pnl = (close_price - entry_price) * position_size
                current_capital += pnl
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "SELL",
                    "price": close_price,
                    "size": position_size,
                    "pnl": pnl,
                    "pnl_pct": (close_price / entry_price - 1) * 100
                })
                
                position = None
                position_size = 0
        
        elif position == "SHORT":  # Short position
            if signal == "EXIT_SHORT" or signal == "BUY":
                # Exit short position
                pnl = (entry_price - close_price) * position_size
                current_capital += pnl
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "BUY",
                    "price": close_price,
                    "size": position_size,
                    "pnl": pnl,
                    "pnl_pct": (entry_price / close_price - 1) * 100
                })
                
                position = None
                position_size = 0
        
        # Track equity
        unrealized_pnl = 0
        if position == "LONG":
            unrealized_pnl = (close_price - entry_price) * position_size
        elif position == "SHORT":
            unrealized_pnl = (entry_price - close_price) * position_size
        
        equity.append(current_capital + unrealized_pnl)
    
    # Close any open position at the end
    if position is not None:
        close_price = data.iloc[-1]["close"]
        
        if position == "LONG":
            pnl = (close_price - entry_price) * position_size
            current_capital += pnl
            
            trades.append({
                "time": data.index[-1],
                "type": "SELL",
                "price": close_price,
                "size": position_size,
                "pnl": pnl,
                "pnl_pct": (close_price / entry_price - 1) * 100
            })
        
        elif position == "SHORT":
            pnl = (entry_price - close_price) * position_size
            current_capital += pnl
            
            trades.append({
                "time": data.index[-1],
                "type": "BUY",
                "price": close_price,
                "size": position_size,
                "pnl": pnl,
                "pnl_pct": (entry_price / close_price - 1) * 100
            })
    
    # Calculate metrics
    profit_loss = current_capital - initial_capital
    profit_loss_pct = profit_loss / initial_capital
    
    # Count winning/losing trades
    winning_trades = [t for t in trades if "pnl" in t and t["pnl"] > 0]
    losing_trades = [t for t in trades if "pnl" in t and t["pnl"] <= 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    # Calculate drawdown
    max_equity = equity[0]
    max_drawdown = 0
    
    for e in equity:
        max_equity = max(max_equity, e)
        drawdown = (max_equity - e) / max_equity
        max_drawdown = max(max_drawdown, drawdown)
    
    metrics = {
        "initial_capital": initial_capital,
        "final_capital": current_capital,
        "profit_loss": profit_loss,
        "profit_loss_pct": profit_loss_pct,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
    }
    
    return profit_loss_pct, metrics

def run_integrated_strategy_backtest(
    data: pd.DataFrame, 
    params: Dict,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> Tuple[float, Dict]:
    """
    Run Integrated strategy backtest with given parameters
    
    Args:
        data (pd.DataFrame): Historical OHLCV data
        params (dict): Strategy parameters
        initial_capital (float): Initial capital
        
    Returns:
        tuple: (profit_percentage, metrics)
    """
    # Create strategy with parameters
    symbol = "SOL/USD"
    strategy = IntegratedStrategy(
        symbol=symbol,
        lookback=params.get("lookback", 24),
        rsi_period=params.get("rsi_period", 14),
        atr_period=params.get("atr_period", 14),
        ema_short=params.get("ema_short", 9),
        ema_long=params.get("ema_long", 21),
        volatility_threshold=params.get("volatility_threshold", 0.005)
    )
    
    # Initialize metrics
    equity = [initial_capital]
    current_capital = initial_capital
    position = None
    entry_price = 0
    position_size = 0
    
    trades = []
    
    # Simulate trading
    for i in range(params.get("lookback", 24), len(data)):
        # Current price data
        current_data = data.iloc[:i+1]
        close_price = current_data.iloc[-1]["close"]
        
        # Process strategy
        strategy.update(current_data)
        signal = strategy.check_strategy()
        
        # Handle trades
        if position is None:  # No position
            if signal == "BUY":
                # Enter long position
                position = "LONG"
                entry_price = close_price
                position_size = (current_capital * 0.2) / entry_price  # 20% of capital
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "BUY",
                    "price": entry_price,
                    "size": position_size,
                })
                
            elif signal == "SELL":
                # Enter short position
                position = "SHORT"
                entry_price = close_price
                position_size = (current_capital * 0.2) / entry_price  # 20% of capital
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "SELL",
                    "price": entry_price,
                    "size": position_size,
                })
        
        elif position == "LONG":  # Long position
            if signal == "EXIT_LONG" or signal == "SELL":
                # Exit long position
                pnl = (close_price - entry_price) * position_size
                current_capital += pnl
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "SELL",
                    "price": close_price,
                    "size": position_size,
                    "pnl": pnl,
                    "pnl_pct": (close_price / entry_price - 1) * 100
                })
                
                position = None
                position_size = 0
        
        elif position == "SHORT":  # Short position
            if signal == "EXIT_SHORT" or signal == "BUY":
                # Exit short position
                pnl = (entry_price - close_price) * position_size
                current_capital += pnl
                
                trades.append({
                    "time": current_data.index[-1],
                    "type": "BUY",
                    "price": close_price,
                    "size": position_size,
                    "pnl": pnl,
                    "pnl_pct": (entry_price / close_price - 1) * 100
                })
                
                position = None
                position_size = 0
        
        # Track equity
        unrealized_pnl = 0
        if position == "LONG":
            unrealized_pnl = (close_price - entry_price) * position_size
        elif position == "SHORT":
            unrealized_pnl = (entry_price - close_price) * position_size
        
        equity.append(current_capital + unrealized_pnl)
    
    # Close any open position at the end
    if position is not None:
        close_price = data.iloc[-1]["close"]
        
        if position == "LONG":
            pnl = (close_price - entry_price) * position_size
            current_capital += pnl
            
            trades.append({
                "time": data.index[-1],
                "type": "SELL",
                "price": close_price,
                "size": position_size,
                "pnl": pnl,
                "pnl_pct": (close_price / entry_price - 1) * 100
            })
        
        elif position == "SHORT":
            pnl = (entry_price - close_price) * position_size
            current_capital += pnl
            
            trades.append({
                "time": data.index[-1],
                "type": "BUY",
                "price": close_price,
                "size": position_size,
                "pnl": pnl,
                "pnl_pct": (entry_price / close_price - 1) * 100
            })
    
    # Calculate metrics
    profit_loss = current_capital - initial_capital
    profit_loss_pct = profit_loss / initial_capital
    
    # Count winning/losing trades
    winning_trades = [t for t in trades if "pnl" in t and t["pnl"] > 0]
    losing_trades = [t for t in trades if "pnl" in t and t["pnl"] <= 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    # Calculate drawdown
    max_equity = equity[0]
    max_drawdown = 0
    
    for e in equity:
        max_equity = max(max_equity, e)
        drawdown = (max_equity - e) / max_equity
        max_drawdown = max(max_drawdown, drawdown)
    
    metrics = {
        "initial_capital": initial_capital,
        "final_capital": current_capital,
        "profit_loss": profit_loss,
        "profit_loss_pct": profit_loss_pct,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
    }
    
    return profit_loss_pct, metrics

def optimize_arima_strategy(
    pair: str,
    timeframe: str = "1h",
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    max_combinations: int = 8
) -> Dict:
    """
    Optimize ARIMA strategy parameters for a given pair and timeframe
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe
        initial_capital (float): Initial capital
        max_combinations (int): Maximum number of parameter combinations to test
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Optimizing ARIMA strategy for {pair} ({timeframe})")
    
    # Load historical data
    data = load_historical_data(pair, timeframe)
    if data.empty:
        logger.error("Failed to load historical data")
        return {}
    
    # Define parameter grid
    param_grid = {
        "lookback_period": [24, 32, 48],
        "arima_order_p": [1, 2],
        "arima_order_d": [0, 1],
        "arima_order_q": [0, 1],
        "atr_trailing_multiplier": [1.5, 2.0, 2.5],
        "entry_atr_multiplier": [0.01, 0.02],
        "risk_buffer_multiplier": [1.0, 1.25, 1.5],
        "max_loss_percent": [3.0, 4.0, 5.0]
    }
    
    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    param_combinations = list(product(*param_values))
    
    # Limit number of combinations if needed
    if len(param_combinations) > max_combinations:
        import random
        random.seed(42)  # For reproducibility
        param_combinations = random.sample(param_combinations, max_combinations)
    
    # Test each parameter combination
    results = []
    best_profit = -float('inf')
    best_params = None
    best_metrics = None
    
    total_combinations = len(param_combinations)
    logger.info(f"Testing {total_combinations} parameter combinations")
    
    start_time = time.time()
    
    for i, combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Run backtest
        profit_pct, metrics = run_arima_strategy_backtest(data, params, initial_capital)
        
        # Store results
        result = {
            "params": params,
            "profit_pct": profit_pct,
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"],
            "max_drawdown": metrics["max_drawdown"],
        }
        
        results.append(result)
        
        # Update best result
        if profit_pct > best_profit:
            best_profit = profit_pct
            best_params = params
            best_metrics = metrics
        
        # Log progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total_combinations - i - 1)
        
        logger.info(f"Combination {i+1}/{total_combinations}: " +
                  f"Profit={profit_pct:.2%}, Win Rate={metrics['win_rate']:.2%}, " +
                  f"Trades={metrics['total_trades']}, Time={avg_time:.1f}s per combo, " +
                  f"ETA: {remaining/60:.1f}m remaining")
    
    # Sort results by profit
    results.sort(key=lambda x: x["profit_pct"], reverse=True)
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, f"arima_optimization_{pair.replace('/', '')}_{timeframe}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "pair": pair,
            "timeframe": timeframe,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": results[:10]  # Only save top 10 results
        }, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_file}")
    
    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_results": results
    }

def optimize_integrated_strategy(
    pair: str,
    timeframe: str = "1h",
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    max_combinations: int = 8
) -> Dict:
    """
    Optimize Integrated strategy parameters for a given pair and timeframe
    
    Args:
        pair (str): Trading pair
        timeframe (str): Timeframe
        initial_capital (float): Initial capital
        max_combinations (int): Maximum number of parameter combinations to test
        
    Returns:
        dict: Optimization results
    """
    logger.info(f"Optimizing Integrated strategy for {pair} ({timeframe})")
    
    # Load historical data
    data = load_historical_data(pair, timeframe)
    if data.empty:
        logger.error("Failed to load historical data")
        return {}
    
    # Define parameter grid
    param_grid = {
        "lookback": [24, 36],
        "rsi_period": [14, 21],
        "atr_period": [14, 21],
        "ema_short": [9, 12],
        "ema_long": [21, 26],
        "volatility_threshold": [0.005, 0.01]
    }
    
    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]
    param_combinations = list(product(*param_values))
    
    # Limit number of combinations if needed
    if len(param_combinations) > max_combinations:
        import random
        random.seed(42)  # For reproducibility
        param_combinations = random.sample(param_combinations, max_combinations)
    
    # Test each parameter combination
    results = []
    best_profit = -float('inf')
    best_params = None
    best_metrics = None
    
    total_combinations = len(param_combinations)
    logger.info(f"Testing {total_combinations} parameter combinations")
    
    start_time = time.time()
    
    for i, combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = {param_keys[j]: combo[j] for j in range(len(param_keys))}
        
        # Run backtest
        profit_pct, metrics = run_integrated_strategy_backtest(data, params, initial_capital)
        
        # Store results
        result = {
            "params": params,
            "profit_pct": profit_pct,
            "win_rate": metrics["win_rate"],
            "total_trades": metrics["total_trades"],
            "max_drawdown": metrics["max_drawdown"],
        }
        
        results.append(result)
        
        # Update best result
        if profit_pct > best_profit:
            best_profit = profit_pct
            best_params = params
            best_metrics = metrics
        
        # Log progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total_combinations - i - 1)
        
        logger.info(f"Combination {i+1}/{total_combinations}: " +
                  f"Profit={profit_pct:.2%}, Win Rate={metrics['win_rate']:.2%}, " +
                  f"Trades={metrics['total_trades']}, Time={avg_time:.1f}s per combo, " +
                  f"ETA: {remaining/60:.1f}m remaining")
    
    # Sort results by profit
    results.sort(key=lambda x: x["profit_pct"], reverse=True)
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, f"integrated_optimization_{pair.replace('/', '')}_{timeframe}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            "pair": pair,
            "timeframe": timeframe,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": results[:10]  # Only save top 10 results
        }, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_file}")
    
    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "all_results": results
    }

def plot_optimization_results(results_file: str, top_n: int = 10) -> None:
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
    profits = [r["profit_pct"] * 100 for r in top_results]  # Convert to percentage
    win_rates = [r["win_rate"] * 100 for r in top_results]  # Convert to percentage
    trades = [r["total_trades"] for r in top_results]
    param_labels = [f"Set {i+1}" for i in range(len(top_results))]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot profit
    bars1 = ax1.bar(param_labels, profits)
    ax1.set_title(f"Top {top_n} Parameter Sets - {pair} ({timeframe})")
    ax1.set_ylabel("Profit (%)")
    ax1.grid(True, alpha=0.3)
    
    # Add profit values
    for i, v in enumerate(profits):
        ax1.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
    
    # Plot win rate
    bars2 = ax2.bar(param_labels, win_rates)
    ax2.set_ylabel("Win Rate (%)")
    ax2.grid(True, alpha=0.3)
    
    # Add win rate values
    for i, v in enumerate(win_rates):
        ax2.text(i, v, f"{v:.1f}%", ha='center', va='bottom')
    
    # Plot trade count
    bars3 = ax3.bar(param_labels, trades)
    ax3.set_xlabel("Parameter Set")
    ax3.set_ylabel("Total Trades")
    ax3.grid(True, alpha=0.3)
    
    # Add trade count values
    for i, v in enumerate(trades):
        ax3.text(i, v, str(v), ha='center', va='bottom')
    
    # Show legend with parameter values
    legend_text = []
    for i, result in enumerate(top_results):
        params_str = ", ".join([f"{k}={v}" for k, v in result["params"].items()])
        legend_text.append(f"Set {i+1}: {params_str}")
    
    fig.text(0.1, 0.01, "\n".join(legend_text), fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    
    # Save plot
    plot_file = os.path.join(RESULTS_DIR, f"optimization_plot_{pair.replace('/', '')}_{timeframe}.png")
    plt.savefig(plot_file)
    logger.info(f"Saved optimization plot to {plot_file}")
    
    plt.close()

def optimize_multi_pair(
    pairs: List[str],
    timeframes: List[str],
    strategies: List[str],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    max_combinations: int = 8
) -> Dict:
    """
    Optimize strategies for multiple pairs and timeframes
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        strategies (list): List of strategies to optimize
        initial_capital (float): Initial capital
        max_combinations (int): Maximum parameter combinations to test per strategy
        
    Returns:
        dict: Optimization results
    """
    results = {}
    
    for pair in pairs:
        for timeframe in timeframes:
            for strategy in strategies:
                if strategy.lower() == "arima":
                    result = optimize_arima_strategy(
                        pair=pair,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        max_combinations=max_combinations
                    )
                    results[f"arima_{pair}_{timeframe}"] = result
                    
                elif strategy.lower() == "integrated":
                    result = optimize_integrated_strategy(
                        pair=pair,
                        timeframe=timeframe,
                        initial_capital=initial_capital,
                        max_combinations=max_combinations
                    )
                    results[f"integrated_{pair}_{timeframe}"] = result
    
    # Aggregate best results
    best_configs = {}
    
    for key, result in results.items():
        if result and "best_params" in result and result["best_params"]:
            best_configs[key] = {
                "params": result["best_params"],
                "profit_pct": result["best_metrics"]["profit_loss_pct"],
                "win_rate": result["best_metrics"]["win_rate"],
                "total_trades": result["best_metrics"]["total_trades"]
            }
    
    # Save aggregated results
    agg_output_file = os.path.join(RESULTS_DIR, "aggregated_optimization_results.json")
    
    with open(agg_output_file, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    logger.info(f"Aggregated results saved to {agg_output_file}")
    
    return results

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimization Tool")
    
    parser.add_argument("--pairs", nargs="+", default=["SOL/USD"],
                      help="Trading pairs to optimize (default: SOL/USD)")
    parser.add_argument("--timeframes", nargs="+", default=["1h"],
                      help="Timeframes to optimize (default: 1h)")
    parser.add_argument("--strategies", nargs="+", default=["arima", "integrated"],
                      help="Strategies to optimize (default: arima, integrated)")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
                      help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL})")
    parser.add_argument("--max-combinations", type=int, default=8,
                      help="Maximum parameter combinations to test per strategy (default: 8)")
    parser.add_argument("--plot", action="store_true",
                      help="Plot optimization results")
    
    args = parser.parse_args()
    
    # Run multi-pair optimization
    results = optimize_multi_pair(
        pairs=args.pairs,
        timeframes=args.timeframes,
        strategies=args.strategies,
        initial_capital=args.capital,
        max_combinations=args.max_combinations
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