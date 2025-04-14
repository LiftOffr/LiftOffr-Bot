#!/usr/bin/env python3
"""
Enhanced Backtesting System Runner

This script provides a user-friendly interface for running the comprehensive
backtesting system with optimized parameters for maximum profitability.

Features:
1. Support for multiple assets (SOL/USD, ETH/USD, BTC/USD)
2. Multiple strategy testing and comparison
3. Parameter optimization with random search
4. Extreme leverage testing up to 125x
5. Detailed performance metrics and visualizations
6. Multi-period analysis (daily, weekly, monthly returns)

Usage:
  python run_enhanced_backtesting.py [options]
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from comprehensive_backtest import (
    run_full_backtest, optimize_strategy_parameters, 
    SUPPORTED_ASSETS, STRATEGIES, INITIAL_CAPITAL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_backtesting.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Backtesting System')
    
    parser.add_argument('--assets', nargs='+', default=SUPPORTED_ASSETS,
                      help=f'Assets to backtest (default: {" ".join(SUPPORTED_ASSETS)})')
    
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES,
                      help=f'Strategies to backtest (default: {" ".join(STRATEGIES)})')
    
    parser.add_argument('--optimize', action='store_true',
                      help='Optimize strategy parameters')
    
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of optimization trials (default: 10)')
    
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                      help=f'Initial capital in USD (default: ${INITIAL_CAPITAL:.2f})')
    
    parser.add_argument('--start-date', type=str,
                      help='Start date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str,
                      help='End date for backtesting (YYYY-MM-DD)')
    
    parser.add_argument('--days', type=int, default=90,
                      help='Number of days to backtest (default: 90)')
    
    parser.add_argument('--allocation', nargs='+', type=float,
                      help='Capital allocation percentages (must match number of assets)')
    
    parser.add_argument('--max-leverage', type=float, default=125.0,
                      help='Maximum leverage to test (default: 125.0x)')
    
    parser.add_argument('--aggressive', action='store_true',
                      help='Use ultra-aggressive position sizing settings')
    
    parser.add_argument('--save-parameters', action='store_true',
                      help='Save optimized parameters for live trading')
    
    return parser.parse_args()

def create_allocation(assets: List[str], allocation_percentages: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Create allocation dictionary from assets and percentages
    
    Args:
        assets: List of trading pairs
        allocation_percentages: List of allocation percentages (must sum to 1.0)
        
    Returns:
        Dict: Dictionary mapping assets to allocation percentages
    """
    if allocation_percentages is None:
        # Default allocations based on volatility and market cap
        default_allocations = {
            "SOL/USD": 0.40,  # 40% to SOL (higher volatility, higher potential returns)
            "ETH/USD": 0.35,  # 35% to ETH 
            "BTC/USD": 0.25   # 25% to BTC (lower volatility, more stable)
        }
        
        # Filter to only include requested assets
        allocation = {asset: default_allocations.get(asset, 1.0 / len(assets)) 
                    for asset in assets}
        
        # Normalize to ensure sum is 1.0
        total = sum(allocation.values())
        allocation = {asset: pct / total for asset, pct in allocation.items()}
    else:
        # Check if number of percentages matches number of assets
        if len(allocation_percentages) != len(assets):
            logger.warning(f"Number of allocation percentages ({len(allocation_percentages)}) " 
                          f"doesn't match number of assets ({len(assets)}). Using default allocation.")
            return create_allocation(assets)
        
        # Create allocation dictionary
        allocation = {asset: pct for asset, pct in zip(assets, allocation_percentages)}
        
        # Normalize to ensure sum is 1.0
        total = sum(allocation.values())
        allocation = {asset: pct / total for asset, pct in allocation.items()}
    
    logger.info(f"Capital allocation: {allocation}")
    return allocation

def save_parameters(params: Dict, filename: str = "optimized_parameters.json"):
    """
    Save optimized parameters to file
    
    Args:
        params: Parameters dictionary
        filename: Output filename
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs("backtest_results", exist_ok=True)
    
    # Save parameters
    filepath = os.path.join("backtest_results", filename)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    logger.info(f"Saved optimized parameters to {filepath}")

def run_interactive_backtest():
    """Run interactive backtest with user inputs"""
    print("\n===== ENHANCED BACKTESTING SYSTEM =====\n")
    
    # Select assets
    print("Available assets:")
    for i, asset in enumerate(SUPPORTED_ASSETS, 1):
        print(f"{i}. {asset}")
    
    assets_input = input("\nEnter asset numbers to backtest (comma-separated) or press Enter for all: ")
    if assets_input.strip():
        asset_indices = [int(idx.strip()) - 1 for idx in assets_input.split(',')]
        assets = [SUPPORTED_ASSETS[idx] for idx in asset_indices if 0 <= idx < len(SUPPORTED_ASSETS)]
    else:
        assets = SUPPORTED_ASSETS
    
    # Select strategies
    print("\nAvailable strategies:")
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"{i}. {strategy}")
    
    strategies_input = input("\nEnter strategy numbers to backtest (comma-separated) or press Enter for all: ")
    if strategies_input.strip():
        strategy_indices = [int(idx.strip()) - 1 for idx in strategies_input.split(',')]
        strategies = [STRATEGIES[idx] for idx in strategy_indices if 0 <= idx < len(STRATEGIES)]
    else:
        strategies = STRATEGIES
    
    # Select date range
    days_input = input("\nEnter number of days to backtest (default: 90): ")
    days = int(days_input) if days_input.strip() else 90
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Select capital
    capital_input = input(f"\nEnter initial capital in USD (default: ${INITIAL_CAPITAL:.2f}): ")
    capital = float(capital_input) if capital_input.strip() else INITIAL_CAPITAL
    
    # Select optimization
    optimize_input = input("\nOptimize strategy parameters? (y/n, default: n): ")
    optimize = optimize_input.lower().startswith('y')
    
    if optimize:
        trials_input = input("\nEnter number of optimization trials (default: 10): ")
        trials = int(trials_input) if trials_input.strip() else 10
    else:
        trials = 10
    
    # Select allocation
    allocation_input = input("\nEnter custom allocation percentages? (y/n, default: n): ")
    if allocation_input.lower().startswith('y'):
        allocation_percentages = []
        for asset in assets:
            pct_input = input(f"Enter allocation percentage for {asset} (0-100): ")
            try:
                pct = float(pct_input) / 100.0  # Convert percentage to decimal
                allocation_percentages.append(pct)
            except ValueError:
                print(f"Invalid input for {asset}, using default allocation.")
                allocation_percentages = None
                break
    else:
        allocation_percentages = None
    
    allocation = create_allocation(assets, allocation_percentages)
    
    # Select leverage
    leverage_input = input("\nEnter maximum leverage to test (default: 125.0): ")
    max_leverage = float(leverage_input) if leverage_input.strip() else 125.0
    
    # Run backtest
    print("\nRunning backtest with the following settings:")
    print(f"Assets: {assets}")
    print(f"Strategies: {strategies}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)")
    print(f"Initial Capital: ${capital:.2f}")
    print(f"Optimization: {'Yes' if optimize else 'No'}")
    if optimize:
        print(f"Optimization Trials: {trials}")
    print(f"Maximum Leverage: {max_leverage}x")
    print(f"Allocation: {allocation}")
    
    confirm = input("\nConfirm settings? (y/n): ")
    if not confirm.lower().startswith('y'):
        print("Backtest cancelled.")
        return
    
    # Run backtest
    results = run_full_backtest(
        assets=assets,
        strategies=strategies,
        allocation=allocation,
        optimize=optimize,
        num_trials=trials,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital
    )
    
    # Save parameters if optimized
    if optimize:
        save_input = input("\nSave optimized parameters for live trading? (y/n): ")
        if save_input.lower().startswith('y'):
            save_parameters(results['best_params'])
    
    print("\nBacktest completed!")
    if optimize:
        print(f"Best Return: ${results['best_metrics']['total_return']:.2f} ({results['best_metrics']['total_return_pct']:.2f}%)")
        print(f"Win Rate: {results['best_metrics']['win_rate']:.2f}%")
        print(f"Profit Factor: {results['best_metrics']['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['best_metrics']['sharpe_ratio']:.2f}")
    else:
        metrics = results['results']['metrics']
        print(f"Return: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Report: {results['results']['report_file']}")

def main():
    """Main function"""
    # Check if running in interactive mode
    if len(sys.argv) == 1:
        run_interactive_backtest()
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create allocation dictionary
    allocation = create_allocation(args.assets, args.allocation)
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    # Log settings
    logger.info("Starting enhanced backtesting with the following settings:")
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Initial Capital: ${args.capital:.2f}")
    logger.info(f"Optimization: {'Yes' if args.optimize else 'No'}")
    if args.optimize:
        logger.info(f"Optimization Trials: {args.trials}")
    logger.info(f"Maximum Leverage: {args.max_leverage}x")
    logger.info(f"Allocation: {allocation}")
    
    # Run backtest
    results = run_full_backtest(
        assets=args.assets,
        strategies=args.strategies,
        allocation=allocation,
        optimize=args.optimize,
        num_trials=args.trials,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital
    )
    
    # Save parameters if requested
    if args.optimize and args.save_parameters:
        save_parameters(results['best_params'])
    
    # Print summary
    if args.optimize:
        logger.info("Optimization completed")
        logger.info(f"Best Return: ${results['best_metrics']['total_return']:.2f} "
                   f"({results['best_metrics']['total_return_pct']:.2f}%)")
        logger.info(f"Win Rate: {results['best_metrics']['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {results['best_metrics']['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {results['best_metrics']['sharpe_ratio']:.2f}")
    else:
        logger.info("Backtest completed")
        metrics = results['results']['metrics']
        logger.info(f"Return: ${metrics['total_return']:.2f} "
                   f"({metrics['total_return_pct']:.2f}%)")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Report: {results['results']['report_file']}")

if __name__ == "__main__":
    main()