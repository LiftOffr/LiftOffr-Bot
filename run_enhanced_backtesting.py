#!/usr/bin/env python3
"""
Run Enhanced Backtesting System

This script runs comprehensive backtests using the enhanced backtesting system
to evaluate and optimize trading strategies with high accuracy.

It provides a convenient CLI interface for running various backtesting scenarios:
1. Single strategy backtests with parameter optimization
2. Multi-strategy and multi-asset backtests
3. ML-enhanced strategy backtests
4. Cross-timeframe backtests
5. Walk-forward optimization

Usage examples:
  python run_enhanced_backtesting.py --strategy arima --optimize
  python run_enhanced_backtesting.py --multi-strategy --multi-asset
  python run_enhanced_backtesting.py --strategy ml_enhanced --walk-forward
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta

# Import enhanced backtesting system
from enhanced_backtesting import (
    EnhancedBacktester,
    load_historical_data,
    load_multi_timeframe_data,
    optimize_strategy_parameters,
    run_cross_timeframe_backtest
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
RESULTS_DIR = "backtest_results/enhanced"
PARAMETER_SETS_DIR = "optimization_results"

# Ensure results directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PARAMETER_SETS_DIR, exist_ok=True)

def run_single_backtest(args):
    """
    Run a single backtest
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running single backtest for {args.symbol} on {args.timeframe} timeframe "
               f"with strategy {args.strategy}")
    
    # Load data
    data = load_historical_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    if data is None:
        logger.error("Cannot proceed: data not found")
        return
    
    # Import strategies
    from trading_strategy import TradingStrategy
    from arima_strategy import ARIMAStrategy
    from integrated_strategy import IntegratedStrategy
    
    try:
        from ml_enhanced_strategy import MLEnhancedStrategy
        ml_available = True
    except ImportError:
        logger.warning("ML enhanced strategy not available, skipping ML-related options")
        ml_available = False
    
    # Select strategy class
    strategy_classes = {
        "arima": ARIMAStrategy,
        "integrated": IntegratedStrategy
    }
    
    if ml_available:
        strategy_classes["ml_enhanced"] = MLEnhancedStrategy
    
    strategy_class = strategy_classes.get(args.strategy.lower())
    if strategy_class is None:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Determine if we should optimize
    if args.optimize:
        # Define parameter grid
        param_grid = {}
        
        if args.strategy.lower() == "arima":
            param_grid = {
                "lookback_period": [20, 30, 40, 50],
                "signal_threshold": [0.1, 0.2, 0.3],
                "atr_period": [14, 21, 28],
                "atr_multiplier": [1.5, 2.0, 2.5, 3.0]
            }
        elif args.strategy.lower() == "integrated":
            param_grid = {
                "signal_smoothing": [2, 3, 5],
                "trend_strength_threshold": [0.3, 0.4, 0.5],
                "volatility_filter_threshold": [0.005, 0.008, 0.01],
                "min_adx_threshold": [20, 25, 30]
            }
        elif args.strategy.lower() == "ml_enhanced" and ml_available:
            param_grid = {
                "ml_influence": [0.3, 0.5, 0.7],
                "confidence_threshold": [0.5, 0.6, 0.7]
            }
        
        # Create backtester
        backtester = EnhancedBacktester(
            data=data,
            trading_pair=args.symbol,
            initial_capital=args.capital,
            timeframe=args.timeframe,
            include_fees=not args.no_fees,
            enable_slippage=not args.no_slippage
        )
        
        # Run walk-forward optimization if requested
        if args.walk_forward:
            logger.info("Running walk-forward optimization")
            
            best_params, cv_results = backtester.run_walk_forward_optimization(
                strategy_class=strategy_class,
                param_grid=param_grid,
                n_splits=5,
                scoring="total_return_pct" if args.strategy.lower() != "ml_enhanced" else "sharpe_ratio"
            )
        else:
            # Run standard optimization
            logger.info("Running standard parameter optimization")
            
            best_params, cv_results = optimize_strategy_parameters(
                symbol=args.symbol,
                timeframe=args.timeframe,
                strategy_class=strategy_class,
                param_grid=param_grid,
                scoring="total_return_pct" if args.strategy.lower() != "ml_enhanced" else "sharpe_ratio",
                initial_capital=args.capital
            )
        
        # Create strategy with optimized parameters
        logger.info(f"Creating strategy with best parameters: {best_params}")
        strategy = strategy_class(args.symbol, **best_params)
        
        # Run backtest with optimized parameters
        logger.info("Running backtest with optimized parameters")
        result = backtester.run_backtest({"strategy": strategy})
        
        # Save optimized parameters
        param_file = os.path.join(
            PARAMETER_SETS_DIR, 
            f"{args.symbol}_{args.timeframe}_{args.strategy}_params.json"
        )
        with open(param_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Optimized parameters saved to {param_file}")
        
    else:
        # Create backtester
        backtester = EnhancedBacktester(
            data=data,
            trading_pair=args.symbol,
            initial_capital=args.capital,
            timeframe=args.timeframe,
            include_fees=not args.no_fees,
            enable_slippage=not args.no_slippage
        )
        
        # Check if saved parameters exist
        param_file = os.path.join(
            PARAMETER_SETS_DIR, 
            f"{args.symbol}_{args.timeframe}_{args.strategy}_params.json"
        )
        
        if os.path.exists(param_file) and args.use_best_params:
            # Load saved parameters
            with open(param_file, 'r') as f:
                best_params = json.load(f)
            logger.info(f"Loaded saved parameters: {best_params}")
            
            # Create strategy with saved parameters
            strategy = strategy_class(args.symbol, **best_params)
        else:
            # Create strategy with default parameters
            strategy = strategy_class(args.symbol)
        
        # Run backtest
        logger.info("Running backtest with standard parameters")
        result = backtester.run_backtest({"strategy": strategy})
    
    # Print summary
    print("\nBacktest Results:")
    print(f"Strategy: {args.strategy}")
    print("-" * 80)
    print(f"Total Return: {result['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {result['win_rate']*100:.2f}%")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print(f"Total Trades: {result['total_trades']}")
    
    # Regime-specific performance
    print("\nPerformance by Market Regime:")
    print("-" * 80)
    print(f"{'Regime':<20} {'Trades':<10} {'Win Rate':<15} {'Profit Factor':<15}")
    print("-" * 80)
    
    for regime, perf in result['regime_performance'].items():
        if perf['trades'] > 0:
            print(f"{regime:<20} {perf['trades']:<10} {perf['win_rate']*100:<15.2f} {perf['profit_factor']:<15.2f}")
    
    # Plot if requested
    if args.plot:
        plot_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_{args.strategy}_backtest.png"
        ) if not args.no_save else None
        
        backtester.plot_results(plot_file)
    
    # Save results
    if not args.no_save:
        result_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_{args.strategy}_results.json"
        )
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {result_file}")


def run_multi_strategy_backtest(args):
    """
    Run a multi-strategy backtest
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running multi-strategy backtest for {args.symbol} on {args.timeframe} timeframe")
    
    # Load data
    data = load_historical_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    if data is None:
        logger.error("Cannot proceed: data not found")
        return
    
    # Import strategies
    from trading_strategy import TradingStrategy
    from arima_strategy import ARIMAStrategy
    from integrated_strategy import IntegratedStrategy
    
    try:
        from ml_enhanced_strategy import MLEnhancedStrategy
        ml_available = True
    except ImportError:
        logger.warning("ML enhanced strategy not available")
        ml_available = False
    
    # Create backtester
    backtester = EnhancedBacktester(
        data=data,
        trading_pair=args.symbol,
        initial_capital=args.capital,
        timeframe=args.timeframe,
        include_fees=not args.no_fees,
        enable_slippage=not args.no_slippage
    )
    
    # Create strategies
    strategies = {}
    
    # Check if we should use saved parameters
    if args.use_best_params:
        # ARIMA strategy
        arima_param_file = os.path.join(
            PARAMETER_SETS_DIR, 
            f"{args.symbol}_{args.timeframe}_arima_params.json"
        )
        
        if os.path.exists(arima_param_file):
            with open(arima_param_file, 'r') as f:
                arima_params = json.load(f)
            logger.info(f"Loaded ARIMA parameters: {arima_params}")
            strategies["arima"] = ARIMAStrategy(args.symbol, **arima_params)
        else:
            logger.info("No saved ARIMA parameters found, using defaults")
            strategies["arima"] = ARIMAStrategy(args.symbol)
        
        # Integrated strategy
        integrated_param_file = os.path.join(
            PARAMETER_SETS_DIR, 
            f"{args.symbol}_{args.timeframe}_integrated_params.json"
        )
        
        if os.path.exists(integrated_param_file):
            with open(integrated_param_file, 'r') as f:
                integrated_params = json.load(f)
            logger.info(f"Loaded Integrated parameters: {integrated_params}")
            strategies["integrated"] = IntegratedStrategy(args.symbol, **integrated_params)
        else:
            logger.info("No saved Integrated parameters found, using defaults")
            strategies["integrated"] = IntegratedStrategy(args.symbol)
        
        # ML Enhanced strategy
        if ml_available and args.ml:
            ml_param_file = os.path.join(
                PARAMETER_SETS_DIR, 
                f"{args.symbol}_{args.timeframe}_ml_enhanced_params.json"
            )
            
            if os.path.exists(ml_param_file):
                with open(ml_param_file, 'r') as f:
                    ml_params = json.load(f)
                logger.info(f"Loaded ML Enhanced parameters: {ml_params}")
                
                # Create base strategy
                base_strategy = ARIMAStrategy(args.symbol)
                
                # Enhance with ML
                strategies["ml_enhanced"] = MLEnhancedStrategy(
                    base_strategy=base_strategy,
                    trading_pair=args.symbol,
                    **ml_params
                )
            else:
                logger.info("No saved ML Enhanced parameters found, using defaults")
                
                # Create base strategy
                base_strategy = ARIMAStrategy(args.symbol)
                
                # Enhance with ML
                strategies["ml_enhanced"] = MLEnhancedStrategy(
                    base_strategy=base_strategy,
                    trading_pair=args.symbol
                )
    else:
        # Use default parameters
        strategies["arima"] = ARIMAStrategy(args.symbol)
        strategies["integrated"] = IntegratedStrategy(args.symbol)
        
        if ml_available and args.ml:
            # Create base strategy
            base_strategy = ARIMAStrategy(args.symbol)
            
            # Enhance with ML
            strategies["ml_enhanced"] = MLEnhancedStrategy(
                base_strategy=base_strategy,
                trading_pair=args.symbol
            )
    
    # Run backtest
    logger.info(f"Running backtest with {len(strategies)} strategies")
    result = backtester.run_backtest(strategies)
    
    # Print summary
    print("\nMulti-Strategy Backtest Results:")
    print("-" * 80)
    print(f"Total Return: {result['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {result['win_rate']*100:.2f}%")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print(f"Total Trades: {result['total_trades']}")
    
    # Plot if requested
    if args.plot:
        plot_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_multi_strategy_backtest.png"
        ) if not args.no_save else None
        
        backtester.plot_results(plot_file)
    
    # Save results
    if not args.no_save:
        result_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_multi_strategy_results.json"
        )
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {result_file}")


def run_cross_timeframe_backtest(args):
    """
    Run a cross-timeframe backtest
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running cross-timeframe backtest for {args.symbol}")
    
    # Define timeframes to use
    timeframes = args.timeframes if args.timeframes else DEFAULT_TIMEFRAMES
    
    # Run cross-timeframe backtest
    from arima_strategy import ARIMAStrategy
    from enhanced_backtesting import run_cross_timeframe_backtest as run_ctf_backtest
    
    results = run_ctf_backtest(
        symbol=args.symbol,
        timeframes=timeframes,
        strategy_class=ARIMAStrategy,
        initial_capital=args.capital
    )
    
    # Print comparative summary
    print("\nCross-timeframe Backtest Results:")
    print("-" * 80)
    print(f"{'Timeframe':<10} {'Return %':<10} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate':<10} {'Profit Factor':<15}")
    print("-" * 80)
    
    for tf, res in results.items():
        print(f"{tf:<10} {res['total_return_pct']:<10.2f} {res['sharpe_ratio']:<10.2f} "
             f"{res['max_drawdown_pct']:<10.2f} {res['win_rate']*100:<10.2f} {res['profit_factor']:<15.2f}")
    
    # Save results
    if not args.no_save:
        result_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_cross_timeframe_results.json"
        )
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {result_file}")


def run_multi_asset_backtest(args):
    """
    Run a multi-asset backtest
    
    Args:
        args: Command line arguments
    """
    # Define symbols to use
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    
    # Define timeframe
    timeframe = args.timeframe
    
    logger.info(f"Running multi-asset backtest for {len(symbols)} symbols on {timeframe} timeframe")
    
    # Import strategies
    from trading_strategy import TradingStrategy
    from arima_strategy import ARIMAStrategy
    from integrated_strategy import IntegratedStrategy
    
    # Results by symbol
    all_results = {}
    
    # Run backtest for each symbol
    for symbol in symbols:
        logger.info(f"Running backtest for {symbol}")
        
        # Load data
        data = load_historical_data(symbol, timeframe, args.start_date, args.end_date)
        if data is None:
            logger.warning(f"Data not found for {symbol}, skipping")
            continue
        
        # Create backtester
        backtester = EnhancedBacktester(
            data=data,
            trading_pair=symbol,
            initial_capital=args.capital / len(symbols),  # Divide capital among assets
            timeframe=timeframe,
            include_fees=not args.no_fees,
            enable_slippage=not args.no_slippage
        )
        
        # Create strategies
        if args.multi_strategy:
            strategies = {
                "arima": ARIMAStrategy(symbol),
                "integrated": IntegratedStrategy(symbol)
            }
        else:
            strategies = {
                "arima": ARIMAStrategy(symbol)
            }
        
        # Run backtest
        result = backtester.run_backtest(strategies)
        
        # Store results
        all_results[symbol] = result
        
        # Print summary
        print(f"\nBacktest Results for {symbol}:")
        print("-" * 80)
        print(f"Total Return: {result['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Total Trades: {result['total_trades']}")
        
        # Plot if requested
        if args.plot:
            plot_file = os.path.join(
                RESULTS_DIR, 
                f"{symbol}_{timeframe}_backtest.png"
            ) if not args.no_save else None
            
            backtester.plot_results(plot_file)
    
    # Calculate portfolio metrics
    if all_results:
        # Combine metrics
        combined_return = sum(res['total_return_pct'] for res in all_results.values()) / len(all_results)
        combined_sharpe = sum(res['sharpe_ratio'] for res in all_results.values()) / len(all_results)
        max_drawdown = max(res['max_drawdown_pct'] for res in all_results.values())
        total_trades = sum(res['total_trades'] for res in all_results.values())
        
        # Print combined summary
        print("\nPortfolio Summary:")
        print("-" * 80)
        print(f"Average Return: {combined_return:.2f}%")
        print(f"Average Sharpe: {combined_sharpe:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {total_trades}")
        
        # Save results
        if not args.no_save:
            portfolio_file = os.path.join(
                RESULTS_DIR, 
                f"multi_asset_{timeframe}_portfolio_results.json"
            )
            with open(portfolio_file, 'w') as f:
                json.dump({
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "average_return": combined_return,
                    "average_sharpe": combined_sharpe,
                    "max_drawdown": max_drawdown,
                    "total_trades": total_trades,
                    "results_by_symbol": {s: {k: str(v) if isinstance(v, datetime) else v 
                                             for k, v in r.items() if k not in ['trades', 'equity_curve']} 
                                        for s, r in all_results.items()}
                }, f, indent=2)
            logger.info(f"Portfolio results saved to {portfolio_file}")


def run_ml_model_backtest(args):
    """
    Run a backtest using ML-enhanced models
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Running ML-enhanced backtest for {args.symbol} on {args.timeframe} timeframe")
    
    # Check if ML modules are available
    try:
        from ml_enhanced_strategy import MLEnhancedStrategy
        from ml_strategy_integrator import MLStrategyIntegrator
    except ImportError:
        logger.error("ML modules not available, cannot run ML-enhanced backtest")
        return
    
    # Load data
    data = load_historical_data(args.symbol, args.timeframe, args.start_date, args.end_date)
    if data is None:
        logger.error("Cannot proceed: data not found")
        return
    
    # Import base strategy
    from arima_strategy import ARIMAStrategy
    
    # Create backtester
    backtester = EnhancedBacktester(
        data=data,
        trading_pair=args.symbol,
        initial_capital=args.capital,
        timeframe=args.timeframe,
        include_fees=not args.no_fees,
        enable_slippage=not args.no_slippage
    )
    
    # Create base strategy
    base_strategy = ARIMAStrategy(args.symbol)
    
    # Determine ML parameters
    ml_params = {}
    
    if args.use_best_params:
        # Try to load saved parameters
        param_file = os.path.join(
            PARAMETER_SETS_DIR, 
            f"{args.symbol}_{args.timeframe}_ml_enhanced_params.json"
        )
        
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                ml_params = json.load(f)
            logger.info(f"Loaded saved ML parameters: {ml_params}")
    
    # Create ML-enhanced strategy
    ml_strategy = MLEnhancedStrategy(
        base_strategy=base_strategy,
        trading_pair=args.symbol,
        **ml_params
    )
    
    # Run backtest
    result = backtester.run_backtest({"ml_enhanced": ml_strategy})
    
    # Print summary
    print("\nML-Enhanced Backtest Results:")
    print("-" * 80)
    print(f"Total Return: {result['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {result['win_rate']*100:.2f}%")
    print(f"Profit Factor: {result['profit_factor']:.2f}")
    print(f"Total Trades: {result['total_trades']}")
    
    # Regime-specific performance
    print("\nPerformance by Market Regime:")
    print("-" * 80)
    print(f"{'Regime':<20} {'Trades':<10} {'Win Rate':<15} {'Profit Factor':<15}")
    print("-" * 80)
    
    for regime, perf in result['regime_performance'].items():
        if perf['trades'] > 0:
            print(f"{regime:<20} {perf['trades']:<10} {perf['win_rate']*100:<15.2f} {perf['profit_factor']:<15.2f}")
    
    # Plot if requested
    if args.plot:
        plot_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_ml_enhanced_backtest.png"
        ) if not args.no_save else None
        
        backtester.plot_results(plot_file)
    
    # Save results
    if not args.no_save:
        result_file = os.path.join(
            RESULTS_DIR, 
            f"{args.symbol}_{args.timeframe}_ml_enhanced_results.json"
        )
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Results saved to {result_file}")


def main():
    """Main function for the enhanced backtesting script"""
    parser = argparse.ArgumentParser(description="Run Enhanced Backtesting System")
    
    # General arguments
    parser.add_argument("--symbol", type=str, default="SOLUSD", 
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       help="Timeframe to use (default: 1h)")
    parser.add_argument("--strategy", type=str, default="arima", 
                       help="Strategy to backtest (default: arima)")
    parser.add_argument("--capital", type=float, default=20000.0, 
                       help="Initial capital (default: 20000.0)")
    parser.add_argument("--start-date", type=str, default=None, 
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, 
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-fees", action="store_true", 
                       help="Disable fee simulation")
    parser.add_argument("--no-slippage", action="store_true", 
                       help="Disable slippage simulation")
    parser.add_argument("--plot", action="store_true", 
                       help="Plot backtest results")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save results to disk")
    
    # Optimization arguments
    parser.add_argument("--optimize", action="store_true", 
                       help="Optimize strategy parameters")
    parser.add_argument("--walk-forward", action="store_true", 
                       help="Use walk-forward optimization")
    parser.add_argument("--use-best-params", action="store_true", 
                       help="Use previously optimized parameters")
    
    # Multi-strategy arguments
    parser.add_argument("--multi-strategy", action="store_true", 
                       help="Use multiple strategies")
    parser.add_argument("--ml", action="store_true", 
                       help="Include ML-enhanced strategies")
    
    # Multi-asset arguments
    parser.add_argument("--multi-asset", action="store_true", 
                       help="Run backtest for multiple assets")
    parser.add_argument("--symbols", nargs="+", default=None, 
                       help="List of symbols to backtest")
    
    # Cross-timeframe arguments
    parser.add_argument("--cross-timeframe", action="store_true", 
                       help="Run cross-timeframe backtest")
    parser.add_argument("--timeframes", nargs="+", default=None, 
                       help="List of timeframes to use")
    
    args = parser.parse_args()
    
    # Determine which type of backtest to run
    if args.multi_asset:
        run_multi_asset_backtest(args)
    elif args.cross_timeframe:
        run_cross_timeframe_backtest(args)
    elif args.multi_strategy:
        run_multi_strategy_backtest(args)
    elif args.ml and args.strategy.lower() != "ml_enhanced":
        run_ml_model_backtest(args)
    else:
        run_single_backtest(args)


if __name__ == "__main__":
    main()