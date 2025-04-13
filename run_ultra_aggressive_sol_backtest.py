#!/usr/bin/env python3
"""
Run Ultra-Aggressive SOL/USD Backtest

This script runs the ultra-aggressive SOL/USD backtest with parameters optimized for
maximum returns while maintaining high accuracy.

Features:
- Increased position sizing (48% vs 20% standard)
- Lower ML confidence threshold (0.65 vs 0.7)
- Reduced signal thresholds (0.15 vs 0.2) for more trading opportunities 
- Increased maximum margin cap to 60% (from 50%)
- Raised base leverage to 3.5x (from 3.0x)
- Increased maximum leverage cap to 12x (from 10x)
- Higher position size multiplier in normal trends
- More concurrent positions (5 vs 3 standard)
- Tighter profit targets and stop losses
- Advanced trailing stop logic with dynamic adjustments
"""

import os
import sys
import logging
from aggressive_sol_backtest import run_aggressive_backtest, create_detailed_report, plot_pnl_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/ultra_aggressive_sol_backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Constants
DEFAULT_INITIAL_CAPITAL = 20000.0
OUTPUT_DIR = 'backtest_results/ultra_aggressive'

def run_ultra_aggressive_backtest():
    """
    Run the ultra-aggressive SOL/USD backtest and generate detailed reports
    """
    logger.info("=" * 80)
    logger.info("Starting ULTRA-AGGRESSIVE SOL/USD Backtest")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run the backtest
    results = run_aggressive_backtest(
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        plot_results=True
    )
    
    if results:
        # Create detailed report
        report_path = os.path.join(OUTPUT_DIR, 'ultra_aggressive_backtest_report_SOLUSD_1h.txt')
        create_detailed_report(results, report_path)
        
        # Create PnL distribution plot
        if 'trades' in results:
            pnl_plot_path = os.path.join(OUTPUT_DIR, 'ultra_aggressive_pnl_distribution_SOLUSD_1h.png')
            plot_pnl_distribution(results['trades'], pnl_plot_path)
        
        logger.info(f"Ultra-aggressive backtest completed")
        logger.info(f"Detailed report available at: {report_path}")
        
        # Display key metrics
        metrics = results['performance']['trading_metrics']
        logger.info("=" * 80)
        logger.info("ULTRA-AGGRESSIVE BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Initial Capital:   ${metrics['initial_capital']:.2f}")
        logger.info(f"Final Capital:     ${metrics['final_capital']:.2f}")
        logger.info(f"Total Return:      {metrics['total_return_pct']*100:.2f}% (${metrics['total_return']:.2f})")
        logger.info(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        logger.info(f"Win Rate:          {metrics['win_rate']*100:.2f}% ({metrics['win_count']}/{metrics['total_trades']})")
        logger.info(f"Profit Factor:     {metrics['profit_factor']:.2f}")
        logger.info(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Avg Win:           ${metrics['average_win']:.2f}")
        logger.info(f"Avg Loss:          ${metrics['average_loss']:.2f}")
        logger.info("=" * 80)
        
        return results
    else:
        logger.error("Backtest failed to produce results")
        return None

if __name__ == "__main__":
    run_ultra_aggressive_backtest()