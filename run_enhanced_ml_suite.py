#!/usr/bin/env python3
"""
Run Enhanced ML Suite

This script provides a unified interface to run all the enhanced ML components:
1. Comprehensive ML backtesting across all trading pairs and timeframes
2. Enhanced auto-pruning for model optimization
3. Advanced visualization dashboard for backtest results
4. ML ensemble weight optimization for different market conditions

Usage:
    python run_enhanced_ml_suite.py [--component COMPONENT]

Available components:
    - all: Run all components in sequence
    - backtest: Run comprehensive ML backtesting
    - prune: Run enhanced auto-pruning
    - visualize: Generate visualization dashboard
    - optimize: Run ensemble weight optimization
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"logs/enhanced_ml_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        'logs',
        'backtest_results',
        'backtest_results/ml_ensemble',
        'backtest_results/visualizations',
        'optimization_results',
        'optimization_results/ensemble_weights',
        'optimization_results/model_pruning'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def run_comprehensive_backtests():
    """Run comprehensive ML backtests"""
    logger.info("=== Running Comprehensive ML Backtests ===")
    
    # Import here to avoid module-level import issues
    from run_ml_backtest import run_comprehensive_backtests as run_backtests
    
    try:
        results = run_backtests()
        logger.info("Comprehensive ML backtests completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error running comprehensive backtests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_enhanced_auto_pruning():
    """Run enhanced auto-pruning"""
    logger.info("=== Running Enhanced Auto-Pruning ===")
    
    # Import here to avoid module-level import issues
    from enhanced_auto_pruning import EnhancedAutoPruner
    from advanced_ensemble_model import DynamicWeightedEnsemble
    
    try:
        # Define pairs and timeframes to run pruning for
        pairs = ['SOL/USD']
        timeframes = ['1h']
        
        all_results = {}
        
        for pair in pairs:
            pair_results = {}
            for timeframe in timeframes:
                logger.info(f"Running auto-pruning for {pair} ({timeframe})")
                
                # Create ensemble model and pruner
                ensemble = DynamicWeightedEnsemble(pair, timeframe)
                pruner = EnhancedAutoPruner(
                    ensemble_model=ensemble,
                    min_models_per_type=1,
                    min_total_models=4,
                    base_performance_threshold=0.55
                )
                
                # Run pruning (force=True to bypass scheduling)
                pruned, kept, details = pruner.run_pruning(force=True)
                
                pair_results[timeframe] = {
                    'pruned_models': pruned,
                    'kept_models': kept,
                    'details': details
                }
                
                logger.info(f"Auto-pruning complete for {pair} ({timeframe})")
                logger.info(f"Pruned {len(pruned)} models, kept {len(kept)} models")
            
            all_results[pair] = pair_results
        
        logger.info("Enhanced auto-pruning completed successfully")
        return all_results
    except Exception as e:
        logger.error(f"Error running enhanced auto-pruning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_visualization_dashboard():
    """Generate advanced visualization dashboard"""
    logger.info("=== Generating Advanced Visualization Dashboard ===")
    
    # Import here to avoid module-level import issues
    from visualization_dashboard import BacktestVisualizationDashboard
    
    try:
        dashboard = BacktestVisualizationDashboard()
        output_dir = dashboard.create_dashboard()
        
        if output_dir:
            logger.info(f"Visualization dashboard created successfully at {output_dir}")
            return output_dir
        else:
            logger.error("Failed to create visualization dashboard")
            return None
    except Exception as e:
        logger.error(f"Error generating visualization dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_ensemble_weight_optimization():
    """Run ensemble weight optimization"""
    logger.info("=== Running Ensemble Weight Optimization ===")
    
    # Import here to avoid module-level import issues
    from optimize_ensemble_weights import EnsembleWeightOptimizer
    
    try:
        # Define pairs and timeframes to optimize
        pairs = ['SOL/USD']
        timeframes = ['1h']
        
        all_results = {}
        
        for pair in pairs:
            pair_results = {}
            for timeframe in timeframes:
                logger.info(f"Optimizing weights for {pair} ({timeframe})")
                
                # Create optimizer with a small population and few generations for demonstration
                optimizer = EnsembleWeightOptimizer(
                    trading_pair=pair,
                    timeframe=timeframe,
                    fitness_metric='f1',
                    population_size=10,  # Small for demo purposes
                    generations=5  # Small for demo purposes
                )
                
                # Run optimization
                optimized_weights = optimizer.optimize_all_regimes()
                
                if optimized_weights:
                    pair_results[timeframe] = optimized_weights
                    logger.info(f"Weight optimization complete for {pair} ({timeframe})")
                else:
                    logger.error(f"Weight optimization failed for {pair} ({timeframe})")
            
            all_results[pair] = pair_results
        
        logger.info("Ensemble weight optimization completed successfully")
        return all_results
    except Exception as e:
        logger.error(f"Error running ensemble weight optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def run_all_components():
    """Run all ML suite components in sequence"""
    logger.info("=== Running All ML Suite Components ===")
    
    results = {}
    
    # Run backtests
    logger.info("Step 1/4: Running comprehensive backtests")
    backtest_results = run_comprehensive_backtests()
    results['backtest'] = backtest_results
    
    # Generate visualizations
    logger.info("Step 2/4: Generating visualization dashboard")
    visualization_results = generate_visualization_dashboard()
    results['visualize'] = visualization_results
    
    # Run auto-pruning
    logger.info("Step 3/4: Running enhanced auto-pruning")
    pruning_results = run_enhanced_auto_pruning()
    results['prune'] = pruning_results
    
    # Run weight optimization
    logger.info("Step 4/4: Running ensemble weight optimization")
    optimization_results = run_ensemble_weight_optimization()
    results['optimize'] = optimization_results
    
    logger.info("All ML suite components completed")
    return results

def main():
    """Main function to run the ML suite"""
    parser = argparse.ArgumentParser(description='Run Enhanced ML Suite')
    parser.add_argument('--component', type=str, default='all',
                       help='Component to run (all, backtest, prune, visualize, optimize)')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    start_time = time.time()
    
    # Run the requested component
    if args.component == 'all':
        results = run_all_components()
    elif args.component == 'backtest':
        results = run_comprehensive_backtests()
    elif args.component == 'prune':
        results = run_enhanced_auto_pruning()
    elif args.component == 'visualize':
        results = generate_visualization_dashboard()
    elif args.component == 'optimize':
        results = run_ensemble_weight_optimization()
    else:
        logger.error(f"Unknown component: {args.component}")
        print(f"Error: Unknown component '{args.component}'")
        print("Available components: all, backtest, prune, visualize, optimize")
        return 1
    
    elapsed_time = time.time() - start_time
    logger.info(f"Component '{args.component}' completed in {elapsed_time:.2f} seconds")
    
    # Print summary
    print(f"\nEnhanced ML Suite - Component '{args.component}' Summary:")
    if results:
        print("✅ Execution completed successfully")
    else:
        print("❌ Execution failed")
    
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Log file: logs/enhanced_ml_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    return 0 if results else 1

if __name__ == "__main__":
    sys.exit(main())