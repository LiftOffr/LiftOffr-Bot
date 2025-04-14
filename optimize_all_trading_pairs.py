#!/usr/bin/env python3
"""
Comprehensive Trading Pair Optimization

This script runs a comprehensive optimization process for all trading pairs:
1. Fetches historical data for all configured trading pairs
2. Runs backtests with different parameter combinations
3. Optimizes parameters per pair to maximize returns
4. Creates ensemble models with optimal weights
5. Saves optimized parameters for live trading

The optimization process allows all parameters to be dynamically adjusted
per trade based on confidence levels and market conditions.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our modules
from dynamic_parameter_optimizer import DynamicParameterOptimizer
from utils.backtest_engine import BacktestEngine
from utils.data_loader import HistoricalDataLoader
from utils.market_analyzer import MarketAnalyzer
from utils.visualization import plot_optimization_results
from ml_models.ensemble_builder import EnsembleModelBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("optimization_results/optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "config/ml_config.json"
OPTIMIZATION_RESULTS_DIR = "optimization_results"
BACKTEST_RESULTS_DIR = "backtest_results"
MODEL_WEIGHTS_DIR = "models/ensemble"
DEFAULT_LOOKBACK_DAYS = 365  # 1 year of historical data
DEFAULT_TEST_RATIO = 0.3  # 30% for testing, 70% for training

class TradingPairOptimizer:
    """
    Comprehensive trading pair optimization system that maximizes returns
    through parameter optimization and ensemble model training.
    """
    
    def __init__(self, config_path: str = CONFIG_PATH):
        """
        Initialize the trading pair optimizer.
        
        Args:
            config_path: Path to the ML configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Create directories if they don't exist
        os.makedirs(OPTIMIZATION_RESULTS_DIR, exist_ok=True)
        os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)
        os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
        
        # Initialize components
        self.parameter_optimizer = DynamicParameterOptimizer(base_config_path=config_path)
        self.backtest_engine = BacktestEngine()
        self.data_loader = HistoricalDataLoader()
        self.market_analyzer = MarketAnalyzer()
        self.ensemble_builder = EnsembleModelBuilder()
        
        # Store optimization results
        self.optimization_results = {}
        
    def load_config(self) -> None:
        """Load the ML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            
            # Extract trading pairs
            self.pairs = self.config.get("pairs", ["SOL/USD", "BTC/USD", "ETH/USD"])
            logger.info(f"Trading pairs to optimize: {self.pairs}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            self.config = {
                "pairs": ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"],
                "base_leverage": 20.0,
                "max_leverage": 125.0,
                "risk_percentage": 0.20,
                "confidence_threshold": 0.65,
                "time_scales": ["1h", "4h", "1d"],
                "use_ensemble": True,
                "ensemble_models": ["tcn", "lstm", "attention_gru"]
            }
            self.pairs = self.config["pairs"]
            logger.warning("Using default configuration")
    
    def fetch_historical_data(self, days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all trading pairs.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping pair symbols to historical data DataFrames
        """
        logger.info(f"Fetching {days} days of historical data for {len(self.pairs)} pairs")
        historical_data = {}
        
        for pair in self.pairs:
            try:
                pair_data = self.data_loader.fetch_historical_data(
                    pair=pair,
                    timeframe="1h",  # Use 1h as base timeframe
                    days=days
                )
                historical_data[pair] = pair_data
                logger.info(f"Fetched {len(pair_data)} candles for {pair}")
            except Exception as e:
                logger.error(f"Error fetching historical data for {pair}: {e}")
                
        return historical_data
    
    def _split_data(self, data: pd.DataFrame, test_ratio: float = DEFAULT_TEST_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            data: Historical price data
            test_ratio: Ratio of data to use for testing
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        split_idx = int(len(data) * (1 - test_ratio))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        return train_data, test_data
    
    def analyze_market_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market regimes for a given dataset.
        
        Args:
            data: Historical price data
            
        Returns:
            Dictionary with market regime analysis
        """
        return self.market_analyzer.analyze_market_regimes(data)
    
    def _generate_parameter_grid(self, pair: str) -> List[Dict[str, Any]]:
        """
        Generate a grid of parameters to test for a given pair.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            List of parameter dictionaries to test
        """
        # Get current parameters for the pair
        current_params = self.parameter_optimizer.get_pair_params(pair)
        
        # Generate a grid of parameters around the current values
        param_grid = []
        
        # Risk percentage variations
        risk_percentages = [
            max(0.05, current_params["risk_percentage"] * 0.7),
            current_params["risk_percentage"],
            min(0.4, current_params["risk_percentage"] * 1.3)
        ]
        
        # Leverage variations
        base_leverages = [
            max(5.0, current_params["base_leverage"] * 0.7),
            current_params["base_leverage"],
            min(100.0, current_params["base_leverage"] * 1.3)
        ]
        
        # Max leverage variations (always higher than base leverage)
        max_leverages = [
            min(125.0, bl * 2.0) for bl in base_leverages
        ]
        
        # Confidence threshold variations
        confidence_thresholds = [
            max(0.55, current_params["confidence_threshold"] * 0.9),
            current_params["confidence_threshold"],
            min(0.95, current_params["confidence_threshold"] * 1.1)
        ]
        
        # Signal strength threshold variations
        signal_strengths = [
            max(0.45, current_params.get("signal_strength_threshold", 0.6) * 0.9),
            current_params.get("signal_strength_threshold", 0.6),
            min(0.95, current_params.get("signal_strength_threshold", 0.6) * 1.1)
        ]
        
        # ATR multiplier variations
        atr_multipliers = [
            max(1.0, current_params.get("trailing_stop_atr_multiplier", 3.0) * 0.8),
            current_params.get("trailing_stop_atr_multiplier", 3.0),
            min(6.0, current_params.get("trailing_stop_atr_multiplier", 3.0) * 1.2)
        ]
        
        # Exit multiplier variations
        exit_multipliers = [
            max(1.0, current_params.get("exit_multiplier", 1.5) * 0.8),
            current_params.get("exit_multiplier", 1.5),
            min(3.0, current_params.get("exit_multiplier", 1.5) * 1.2)
        ]
        
        # Strategy weight variations
        strategy_weights = [
            {"arima": 0.3, "adaptive": 0.7},
            {"arima": 0.5, "adaptive": 0.5},
            {"arima": 0.7, "adaptive": 0.3}
        ]
        
        # Generate all combinations (using a simplified approach to avoid a huge grid)
        # This generates ~27 parameter combinations instead of 3^7 = 2187
        for risk in risk_percentages:
            for i, base_lev in enumerate(base_leverages):
                max_lev = max_leverages[i]
                for conf in confidence_thresholds:
                    for signal in signal_strengths:
                        # Use index approach to keep related parameters together
                        for idx in range(3):
                            atr_mult = atr_multipliers[min(idx, len(atr_multipliers)-1)]
                            exit_mult = exit_multipliers[min(idx, len(exit_multipliers)-1)]
                            strat_weight = strategy_weights[min(idx, len(strategy_weights)-1)]
                            
                            param_grid.append({
                                "risk_percentage": risk,
                                "base_leverage": base_lev,
                                "max_leverage": max_lev,
                                "confidence_threshold": conf,
                                "signal_strength_threshold": signal,
                                "trailing_stop_atr_multiplier": atr_mult,
                                "exit_multiplier": exit_mult,
                                "strategy_weights": strat_weight
                            })
        
        return param_grid
    
    def backtest_pair(self, pair: str, historical_data: pd.DataFrame, 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Backtest a single parameter set for a trading pair.
        
        Args:
            pair: Trading pair symbol
            historical_data: Historical price data
            parameters: Dictionary of parameters to test
            
        Returns:
            Dictionary with backtest results
        """
        # Split data into training and testing sets
        train_data, test_data = self._split_data(historical_data)
        
        # Train ML models on training data
        ml_model_results = self.train_ml_models(pair, train_data)
        
        # Run backtest on test data
        backtest_results = self.backtest_engine.run_backtest(
            pair=pair,
            data=test_data,
            parameters=parameters,
            ml_models=ml_model_results["models"]
        )
        
        # Add ML model metrics to results
        backtest_results.update({
            "ml_accuracy": ml_model_results.get("accuracy", 0.0),
            "ml_precision": ml_model_results.get("precision", 0.0),
            "ml_recall": ml_model_results.get("recall", 0.0),
            "parameters": parameters
        })
        
        return backtest_results
    
    def train_ml_models(self, pair: str, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models for a trading pair.
        
        Args:
            pair: Trading pair symbol
            train_data: Training data
            
        Returns:
            Dictionary with trained models and metrics
        """
        # Prepare features
        features = self.extract_features(train_data)
        
        # Train models
        models = {}
        metrics = {}
        
        # Train individual models
        for model_type in self.config.get("ensemble_models", ["tcn", "lstm", "attention_gru"]):
            model, model_metrics = self.train_model(pair, model_type, features)
            models[model_type] = model
            metrics[model_type] = model_metrics
        
        # Combine metrics
        overall_metrics = {
            "accuracy": np.mean([m.get("accuracy", 0.0) for m in metrics.values()]),
            "precision": np.mean([m.get("precision", 0.0) for m in metrics.values()]),
            "recall": np.mean([m.get("recall", 0.0) for m in metrics.values()])
        }
        
        return {
            "models": models,
            **overall_metrics
        }
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from historical data for ML models.
        
        Args:
            data: Historical price data
            
        Returns:
            DataFrame with extracted features
        """
        # This would be implemented in a real system to extract technical indicators,
        # price patterns, volatility metrics, etc.
        # For this implementation, we'll assume it's done and return the same data
        return data
    
    def train_model(self, pair: str, model_type: str, features: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
        """
        Train a specific ML model.
        
        Args:
            pair: Trading pair symbol
            model_type: Type of model to train
            features: Feature data
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        # This would be implemented to train the specific model type
        # For this implementation, we'll return mock results
        
        # In a real implementation, this would train the model and return actual metrics
        metrics = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85
        }
        
        return None, metrics  # Model would be returned in real implementation
    
    def optimize_pair(self, pair: str, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize parameters for a single trading pair.
        
        Args:
            pair: Trading pair symbol
            historical_data: Dictionary of historical data for all pairs
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing parameters for {pair}")
        
        # Get pair data
        pair_data = historical_data.get(pair)
        if pair_data is None:
            logger.error(f"No historical data available for {pair}")
            return {"error": "No historical data available"}
        
        # Analyze market regimes
        market_analysis = self.analyze_market_regimes(pair_data)
        logger.info(f"Market regime analysis for {pair}: {market_analysis}")
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(pair)
        logger.info(f"Generated {len(param_grid)} parameter combinations to test for {pair}")
        
        # Run backtests for each parameter set
        backtest_results = []
        
        for i, params in enumerate(param_grid):
            logger.info(f"Running backtest {i+1}/{len(param_grid)} for {pair}")
            
            try:
                result = self.backtest_pair(pair, pair_data, params)
                backtest_results.append({
                    "params": params,
                    "results": result
                })
                
                logger.info(f"Backtest {i+1} for {pair}: Return: {result.get('total_return', 0):.2f}, "
                           f"Win Rate: {result.get('win_rate', 0):.2f}, "
                           f"Profit Factor: {result.get('profit_factor', 0):.2f}")
            except Exception as e:
                logger.error(f"Error in backtest {i+1} for {pair}: {e}")
        
        # Find the best parameter set
        if backtest_results:
            # Sort by total return
            backtest_results = sorted(
                backtest_results, 
                key=lambda x: (
                    x["results"].get("total_return", 0) * 0.6 + 
                    x["results"].get("sharpe_ratio", 0) * 0.4
                ),
                reverse=True
            )
            
            best_result = backtest_results[0]
            best_params = best_result["params"]
            best_metrics = best_result["results"]
            
            logger.info(f"Best parameters for {pair}:")
            logger.info(f"  Return: {best_metrics.get('total_return', 0):.2f}")
            logger.info(f"  Win Rate: {best_metrics.get('win_rate', 0):.2f}")
            logger.info(f"  Profit Factor: {best_metrics.get('profit_factor', 0):.2f}")
            logger.info(f"  Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.2f}")
            
            # Save results
            self.save_optimization_results(pair, backtest_results)
            
            # Update optimized parameters
            self.parameter_optimizer.optimize_parameters_for_pair(pair, best_metrics)
            
            return {
                "pair": pair,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "all_results": backtest_results
            }
        else:
            logger.error(f"No successful backtest results for {pair}")
            return {"error": "No successful backtest results"}
    
    def save_optimization_results(self, pair: str, results: List[Dict]) -> None:
        """
        Save optimization results to file.
        
        Args:
            pair: Trading pair symbol
            results: List of backtest results
        """
        pair_safe = pair.replace('/', '_')
        filename = f"{OPTIMIZATION_RESULTS_DIR}/{pair_safe}_optimization.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "pair": pair,
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                }, f, indent=2, default=str)
            logger.info(f"Saved optimization results to {filename}")
        except Exception as e:
            logger.error(f"Error saving optimization results for {pair}: {e}")
    
    def create_ensemble_models(self, pair: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create ensemble models for a trading pair.
        
        Args:
            pair: Trading pair symbol
            historical_data: Historical price data
            
        Returns:
            Dictionary with ensemble model information
        """
        # Split data for training and validation
        train_data, val_data = self._split_data(historical_data, test_ratio=0.2)
        
        # Train individual models
        ml_model_results = self.train_ml_models(pair, train_data)
        models = ml_model_results.get("models", {})
        
        # Create and train ensemble model
        ensemble_model = self.ensemble_builder.build_ensemble(
            pair=pair,
            models=models,
            weights=None  # Auto-determine weights
        )
        
        # Evaluate ensemble model
        ensemble_metrics = self.ensemble_builder.evaluate_ensemble(
            ensemble_model=ensemble_model,
            validation_data=val_data
        )
        
        # Save ensemble model weights
        pair_safe = pair.replace('/', '_')
        weights_path = f"{MODEL_WEIGHTS_DIR}/{pair_safe}_weights.json"
        self.ensemble_builder.save_weights(ensemble_model, weights_path)
        
        logger.info(f"Created ensemble model for {pair} with accuracy: {ensemble_metrics.get('accuracy', 0):.4f}")
        
        return {
            "pair": pair,
            "ensemble_model": ensemble_model,
            "metrics": ensemble_metrics,
            "weights_path": weights_path
        }
    
    def run_optimization_for_all_pairs(self, days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, Any]:
        """
        Run optimization for all trading pairs.
        
        Args:
            days: Number of days of historical data to use
            
        Returns:
            Dictionary with optimization results for all pairs
        """
        start_time = time.time()
        logger.info(f"Starting optimization for {len(self.pairs)} trading pairs")
        
        # Fetch historical data for all pairs
        historical_data = self.fetch_historical_data(days=days)
        
        # Run optimization for each pair
        optimization_results = {}
        
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(self.pairs))) as executor:
            future_to_pair = {
                executor.submit(self.optimize_pair, pair, historical_data): pair 
                for pair in self.pairs
            }
            
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    optimization_results[pair] = result
                except Exception as e:
                    logger.error(f"Error optimizing {pair}: {e}")
                    optimization_results[pair] = {"error": str(e)}
        
        # Create ensemble models for each pair
        for pair in self.pairs:
            if pair in historical_data:
                try:
                    ensemble_result = self.create_ensemble_models(pair, historical_data[pair])
                    if pair in optimization_results:
                        optimization_results[pair]["ensemble"] = ensemble_result
                except Exception as e:
                    logger.error(f"Error creating ensemble model for {pair}: {e}")
                    if pair in optimization_results:
                        optimization_results[pair]["ensemble_error"] = str(e)
        
        # Save summary of all results
        summary_file = f"{OPTIMIZATION_RESULTS_DIR}/optimization_summary.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "pairs": self.pairs,
                    "results": {
                        pair: {
                            "best_metrics": result.get("best_metrics", {}),
                            "ensemble_accuracy": result.get("ensemble", {}).get("metrics", {}).get("accuracy", 0)
                        } for pair, result in optimization_results.items()
                    }
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving optimization summary: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Optimization completed in {duration:.2f} seconds")
        
        return optimization_results
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> None:
        """
        Generate a human-readable optimization report.
        
        Args:
            results: Optimization results for all pairs
        """
        report_file = f"{OPTIMIZATION_RESULTS_DIR}/optimization_report.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("TRADING PAIR OPTIMIZATION REPORT\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for pair, result in results.items():
                    f.write(f"PAIR: {pair}\n")
                    f.write("-" * 40 + "\n")
                    
                    if "error" in result:
                        f.write(f"ERROR: {result['error']}\n\n")
                        continue
                    
                    best_metrics = result.get("best_metrics", {})
                    best_params = result.get("best_params", {})
                    
                    f.write("Performance Metrics:\n")
                    f.write(f"  Total Return: {best_metrics.get('total_return', 0):.2f}\n")
                    f.write(f"  Win Rate: {best_metrics.get('win_rate', 0):.2f}\n")
                    f.write(f"  Profit Factor: {best_metrics.get('profit_factor', 0):.2f}\n")
                    f.write(f"  Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.2f}\n")
                    f.write(f"  Max Drawdown: {best_metrics.get('max_drawdown', 0):.2f}\n")
                    
                    f.write("\nOptimized Parameters:\n")
                    f.write(f"  Risk Percentage: {best_params.get('risk_percentage', 0):.2f}\n")
                    f.write(f"  Base Leverage: {best_params.get('base_leverage', 0):.1f}\n")
                    f.write(f"  Max Leverage: {best_params.get('max_leverage', 0):.1f}\n")
                    f.write(f"  Confidence Threshold: {best_params.get('confidence_threshold', 0):.2f}\n")
                    f.write(f"  Signal Strength Threshold: {best_params.get('signal_strength_threshold', 0):.2f}\n")
                    
                    strategy_weights = best_params.get("strategy_weights", {})
                    f.write(f"  Strategy Weights: ARIMA={strategy_weights.get('arima', 0):.2f}, "
                           f"Adaptive={strategy_weights.get('adaptive', 0):.2f}\n")
                    
                    if "ensemble" in result:
                        ensemble = result["ensemble"]
                        ensemble_metrics = ensemble.get("metrics", {})
                        f.write("\nEnsemble Model:\n")
                        f.write(f"  Accuracy: {ensemble_metrics.get('accuracy', 0):.4f}\n")
                        f.write(f"  Precision: {ensemble_metrics.get('precision', 0):.4f}\n")
                        f.write(f"  Recall: {ensemble_metrics.get('recall', 0):.4f}\n")
                        f.write(f"  F1 Score: {ensemble_metrics.get('f1_score', 0):.4f}\n")
                        f.write(f"  Weights Path: {ensemble.get('weights_path', 'N/A')}\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
                
                # Overall summary
                f.write("OPTIMIZATION SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write("Average Metrics Across All Pairs:\n")
                
                avg_return = np.mean([
                    result.get("best_metrics", {}).get("total_return", 0) 
                    for result in results.values() if "error" not in result
                ])
                
                avg_win_rate = np.mean([
                    result.get("best_metrics", {}).get("win_rate", 0) 
                    for result in results.values() if "error" not in result
                ])
                
                avg_sharpe = np.mean([
                    result.get("best_metrics", {}).get("sharpe_ratio", 0) 
                    for result in results.values() if "error" not in result
                ])
                
                avg_ensemble_accuracy = np.mean([
                    result.get("ensemble", {}).get("metrics", {}).get("accuracy", 0) 
                    for result in results.values() 
                    if "error" not in result and "ensemble" in result
                ])
                
                f.write(f"  Average Return: {avg_return:.2f}\n")
                f.write(f"  Average Win Rate: {avg_win_rate:.2f}\n")
                f.write(f"  Average Sharpe Ratio: {avg_sharpe:.2f}\n")
                f.write(f"  Average Ensemble Accuracy: {avg_ensemble_accuracy:.4f}\n")
            
            logger.info(f"Generated optimization report: {report_file}")
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimize trading parameters for all pairs")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                       help=f"Number of days of historical data to use (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--pairs", type=str, default=None,
                       help="Comma-separated list of pairs to optimize (default: all pairs in config)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                       help=f"Path to ML configuration file (default: {CONFIG_PATH})")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting trading pair optimization")
    
    # Initialize optimizer
    optimizer = TradingPairOptimizer(config_path=args.config)
    
    # Override pairs if specified
    if args.pairs:
        pairs = args.pairs.split(',')
        optimizer.pairs = pairs
        logger.info(f"Overriding pairs to optimize: {pairs}")
    
    # Run optimization
    results = optimizer.run_optimization_for_all_pairs(days=args.days)
    
    # Generate report
    optimizer.generate_optimization_report(results)
    
    logger.info("Trading pair optimization completed")

if __name__ == "__main__":
    main()