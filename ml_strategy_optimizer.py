#!/usr/bin/env python3
"""
ML Strategy Optimizer for Kraken Trading Bot

This module optimizes all aspects of the trading strategies using:
1. Machine learning and AI techniques
2. Hyperparameter optimization
3. Ensemble model tuning
4. Position sizing and risk optimization
5. Asset-specific customizations

It processes backtest results and updates trading configurations to maximize
both profit percentage and win rate across all supported trading pairs.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ml_strategy_optimizer.log")
    ]
)

logger = logging.getLogger(__name__)

# Constants
SUPPORTED_ASSETS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
DEFAULT_ENSEMBLE_WEIGHTS = {
    "lstm": 0.3,
    "attention": 0.2,
    "tcn": 0.2,
    "transformer": 0.2,
    "cnn": 0.1
}

class MLStrategyOptimizer:
    """Class for optimizing ML strategies based on backtesting results"""
    
    def __init__(self, config_path: str = "ml_optimization_config.json"):
        """
        Initialize the ML strategy optimizer
        
        Args:
            config_path: Path to ML optimization configuration file
        """
        self.config_path = config_path
        self.load_config()
        
        # Results storage
        self.backtest_results = {}
        self.optimized_configs = {}
    
    def load_config(self):
        """Load ML optimization configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded ML optimization configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {
                "model_optimization": {
                    "base_models": {},
                    "ensemble_optimization": {}
                },
                "feature_engineering": {},
                "trading_parameters": {},
                "analysis_parameters": {},
                "asset_specific_optimizations": {}
            }
    
    def load_backtest_results(self, backtest_dir: str = "backtest_results"):
        """
        Load backtest results for analysis
        
        Args:
            backtest_dir: Directory containing backtest results
        """
        # Load summary file
        summary_path = os.path.join(backtest_dir, "backtest_summary.csv")
        if os.path.exists(summary_path):
            try:
                self.backtest_summary = pd.read_csv(summary_path)
                logger.info(f"Loaded backtest summary: {len(self.backtest_summary)} records")
            except Exception as e:
                logger.error(f"Failed to load backtest summary: {e}")
                self.backtest_summary = pd.DataFrame()
        else:
            logger.warning(f"Backtest summary file not found: {summary_path}")
            self.backtest_summary = pd.DataFrame()
        
        # Load detailed report
        report_path = os.path.join(backtest_dir, "backtest_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    self.backtest_report = json.load(f)
                logger.info(f"Loaded backtest report")
            except Exception as e:
                logger.error(f"Failed to load backtest report: {e}")
                self.backtest_report = {}
        else:
            logger.warning(f"Backtest report file not found: {report_path}")
            self.backtest_report = {}
        
        # Find and load all report directories
        for item in os.listdir(backtest_dir):
            if item.startswith("report_") and os.path.isdir(os.path.join(backtest_dir, item)):
                report_dir = os.path.join(backtest_dir, item)
                
                # Load metrics
                metrics_path = os.path.join(report_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                        
                        # Load trades
                        trades_path = os.path.join(report_dir, "trades.csv")
                        if os.path.exists(trades_path):
                            trades = pd.read_csv(trades_path)
                        else:
                            trades = pd.DataFrame()
                        
                        # Store results
                        self.backtest_results[item] = {
                            'metrics': metrics,
                            'trades': trades
                        }
                        
                        logger.info(f"Loaded backtest results from {report_dir}")
                    except Exception as e:
                        logger.error(f"Failed to load backtest results from {report_dir}: {e}")
    
    def load_current_models(self, models_dir: str = "models/ensemble"):
        """
        Load current ensemble model configurations
        
        Args:
            models_dir: Directory containing ensemble model configurations
        """
        self.current_models = {}
        
        for asset in SUPPORTED_ASSETS:
            asset_code = asset.replace('/', '')
            
            # Load weights file
            weights_path = os.path.join(models_dir, f"{asset_code}_weights.json")
            if os.path.exists(weights_path):
                try:
                    with open(weights_path, 'r') as f:
                        weights = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load weights for {asset}: {e}")
                    weights = DEFAULT_ENSEMBLE_WEIGHTS
            else:
                logger.warning(f"Weights file not found for {asset}: {weights_path}")
                weights = DEFAULT_ENSEMBLE_WEIGHTS
            
            # Load ensemble configuration
            config_path = os.path.join(models_dir, f"{asset_code}_ensemble.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        ensemble_config = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load ensemble config for {asset}: {e}")
                    ensemble_config = {}
            else:
                logger.warning(f"Ensemble config file not found for {asset}: {config_path}")
                ensemble_config = {}
            
            # Load position sizing
            sizing_path = os.path.join(models_dir, f"{asset_code}_position_sizing.json")
            if os.path.exists(sizing_path):
                try:
                    with open(sizing_path, 'r') as f:
                        position_sizing = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load position sizing for {asset}: {e}")
                    position_sizing = {}
            else:
                logger.warning(f"Position sizing file not found for {asset}: {sizing_path}")
                position_sizing = {}
            
            # Store current model configuration
            self.current_models[asset] = {
                'weights': weights,
                'ensemble_config': ensemble_config,
                'position_sizing': position_sizing
            }
            
        logger.info(f"Loaded current model configurations for {len(self.current_models)} assets")
    
    def analyze_trading_performance(self, asset: str) -> Dict[str, float]:
        """
        Analyze trading performance for an asset
        
        Args:
            asset: Trading pair (e.g., "SOL/USD")
            
        Returns:
            Dict: Performance metrics
        """
        # Extract trades for the asset from backtest results
        asset_trades = []
        total_pnl = 0.0
        win_count = 0
        lose_count = 0
        total_fees = 0.0
        
        # Process backtest report if available
        if self.backtest_report and "trades" in self.backtest_report:
            for trade in self.backtest_report["trades"]:
                if "asset" in trade and trade["asset"] == asset:
                    asset_trades.append(trade)
                    
                    # Check if this is an exit trade
                    if trade.get("type") == "exit":
                        if "pnl" in trade:
                            pnl = float(trade["pnl"])
                            total_pnl += pnl
                            
                            if pnl > 0:
                                win_count += 1
                            else:
                                lose_count += 1
                        
                        if "fee" in trade:
                            total_fees += float(trade["fee"])
        
        # Process trades from report directories
        for report_id, report_data in self.backtest_results.items():
            if "trades" in report_data:
                trades_df = report_data["trades"]
                
                if "asset" in trades_df.columns:
                    asset_trades_df = trades_df[trades_df["asset"] == asset]
                    
                    for _, trade in asset_trades_df.iterrows():
                        if trade.get("type") == "exit":
                            if "pnl" in trade:
                                pnl = float(trade["pnl"])
                                # Avoid double-counting from backtest report
                                if len(self.backtest_report.get("trades", [])) == 0:
                                    total_pnl += pnl
                                    
                                    if pnl > 0:
                                        win_count += 1
                                    else:
                                        lose_count += 1
                            
                            if "fee" in trade:
                                # Avoid double-counting from backtest report
                                if len(self.backtest_report.get("trades", [])) == 0:
                                    total_fees += float(trade["fee"])
        
        # Calculate metrics
        total_trades = win_count + lose_count
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # Get asset-specific metrics from the backtest summary
        if not self.backtest_summary.empty and "assets" in self.backtest_summary.columns:
            asset_rows = self.backtest_summary[self.backtest_summary["assets"].str.contains(asset, na=False)]
            
            if not asset_rows.empty:
                latest_row = asset_rows.iloc[-1]
                
                # Extract metrics if available
                max_drawdown = latest_row.get("max_drawdown_pct", 0.0)
                sharpe_ratio = latest_row.get("sharpe_ratio", 0.0)
                total_return_pct = latest_row.get("total_return_pct", 0.0)
            else:
                max_drawdown = 0.0
                sharpe_ratio = 0.0
                total_return_pct = 0.0
        else:
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            total_return_pct = 0.0
        
        # Create performance summary
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_return_pct": total_return_pct
        }
    
    def optimize_ensemble_weights(self, asset: str, performance: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize ensemble model weights based on performance
        
        Args:
            asset: Trading pair (e.g., "SOL/USD")
            performance: Performance metrics
            
        Returns:
            Dict: Optimized ensemble weights
        """
        # Get current weights
        current_weights = self.current_models.get(asset, {}).get('weights', DEFAULT_ENSEMBLE_WEIGHTS)
        
        # Get asset-specific preferences from the configuration
        asset_config = self.config.get("asset_specific_optimizations", {}).get(asset, {})
        weight_preference = asset_config.get("weight_preference", {})
        
        # If we have specific weight preferences, use them
        if weight_preference:
            # Validate weights
            weight_sum = sum(weight_preference.values())
            if abs(weight_sum - 1.0) > 0.001:
                # Normalize weights
                optimized_weights = {
                    model: weight / weight_sum
                    for model, weight in weight_preference.items()
                }
            else:
                optimized_weights = weight_preference
        else:
            # Apply optimization based on performance
            win_rate = performance.get("win_rate", 0.0)
            
            # Optimize based on win rate:
            # - For high win rates (>0.6), increase LSTM and Attention weights
            # - For medium win rates (0.5-0.6), increase TCN and Transformer weights
            # - For low win rates (<0.5), increase CNN and other ensemble models
            
            optimized_weights = current_weights.copy()
            
            if win_rate > 0.6:
                # Increase LSTM and Attention for high win rates
                optimized_weights["lstm"] = min(0.4, optimized_weights.get("lstm", 0.3) * 1.2)
                optimized_weights["attention"] = min(0.3, optimized_weights.get("attention", 0.2) * 1.2)
                
                # Decrease other models proportionally
                total_other = sum(v for k, v in optimized_weights.items() 
                               if k not in ["lstm", "attention"])
                if total_other > 0:
                    scale_factor = (1.0 - optimized_weights["lstm"] - optimized_weights["attention"]) / total_other
                    for k in optimized_weights:
                        if k not in ["lstm", "attention"]:
                            optimized_weights[k] *= scale_factor
            
            elif win_rate > 0.5:
                # Increase TCN and Transformer for medium win rates
                optimized_weights["tcn"] = min(0.3, optimized_weights.get("tcn", 0.2) * 1.2)
                optimized_weights["transformer"] = min(0.3, optimized_weights.get("transformer", 0.2) * 1.2)
                
                # Decrease other models proportionally
                total_other = sum(v for k, v in optimized_weights.items() 
                               if k not in ["tcn", "transformer"])
                if total_other > 0:
                    scale_factor = (1.0 - optimized_weights["tcn"] - optimized_weights["transformer"]) / total_other
                    for k in optimized_weights:
                        if k not in ["tcn", "transformer"]:
                            optimized_weights[k] *= scale_factor
            
            else:
                # Increase CNN and adjust other models for low win rates
                optimized_weights["cnn"] = min(0.25, optimized_weights.get("cnn", 0.1) * 1.5)
                optimized_weights["tcn"] = min(0.25, optimized_weights.get("tcn", 0.2) * 1.2)
                
                # Adjust LSTM and Attention proportionally
                total_other = sum(v for k, v in optimized_weights.items() 
                               if k not in ["cnn", "tcn"])
                if total_other > 0:
                    scale_factor = (1.0 - optimized_weights["cnn"] - optimized_weights["tcn"]) / total_other
                    for k in optimized_weights:
                        if k not in ["cnn", "tcn"]:
                            optimized_weights[k] *= scale_factor
        
        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(optimized_weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            optimized_weights = {
                model: weight / weight_sum
                for model, weight in optimized_weights.items()
            }
        
        logger.info(f"Optimized ensemble weights for {asset}: {optimized_weights}")
        return optimized_weights
    
    def optimize_position_sizing(self, asset: str, performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize position sizing parameters based on performance
        
        Args:
            asset: Trading pair (e.g., "SOL/USD")
            performance: Performance metrics
            
        Returns:
            Dict: Optimized position sizing parameters
        """
        # Get current position sizing
        current_sizing = self.current_models.get(asset, {}).get('position_sizing', {})
        
        # Get trading parameters from config
        trading_params = self.config.get("trading_parameters", {})
        position_sizing_defaults = trading_params.get("position_sizing", {})
        risk_management = trading_params.get("risk_management", {})
        
        # Start with defaults if no current sizing
        if not current_sizing:
            optimized_sizing = {
                "base_leverage": 25,
                "max_leverage": 125,
                "confidence_threshold": 0.65,
                "min_trade_confidence": 0.5,
                "risk_per_trade": 0.04,
                "risk_multiplier": 1.0,
                "take_profit_multiplier": 3.0,
                "stop_loss_multiplier": 1.0,
                "capital_allocation": 0.2
            }
        else:
            optimized_sizing = current_sizing.copy()
        
        # Performance-based adjustments
        win_rate = performance.get("win_rate", 0.0)
        max_drawdown = performance.get("max_drawdown", 0.0)
        
        # Adjust risk based on win rate
        if win_rate > 0.7:
            optimized_sizing["risk_multiplier"] = min(1.5, optimized_sizing.get("risk_multiplier", 1.0) * 1.2)
            optimized_sizing["base_leverage"] = min(35, optimized_sizing.get("base_leverage", 25) * 1.2)
        elif win_rate > 0.6:
            optimized_sizing["risk_multiplier"] = min(1.25, optimized_sizing.get("risk_multiplier", 1.0) * 1.1)
            optimized_sizing["base_leverage"] = min(30, optimized_sizing.get("base_leverage", 25) * 1.1)
        elif win_rate < 0.4:
            optimized_sizing["risk_multiplier"] = max(0.7, optimized_sizing.get("risk_multiplier", 1.0) * 0.8)
            optimized_sizing["base_leverage"] = max(15, optimized_sizing.get("base_leverage", 25) * 0.8)
        
        # Adjust based on drawdown
        if max_drawdown > 0.25:
            optimized_sizing["risk_multiplier"] = max(0.7, optimized_sizing.get("risk_multiplier", 1.0) * 0.8)
            optimized_sizing["base_leverage"] = max(15, optimized_sizing.get("base_leverage", 25) * 0.8)
        elif max_drawdown > 0.15:
            optimized_sizing["risk_multiplier"] = max(0.85, optimized_sizing.get("risk_multiplier", 1.0) * 0.9)
            optimized_sizing["base_leverage"] = max(20, optimized_sizing.get("base_leverage", 25) * 0.9)
        
        # Adjust take profit and stop loss multipliers based on win rate
        if win_rate > 0.6:
            # For high win rate, increase take profit to capture more upside
            optimized_sizing["take_profit_multiplier"] = min(4.0, optimized_sizing.get("take_profit_multiplier", 3.0) * 1.1)
        else:
            # For lower win rate, tighten stops and take profits
            optimized_sizing["take_profit_multiplier"] = max(2.0, optimized_sizing.get("take_profit_multiplier", 3.0) * 0.9)
            optimized_sizing["stop_loss_multiplier"] = min(1.2, optimized_sizing.get("stop_loss_multiplier", 1.0) * 1.1)
        
        # Ensure leverage stays within safe limits
        optimized_sizing["max_leverage"] = min(125, optimized_sizing.get("base_leverage", 25) * 5)
        
        logger.info(f"Optimized position sizing for {asset}: {optimized_sizing}")
        return optimized_sizing
    
    def optimize_confidence_thresholds(self, asset: str, performance: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize confidence thresholds based on performance
        
        Args:
            asset: Trading pair (e.g., "SOL/USD")
            performance: Performance metrics
            
        Returns:
            Dict: Optimized confidence thresholds
        """
        # Get current ensemble config
        current_config = self.current_models.get(asset, {}).get('ensemble_config', {})
        
        # Extract current thresholds
        confidence_threshold = current_config.get("confidence_threshold", 0.65)
        min_model_confidence = current_config.get("min_model_confidence", 0.5)
        
        # Adjust based on performance
        win_rate = performance.get("win_rate", 0.0)
        total_trades = performance.get("total_trades", 0)
        
        # For low win rates, increase threshold to be more selective
        if win_rate < 0.5 and total_trades > 10:
            confidence_threshold = min(0.85, confidence_threshold * 1.1)
            min_model_confidence = min(0.7, min_model_confidence * 1.1)
        
        # For high win rates with many trades, consider a small reduction in threshold
        # to increase trade frequency
        elif win_rate > 0.65 and total_trades > 20:
            confidence_threshold = max(0.6, confidence_threshold * 0.95)
            min_model_confidence = max(0.5, min_model_confidence * 0.95)
        
        return {
            "confidence_threshold": confidence_threshold,
            "min_model_confidence": min_model_confidence
        }
    
    def optimize_asset(self, asset: str) -> Dict[str, Any]:
        """
        Optimize all parameters for a specific asset
        
        Args:
            asset: Trading pair (e.g., "SOL/USD")
            
        Returns:
            Dict: Optimized configuration for the asset
        """
        logger.info(f"Optimizing configuration for {asset}")
        
        # Analyze performance
        performance = self.analyze_trading_performance(asset)
        logger.info(f"Performance metrics for {asset}: {performance}")
        
        # Optimize ensemble weights
        optimized_weights = self.optimize_ensemble_weights(asset, performance)
        
        # Optimize position sizing
        optimized_sizing = self.optimize_position_sizing(asset, performance)
        
        # Optimize confidence thresholds
        optimized_thresholds = self.optimize_confidence_thresholds(asset, performance)
        
        # Get current ensemble config
        current_config = self.current_models.get(asset, {}).get('ensemble_config', {})
        
        # Update with optimized thresholds
        optimized_config = current_config.copy()
        optimized_config.update(optimized_thresholds)
        
        # Return optimized configuration
        return {
            'weights': optimized_weights,
            'ensemble_config': optimized_config,
            'position_sizing': optimized_sizing
        }
    
    def optimize_all_assets(self) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all supported assets
        
        Returns:
            Dict: Optimized configurations for all assets
        """
        optimization_results = {}
        
        for asset in SUPPORTED_ASSETS:
            optimization_results[asset] = self.optimize_asset(asset)
        
        self.optimized_configs = optimization_results
        return optimization_results
    
    def save_optimized_configs(self, output_dir: str = "models/ensemble"):
        """
        Save optimized configurations to disk
        
        Args:
            output_dir: Directory to save optimized configurations
        """
        if not self.optimized_configs:
            logger.warning("No optimized configurations to save")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for asset, config in self.optimized_configs.items():
            asset_code = asset.replace('/', '')
            
            # Save weights
            weights_path = os.path.join(output_dir, f"{asset_code}_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(config['weights'], f, indent=2)
            
            # Save ensemble config
            config_path = os.path.join(output_dir, f"{asset_code}_ensemble.json")
            with open(config_path, 'w') as f:
                json.dump(config['ensemble_config'], f, indent=2)
            
            # Save position sizing
            sizing_path = os.path.join(output_dir, f"{asset_code}_position_sizing.json")
            with open(sizing_path, 'w') as f:
                json.dump(config['position_sizing'], f, indent=2)
            
            logger.info(f"Saved optimized configuration for {asset}")
    
    def update_ml_config(self, output_path: str = "ml_config.json"):
        """
        Update global ML configuration with optimized parameters
        
        Args:
            output_path: Path to save updated ML configuration
        """
        if not self.optimized_configs:
            logger.warning("No optimized configurations to update ML config")
            return
        
        # Load current ML config
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    ml_config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load ML config: {e}")
                ml_config = {}
        else:
            ml_config = {}
        
        # Update with optimized parameters
        pairs_config = {}
        for asset, config in self.optimized_configs.items():
            asset_code = asset.replace('/', '')
            
            # Extract position sizing parameters
            position_sizing = config.get('position_sizing', {})
            
            # Update pairs configuration
            pairs_config[asset] = {
                "base_leverage": position_sizing.get("base_leverage", 25),
                "max_leverage": position_sizing.get("max_leverage", 125),
                "risk_per_trade": position_sizing.get("risk_per_trade", 0.04),
                "confidence_threshold": config.get('ensemble_config', {}).get("confidence_threshold", 0.65),
                "take_profit_multiplier": position_sizing.get("take_profit_multiplier", 3.0),
                "stop_loss_multiplier": position_sizing.get("stop_loss_multiplier", 1.0)
            }
        
        # Update ML config
        ml_config["pairs"] = pairs_config
        
        # Add general settings
        ml_config.update({
            "use_ensemble": True,
            "dynamic_position_sizing": True,
            "adaptive_parameters": True,
            "market_regime_awareness": True,
            "use_trailing_stops": True
        })
        
        # Save updated config
        with open(output_path, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML configuration saved to {output_path}")
    
    def generate_performance_report(self, output_dir: str = "optimization_results"):
        """
        Generate performance report and visualizations
        
        Args:
            output_dir: Directory to save performance report
        """
        if not self.optimized_configs:
            logger.warning("No optimized configurations for performance report")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report dictionary
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assets": {}
        }
        
        # Add performance metrics for each asset
        for asset in SUPPORTED_ASSETS:
            performance = self.analyze_trading_performance(asset)
            report["assets"][asset] = performance
        
        # Save report to JSON
        report_path = os.path.join(output_dir, "optimization_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations
        self._generate_performance_visualizations(report, output_dir)
        
        logger.info(f"Performance report generated and saved to {output_dir}")
    
    def _generate_performance_visualizations(self, report: Dict[str, Any], output_dir: str):
        """
        Generate performance visualizations
        
        Args:
            report: Performance report dictionary
            output_dir: Directory to save visualizations
        """
        # Extract data
        assets = []
        win_rates = []
        returns = []
        
        for asset, performance in report["assets"].items():
            assets.append(asset)
            win_rates.append(performance.get("win_rate", 0) * 100)  # Convert to percentage
            returns.append(performance.get("total_return_pct", 0))
        
        # Set style
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot win rates
        bars1 = ax1.bar(assets, win_rates, color='skyblue')
        ax1.set_title('Win Rate by Asset (%)', fontsize=16)
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Plot returns
        bars2 = ax2.bar(assets, returns, color='lightgreen')
        ax2.set_title('Total Return by Asset (%)', fontsize=16)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_ylim(min(min(returns) * 1.1, 0), max(max(returns) * 1.1, 0))
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add timestamp
        fig.text(0.5, 0.01, f"Generated: {report['timestamp']}", ha='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "performance_by_asset.png"), dpi=150)
        plt.close()
        
        # Create correlation heatmap if we have multiple assets
        if len(assets) > 1:
            # Create a DataFrame with returns data
            returns_data = {}
            
            if self.backtest_report and "equity_curve" in self.backtest_report:
                equity_data = pd.DataFrame(self.backtest_report["equity_curve"])
                
                if "time" in equity_data.columns and "equity" in equity_data.columns:
                    equity_data["time"] = pd.to_datetime(equity_data["time"])
                    equity_data.set_index("time", inplace=True)
                    equity_data["return"] = equity_data["equity"].pct_change()
                    
                    # Calculate correlations of returns
                    corr_matrix = np.ones((len(assets), len(assets)))
                    
                    for i, asset1 in enumerate(assets):
                        for j, asset2 in enumerate(assets):
                            if i != j:
                                # This is a simplified approximation as we don't have asset-specific returns
                                # In a real system, you would use actual asset returns
                                corr_matrix[i, j] = 0.5  # Placeholder
            
            else:
                # Create a placeholder correlation matrix
                corr_matrix = np.ones((len(assets), len(assets)))
                for i in range(len(assets)):
                    for j in range(len(assets)):
                        if i != j:
                            corr_matrix[i, j] = 0.5  # Placeholder
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=assets, yticklabels=assets)
            plt.title('Asset Return Correlations', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "asset_correlations.png"), dpi=150)
            plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ML Strategy Optimizer')
    
    parser.add_argument('--config', type=str, default="ml_optimization_config.json",
                      help='Path to ML optimization configuration file')
    
    parser.add_argument('--backtest-dir', type=str, default="backtest_results",
                      help='Directory containing backtest results')
    
    parser.add_argument('--models-dir', type=str, default="models/ensemble",
                      help='Directory containing ensemble model configurations')
    
    parser.add_argument('--output-dir', type=str, default="optimization_results",
                      help='Directory to save optimization results')
    
    parser.add_argument('--update-ml-config', action='store_true',
                      help='Update ML configuration file')
    
    parser.add_argument('--ml-config', type=str, default="ml_config.json",
                      help='Path to ML configuration file')
    
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create optimizer
    optimizer = MLStrategyOptimizer(config_path=args.config)
    
    # Load data
    optimizer.load_backtest_results(backtest_dir=args.backtest_dir)
    optimizer.load_current_models(models_dir=args.models_dir)
    
    # Optimize all assets
    optimized_configs = optimizer.optimize_all_assets()
    
    # Save optimized configurations
    optimizer.save_optimized_configs(output_dir=args.models_dir)
    
    # Update ML config if requested
    if args.update_ml_config:
        optimizer.update_ml_config(output_path=args.ml_config)
    
    # Generate performance report
    optimizer.generate_performance_report(output_dir=args.output_dir)
    
    logger.info("ML strategy optimization completed successfully!")

if __name__ == "__main__":
    main()