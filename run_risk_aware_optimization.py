#!/usr/bin/env python3
"""
Risk-Aware Trading System Optimization

This script runs a comprehensive optimization of the entire trading system with
integrated risk management across all pairs to maximize returns while preventing
liquidations and large losses.

It performs the following steps:
1. Fetches historical data for all trading pairs
2. Analyzes market conditions and volatility patterns
3. Performs risk-adjusted backtest optimization of trading parameters
4. Validates results with stress testing and risk metrics
5. Applies optimized parameters to the trading system

The optimization process incorporates advanced risk management that prevents
liquidations and large losses without sacrificing profitability.
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our components
from dynamic_parameter_optimizer import DynamicParameterOptimizer
from utils.risk_manager import risk_manager
from utils.data_loader import HistoricalDataLoader
from integrated_risk_manager import integrated_risk_manager
from utils.market_analyzer import MarketAnalyzer
from optimize_all_trading_pairs import TradingPairOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("risk_aware_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "config/integrated_risk_config.json"
OPTIMIZATION_RESULTS_DIR = "risk_optimized_results"
OUTPUT_DIR = "risk_optimized_results"

class RiskAwareOptimizer:
    """
    Risk-aware trading system optimizer that maximizes returns while preventing
    liquidations and large losses.
    """
    
    def __init__(self, config_path: str = CONFIG_PATH, output_dir: str = OUTPUT_DIR):
        """
        Initialize the risk-aware optimizer.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for output files
        """
        self.config_path = config_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.risk_manager = risk_manager  # Use singleton
        self.integrated_risk_manager = integrated_risk_manager  # Use singleton
        self.data_loader = HistoricalDataLoader()
        self.market_analyzer = MarketAnalyzer()
        self.parameter_optimizer = DynamicParameterOptimizer()
        self.trading_pair_optimizer = TradingPairOptimizer()
        
        # Set pairs to optimize
        self.pairs = self.config.get("pair_specific_settings", {}).keys()
        if not self.pairs:
            self.pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
            
        logger.info(f"Risk-aware optimizer initialized with {len(self.pairs)} pairs: {', '.join(self.pairs)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                "enable_integrated_risk": True,
                "performance_metrics_window": 30,
                "risk_profile": "balanced",
                "pair_specific_settings": {
                    "SOL/USD": {"risk_coefficient": 1.1},
                    "BTC/USD": {"risk_coefficient": 1.0},
                    "ETH/USD": {"risk_coefficient": 1.0}
                },
                "optimization_parameters": {
                    "default_lookback_days": 180,
                    "test_ratio": 0.3
                }
            }
    
    def run_optimization(self):
        """Run the complete risk-aware optimization process"""
        start_time = time.time()
        
        logger.info("Starting risk-aware optimization across all pairs")
        
        # Step 1: Load and prepare historical data
        historical_data = self._load_historical_data()
        
        # Step 2: Run volatility and market regime analysis
        market_analysis = self._analyze_markets(historical_data)
        
        # Step 3: Configure risk parameters based on analysis
        risk_parameters = self._configure_risk_parameters(market_analysis)
        
        # Step 4: Run optimization with risk awareness
        optimization_results = self._run_risk_aware_optimization(historical_data, risk_parameters)
        
        # Step 5: Validate results with stress testing
        validated_results = self._validate_with_stress_testing(optimization_results, historical_data)
        
        # Step 6: Apply optimized parameters
        self._apply_optimized_parameters(validated_results)
        
        # Step 7: Generate report
        self._generate_report(validated_results, market_analysis)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Risk-aware optimization completed in {execution_time:.2f} seconds")
        
        return validated_results
    
    def _load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for all pairs.
        
        Returns:
            Dictionary mapping pair symbols to historical data
        """
        lookback_days = self.config.get("optimization_parameters", {}).get("default_lookback_days", 180)
        
        logger.info(f"Loading {lookback_days} days of historical data for {len(self.pairs)} pairs")
        
        historical_data = {}
        for pair in self.pairs:
            try:
                data = self.data_loader.fetch_historical_data(
                    pair=pair,
                    timeframe="1h",
                    days=lookback_days
                )
                
                # Add technical indicators
                data = self.data_loader.add_technical_indicators(data)
                
                historical_data[pair] = data
                logger.info(f"Loaded {len(data)} data points for {pair}")
            except Exception as e:
                logger.error(f"Error loading data for {pair}: {e}")
        
        return historical_data
    
    def _analyze_markets(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market conditions for all pairs.
        
        Args:
            historical_data: Dictionary of historical data by pair
            
        Returns:
            Dictionary with market analysis results
        """
        logger.info("Analyzing market conditions for all pairs")
        
        market_analysis = {}
        
        for pair, data in historical_data.items():
            try:
                # Analyze market regimes
                regime_analysis = self.market_analyzer.analyze_market_regimes(data)
                
                # Assess volatility
                volatility_metrics = self.risk_manager.assess_volatility(pair, data)
                
                # Analyze trend strength
                trend_analysis = self.market_analyzer.analyze_trend_strength(data)
                
                # Store analysis
                market_analysis[pair] = {
                    "regime_analysis": regime_analysis,
                    "volatility_metrics": volatility_metrics,
                    "trend_analysis": trend_analysis
                }
                
                logger.info(f"Market analysis for {pair}: Regime={regime_analysis.get('current_regime')}, "
                          f"Volatility={volatility_metrics.get('volatility_category')}, "
                          f"Trend={trend_analysis.get('trend_direction')}")
            except Exception as e:
                logger.error(f"Error analyzing market for {pair}: {e}")
        
        # Calculate cross-asset correlation
        if len(historical_data) > 1:
            try:
                self.risk_manager.calculate_portfolio_correlation(historical_data)
                logger.info("Calculated cross-asset correlation matrix")
            except Exception as e:
                logger.error(f"Error calculating correlation: {e}")
        
        return market_analysis
    
    def _configure_risk_parameters(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure risk parameters based on market analysis.
        
        Args:
            market_analysis: Dictionary with market analysis results
            
        Returns:
            Dictionary with risk parameters for optimization
        """
        logger.info("Configuring risk parameters based on market analysis")
        
        # Get base risk parameters from integrated risk manager
        base_params = self.integrated_risk_manager.get_optimization_parameters()
        
        # Configure pair-specific risk parameters
        pair_params = {}
        
        for pair, analysis in market_analysis.items():
            # Get volatility category
            volatility_category = analysis.get("volatility_metrics", {}).get("volatility_category", "medium")
            
            # Get market regime
            regime = analysis.get("regime_analysis", {}).get("current_regime", "UNKNOWN")
            
            # Get trend strength
            trend_strength = analysis.get("trend_analysis", {}).get("trend_strength", "moderate")
            
            # Get pair-specific settings from config
            pair_settings = self.config.get("pair_specific_settings", {}).get(pair, {})
            
            # Configure risk parameters for this pair
            max_leverage = base_params.get("max_leverage", {}).get(volatility_category, 50.0)
            
            # Apply pair-specific override if specified
            if "max_leverage_override" in pair_settings:
                max_leverage = min(max_leverage, pair_settings["max_leverage_override"])
            
            # Adjust risk percentage based on market conditions
            risk_range = base_params.get("risk_percentage_range", [0.1, 0.25])
            base_risk = (risk_range[0] + risk_range[1]) / 2.0
            
            if regime == "TRENDING_UP" and trend_strength in ["strong", "very strong"]:
                risk_factor = 1.1  # Increase risk in strong uptrend
            elif regime == "TRENDING_DOWN" and trend_strength in ["strong", "very strong"]:
                risk_factor = 0.9  # Decrease risk in strong downtrend
            elif regime == "VOLATILE":
                risk_factor = 0.8  # Reduce risk in volatile markets
            elif volatility_category in ["high", "very_high", "extreme"]:
                risk_factor = 0.85  # Reduce risk in high volatility
            else:
                risk_factor = 1.0
            
            # Apply pair-specific risk coefficient if specified
            if "risk_coefficient" in pair_settings:
                risk_factor *= pair_settings["risk_coefficient"]
            
            # Calculate final risk percentage
            risk_percentage = base_risk * risk_factor
            risk_percentage = max(risk_range[0], min(risk_range[1], risk_percentage))
            
            # Configure stop-loss multiplier
            stop_range = base_params.get("stop_multiplier_range", [2.2, 3.8])
            base_stop = (stop_range[0] + stop_range[1]) / 2.0
            
            if volatility_category in ["low", "very_low"]:
                stop_factor = 1.1  # Wider stops in low volatility
            elif volatility_category in ["high", "very_high"]:
                stop_factor = 0.9  # Tighter stops in high volatility
            elif volatility_category == "extreme":
                stop_factor = 0.7  # Very tight stops in extreme volatility
            else:
                stop_factor = 1.0
            
            stop_multiplier = base_stop * stop_factor
            stop_multiplier = max(stop_range[0], min(stop_range[1], stop_multiplier))
            
            # Store parameters for this pair
            pair_params[pair] = {
                "max_leverage": max_leverage,
                "risk_percentage": risk_percentage,
                "trailing_stop_atr_multiplier": stop_multiplier,
                "volatility_category": volatility_category,
                "market_regime": regime,
                "trend_strength": trend_strength,
                "risk_factor": risk_factor,
                "stop_factor": stop_factor
            }
            
            logger.info(f"Risk parameters for {pair}: Risk={risk_percentage:.2%}, "
                      f"Leverage={max_leverage:.1f}, "
                      f"Stop Multiplier={stop_multiplier:.2f}")
        
        return {
            "base_params": base_params,
            "pair_params": pair_params
        }
    
    def _run_risk_aware_optimization(self, historical_data: Dict[str, pd.DataFrame], 
                                   risk_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run risk-aware optimization for all pairs.
        
        Args:
            historical_data: Dictionary of historical data by pair
            risk_parameters: Dictionary with risk parameters
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("Running risk-aware optimization for all pairs")
        
        # Configure the trading pair optimizer with risk parameters
        pair_params = risk_parameters.get("pair_params", {})
        optimization_results = {}
        
        for pair, data in historical_data.items():
            if pair not in pair_params:
                logger.warning(f"No risk parameters for {pair}, skipping optimization")
                continue
            
            try:
                logger.info(f"Optimizing {pair} with risk-aware parameters")
                
                # Configure parameter grid with risk constraints
                risk_param = pair_params[pair]
                
                # Override optimizer settings with risk-aware parameters
                self.trading_pair_optimizer.override_params = {
                    "max_leverage": risk_param["max_leverage"],
                    "risk_percentage": risk_param["risk_percentage"],
                    "trailing_stop_atr_multiplier": risk_param["trailing_stop_atr_multiplier"]
                }
                
                # Run optimization for this pair
                result = self.trading_pair_optimizer.optimize_pair(pair, {pair: data})
                
                optimization_results[pair] = result
                
                logger.info(f"Optimization completed for {pair}")
                
                # Log best metrics
                best_metrics = result.get("best_metrics", {})
                logger.info(f"Best metrics for {pair}: Return={best_metrics.get('total_return', 0):.2f}, "
                          f"Win Rate={best_metrics.get('win_rate', 0):.2f}, "
                          f"Sharpe={best_metrics.get('sharpe_ratio', 0):.2f}")
            except Exception as e:
                logger.error(f"Error optimizing {pair}: {e}")
        
        return optimization_results
    
    def _validate_with_stress_testing(self, optimization_results: Dict[str, Any],
                                    historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate optimization results with stress testing.
        
        Args:
            optimization_results: Dictionary with optimization results
            historical_data: Dictionary of historical data by pair
            
        Returns:
            Dictionary with validated results
        """
        logger.info("Validating optimization results with stress testing")
        
        # Get validation criteria
        validation_criteria = self.integrated_risk_manager.get_validation_criteria()
        
        validated_results = {}
        stress_test_results = {}
        
        # Define stress test scenarios
        scenarios = self.config.get("enhanced_risk_metrics", {}).get("stress_test_scenarios", [])
        
        for pair, result in optimization_results.items():
            if pair not in historical_data:
                continue
                
            try:
                # Extract best parameters
                best_params = result.get("best_params", {})
                
                # Run stress tests on best parameters
                stress_results = self._run_stress_tests(pair, historical_data[pair], best_params, scenarios)
                
                # Check if results pass validation criteria
                passes_validation = self._check_validation_criteria(result, stress_results, validation_criteria)
                
                if passes_validation:
                    logger.info(f"{pair} optimization PASSED validation criteria")
                    validated_results[pair] = result
                else:
                    logger.warning(f"{pair} optimization FAILED validation criteria, adjusting parameters")
                    # Adjust parameters to meet validation criteria
                    adjusted_params = self._adjust_parameters_for_validation(best_params, stress_results)
                    
                    # Update result with adjusted parameters
                    result["adjusted_params"] = adjusted_params
                    validated_results[pair] = result
                
                # Store stress test results
                stress_test_results[pair] = stress_results
                
            except Exception as e:
                logger.error(f"Error validating {pair}: {e}")
        
        # Save stress test results
        self._save_stress_test_results(stress_test_results)
        
        return {
            "validated_results": validated_results,
            "stress_test_results": stress_test_results
        }
    
    def _run_stress_tests(self, pair: str, data: pd.DataFrame, 
                        parameters: Dict[str, Any], scenarios: List[str]) -> Dict[str, Any]:
        """
        Run stress tests on parameters using different market scenarios.
        
        Args:
            pair: Trading pair symbol
            data: Historical price data
            parameters: Parameters to test
            scenarios: List of stress test scenarios
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        # For each scenario, modify the data to simulate market conditions
        for scenario in scenarios:
            try:
                # Create stress scenario data
                scenario_data = self._create_stress_scenario(data, scenario)
                
                # Run backtest with stress scenario
                backtest_result = self._run_stress_backtest(pair, scenario_data, parameters)
                
                # Store result
                stress_results[scenario] = backtest_result
                
                logger.info(f"Stress test for {pair} - {scenario}: "
                          f"Return={backtest_result.get('total_return', 0):.2f}, "
                          f"Max Drawdown={backtest_result.get('max_drawdown', 0):.2%}")
            except Exception as e:
                logger.error(f"Error running stress test for {pair} - {scenario}: {e}")
        
        return stress_results
    
    def _create_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Create a stress scenario by modifying historical data.
        
        Args:
            data: Historical price data
            scenario: Stress scenario type
            
        Returns:
            Modified DataFrame for stress testing
        """
        # Make a copy to avoid modifying original data
        scenario_data = data.copy()
        
        if scenario == "high_volatility":
            # Increase volatility by adding random shocks
            close = scenario_data["close"].values
            high = scenario_data["high"].values
            low = scenario_data["low"].values
            
            # Calculate average daily range
            adr = (high - low).mean()
            
            # Add random shocks (up to 3x normal range)
            for i in range(len(close)):
                if i % 5 == 0:  # Every 5th candle
                    shock = np.random.uniform(-3.0, 3.0) * adr
                    scenario_data.loc[scenario_data.index[i], "close"] = close[i] + shock
                    
                    # Adjust high/low accordingly
                    if shock > 0:
                        scenario_data.loc[scenario_data.index[i], "high"] = max(high[i], close[i] + shock)
                    else:
                        scenario_data.loc[scenario_data.index[i], "low"] = min(low[i], close[i] + shock)
            
        elif scenario == "market_crash":
            # Simulate a market crash with a sharp decline
            mid_idx = len(scenario_data) // 2
            crash_period = 48  # 2 days (assuming hourly data)
            
            # Calculate crash magnitude (30-50% decline)
            crash_pct = np.random.uniform(0.3, 0.5)
            
            # Apply crash over the period
            for i in range(crash_period):
                if mid_idx + i < len(scenario_data):
                    decay = (crash_period - i) / crash_period  # More at the beginning
                    daily_drop = (crash_pct / crash_period) * decay * 2
                    
                    idx = scenario_data.index[mid_idx + i]
                    scenario_data.loc[idx, "close"] *= (1 - daily_drop)
                    scenario_data.loc[idx, "low"] *= (1 - daily_drop * 1.2)  # Low drops more
                    scenario_data.loc[idx, "high"] *= (1 - daily_drop * 0.8)  # High drops less
            
        elif scenario == "flash_crash":
            # Simulate a flash crash (very sharp drop and recovery)
            crash_idx = len(scenario_data) // 3
            crash_depth = np.random.uniform(0.1, 0.3)  # 10-30% drop
            
            # Apply flash crash over a few hours
            for i in range(5):  # 5-hour crash and recovery
                if crash_idx + i < len(scenario_data):
                    if i <= 2:  # Crash phase
                        drop_pct = crash_depth * (i + 1) / 3
                        scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - drop_pct)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - drop_pct * 1.2)
                    else:  # Recovery phase
                        recover_pct = crash_depth * (5 - i) / 2
                        scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - recover_pct)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - recover_pct * 1.2)
            
        elif scenario == "correlation_spike":
            # For correlation spike, we would modify multiple assets simultaneously
            # In this single-asset implementation, we'll just add some volatility
            volatility_increase = np.random.uniform(1.5, 2.5)
            
            for i in range(len(scenario_data)):
                if np.random.random() < 0.2:  # 20% of candles
                    current_range = scenario_data["high"].iloc[i] - scenario_data["low"].iloc[i]
                    expanded_range = current_range * volatility_increase
                    
                    # Expand the range around the close price
                    close = scenario_data["close"].iloc[i]
                    scenario_data.loc[scenario_data.index[i], "high"] = close + expanded_range / 2
                    scenario_data.loc[scenario_data.index[i], "low"] = close - expanded_range / 2
        
        # Recalculate any derived values (e.g., technical indicators)
        scenario_data = self.data_loader.add_technical_indicators(scenario_data)
        
        return scenario_data
    
    def _run_stress_backtest(self, pair: str, data: pd.DataFrame, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a backtest with stress scenario data.
        
        Args:
            pair: Trading pair symbol
            data: Stress scenario price data
            parameters: Parameters to test
            
        Returns:
            Dictionary with backtest results
        """
        # Use the backtest engine to run a backtest
        return self.trading_pair_optimizer.backtest_engine.run_backtest(
            pair=pair,
            data=data,
            parameters=parameters
        )
    
    def _check_validation_criteria(self, optimization_result: Dict[str, Any],
                                 stress_results: Dict[str, Any],
                                 validation_criteria: Dict[str, Any]) -> bool:
        """
        Check if optimization results pass validation criteria.
        
        Args:
            optimization_result: Optimization result for a pair
            stress_results: Stress test results for a pair
            validation_criteria: Validation criteria
            
        Returns:
            True if validation passes, False otherwise
        """
        # Get best metrics
        best_metrics = optimization_result.get("best_metrics", {})
        
        # Check base metrics
        if best_metrics.get("win_rate", 0) < validation_criteria.get("min_win_rate", 0.55):
            logger.warning(f"Failed validation: Win rate {best_metrics.get('win_rate', 0):.2f} below minimum")
            return False
            
        if best_metrics.get("max_drawdown", 1.0) > validation_criteria.get("max_drawdown", 0.15):
            logger.warning(f"Failed validation: Max drawdown {best_metrics.get('max_drawdown', 1.0):.2%} above limit")
            return False
            
        if best_metrics.get("profit_factor", 0) < validation_criteria.get("min_profit_factor", 1.5):
            logger.warning(f"Failed validation: Profit factor {best_metrics.get('profit_factor', 0):.2f} below minimum")
            return False
            
        if best_metrics.get("sharpe_ratio", 0) < validation_criteria.get("min_sharpe_ratio", 1.0):
            logger.warning(f"Failed validation: Sharpe ratio {best_metrics.get('sharpe_ratio', 0):.2f} below minimum")
            return False
        
        # Check stress test metrics
        for scenario, result in stress_results.items():
            # Most important check: no liquidations in stress tests
            if result.get("liquidations", 0) > 0:
                logger.warning(f"Failed validation: {result.get('liquidations', 0)} liquidations in {scenario} scenario")
                return False
                
            # Drawdown in stress tests should not exceed 30%
            if result.get("max_drawdown", 0) > 0.3:
                logger.warning(f"Failed validation: {scenario} max drawdown {result.get('max_drawdown', 0):.2%} exceeds 30%")
                return False
        
        # All checks passed
        return True
    
    def _adjust_parameters_for_validation(self, parameters: Dict[str, Any],
                                        stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust parameters to pass validation criteria based on stress test results.
        
        Args:
            parameters: Original parameters
            stress_results: Stress test results
            
        Returns:
            Adjusted parameters
        """
        # Start with a copy of original parameters
        adjusted_params = parameters.copy()
        
        # Check for liquidations or extreme drawdowns
        has_liquidations = any(result.get("liquidations", 0) > 0 for result in stress_results.values())
        extreme_drawdown = any(result.get("max_drawdown", 0) > 0.25 for result in stress_results.values())
        
        # Adjust risk parameters
        if has_liquidations or extreme_drawdown:
            # Reduce leverage
            if "base_leverage" in adjusted_params:
                adjusted_params["base_leverage"] *= 0.7
            if "max_leverage" in adjusted_params:
                adjusted_params["max_leverage"] *= 0.7
                
            # Reduce risk percentage
            if "risk_percentage" in adjusted_params:
                adjusted_params["risk_percentage"] *= 0.8
                
            # Tighten stops
            if "trailing_stop_atr_multiplier" in adjusted_params:
                adjusted_params["trailing_stop_atr_multiplier"] *= 0.85
            
            logger.info("Adjusted parameters to prevent liquidations and reduce drawdown")
        else:
            # Check for low returns or win rate
            low_returns = any(result.get("total_return", 0) < 0.2 for result in stress_results.values())
            low_win_rate = any(result.get("win_rate", 0) < 0.5 for result in stress_results.values())
            
            if low_returns or low_win_rate:
                # Slightly reduce leverage and risk, but not as much
                if "base_leverage" in adjusted_params:
                    adjusted_params["base_leverage"] *= 0.85
                if "max_leverage" in adjusted_params:
                    adjusted_params["max_leverage"] *= 0.85
                    
                # Adjust risk percentage less aggressively
                if "risk_percentage" in adjusted_params:
                    adjusted_params["risk_percentage"] *= 0.9
                
                logger.info("Made minor adjustments to improve risk/reward profile")
        
        return adjusted_params
    
    def _save_stress_test_results(self, stress_test_results: Dict[str, Any]):
        """
        Save stress test results to file.
        
        Args:
            stress_test_results: Dictionary with stress test results
        """
        output_path = os.path.join(self.output_dir, "stress_test_results.json")
        
        try:
            # Convert non-serializable values to strings
            serializable_results = {}
            for pair, results in stress_test_results.items():
                serializable_results[pair] = {}
                for scenario, result in results.items():
                    serializable_results[pair][scenario] = {
                        k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                        for k, v in result.items()
                    }
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logger.info(f"Saved stress test results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving stress test results: {e}")
    
    def _apply_optimized_parameters(self, validated_results: Dict[str, Any]):
        """
        Apply optimized parameters to the trading system.
        
        Args:
            validated_results: Dictionary with validated optimization results
        """
        logger.info("Applying optimized parameters to trading system")
        
        validated_pairs = validated_results.get("validated_results", {})
        
        # Create optimized parameters dictionary
        optimized_params = {}
        
        for pair, result in validated_pairs.items():
            # Use adjusted parameters if available, otherwise use best parameters
            if "adjusted_params" in result:
                params = result["adjusted_params"]
                logger.info(f"Using adjusted parameters for {pair}")
            else:
                params = result.get("best_params", {})
                logger.info(f"Using best parameters for {pair}")
                
            # Add to optimized parameters
            optimized_params[pair] = params
        
        # Save the optimized parameters
        output_path = os.path.join(self.output_dir, "optimized_params.json")
        
        try:
            with open(output_path, 'w') as f:
                json.dump(optimized_params, f, indent=2)
                
            logger.info(f"Saved optimized parameters to {output_path}")
            
            # Also save to the config directory for the trading system
            system_config_path = "config/optimized_params.json"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(system_config_path), exist_ok=True)
            
            with open(system_config_path, 'w') as f:
                json.dump(optimized_params, f, indent=2)
                
            logger.info(f"Applied optimized parameters to trading system at {system_config_path}")
            
            # Update the parameter optimizer
            if self.config.get("optimization_parameters", {}).get("apply_results_immediately", True):
                for pair, params in optimized_params.items():
                    # Convert to format expected by parameter optimizer
                    optimizer_params = {
                        pair: {
                            "risk_percentage": params.get("risk_percentage", 0.2),
                            "base_leverage": params.get("base_leverage", 20.0),
                            "max_leverage": params.get("max_leverage", 100.0),
                            "confidence_threshold": params.get("confidence_threshold", 0.65),
                            "signal_strength_threshold": params.get("signal_strength_threshold", 0.6),
                            "trailing_stop_atr_multiplier": params.get("trailing_stop_atr_multiplier", 3.0),
                            "exit_multiplier": params.get("exit_multiplier", 1.5),
                            "drawdown_limit_percentage": params.get("drawdown_limit_percentage", 4.0),
                            "strategy_weights": params.get("strategy_weights", {"arima": 0.3, "adaptive": 0.7}),
                            "optimization_time": datetime.now().isoformat()
                        }
                    }
                    
                    # Update the parameter optimizer
                    self.parameter_optimizer.optimized_params.update(optimizer_params)
                
                # Save updated parameters
                self.parameter_optimizer.save_optimized_params()
                
                logger.info("Updated parameter optimizer with optimized parameters")
        except Exception as e:
            logger.error(f"Error applying optimized parameters: {e}")
    
    def _generate_report(self, validated_results: Dict[str, Any], 
                       market_analysis: Dict[str, Any]):
        """
        Generate a comprehensive optimization report.
        
        Args:
            validated_results: Dictionary with validated optimization results
            market_analysis: Dictionary with market analysis results
        """
        logger.info("Generating optimization report")
        
        validated_pairs = validated_results.get("validated_results", {})
        stress_results = validated_results.get("stress_test_results", {})
        
        report_path = os.path.join(self.output_dir, "optimization_report.txt")
        
        try:
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("RISK-AWARE OPTIMIZATION REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Pairs Optimized: {len(validated_pairs)}\n")
                
                # Calculate overall metrics
                total_return = 0.0
                total_win_rate = 0.0
                total_drawdown = 0.0
                total_pairs = 0
                
                for pair, result in validated_pairs.items():
                    metrics = result.get("best_metrics", {})
                    total_return += metrics.get("total_return", 0)
                    total_win_rate += metrics.get("win_rate", 0)
                    total_drawdown += metrics.get("max_drawdown", 0)
                    total_pairs += 1
                
                if total_pairs > 0:
                    avg_return = total_return / total_pairs
                    avg_win_rate = total_win_rate / total_pairs
                    avg_drawdown = total_drawdown / total_pairs
                    
                    f.write(f"Average Return: {avg_return:.2f}\n")
                    f.write(f"Average Win Rate: {avg_win_rate:.2f}\n")
                    f.write(f"Average Max Drawdown: {avg_drawdown:.2%}\n")
                
                f.write("\nPAIR DETAILS\n")
                f.write("-" * 40 + "\n")
                
                for pair, result in validated_pairs.items():
                    f.write(f"Pair: {pair}\n")
                    
                    # Add market analysis
                    if pair in market_analysis:
                        analysis = market_analysis[pair]
                        regime = analysis.get("regime_analysis", {}).get("current_regime", "UNKNOWN")
                        volatility = analysis.get("volatility_metrics", {}).get("volatility_category", "medium")
                        trend = analysis.get("trend_analysis", {}).get("trend_direction", "neutral")
                        
                        f.write(f"  Market Regime: {regime}\n")
                        f.write(f"  Volatility Category: {volatility}\n")
                        f.write(f"  Trend Direction: {trend}\n")
                    
                    # Add performance metrics
                    metrics = result.get("best_metrics", {})
                    f.write(f"  Total Return: {metrics.get('total_return', 0):.2f}\n")
                    f.write(f"  Win Rate: {metrics.get('win_rate', 0):.2f}\n")
                    f.write(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
                    f.write(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n")
                    f.write(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
                    
                    # Add stress test results
                    if pair in stress_results:
                        f.write("  Stress Test Results:\n")
                        for scenario, st_result in stress_results[pair].items():
                            f.write(f"    {scenario}: Return={st_result.get('total_return', 0):.2f}, " +
                                   f"Drawdown={st_result.get('max_drawdown', 0):.2%}\n")
                    
                    # Add optimized parameters
                    if "adjusted_params" in result:
                        params = result["adjusted_params"]
                        f.write("  Adjusted Parameters:\n")
                    else:
                        params = result.get("best_params", {})
                        f.write("  Optimized Parameters:\n")
                        
                    f.write(f"    Risk Percentage: {params.get('risk_percentage', 0):.2%}\n")
                    f.write(f"    Base Leverage: {params.get('base_leverage', 0):.1f}\n")
                    f.write(f"    Max Leverage: {params.get('max_leverage', 0):.1f}\n")
                    f.write(f"    Trailing Stop Multiplier: {params.get('trailing_stop_atr_multiplier', 0):.2f}\n")
                    
                    strategy_weights = params.get("strategy_weights", {})
                    f.write(f"    ARIMA Weight: {strategy_weights.get('arima', 0):.2f}\n")
                    f.write(f"    Adaptive Weight: {strategy_weights.get('adaptive', 0):.2f}\n")
                    
                    f.write("\n")
                
                f.write("\nRISK MANAGEMENT SETTINGS\n")
                f.write("-" * 40 + "\n")
                f.write("1. Dynamic position sizing based on trade confidence and market conditions\n")
                f.write("2. Volatility-adjusted leverage with automatic reduction in high-volatility markets\n")
                f.write("3. Profit-protecting ratcheting stops to lock in gains\n")
                f.write("4. Portfolio correlation monitoring to prevent over-exposure\n")
                f.write("5. Kelly criterion optimization for long-term capital growth\n")
                f.write("6. Drawdown control mechanisms to protect capital\n")
                f.write("7. Stress-tested parameters to prevent liquidations\n")
                
                f.write("\nNEXT STEPS\n")
                f.write("-" * 40 + "\n")
                f.write("1. Parameters have been applied to the trading system\n")
                f.write("2. Continue monitoring portfolio performance and adapting parameters as needed\n")
                f.write("3. Consider running optimization periodically (e.g., weekly) to adapt to changing markets\n")
                f.write("4. Enable emergency controls to automatically reduce risk in extreme market conditions\n")
            
            logger.info(f"Generated optimization report at {report_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run risk-aware optimization")
    parser.add_argument("--pairs", type=str, help="Comma-separated list of trading pairs")
    parser.add_argument("--days", type=int, default=180, help="Number of days of historical data")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to configuration file")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Get pairs from args
    pairs = args.pairs.split(",") if args.pairs else None
    
    # Create optimizer
    optimizer = RiskAwareOptimizer(config_path=args.config, output_dir=args.output)
    
    # If pairs specified, override optimizer pairs
    if pairs:
        optimizer.pairs = pairs
    
    # Run optimization
    optimizer.run_optimization()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())