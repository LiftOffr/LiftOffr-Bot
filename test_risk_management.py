#!/usr/bin/env python3
"""
Risk Management System Test

This script tests the risk management system with sample data to verify its effectiveness
in preventing liquidations and large losses while maintaining profitability.

It performs the following steps:
1. Generates sample data for multiple trading pairs
2. Simulates various market scenarios including flash crashes and volatility spikes
3. Tests the risk management system's response to these scenarios
4. Compares performance with and without risk management
5. Generates detailed reports on drawdowns, win rates, and profitability
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import our components
from utils.risk_manager import risk_manager
from utils.data_loader import HistoricalDataLoader
from utils.market_analyzer import MarketAnalyzer
from integrated_risk_manager import integrated_risk_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("risk_management_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TEST_DATA_DIR = "test_data"
RESULTS_DIR = "test_results"
PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
SCENARIO_TYPES = ["normal", "flash_crash", "volatility_spike", "sustained_downtrend"]

class RiskManagementTester:
    """
    Tests the risk management system with various market scenarios.
    """
    
    def __init__(self, data_dir: str = TEST_DATA_DIR, results_dir: str = RESULTS_DIR):
        """
        Initialize the tester.
        
        Args:
            data_dir: Directory for test data
            results_dir: Directory for test results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = HistoricalDataLoader(data_dir=data_dir)
        self.market_analyzer = MarketAnalyzer()
        
        # Initial portfolio
        self.initial_portfolio = 10000.0
        
        logger.info("Risk management tester initialized")
    
    def generate_test_data(self, pairs: List[str] = PAIRS, days: int = 180):
        """
        Generate test data for multiple pairs and scenarios.
        
        Args:
            pairs: List of trading pairs
            days: Number of days of data
        """
        logger.info(f"Generating test data for {len(pairs)} pairs")
        
        # For each pair, generate base data
        for pair in pairs:
            # Generate normal data
            data = self.data_loader.fetch_historical_data(pair, days=days)
            
            # Add technical indicators
            data = self.data_loader.add_technical_indicators(data)
            
            # Save with 'normal' scenario tag
            self._save_scenario_data(pair, data, "normal")
            
            # Create and save stress scenarios
            for scenario in SCENARIO_TYPES[1:]:  # Skip 'normal'
                scenario_data = self._create_stress_scenario(data, scenario)
                self._save_scenario_data(pair, scenario_data, scenario)
        
        logger.info("Test data generation completed")
    
    def _create_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Create a stress scenario by modifying price data.
        
        Args:
            data: Original price data
            scenario: Scenario type
            
        Returns:
            Modified DataFrame for the scenario
        """
        # Make a copy to avoid modifying original data
        scenario_data = data.copy()
        
        if scenario == "flash_crash":
            # Create a flash crash (30-50% drop in a short time)
            crash_idx = len(scenario_data) // 3
            crash_depth = np.random.uniform(0.3, 0.5)
            
            # Apply crash over a few candles
            for i in range(5):
                if crash_idx + i < len(scenario_data):
                    if i <= 2:  # Crash phase
                        drop_pct = crash_depth * (i + 1) / 3
                        scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - drop_pct)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - drop_pct * 1.2)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "high"] *= (1 - drop_pct * 0.8)
                    else:  # Recovery phase
                        recover_pct = crash_depth * (5 - i) / 2
                        scenario_data.loc[scenario_data.index[crash_idx + i], "close"] *= (1 - recover_pct)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "low"] *= (1 - recover_pct * 1.2)
                        scenario_data.loc[scenario_data.index[crash_idx + i], "high"] *= (1 - recover_pct * 0.8)
            
        elif scenario == "volatility_spike":
            # Increase volatility throughout the dataset
            for i in range(len(scenario_data)):
                if np.random.random() < 0.3:  # 30% of candles affected
                    # Calculate volatility multiplier (1.5-3x normal range)
                    vol_mult = np.random.uniform(1.5, 3.0)
                    
                    # Get normal candle range
                    normal_range = scenario_data["high"].iloc[i] - scenario_data["low"].iloc[i]
                    
                    # Calculate extended range
                    extended_range = normal_range * vol_mult
                    
                    # Center price (to expand around)
                    center = scenario_data["close"].iloc[i]
                    
                    # Create new high/low with extended range
                    scenario_data.loc[scenario_data.index[i], "high"] = center + (extended_range / 2)
                    scenario_data.loc[scenario_data.index[i], "low"] = center - (extended_range / 2)
        
        elif scenario == "sustained_downtrend":
            # Create a sustained downtrend (30-40% drop over time)
            total_drop = np.random.uniform(0.3, 0.4)
            start_idx = len(scenario_data) // 4
            duration = len(scenario_data) // 2
            
            for i in range(duration):
                if start_idx + i < len(scenario_data):
                    # Calculate cumulative drop for this candle
                    drop_pct = total_drop * (i + 1) / duration
                    
                    # Apply drop to prices
                    idx = scenario_data.index[start_idx + i]
                    orig_close = data["close"].iloc[start_idx + i]
                    adjusted_close = orig_close * (1 - drop_pct)
                    
                    scenario_data.loc[idx, "close"] = adjusted_close
                    
                    # Adjust high/low proportionally
                    range_pct = (data["high"].iloc[start_idx + i] - data["low"].iloc[start_idx + i]) / orig_close
                    scenario_data.loc[idx, "high"] = adjusted_close * (1 + range_pct/2)
                    scenario_data.loc[idx, "low"] = adjusted_close * (1 - range_pct/2)
        
        # Recalculate indicators
        scenario_data = self.data_loader.add_technical_indicators(scenario_data)
        
        logger.info(f"Created {scenario} scenario with {len(scenario_data)} data points")
        return scenario_data
    
    def _save_scenario_data(self, pair: str, data: pd.DataFrame, scenario: str):
        """
        Save scenario data to file.
        
        Args:
            pair: Trading pair
            data: DataFrame with price data
            scenario: Scenario type
        """
        # Create filename with scenario tag
        pair_filename = pair.replace("/", "_")
        filename = f"{self.data_dir}/{pair_filename}_{scenario}_1h.csv"
        
        # Save to CSV
        data.reset_index().to_csv(filename, index=False)
        
        logger.info(f"Saved {scenario} scenario data for {pair} to {filename}")
    
    def _load_scenario_data(self, pair: str, scenario: str) -> pd.DataFrame:
        """
        Load scenario data from file.
        
        Args:
            pair: Trading pair
            scenario: Scenario type
            
        Returns:
            DataFrame with scenario data
        """
        # Create filename with scenario tag
        pair_filename = pair.replace("/", "_")
        filename = f"{self.data_dir}/{pair_filename}_{scenario}_1h.csv"
        
        # Load data
        if os.path.exists(filename):
            data = pd.read_csv(filename, parse_dates=["timestamp"])
            data.set_index("timestamp", inplace=True)
            return data
        else:
            logger.warning(f"Scenario data file not found: {filename}")
            return pd.DataFrame()
    
    def run_backtest(self, pair: str, scenario: str, use_risk_management: bool = True) -> Dict[str, Any]:
        """
        Run a backtest for a specific pair and scenario.
        
        Args:
            pair: Trading pair
            scenario: Scenario type
            use_risk_management: Whether to use risk management
            
        Returns:
            Dictionary with backtest results
        """
        # Load scenario data
        data = self._load_scenario_data(pair, scenario)
        
        if data.empty:
            logger.error(f"No data available for {pair} {scenario}")
            return {"error": "No data available"}
        
        logger.info(f"Running backtest for {pair} {scenario} with " + 
                  f"risk management {'enabled' if use_risk_management else 'disabled'}")
        
        # Initialize portfolio and positions
        portfolio_value = self.initial_portfolio
        cash = self.initial_portfolio
        positions = {}
        trades = []
        portfolio_history = []
        
        # Trade parameters
        leverage = 20.0
        risk_percentage = 0.2
        
        # Tracking metrics
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        liquidations = 0
        peak_value = portfolio_value
        max_drawdown = 0.0
        
        # Generate trading signals
        for i in range(50, len(data)-1):  # Start after enough data for indicators
            # Get current timestamp and price
            timestamp = data.index[i]
            current_price = data["close"].iloc[i]
            
            # Simple trading signal based on EMAs
            ema9 = data["ema9"].iloc[i]
            ema21 = data["ema21"].iloc[i]
            
            # Update portfolio value
            portfolio_value = cash
            for pos_id, position in list(positions.items()):
                # Calculate unrealized P&L
                entry_price = position["entry_price"]
                size = position["size"]
                pos_leverage = position["leverage"]
                direction = position["direction"]
                
                if direction == "long":
                    price_change_pct = (current_price / entry_price) - 1
                else:  # short
                    price_change_pct = (entry_price / current_price) - 1
                
                pnl_pct = price_change_pct * pos_leverage
                position_value = position["margin"] * (1 + pnl_pct)
                
                # Update portfolio value
                portfolio_value += position_value
                
                # Check for stop-loss hit (with or without risk management)
                if use_risk_management:
                    # Update position risk
                    position = risk_manager.update_position_risk(
                        position_id=pos_id,
                        current_price=current_price,
                        position_data=position
                    )
                    positions[pos_id] = position
                    
                    # Check if stop loss hit
                    stop_loss = position["stop_loss"]
                    if (direction == "long" and data["low"].iloc[i] <= stop_loss) or \
                       (direction == "short" and data["high"].iloc[i] >= stop_loss):
                        # Close position at stop loss
                        exit_price = stop_loss
                        self._close_position(
                            trade_id=pos_id,
                            position=position,
                            exit_price=exit_price,
                            exit_reason="stop_loss",
                            exit_time=timestamp,
                            trades=trades,
                            positions=positions,
                            cash=cash
                        )
                        cash += position_value
                else:
                    # Basic stop-loss at 4% for non-risk-managed version
                    max_loss_pct = -0.04
                    if pnl_pct <= max_loss_pct:
                        # Close position with loss
                        exit_price = current_price
                        self._close_position(
                            trade_id=pos_id,
                            position=position,
                            exit_price=exit_price,
                            exit_reason="max_loss",
                            exit_time=timestamp,
                            trades=trades,
                            positions=positions,
                            cash=cash
                        )
                        cash += position_value
                
                # Check for liquidation (different thresholds with/without risk management)
                liquidation_threshold = -0.9 if use_risk_management else -0.95
                if pnl_pct <= liquidation_threshold:
                    # Position liquidated
                    liquidations += 1
                    logger.warning(f"Position {pos_id} liquidated with {pnl_pct:.2%} loss")
                    
                    # Add to closed trades
                    trades.append({
                        "trade_id": pos_id,
                        "pair": pair,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "size": size,
                        "leverage": pos_leverage,
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "pnl_percentage": liquidation_threshold,
                        "pnl_amount": position["margin"] * liquidation_threshold,
                        "exit_reason": "liquidation"
                    })
                    
                    # Remove position
                    del positions[pos_id]
                    
                    # No capital returned on liquidation
            
            # Update peak value and drawdown
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Record portfolio history
            portfolio_history.append({
                "timestamp": timestamp,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "positions": len(positions),
                "drawdown": current_drawdown
            })
            
            # Trading logic - simple EMA crossover
            signal = "hold"
            if ema9 > ema21:
                signal = "buy"  # Go long
            elif ema9 < ema21:
                signal = "sell"  # Go short
                
            # Don't trade if we already have a position in the same direction
            # or if we have no cash (simplified logic for test)
            if signal == "hold" or cash < 100:
                continue
                
            # Check existing positions
            existing_long = any(p["direction"] == "long" for p in positions.values())
            existing_short = any(p["direction"] == "short" for p in positions.values())
            
            if (signal == "buy" and existing_long) or (signal == "sell" and existing_short):
                continue
            
            # Open new position with or without risk management
            if use_risk_management:
                # Calculate position size with risk management
                market_data_window = data.iloc[i-50:i+1]
                
                # Update market data for risk management
                risk_manager.assess_volatility(pair, market_data_window)
                market_regime = self.market_analyzer.analyze_market_regimes(market_data_window)
                
                # Calculate confidence based on signal strength
                confidence = 0.7  # Default medium confidence
                
                # Get position parameters from risk manager
                position_params = risk_manager.calculate_position_size(
                    pair=pair,
                    price=current_price,
                    strategy="ema_crossover",
                    confidence=confidence,
                    win_rate=winning_trades / max(1, total_trades),
                    portfolio_value=portfolio_value
                )
                
                actual_leverage = position_params["leverage"]
                margin_amount = position_params["margin_amount"]
                position_size = position_params["position_size"]
                
                # Calculate stop loss with risk management
                if signal == "buy":
                    stop_loss = current_price - position_params["stop_loss_distance"]
                else:  # sell
                    stop_loss = current_price + position_params["stop_loss_distance"]
            else:
                # Fixed parameters without risk management
                actual_leverage = leverage
                margin_amount = portfolio_value * risk_percentage
                position_size = (margin_amount * actual_leverage) / current_price
                stop_loss = None  # Will use fixed 4% loss threshold
            
            # Check if we have enough cash
            if margin_amount > cash:
                continue
            
            # Open position
            direction = "long" if signal == "buy" else "short"
            trade_id = f"{pair}_{timestamp.strftime('%Y%m%d%H%M%S')}_{direction}"
            
            position = {
                "trade_id": trade_id,
                "pair": pair,
                "direction": direction,
                "entry_price": current_price,
                "entry_time": timestamp,
                "size": position_size,
                "leverage": actual_leverage,
                "margin": margin_amount,
                "position_value": position_size * current_price,
                "initial_stop_loss": stop_loss,
                "stop_loss": stop_loss
            }
            
            # Add to positions and update cash
            positions[trade_id] = position
            cash -= margin_amount
            total_trades += 1
            
            logger.info(f"Opened {direction} position: Price={current_price:.2f}, "
                      f"Leverage={actual_leverage:.1f}x, Margin={margin_amount:.2f}")
        
        # Close any remaining positions at the end
        final_price = data["close"].iloc[-1]
        for pos_id, position in list(positions.items()):
            self._close_position(
                trade_id=pos_id,
                position=position,
                exit_price=final_price,
                exit_reason="end_of_test",
                exit_time=data.index[-1],
                trades=trades,
                positions=positions,
                cash=cash
            )
            
            # Calculate final position value
            direction = position["direction"]
            entry_price = position["entry_price"]
            pos_leverage = position["leverage"]
            margin = position["margin"]
            
            if direction == "long":
                price_change_pct = (final_price / entry_price) - 1
            else:  # short
                price_change_pct = (entry_price / final_price) - 1
                
            pnl_pct = price_change_pct * pos_leverage
            position_value = margin * (1 + pnl_pct)
            
            cash += position_value
        
        # Calculate final portfolio value
        portfolio_value = cash
        
        # Calculate performance metrics
        for trade in trades:
            pnl = trade.get("pnl_amount", 0)
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                losing_trades += 1
                total_loss += abs(pnl)
        
        # Compile results
        results = {
            "pair": pair,
            "scenario": scenario,
            "risk_management": use_risk_management,
            "initial_portfolio": self.initial_portfolio,
            "final_portfolio": portfolio_value,
            "total_return": (portfolio_value / self.initial_portfolio - 1) * 100,
            "max_drawdown": max_drawdown * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float('inf'),
            "liquidations": liquidations,
            "trades": trades,
            "portfolio_history": portfolio_history
        }
        
        # Save results
        self._save_test_results(results)
        
        return results
    
    def _close_position(self, trade_id: str, position: Dict[str, Any], exit_price: float,
                      exit_reason: str, exit_time, trades: List, positions: Dict, cash: float):
        """Helper to close a position and record the trade"""
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]
        leverage = position["leverage"]
        margin = position["margin"]
        
        # Calculate P&L
        if direction == "long":
            price_change_pct = (exit_price / entry_price) - 1
        else:  # short
            price_change_pct = (entry_price / exit_price) - 1
            
        pnl_pct = price_change_pct * leverage
        pnl_amount = margin * pnl_pct
        
        # Record trade
        trades.append({
            "trade_id": trade_id,
            "pair": position["pair"],
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "leverage": leverage,
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "pnl_percentage": pnl_pct,
            "pnl_amount": pnl_amount,
            "exit_reason": exit_reason
        })
        
        # Remove position
        del positions[trade_id]
    
    def _save_test_results(self, results: Dict[str, Any]):
        """
        Save test results to file.
        
        Args:
            results: Dictionary with test results
        """
        # Create filename
        pair = results["pair"]
        scenario = results["scenario"]
        risk_tag = "with_risk" if results["risk_management"] else "no_risk"
        
        pair_filename = pair.replace("/", "_")
        filename = f"{self.results_dir}/{pair_filename}_{scenario}_{risk_tag}.json"
        
        # Create results object with metadata
        save_results = {
            "pair": pair,
            "scenario": scenario,
            "risk_management": results["risk_management"],
            "metrics": {
                "initial_portfolio": results["initial_portfolio"],
                "final_portfolio": results["final_portfolio"],
                "total_return": results["total_return"],
                "max_drawdown": results["max_drawdown"],
                "total_trades": results["total_trades"],
                "winning_trades": results["winning_trades"],
                "losing_trades": results["losing_trades"],
                "win_rate": results["win_rate"],
                "profit_factor": results["profit_factor"],
                "liquidations": results["liquidations"]
            },
            # Add simplified trade history
            "trades": [{
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "leverage": trade["leverage"],
                "pnl_percentage": trade["pnl_percentage"],
                "exit_reason": trade["exit_reason"]
            } for trade in results["trades"]],
            # Add portfolio history
            "portfolio_history": [{
                "timestamp": str(entry["timestamp"]),
                "portfolio_value": entry["portfolio_value"],
                "drawdown": entry["drawdown"]
            } for entry in results["portfolio_history"][::10]]  # Save every 10th data point to reduce size
        }
        
        # Save to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        logger.info(f"Saved test results to {filename}")
    
    def plot_test_results(self, pair: str, scenario: str):
        """
        Plot comparison of test results with and without risk management.
        
        Args:
            pair: Trading pair
            scenario: Scenario type
        """
        # Load results
        pair_filename = pair.replace("/", "_")
        with_risk_file = f"{self.results_dir}/{pair_filename}_{scenario}_with_risk.json"
        no_risk_file = f"{self.results_dir}/{pair_filename}_{scenario}_no_risk.json"
        
        if not os.path.exists(with_risk_file) or not os.path.exists(no_risk_file):
            logger.error(f"Results files not found for {pair} {scenario}")
            return
        
        import json
        
        with open(with_risk_file, 'r') as f:
            with_risk_results = json.load(f)
            
        with open(no_risk_file, 'r') as f:
            no_risk_results = json.load(f)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value over time
        plt.subplot(2, 1, 1)
        
        # Extract portfolio values
        with_risk_history = with_risk_results["portfolio_history"]
        no_risk_history = no_risk_results["portfolio_history"]
        
        with_risk_values = [entry["portfolio_value"] for entry in with_risk_history]
        no_risk_values = [entry["portfolio_value"] for entry in no_risk_history]
        
        # Plot
        plt.plot(range(len(with_risk_values)), with_risk_values, 'b-', label="With Risk Management")
        plt.plot(range(len(no_risk_values)), no_risk_values, 'r-', label="Without Risk Management")
        
        plt.title(f"Portfolio Performance: {pair} {scenario}")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        
        with_risk_drawdowns = [entry["drawdown"] * 100 for entry in with_risk_history]
        no_risk_drawdowns = [entry["drawdown"] * 100 for entry in no_risk_history]
        
        plt.plot(range(len(with_risk_drawdowns)), with_risk_drawdowns, 'b-', label="With Risk Management")
        plt.plot(range(len(no_risk_drawdowns)), no_risk_drawdowns, 'r-', label="Without Risk Management")
        
        plt.title("Portfolio Drawdown")
        plt.xlabel("Time Period")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = f"{self.results_dir}/{pair_filename}_{scenario}_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved comparison plot to {plot_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"COMPARISON RESULTS: {pair} - {scenario}")
        print("=" * 80)
        print(f"WITH RISK MANAGEMENT:")
        print(f"  Return: {with_risk_results['metrics']['total_return']:.2f}%")
        print(f"  Max Drawdown: {with_risk_results['metrics']['max_drawdown']:.2f}%")
        print(f"  Win Rate: {with_risk_results['metrics']['win_rate']:.2f}")
        print(f"  Liquidations: {with_risk_results['metrics']['liquidations']}")
        print()
        print(f"WITHOUT RISK MANAGEMENT:")
        print(f"  Return: {no_risk_results['metrics']['total_return']:.2f}%")
        print(f"  Max Drawdown: {no_risk_results['metrics']['max_drawdown']:.2f}%")
        print(f"  Win Rate: {no_risk_results['metrics']['win_rate']:.2f}")
        print(f"  Liquidations: {no_risk_results['metrics']['liquidations']}")
        print("=" * 80)
    
    def run_all_tests(self, pairs: List[str] = PAIRS, scenarios: List[str] = SCENARIO_TYPES):
        """
        Run all tests for specified pairs and scenarios.
        
        Args:
            pairs: List of trading pairs
            scenarios: List of scenarios
        """
        logger.info(f"Running all tests for {len(pairs)} pairs and {len(scenarios)} scenarios")
        
        # First, generate test data if needed
        self.generate_test_data(pairs)
        
        # Run tests with and without risk management
        results = []
        
        for pair in pairs:
            for scenario in scenarios:
                # Run with risk management
                with_risk_results = self.run_backtest(pair, scenario, use_risk_management=True)
                
                # Run without risk management
                no_risk_results = self.run_backtest(pair, scenario, use_risk_management=False)
                
                # Plot comparison
                self.plot_test_results(pair, scenario)
                
                # Store results
                results.append({
                    "pair": pair,
                    "scenario": scenario,
                    "with_risk": with_risk_results,
                    "no_risk": no_risk_results
                })
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict[str, Any]]):
        """
        Generate summary report of all test results.
        
        Args:
            results: List of test result pairs
        """
        # Create summary file
        filename = f"{self.results_dir}/summary_report.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RISK MANAGEMENT SYSTEM TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-" * 50 + "\n")
            
            # Calculate averages
            with_risk_returns = [r["with_risk"]["total_return"] for r in results if "error" not in r["with_risk"]]
            no_risk_returns = [r["no_risk"]["total_return"] for r in results if "error" not in r["no_risk"]]
            
            with_risk_drawdowns = [r["with_risk"]["max_drawdown"] for r in results if "error" not in r["with_risk"]]
            no_risk_drawdowns = [r["no_risk"]["max_drawdown"] for r in results if "error" not in r["no_risk"]]
            
            with_risk_liquidations = [r["with_risk"]["liquidations"] for r in results if "error" not in r["with_risk"]]
            no_risk_liquidations = [r["no_risk"]["liquidations"] for r in results if "error" not in r["no_risk"]]
            
            # Write averages
            f.write(f"Average Return (With Risk): {sum(with_risk_returns) / len(with_risk_returns):.2f}%\n")
            f.write(f"Average Return (No Risk): {sum(no_risk_returns) / len(no_risk_returns):.2f}%\n\n")
            
            f.write(f"Average Max Drawdown (With Risk): {sum(with_risk_drawdowns) / len(with_risk_drawdowns):.2f}%\n")
            f.write(f"Average Max Drawdown (No Risk): {sum(no_risk_drawdowns) / len(no_risk_drawdowns):.2f}%\n\n")
            
            f.write(f"Total Liquidations (With Risk): {sum(with_risk_liquidations)}\n")
            f.write(f"Total Liquidations (No Risk): {sum(no_risk_liquidations)}\n\n")
            
            # Write detailed results by pair and scenario
            f.write("DETAILED RESULTS BY PAIR AND SCENARIO\n")
            f.write("-" * 50 + "\n")
            
            for result in results:
                pair = result["pair"]
                scenario = result["scenario"]
                
                f.write(f"{pair} - {scenario}\n")
                
                if "error" in result["with_risk"]:
                    f.write("  Error in test with risk management\n")
                else:
                    with_risk = result["with_risk"]
                    f.write(f"  With Risk Management:\n")
                    f.write(f"    Return: {with_risk['total_return']:.2f}%\n")
                    f.write(f"    Max Drawdown: {with_risk['max_drawdown']:.2f}%\n")
                    f.write(f"    Win Rate: {with_risk['win_rate']:.2f}\n")
                    f.write(f"    Liquidations: {with_risk['liquidations']}\n")
                
                if "error" in result["no_risk"]:
                    f.write("  Error in test without risk management\n")
                else:
                    no_risk = result["no_risk"]
                    f.write(f"  Without Risk Management:\n")
                    f.write(f"    Return: {no_risk['total_return']:.2f}%\n")
                    f.write(f"    Max Drawdown: {no_risk['max_drawdown']:.2f}%\n")
                    f.write(f"    Win Rate: {no_risk['win_rate']:.2f}\n")
                    f.write(f"    Liquidations: {no_risk['liquidations']}\n")
                
                f.write("\n")
            
            # Write conclusions
            f.write("CONCLUSIONS\n")
            f.write("-" * 50 + "\n")
            
            # Compare performance
            avg_with_risk_return = sum(with_risk_returns) / len(with_risk_returns)
            avg_no_risk_return = sum(no_risk_returns) / len(no_risk_returns)
            
            if avg_with_risk_return > avg_no_risk_return:
                return_diff = avg_with_risk_return - avg_no_risk_return
                f.write(f"1. Risk management improved returns by {return_diff:.2f}% on average\n")
            else:
                return_diff = avg_no_risk_return - avg_with_risk_return
                f.write(f"1. Risk management reduced returns by {return_diff:.2f}% on average\n")
            
            # Compare drawdowns
            avg_with_risk_dd = sum(with_risk_drawdowns) / len(with_risk_drawdowns)
            avg_no_risk_dd = sum(no_risk_drawdowns) / len(no_risk_drawdowns)
            
            dd_reduction = (avg_no_risk_dd - avg_with_risk_dd) / avg_no_risk_dd * 100
            f.write(f"2. Risk management reduced maximum drawdowns by {dd_reduction:.2f}%\n")
            
            # Compare liquidations
            total_with_risk_liq = sum(with_risk_liquidations)
            total_no_risk_liq = sum(no_risk_liquidations)
            
            if total_with_risk_liq < total_no_risk_liq:
                liq_reduction = (total_no_risk_liq - total_with_risk_liq) / max(1, total_no_risk_liq) * 100
                f.write(f"3. Risk management prevented {liq_reduction:.2f}% of liquidations\n")
            else:
                f.write(f"3. Risk management did not reduce liquidations\n")
            
            # Overall assessment
            if avg_with_risk_return >= avg_no_risk_return * 0.9 and avg_with_risk_dd < avg_no_risk_dd:
                f.write("\nOVERALL ASSESSMENT: Risk management system is effective at reducing risk ")
                f.write("while maintaining profitability.\n")
            elif avg_with_risk_return < avg_no_risk_return * 0.8:
                f.write("\nOVERALL ASSESSMENT: Risk management system reduces risk but significantly ")
                f.write("impacts profitability. Further optimization needed.\n")
            else:
                f.write("\nOVERALL ASSESSMENT: Risk management shows mixed results. ")
                f.write("Consider scenario-specific adjustments.\n")
        
        logger.info(f"Generated summary report at {filename}")
        
        # Print summary to console
        with open(filename, 'r') as f:
            print(f.read())

def main():
    """Main function"""
    # Create tester
    tester = RiskManagementTester()
    
    # Run all tests
    tester.run_all_tests()
    
    return 0

if __name__ == "__main__":
    main()