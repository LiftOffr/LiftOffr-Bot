#!/usr/bin/env python3
"""
Simulate High Leverage Risk

This script simulates the risk of high leverage trading by:
1. Calculating liquidation prices for different leverage levels
2. Simulating price movements to find likelihood of liquidation
3. Calculating expected returns vs drawdowns

Usage:
    python simulate_high_leverage_risk.py
"""
import json
import os
import sys
import time
import random
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("leverage_risk_simulator")

# Constants
DATA_DIR = "data"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

class LeverageRiskSimulator:
    """Simulates liquidation risk for high leverage positions"""
    
    def __init__(self):
        """Initialize the simulator"""
        self.maintenance_margin = 0.01  # 1%
        self.liquidation_fee = 0.0075   # 0.75%
        self.positions = self._load_positions()
        self.portfolio = self._load_portfolio()
        self.trades = self._load_trades()
        
        # Store original positions for comparison
        self.original_positions = self.positions.copy() if self.positions else []
        
    def _load_json(self, filepath: str, default: Any = None) -> Any:
        """Load JSON data from file or return default if file doesn't exist"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default
    
    def _load_positions(self) -> List[Dict]:
        """Load current positions"""
        return self._load_json(POSITIONS_FILE, [])
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio data"""
        return self._load_json(PORTFOLIO_FILE, {"balance": 20000.0})
    
    def _load_trades(self) -> List[Dict]:
        """Load trade history"""
        return self._load_json(TRADES_FILE, [])
    
    def _save_json(self, filepath: str, data: Any) -> None:
        """Save JSON data to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_liquidation_price(
        self, 
        entry_price: float, 
        leverage: float, 
        direction: str
    ) -> float:
        """
        Calculate liquidation price based on leverage and direction
        
        Args:
            entry_price: Entry price
            leverage: Leverage used
            direction: 'Long' or 'Short'
            
        Returns:
            Liquidation price
        """
        if direction.lower() == "long":
            liquidation_price = entry_price * (1 - (1 / leverage) + self.maintenance_margin)
        else:  # Short
            liquidation_price = entry_price * (1 + (1 / leverage) - self.maintenance_margin)
        
        return liquidation_price
    
    def calculate_price_move_for_liquidation(
        self, 
        entry_price: float, 
        leverage: float, 
        direction: str
    ) -> float:
        """
        Calculate the percentage price move needed for liquidation
        
        Args:
            entry_price: Entry price
            leverage: Leverage used
            direction: 'Long' or 'Short'
            
        Returns:
            Percentage price move needed for liquidation (negative for price decrease)
        """
        liquidation_price = self.calculate_liquidation_price(entry_price, leverage, direction)
        
        if direction.lower() == "long":
            return (liquidation_price / entry_price - 1) * 100  # Negative percentage
        else:  # Short
            return (liquidation_price / entry_price - 1) * 100  # Positive percentage
    
    def simulate_price_volatility(
        self,
        current_price: float, 
        volatility: float,
        num_samples: int = 1000,
        time_horizon_hours: int = 24
    ) -> np.ndarray:
        """
        Simulate price movements based on volatility
        
        Args:
            current_price: Current price
            volatility: Daily volatility (percentage)
            num_samples: Number of price paths to simulate
            time_horizon_hours: Time horizon in hours
            
        Returns:
            Array of simulated prices
        """
        # Convert daily volatility to hourly
        hourly_volatility = volatility / np.sqrt(24)
        
        # Simulate using geometric Brownian motion
        drift = 0  # Assume zero drift for short-term simulations
        dt = 1  # 1 hour steps
        
        # Create empty array for price paths
        price_paths = np.zeros((num_samples, time_horizon_hours + 1))
        price_paths[:, 0] = current_price
        
        # Simulate price paths
        for t in range(1, time_horizon_hours + 1):
            z = np.random.normal(0, 1, num_samples)
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (drift - 0.5 * hourly_volatility**2) * dt +
                hourly_volatility * np.sqrt(dt) * z
            )
            
        return price_paths
    
    def estimate_liquidation_probability(
        self,
        entry_price: float,
        leverage: float,
        direction: str,
        volatility: float,
        time_horizon_hours: int = 24,
        num_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Estimate the probability of liquidation within a time horizon
        
        Args:
            entry_price: Entry price
            leverage: Leverage used
            direction: 'Long' or 'Short'
            volatility: Daily volatility (percentage)
            time_horizon_hours: Time horizon in hours
            num_samples: Number of price paths to simulate
            
        Returns:
            Tuple of (liquidation probability, expected return)
        """
        liquidation_price = self.calculate_liquidation_price(entry_price, leverage, direction)
        price_paths = self.simulate_price_volatility(
            entry_price, volatility/100, num_samples, time_horizon_hours
        )
        
        # Count liquidations
        liquidations = 0
        final_returns = []
        
        for path in price_paths:
            liquidated = False
            
            # Check each timepoint for liquidation
            for price in path[1:]:  # Skip initial price
                if (direction.lower() == "long" and price <= liquidation_price) or \
                   (direction.lower() == "short" and price >= liquidation_price):
                    liquidations += 1
                    liquidated = True
                    # Add a -100% return (total loss) for liquidated positions
                    final_returns.append(-1.0)
                    break
            
            # If not liquidated, calculate return at end of period
            if not liquidated:
                final_price = path[-1]
                if direction.lower() == "long":
                    pnl = (final_price / entry_price - 1) * leverage
                else:  # Short
                    pnl = (entry_price / final_price - 1) * leverage
                final_returns.append(pnl)
        
        # Calculate probability and expected return
        liquidation_prob = liquidations / num_samples
        expected_return = np.mean(final_returns)
        
        return liquidation_prob, expected_return
    
    def analyze_position_risk(self, position: Dict) -> Dict:
        """
        Analyze the risk of a position
        
        Args:
            position: Position data
            
        Returns:
            Risk analysis results
        """
        # Extract position data
        pair = position["pair"]
        direction = position["direction"]
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        leverage = position["leverage"]
        size = position["size"]
        
        # Calculate metrics
        liquidation_price = self.calculate_liquidation_price(entry_price, leverage, direction)
        price_move_for_liquidation = self.calculate_price_move_for_liquidation(entry_price, leverage, direction)
        
        # Estimate volatility based on pair (in percentage)
        # This is a simplification - in a real system you would calculate this from historical data
        volatility_map = {
            "BTC/USD": 3.0,   # 3% daily volatility
            "ETH/USD": 4.0,   # 4% daily volatility
            "SOL/USD": 7.0,   # 7% daily volatility
            "ADA/USD": 6.0,   # 6% daily volatility
            "AVAX/USD": 8.0,  # 8% daily volatility
            "LINK/USD": 5.5,  # 5.5% daily volatility
            "MATIC/USD": 7.5, # 7.5% daily volatility
            "UNI/USD": 6.5,   # 6.5% daily volatility
            "ATOM/USD": 7.0,  # 7% daily volatility
            "DOT/USD": 6.0    # 6% daily volatility
        }
        
        volatility = volatility_map.get(pair, 5.0)  # Default 5% if pair not found
        
        # Estimate liquidation probability for different time horizons
        liq_prob_24h, expected_return_24h = self.estimate_liquidation_probability(
            entry_price, leverage, direction, volatility, 24
        )
        
        liq_prob_48h, expected_return_48h = self.estimate_liquidation_probability(
            entry_price, leverage, direction, volatility, 48
        )
        
        liq_prob_7d, expected_return_7d = self.estimate_liquidation_probability(
            entry_price, leverage, direction, volatility, 24 * 7
        )
        
        # Calculate distance to liquidation
        if direction.lower() == "long":
            current_distance_pct = ((current_price - liquidation_price) / current_price) * 100
        else:  # Short
            current_distance_pct = ((liquidation_price - current_price) / current_price) * 100
        
        return {
            "pair": pair,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": current_price,
            "leverage": leverage,
            "liquidation_price": liquidation_price,
            "price_move_for_liquidation": price_move_for_liquidation,
            "distance_to_liquidation_pct": current_distance_pct,
            "estimated_volatility": volatility,
            "liquidation_probability_24h": liq_prob_24h,
            "expected_return_24h": expected_return_24h,
            "liquidation_probability_48h": liq_prob_48h,
            "expected_return_48h": expected_return_48h,
            "liquidation_probability_7d": liq_prob_7d,
            "expected_return_7d": expected_return_7d,
            "size": size,
            "margin": size / leverage
        }
    
    def simulate_flash_crash(
        self, 
        crash_pct: float = 15.0,
        recovery_pct: Optional[float] = None
    ) -> Dict:
        """
        Simulate a flash crash across all assets
        
        Args:
            crash_pct: Percentage price drop in the flash crash
            recovery_pct: Percentage price recovery after crash (None for no recovery)
            
        Returns:
            Simulation results
        """
        logger.info(f"Simulating flash crash of {crash_pct}%...")
        
        # Store original positions and portfolio
        original_positions = self.positions.copy() if self.positions else []
        original_portfolio = self.portfolio.copy()
        original_balance = original_portfolio.get("balance", 20000.0)
        
        # Simulate crash for each position
        liquidated_positions = []
        survived_positions = []
        total_loss = 0.0
        
        for position in original_positions:
            pair = position["pair"]
            direction = position["direction"]
            entry_price = position["entry_price"]
            current_price = position["current_price"]
            leverage = position["leverage"]
            margin = position["size"] / leverage
            
            # Calculate liquidation price
            liquidation_price = self.calculate_liquidation_price(entry_price, leverage, direction)
            
            # Apply flash crash to price
            if direction.lower() == "long":
                crash_price = current_price * (1 - crash_pct / 100)
                # Check if liquidated
                if crash_price <= liquidation_price:
                    liquidated_positions.append({
                        "pair": pair,
                        "direction": direction,
                        "leverage": leverage,
                        "entry_price": entry_price,
                        "liquidation_price": liquidation_price,
                        "crash_price": crash_price,
                        "margin": margin
                    })
                    total_loss += margin + (margin * self.liquidation_fee)
                else:
                    survived_positions.append({
                        "pair": pair,
                        "direction": direction,
                        "leverage": leverage,
                        "entry_price": entry_price,
                        "liquidation_price": liquidation_price,
                        "crash_price": crash_price,
                        "margin": margin
                    })
            else:  # Short
                crash_price = current_price * (1 - crash_pct / 100)
                # Shorts benefit from crash, won't be liquidated
                survived_positions.append({
                    "pair": pair,
                    "direction": direction,
                    "leverage": leverage,
                    "entry_price": entry_price,
                    "liquidation_price": liquidation_price,
                    "crash_price": crash_price,
                    "margin": margin,
                    "pnl": margin * leverage * (crash_pct / 100)  # Estimated PnL from crash
                })
        
        # Calculate new portfolio value
        new_balance = original_balance - total_loss
        
        # If recovery is specified, calculate recovery for survived positions
        recovery_gain = 0.0
        if recovery_pct is not None and recovery_pct > 0:
            logger.info(f"Simulating recovery of {recovery_pct}%...")
            for position in survived_positions:
                direction = position["direction"]
                margin = position["margin"]
                leverage = position["leverage"]
                
                if direction.lower() == "long":
                    # Longs benefit from recovery
                    position["recovery_pnl"] = margin * leverage * (recovery_pct / 100)
                    recovery_gain += position["recovery_pnl"]
                else:  # Short
                    # Shorts lose during recovery
                    max_loss = margin  # Can't lose more than margin
                    position["recovery_pnl"] = max(-max_loss, -margin * leverage * (recovery_pct / 100))
                    recovery_gain += position["recovery_pnl"]
            
            # Update balance with recovery gains/losses
            final_balance = new_balance + recovery_gain
        else:
            final_balance = new_balance
        
        # Prepare result
        result = {
            "original_balance": original_balance,
            "crash_pct": crash_pct,
            "recovery_pct": recovery_pct,
            "liquidated_positions": liquidated_positions,
            "survived_positions": survived_positions,
            "total_positions": len(original_positions),
            "liquidated_count": len(liquidated_positions),
            "survival_rate": len(survived_positions) / len(original_positions) if original_positions else 1.0,
            "total_loss_from_liquidations": total_loss,
            "balance_after_crash": new_balance,
            "balance_after_recovery": final_balance,
            "total_drawdown_pct": ((original_balance - new_balance) / original_balance) * 100,
            "final_drawdown_pct": ((original_balance - final_balance) / original_balance) * 100
        }
        
        return result
    
    def analyze_all_positions(self) -> None:
        """Analyze risks for all current positions"""
        if not self.positions:
            logger.info("No open positions to analyze")
            return
        
        logger.info(f"Analyzing {len(self.positions)} open positions for liquidation risk...")
        
        # Analyze each position
        position_risks = []
        high_risk_positions = []
        
        for position in self.positions:
            risk_analysis = self.analyze_position_risk(position)
            position_risks.append(risk_analysis)
            
            # Identify high-risk positions (>10% chance of liquidation in 24h)
            if risk_analysis["liquidation_probability_24h"] > 0.1:
                high_risk_positions.append(risk_analysis)
        
        # Print summary
        print("\n" + "="*80)
        print("POSITION RISK ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nTotal positions: {len(self.positions)}")
        print(f"High-risk positions (>10% liquidation chance in 24h): {len(high_risk_positions)}")
        
        # Calculate portfolio-level statistics
        avg_leverage = sum(p["leverage"] for p in self.positions) / len(self.positions)
        avg_liquidation_prob_24h = sum(r["liquidation_probability_24h"] for r in position_risks) / len(position_risks)
        max_liquidation_prob_24h = max(r["liquidation_probability_24h"] for r in position_risks)
        
        print(f"\nAverage leverage: {avg_leverage:.2f}x")
        print(f"Average 24h liquidation probability: {avg_liquidation_prob_24h:.2%}")
        print(f"Maximum 24h liquidation probability: {max_liquidation_prob_24h:.2%}")
        
        # Print detailed position risks
        print("\n" + "-"*80)
        print("DETAILED POSITION RISK ANALYSIS")
        print("-"*80)
        
        for risk in position_risks:
            print(f"\n{risk['pair']} - {risk['direction']} @ {risk['leverage']:.1f}x leverage:")
            print(f"  Entry: ${risk['entry_price']:.4f}, Current: ${risk['current_price']:.4f}")
            print(f"  Liquidation price: ${risk['liquidation_price']:.4f}")
            print(f"  Distance to liquidation: {risk['distance_to_liquidation_pct']:.2f}%")
            print(f"  Liquidation probabilities:")
            print(f"    24h: {risk['liquidation_probability_24h']:.2%}")
            print(f"    48h: {risk['liquidation_probability_48h']:.2%}")
            print(f"    7d: {risk['liquidation_probability_7d']:.2%}")
            print(f"  Expected returns (including liquidation risk):")
            print(f"    24h: {risk['expected_return_24h']:.2%}")
            print(f"    48h: {risk['expected_return_48h']:.2%}")
            print(f"    7d: {risk['expected_return_7d']:.2%}")
        
        # Simulate flash crashes of different magnitudes
        print("\n" + "-"*80)
        print("FLASH CRASH SIMULATION")
        print("-"*80)
        
        for crash_pct in [5, 10, 15, 20, 30]:
            crash_result = self.simulate_flash_crash(crash_pct)
            print(f"\nFlash crash of {crash_pct}%:")
            print(f"  Positions liquidated: {crash_result['liquidated_count']}/{crash_result['total_positions']} ({crash_result['survival_rate']:.2%} survived)")
            print(f"  Portfolio drawdown: {crash_result['total_drawdown_pct']:.2f}%")
            print(f"  Balance after crash: ${crash_result['balance_after_crash']:.2f}")
            
            # Simulate recovery after crash
            recovery_result = self.simulate_flash_crash(crash_pct, recovery_pct=crash_pct/2)
            print(f"\nFlash crash of {crash_pct}% with {crash_pct/2}% recovery:")
            print(f"  Final portfolio drawdown: {recovery_result['final_drawdown_pct']:.2f}%")
            print(f"  Final balance: ${recovery_result['balance_after_recovery']:.2f}")
        
        print("\n" + "="*80)

def analyze_leverage_risk_profiles():
    """Analyze risk profiles of different leverage levels"""
    print("\n" + "="*80)
    print("LEVERAGE RISK PROFILES")
    print("="*80)
    
    # Asset volatilities (daily %)
    volatilities = {
        "BTC/USD": 3.0,
        "ETH/USD": 4.0,
        "SOL/USD": 7.0,
        "AVAX/USD": 8.0,
    }
    
    # Leverage levels to analyze
    leverage_levels = [5, 10, 20, 30, 50, 75, 100, 125]
    
    # Time horizons
    time_horizons = [24, 48, 24*7]  # 1 day, 2 days, 1 week
    
    for pair, volatility in volatilities.items():
        print(f"\n{pair} (Daily Volatility: {volatility}%):")
        
        # Table header
        print(f"\n{'Leverage':>8} | {'24h Liq%':>8} | {'48h Liq%':>8} | {'7d Liq%':>8} | {'24h Ret%':>8} | {'7d Ret%':>8}")
        print("-"*65)
        
        simulator = LeverageRiskSimulator()
        
        for leverage in leverage_levels:
            # For both Long and Short directions
            for direction in ["Long", "Short"]:
                # Calculate liquidation probabilities and expected returns
                liq_probs = []
                exp_returns = []
                
                for hours in time_horizons:
                    liq_prob, exp_return = simulator.estimate_liquidation_probability(
                        100.0,  # Arbitrary entry price
                        leverage,
                        direction,
                        volatility,
                        hours
                    )
                    liq_probs.append(liq_prob)
                    exp_returns.append(exp_return)
                
                # Print results
                print(f"{leverage:>7}x {direction[0]} | {liq_probs[0]:>7.2%} | {liq_probs[1]:>7.2%} | {liq_probs[2]:>7.2%} | {exp_returns[0]:>7.2%} | {exp_returns[2]:>7.2%}")

def calculate_optimal_leverage():
    """Calculate optimal leverage for maximum expected return"""
    print("\n" + "="*80)
    print("OPTIMAL LEVERAGE CALCULATION")
    print("="*80)
    
    # Asset volatilities (daily %)
    volatilities = {
        "BTC/USD": 3.0,
        "ETH/USD": 4.0,
        "SOL/USD": 7.0,
        "AVAX/USD": 8.0,
    }
    
    # Time horizons
    time_horizons = [24, 24*7, 24*30]  # 1 day, 1 week, 1 month
    
    for pair, volatility in volatilities.items():
        print(f"\n{pair} (Daily Volatility: {volatility}%):")
        
        # Table header
        print(f"\n{'Time Horizon':>12} | {'Opt Leverage':>12} | {'Max Exp Ret%':>12} | {'Liq Prob':>10}")
        print("-"*55)
        
        simulator = LeverageRiskSimulator()
        
        for hours in time_horizons:
            # Test a range of leverage values to find optimal
            leverage_values = list(range(1, 126, 5))  # 1 to 125 in steps of 5
            best_return = -float('inf')
            optimal_leverage = 0
            optimal_liq_prob = 0
            
            for leverage in leverage_values:
                # Check both directions
                for direction in ["Long", "Short"]:
                    liq_prob, exp_return = simulator.estimate_liquidation_probability(
                        100.0,  # Arbitrary entry price
                        leverage,
                        direction,
                        volatility,
                        hours
                    )
                    
                    if exp_return > best_return:
                        best_return = exp_return
                        optimal_leverage = leverage
                        optimal_liq_prob = liq_prob
            
            # Convert time horizon to readable format
            if hours == 24:
                horizon_str = "1 day"
            elif hours == 24*7:
                horizon_str = "1 week"
            else:
                horizon_str = "1 month"
            
            # Print results
            print(f"{horizon_str:>12} | {optimal_leverage:>12}x | {best_return:>11.2%} | {optimal_liq_prob:>9.2%}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simulate high leverage trading risk")
    parser.add_argument("--analyze-risk", action="store_true", help="Analyze risk of current positions")
    parser.add_argument("--leverage-profiles", action="store_true", help="Show risk profiles for different leverage levels")
    parser.add_argument("--optimal-leverage", action="store_true", help="Calculate optimal leverage for maximum expected return")
    parser.add_argument("--crash", type=float, help="Simulate a flash crash of specified percentage")
    args = parser.parse_args()
    
    simulator = LeverageRiskSimulator()
    
    if args.analyze_risk:
        simulator.analyze_all_positions()
    
    if args.leverage_profiles:
        analyze_leverage_risk_profiles()
    
    if args.optimal_leverage:
        calculate_optimal_leverage()
    
    if args.crash:
        result = simulator.simulate_flash_crash(args.crash)
        print(f"\nSimulated flash crash of {args.crash}%:")
        print(f"Positions liquidated: {result['liquidated_count']}/{result['total_positions']}")
        print(f"Portfolio drawdown: {result['total_drawdown_pct']:.2f}%")
        print(f"Balance after crash: ${result['balance_after_crash']:.2f}")
    
    # If no specific action was requested, run all analyses
    if not (args.analyze_risk or args.leverage_profiles or args.optimal_leverage or args.crash):
        simulator.analyze_all_positions()
        analyze_leverage_risk_profiles()
        calculate_optimal_leverage()

if __name__ == "__main__":
    main()