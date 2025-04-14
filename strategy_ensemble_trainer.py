#!/usr/bin/env python3
"""
Strategy Ensemble Trainer

This module trains multiple trading strategies to work together efficiently,
optimizing how they collaborate in different market regimes. It includes:

1. Market regime detection and labeling
2. Strategy role assignment based on capabilities
3. Strategy weight optimization by regime
4. Performance backtesting and visualization
5. Configuration generation for the model collaboration integrator
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy_ensemble_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import ML components (will be imported from proper modules in full implementation)
def prepare_data(df, lookback=10): return df, df.index[-100:], ["close"]
def train_model(X, y, **kwargs): return {"model": "dummy"}
def evaluate_model(model, X, y): return {"accuracy": 0.85, "f1": 0.82}

class StrategyEnsembleTrainer:
    """
    Trains a strategy ensemble for collaborative trading
    """
    
    def __init__(
        self,
        strategies: List[str] = ["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"],
        assets: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        data_dir: str = "historical_data",
        timeframes: List[str] = ["5m", "15m", "1h", "4h"],
        training_days: int = 60,
        validation_days: int = 14,
        ensemble_output_dir: str = "models/ensemble",
        min_regime_samples: int = 50
    ):
        """
        Initialize the strategy ensemble trainer
        
        Args:
            strategies: List of strategies to include in ensemble
            assets: List of assets to train on
            data_dir: Directory containing historical data
            timeframes: Timeframes to use for training
            training_days: Days of historical data for training
            validation_days: Days of historical data for validation
            ensemble_output_dir: Directory to save ensemble results
            min_regime_samples: Minimum samples required for a regime
        """
        self.strategies = strategies
        self.assets = assets
        self.data_dir = data_dir
        self.timeframes = timeframes
        self.training_days = training_days
        self.validation_days = validation_days
        self.ensemble_output_dir = ensemble_output_dir
        self.min_regime_samples = min_regime_samples
        
        # Ensure output directory exists
        os.makedirs(ensemble_output_dir, exist_ok=True)
        
        # Define market regimes
        self.market_regimes = [
            "trending_bullish",
            "trending_bearish",
            "volatile",
            "neutral",
            "ranging"
        ]
        
        # Strategy capabilities by regime (expert knowledge)
        self.strategy_capabilities = {
            "ARIMAStrategy": {
                "trending_bullish": 0.7,
                "trending_bearish": 0.7,
                "volatile": 0.4,
                "neutral": 0.5,
                "ranging": 0.6
            },
            "AdaptiveStrategy": {
                "trending_bullish": 0.5,
                "trending_bearish": 0.5,
                "volatile": 0.6,
                "neutral": 0.6,
                "ranging": 0.8
            },
            "IntegratedStrategy": {
                "trending_bullish": 0.6,
                "trending_bearish": 0.6,
                "volatile": 0.8,
                "neutral": 0.7,
                "ranging": 0.5
            },
            "MLStrategy": {
                "trending_bullish": 0.8,
                "trending_bearish": 0.7,
                "volatile": 0.7,
                "neutral": 0.7,
                "ranging": 0.6
            }
        }
        
        logger.info(f"Strategy Ensemble Trainer initialized with {len(strategies)} strategies and {len(assets)} assets")
    
    def load_historical_data(
        self,
        asset: str,
        timeframe: str = "1h",
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a trading pair
        
        Args:
            asset: Trading pair
            timeframe: Timeframe
            days: Number of days of data to load (None for all)
            
        Returns:
            DataFrame: Historical data
        """
        clean_asset = asset.replace("/", "")
        file_path = os.path.join(self.data_dir, f"{clean_asset}_{timeframe}.csv")
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"No historical data found for {asset} ({timeframe})")
                return pd.DataFrame()
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Filter to the requested number of days if specified
            if days:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = df[df.index >= start_date]
            
            logger.info(f"Loaded {len(df)} historical data points for {asset} ({timeframe})")
            return df
        
        except Exception as e:
            logger.error(f"Error loading historical data for {asset} ({timeframe}): {e}")
            return pd.DataFrame()
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect market regime from price data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            str: Detected market regime
        """
        if df.empty:
            return "neutral"
        
        # Calculate indicators for regime detection
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility (rolling standard deviation of returns)
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Calculate trend (simple moving averages)
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['sma50'] = df['close'].rolling(window=50).mean()
            
            # Get latest data point
            latest = df.iloc[-1]
            
            # Volatility threshold for volatile regime
            volatility_threshold = 0.03  # 3% daily volatility
            
            # Check for volatility
            if latest['volatility'] > volatility_threshold:
                return "volatile"
            
            # Check for trend
            if latest['sma20'] > latest['sma50'] * 1.02:  # 2% above
                return "trending_bullish"
            elif latest['sma20'] < latest['sma50'] * 0.98:  # 2% below
                return "trending_bearish"
            
            # Check for range-bound
            # Calculate recent high and low
            recent_high = df['close'].iloc[-20:].max()
            recent_low = df['close'].iloc[-20:].min()
            range_pct = (recent_high - recent_low) / recent_low
            
            if range_pct < 0.05:  # Less than 5% range
                return "ranging"
            
            # Default to neutral
            return "neutral"
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "neutral"
    
    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label market regimes in historical data
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame: Data with regime labels
        """
        if df.empty:
            return df
        
        # Copy dataframe to avoid modifying original
        result = df.copy()
        
        # Calculate returns
        result['returns'] = result['close'].pct_change()
        
        # Calculate volatility (rolling standard deviation of returns)
        result['volatility'] = result['returns'].rolling(window=20).std()
        
        # Calculate moving averages for trend detection
        result['sma20'] = result['close'].rolling(window=20).mean()
        result['sma50'] = result['close'].rolling(window=50).mean()
        
        # Initialize regime column
        result['regime'] = 'neutral'
        
        # Label volatile periods
        volatility_threshold = 0.03
        result.loc[result['volatility'] > volatility_threshold, 'regime'] = 'volatile'
        
        # Label trending periods
        result.loc[(result['sma20'] > result['sma50'] * 1.02) & (result['regime'] == 'neutral'), 'regime'] = 'trending_bullish'
        result.loc[(result['sma20'] < result['sma50'] * 0.98) & (result['regime'] == 'neutral'), 'regime'] = 'trending_bearish'
        
        # Label ranging periods (more complex)
        for i in range(20, len(result)):
            if result.iloc[i]['regime'] == 'neutral':
                # Get recent window
                window = result.iloc[i-20:i]
                recent_high = window['close'].max()
                recent_low = window['close'].min()
                range_pct = (recent_high - recent_low) / recent_low
                
                if range_pct < 0.05:
                    result.iloc[i, result.columns.get_loc('regime')] = 'ranging'
        
        return result
    
    def backtest_strategy(
        self,
        strategy: str,
        asset: str,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Backtest a specific strategy on historical data
        
        Args:
            strategy: Strategy name
            asset: Asset being traded
            data: Historical data with regime labels
            params: Optional strategy parameters
            
        Returns:
            Dict: Backtest results
        """
        if data.empty:
            logger.warning(f"Empty data for {strategy} on {asset}, skipping backtest")
            return {
                "strategy": strategy,
                "asset": asset,
                "performance": {},
                "trades": [],
                "equity_curve": []
            }
        
        # This would be implemented with actual strategy backtesting logic
        # For this example, we'll simulate performance based on strategy capabilities
        
        # Get strategy capabilities
        capabilities = self.strategy_capabilities.get(strategy, {})
        
        # Results by regime
        results_by_regime = {}
        trades = []
        equity_curve = [100.0]  # Start with $100
        
        # Group data by regime
        regime_groups = data.groupby('regime')
        
        for regime, group in regime_groups:
            if len(group) < self.min_regime_samples:
                logger.info(f"Not enough samples for {regime} regime on {asset}, skipping")
                continue
            
            # Simulate performance based on capabilities
            capability = capabilities.get(regime, 0.5)
            
            # Add some randomness around the capability
            win_rate = max(0.4, min(0.9, capability + np.random.normal(0, 0.1)))
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss
            
            # Expected return
            expected_return = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Total return for this regime
            total_return = len(group) * expected_return * 0.1  # Assuming 10% of bars generate trades
            
            # Record results
            results_by_regime[regime] = {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expected_return": expected_return,
                "total_return": total_return,
                "samples": len(group)
            }
            
            # Generate some sample trades for this regime
            for i in range(int(len(group) * 0.1)):  # 10% of bars generate trades
                # Randomly determine if win or loss
                is_win = np.random.random() < win_rate
                
                # Calculate return
                trade_return = avg_win if is_win else -avg_loss
                
                # Add to equity curve
                equity_curve.append(equity_curve[-1] * (1 + trade_return))
                
                # Record trade
                trades.append({
                    "timestamp": group.index[min(i, len(group) - 1)].isoformat(),
                    "regime": regime,
                    "type": "win" if is_win else "loss",
                    "return": trade_return,
                    "equity": equity_curve[-1]
                })
        
        # Calculate overall performance
        if trades:
            win_trades = [t for t in trades if t["type"] == "win"]
            loss_trades = [t for t in trades if t["type"] == "loss"]
            
            overall_win_rate = len(win_trades) / len(trades) if trades else 0.0
            overall_return = (equity_curve[-1] / equity_curve[0]) - 1.0
            
            overall_performance = {
                "win_rate": overall_win_rate,
                "total_return": overall_return,
                "total_trades": len(trades),
                "by_regime": results_by_regime
            }
        else:
            overall_performance = {
                "win_rate": 0.0,
                "total_return": 0.0,
                "total_trades": 0,
                "by_regime": {}
            }
        
        return {
            "strategy": strategy,
            "asset": asset,
            "performance": overall_performance,
            "trades": trades,
            "equity_curve": equity_curve
        }
    
    def optimize_strategy_weights(
        self,
        backtest_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize strategy weights based on backtest results
        
        Args:
            backtest_results: Backtest results by asset and strategy
            
        Returns:
            Dict: Optimized strategy weights by regime
        """
        # Initialize weights for each regime
        optimized_weights = {regime: {} for regime in self.market_regimes}
        
        # Process each asset
        for asset, asset_results in backtest_results.items():
            # For each strategy
            for strategy, results in asset_results.items():
                # Get performance by regime
                regime_performance = results.get("performance", {}).get("by_regime", {})
                
                # Update weights for each regime
                for regime, perf in regime_performance.items():
                    if regime not in optimized_weights:
                        continue
                    
                    # Calculate score based on win rate and return
                    win_rate = perf.get("win_rate", 0.5)
                    total_return = perf.get("total_return", 0.0)
                    
                    # Score formula: balance win rate and returns
                    score = (win_rate * 0.7) + (min(1.0, max(0.0, total_return)) * 0.3)
                    
                    # Update regime weights
                    if strategy not in optimized_weights[regime]:
                        optimized_weights[regime][strategy] = 0.0
                    
                    optimized_weights[regime][strategy] += score
        
        # Normalize weights for each regime
        for regime in optimized_weights:
            total = sum(optimized_weights[regime].values())
            if total > 0:
                optimized_weights[regime] = {
                    strategy: weight / total
                    for strategy, weight in optimized_weights[regime].items()
                }
            else:
                # Fallback to equal weights
                strategies = list(self.strategy_capabilities.keys())
                weight = 1.0 / len(strategies)
                optimized_weights[regime] = {strategy: weight for strategy in strategies}
        
        return optimized_weights
    
    def train_strategy_ensemble(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Train the strategy ensemble
        
        Returns:
            Dict: Training results
        """
        logger.info("Training strategy ensemble")
        
        # Results by asset
        asset_results = {}
        
        # Process each asset
        for asset in self.assets:
            logger.info(f"Processing asset: {asset}")
            
            # Load historical data (use 1h as primary timeframe)
            data = self.load_historical_data(
                asset=asset,
                timeframe="1h",
                days=self.training_days + self.validation_days
            )
            
            if data.empty:
                logger.warning(f"No data available for {asset}, skipping")
                continue
            
            # Label market regimes
            data_with_regimes = self.label_regimes(data)
            
            # Backtest each strategy
            strategy_results = {}
            
            for strategy in self.strategies:
                logger.info(f"Backtesting {strategy} on {asset}")
                
                backtest_result = self.backtest_strategy(
                    strategy=strategy,
                    asset=asset,
                    data=data_with_regimes
                )
                
                strategy_results[strategy] = backtest_result
            
            # Store results for this asset
            asset_results[asset] = strategy_results
        
        # Optimize strategy weights
        if asset_results:
            optimized_weights = self.optimize_strategy_weights(asset_results)
            
            # Save weights to file
            weights_path = os.path.join(self.ensemble_output_dir, "strategy_ensemble_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(optimized_weights, f, indent=4)
            
            logger.info(f"Saved optimized strategy weights to {weights_path}")
        else:
            logger.warning("No results to optimize weights, using default weights")
            
            # Create default weights
            optimized_weights = {}
            for regime in self.market_regimes:
                strategy_weights = {}
                total_capability = 0.0
                
                for strategy, capabilities in self.strategy_capabilities.items():
                    if strategy in self.strategies:
                        capability = capabilities.get(regime, 0.5)
                        strategy_weights[strategy] = capability
                        total_capability += capability
                
                # Normalize weights
                if total_capability > 0:
                    optimized_weights[regime] = {
                        strategy: weight / total_capability
                        for strategy, weight in strategy_weights.items()
                    }
                else:
                    # Equal weights
                    weight = 1.0 / len(self.strategies)
                    optimized_weights[regime] = {strategy: weight for strategy in self.strategies}
            
            # Save default weights
            weights_path = os.path.join(self.ensemble_output_dir, "strategy_ensemble_weights.json")
            with open(weights_path, 'w') as f:
                json.dump(optimized_weights, f, indent=4)
            
            logger.info(f"Saved default strategy weights to {weights_path}")
        
        # Return results by asset and regime
        results_by_asset_regime = {}
        
        for asset, strategies in asset_results.items():
            results_by_asset_regime[asset] = {}
            
            for strategy, results in strategies.items():
                for regime, perf in results.get("performance", {}).get("by_regime", {}).items():
                    if regime not in results_by_asset_regime[asset]:
                        results_by_asset_regime[asset][regime] = {}
                    
                    results_by_asset_regime[asset][regime][strategy] = {
                        "win_rate": perf.get("win_rate", 0.0),
                        "total_return": perf.get("total_return", 0.0),
                        "samples": perf.get("samples", 0),
                        "weight": optimized_weights.get(regime, {}).get(strategy, 0.0)
                    }
        
        return results_by_asset_regime
    
    def visualize_ensemble_performance(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        save_path: Optional[str] = None
    ):
        """
        Visualize ensemble performance
        
        Args:
            results: Results from train_strategy_ensemble
            save_path: Path to save visualization (None for auto-generate)
        """
        # Auto-generate save path if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.ensemble_output_dir, f"ensemble_performance_{timestamp}.png")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set up the figure
        fig, axes = plt.subplots(len(self.assets), len(self.market_regimes), figsize=(20, 12))
        fig.suptitle("Strategy Ensemble Performance by Asset and Market Regime", fontsize=16)
        
        # If only one asset, convert axes to 2D array
        if len(self.assets) == 1:
            axes = np.array([axes])
        
        # Flatten if only one regime
        if len(self.market_regimes) == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each asset and regime
        for i, asset in enumerate(self.assets):
            for j, regime in enumerate(self.market_regimes):
                # Get data for this asset and regime
                if asset in results and regime in results[asset]:
                    regime_data = results[asset][regime]
                    
                    # Extract strategies, returns and weights
                    strategies = list(regime_data.keys())
                    returns = [regime_data[s].get("total_return", 0.0) for s in strategies]
                    weights = [regime_data[s].get("weight", 0.0) for s in strategies]
                    win_rates = [regime_data[s].get("win_rate", 0.0) for s in strategies]
                    
                    # Plot returns
                    ax = axes[i, j]
                    bars = ax.bar(strategies, returns, alpha=0.7)
                    
                    # Annotate bars with weights
                    for k, bar in enumerate(bars):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{weights[k]:.2f}",
                            ha='center', va='bottom', fontsize=8
                        )
                    
                    # Add win rates as a second plot
                    ax2 = ax.twinx()
                    ax2.plot(strategies, win_rates, 'ro-', alpha=0.6)
                    ax2.set_ylim(0, 1)
                    ax2.set_ylabel('Win Rate', color='r')
                    
                    # Set titles and labels
                    ax.set_title(f"{asset} - {regime}")
                    ax.set_ylabel('Return')
                    ax.set_ylim(bottom=min(0, min(returns) * 1.2))
                    
                    # Rotate x-axis labels
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                else:
                    # No data for this asset and regime
                    ax = axes[i, j]
                    ax.text(0.5, 0.5, f"No data for {asset} in {regime} regime",
                           ha='center', va='center')
                    ax.set_title(f"{asset} - {regime}")
                    ax.axis('off')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(save_path)
        logger.info(f"Saved ensemble performance visualization to {save_path}")
        
        # Close figure to free memory
        plt.close(fig)

def main():
    """Test the strategy ensemble trainer"""
    logging.basicConfig(level=logging.INFO)
    
    trainer = StrategyEnsembleTrainer(
        strategies=["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"],
        assets=["SOL/USD", "ETH/USD", "BTC/USD"],
        data_dir="historical_data",
        ensemble_output_dir="models/ensemble"
    )
    
    results = trainer.train_strategy_ensemble()
    trainer.visualize_ensemble_performance(results)

if __name__ == "__main__":
    main()