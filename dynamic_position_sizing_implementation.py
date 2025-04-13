"""
Dynamic Position Sizing Implementation

This module integrates the dynamic position sizing functionality with the ML-enhanced trading bot.
It connects the position sizer with the ML ensemble and trade execution components.
"""

import os
import logging
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt

from dynamic_position_sizing import DynamicPositionSizer
from advanced_ensemble_model import DynamicWeightedEnsemble

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class MLDrivenPositionManager:
    """
    Integrates ML predictions with dynamic position sizing
    
    This class serves as a bridge between ML ensemble predictions,
    market data, and the position sizer to determine optimal trade sizes.
    """
    
    def __init__(self, trading_pair="SOL/USD", timeframe="1h", config_file=None):
        """
        Initialize the position manager
        
        Args:
            trading_pair (str): Trading pair
            timeframe (str): Timeframe for analysis
            config_file (str, optional): Path to configuration file
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        
        # Initialize position sizer
        self.position_sizer = DynamicPositionSizer(config_file)
        
        # Initialize ML ensemble
        self.ensemble = DynamicWeightedEnsemble(trading_pair, timeframe)
        
        # Historical performance tracking
        self.trade_history = []
        
        # Create results directory
        os.makedirs('optimization_results', exist_ok=True)
        
        logger.info(f"ML-driven position manager initialized for {trading_pair} on {timeframe}")
    
    def calculate_position_size(self, 
                               available_capital: float,
                               market_data: pd.DataFrame,
                               signal_strength: float,
                               current_exposure: float = 0.0,
                               atr_value: Optional[float] = None,
                               trailing_stop_distance: Optional[float] = None) -> Tuple[float, float, Dict]:
        """
        Calculate optimal position size using ML ensemble and position sizer
        
        Args:
            available_capital: Available capital for trading
            market_data: Recent market data with indicators
            signal_strength: Signal strength from strategy (0.0 to 1.0)
            current_exposure: Current portfolio exposure (0.0 to 1.0)
            atr_value: Current ATR value if available
            trailing_stop_distance: Trailing stop distance if applicable
            
        Returns:
            tuple: (position_size_pct, position_size_value, metadata)
        """
        # Get ML prediction and confidence
        prediction, confidence, details = self.ensemble.predict(market_data)
        
        # Detect market regime
        market_regime = self.ensemble.detect_market_regime(market_data)
        
        # Calculate market volatility from recent price data
        if 'volatility' in market_data.columns:
            market_volatility = market_data['volatility'].iloc[-1]
        else:
            # Calculate simple volatility measure if not available
            recent_prices = market_data['close'].tail(20)
            market_volatility = recent_prices.pct_change().std()
        
        # Use ATR from market data if not provided
        if atr_value is None and 'atr' in market_data.columns:
            atr_value = market_data['atr'].iloc[-1]
        
        # Get position size from position sizer
        position_size_pct, position_size_value, reasoning = self.position_sizer.calculate_position_size(
            available_capital=available_capital,
            ml_confidence=confidence,
            signal_strength=signal_strength,
            market_volatility=market_volatility,
            market_regime=market_regime,
            atr_value=atr_value,
            current_exposure=current_exposure,
            trailing_stop_distance=trailing_stop_distance
        )
        
        # Add ML details to reasoning
        metadata = {
            'position_details': reasoning,
            'ml_details': details,
            'prediction': prediction,
            'confidence': confidence,
            'market_regime': market_regime,
            'signal_strength': signal_strength
        }
        
        logger.info(f"Position size for {self.trading_pair}: {position_size_pct:.2%} (${position_size_value:.2f})")
        logger.info(f"ML prediction: {prediction} with {confidence:.2%} confidence in {market_regime} regime")
        
        return position_size_pct, position_size_value, metadata
    
    def record_trade_performance(self, 
                                trade_result: float, 
                                position_size: float,
                                metadata: Dict):
        """
        Record trade performance for optimization feedback
        
        Args:
            trade_result: Profit/loss from trade
            position_size: Position size used
            metadata: Trade metadata including ML details
        """
        timestamp = datetime.now()
        
        trade_record = {
            'timestamp': timestamp,
            'trading_pair': self.trading_pair,
            'timeframe': self.timeframe,
            'position_size': position_size,
            'trade_result': trade_result,
            'risk_adjusted_return': trade_result / position_size if position_size > 0 else 0,
            'market_regime': metadata.get('market_regime', 'unknown'),
            'ml_confidence': metadata.get('confidence', 0),
            'signal_strength': metadata.get('signal_strength', 0),
            'prediction': metadata.get('prediction', 0)
        }
        
        self.trade_history.append(trade_record)
        
        # Update position sizer with performance data
        self.position_sizer.track_performance(
            timestamp=timestamp,
            position_size=position_size,
            trade_result=trade_result,
            parameters=metadata.get('position_details', {})
        )
        
        logger.info(f"Recorded trade performance: {trade_result:.2%} profit on {position_size:.2%} position")
        
        # Every 10 trades, run optimization and save history
        if len(self.trade_history) % 10 == 0:
            self.optimize_position_sizing()
            self.save_trade_history()
    
    def optimize_position_sizing(self):
        """
        Analyze trade history and optimize position sizing parameters
        """
        if len(self.trade_history) < 20:
            logger.info("Not enough trade history for optimization")
            return
        
        # Run position sizer optimization
        optimized_config = self.position_sizer.optimize_configuration()
        
        # Save optimized configuration
        self.position_sizer.save_configuration('config/position_sizing_config_optimized.json')
        
        logger.info("Position sizing parameters optimized")
    
    def save_trade_history(self):
        """Save trade history to disk"""
        try:
            df = pd.DataFrame(self.trade_history)
            filename = f"optimization_results/position_sizing_trades_{self.trading_pair}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Trade history saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def analyze_position_sizing_performance(self):
        """
        Generate performance analysis and visualization
        
        Returns:
            dict: Performance metrics
        """
        if len(self.trade_history) < 5:
            logger.info("Not enough trade history for analysis")
            return {}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.trade_history)
            
            # Calculate metrics
            win_rate = (df['trade_result'] > 0).mean()
            avg_win = df[df['trade_result'] > 0]['trade_result'].mean()
            avg_loss = abs(df[df['trade_result'] < 0]['trade_result'].mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Risk-adjusted metrics
            risk_adjusted_return = df['risk_adjusted_return'].mean()
            
            # Group by position size buckets
            df['position_size_bucket'] = pd.cut(df['position_size'], bins=5)
            size_performance = df.groupby('position_size_bucket')['trade_result'].mean()
            
            # Group by market regime
            regime_performance = df.groupby('market_regime')['trade_result'].mean()
            
            # Generate visualization
            self._generate_position_sizing_visualization(df)
            
            metrics = {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'risk_adjusted_return': risk_adjusted_return,
                'size_performance': size_performance.to_dict(),
                'regime_performance': regime_performance.to_dict()
            }
            
            logger.info(f"Position sizing performance analysis completed for {self.trading_pair}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing position sizing performance: {e}")
            return {}
    
    def _generate_position_sizing_visualization(self, df):
        """
        Generate visualization of position sizing performance
        
        Args:
            df (pd.DataFrame): Trade history DataFrame
        """
        try:
            # Create figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Position size vs. return
            axes[0, 0].scatter(df['position_size'], df['trade_result'], alpha=0.6)
            axes[0, 0].set_title('Position Size vs. Return')
            axes[0, 0].set_xlabel('Position Size (% of Capital)')
            axes[0, 0].set_ylabel('Trade Return (%)')
            
            # Plot 2: Market regime performance
            regime_perf = df.groupby('market_regime')['trade_result'].mean().sort_values()
            regime_perf.plot(kind='barh', ax=axes[0, 1])
            axes[0, 1].set_title('Average Return by Market Regime')
            axes[0, 1].set_xlabel('Average Return (%)')
            
            # Plot 3: ML confidence vs. return
            axes[1, 0].scatter(df['ml_confidence'], df['trade_result'], alpha=0.6)
            axes[1, 0].set_title('ML Confidence vs. Return')
            axes[1, 0].set_xlabel('ML Confidence')
            axes[1, 0].set_ylabel('Trade Return (%)')
            
            # Plot 4: Performance over time
            df_sorted = df.sort_values('timestamp')
            df_sorted['cumulative_return'] = (1 + df_sorted['trade_result']).cumprod() - 1
            df_sorted['cumulative_return'].plot(ax=axes[1, 1])
            axes[1, 1].set_title('Cumulative Performance')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('Cumulative Return (%)')
            
            plt.tight_layout()
            
            # Save figure
            filename = f"optimization_results/position_sizing_analysis_{self.trading_pair}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Position sizing visualization saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")

# Sample usage
if __name__ == "__main__":
    # Example usage
    position_manager = MLDrivenPositionManager()
    
    # Load some historical data
    try:
        data_file = f"historical_data/SOLUSD_1h.csv"
        if os.path.exists(data_file):
            market_data = pd.read_csv(data_file)
            
            # Calculate position size
            position_size_pct, position_value, metadata = position_manager.calculate_position_size(
                available_capital=10000,
                market_data=market_data.tail(50),
                signal_strength=0.85,
                current_exposure=0.2,
                atr_value=0.22,
                trailing_stop_distance=0.01
            )
            
            print(f"Calculated position size: {position_size_pct:.2%} (${position_value:.2f})")
            
            # Simulate some trades
            for i in range(10):
                trade_result = np.random.normal(0.01, 0.03)  # Random return
                position_manager.record_trade_performance(
                    trade_result=trade_result,
                    position_size=position_size_pct,
                    metadata=metadata
                )
                
            # Analyze performance
            metrics = position_manager.analyze_position_sizing_performance()
            print("Performance metrics:", json.dumps(metrics, indent=2))
        
        else:
            print(f"No historical data found at {data_file}")
            
    except Exception as e:
        print(f"Error in sample usage: {e}")