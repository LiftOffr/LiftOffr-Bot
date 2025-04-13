#!/usr/bin/env python3
"""
Comprehensive Backtesting Script

This script runs a comprehensive backtest for the trading bot using enhanced backtesting 
and ML features to achieve 90%+ accuracy.

Starting with a $20,000 portfolio, it tests all strategies with realistic
trade execution, fees, and slippage.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest_results/comprehensive/backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ComprehensiveBacktest")

# Constants
INITIAL_CAPITAL = 20000.0
TIMEFRAMES = ["1h", "4h", "1d"]
SYMBOLS = ["SOLUSD", "BTCUSD", "ETHUSD"]
PRIMARY_SYMBOL = "SOLUSD"
PRIMARY_TIMEFRAME = "1h"
RESULTS_DIR = "backtest_results/comprehensive"

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)

class ComprehensiveBacktester:
    """Comprehensive backtester with ML integration and realistic execution"""
    
    def __init__(self, initial_capital=INITIAL_CAPITAL, symbol=PRIMARY_SYMBOL, 
                 timeframe=PRIMARY_TIMEFRAME, start_date=None, end_date=None):
        """Initialize backtester"""
        self.initial_capital = initial_capital
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.current_capital = initial_capital
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        self.ml_models = {}
        
        # Performance metrics
        self.performance = {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'regime_performance': {}
        }
        
        # Load data
        self.data = self._load_data()
        
        if self.data is None or len(self.data) == 0:
            raise ValueError(f"No data available for {symbol} on {timeframe} timeframe")
        
        logger.info(f"Loaded {len(self.data)} candles for {symbol} on {timeframe} timeframe")
        
        # Prepare features
        self._prepare_features()
        
        logger.info(f"Prepared features for {symbol} on {timeframe} timeframe")
        
        # Detect market regimes
        self._detect_market_regimes()
        
        logger.info(f"Detected market regimes for {symbol} on {timeframe} timeframe")
    
    def _load_data(self):
        """Load historical data"""
        try:
            # Try to load data from local files first
            data_path = f"historical_data/{self.symbol}_{self.timeframe}.csv"
            
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Filter by date if specified
                if self.start_date:
                    df = df[df.index >= pd.to_datetime(self.start_date)]
                
                if self.end_date:
                    df = df[df.index <= pd.to_datetime(self.end_date)]
                
                return df
            
            # If unable to load from files, return None
            logger.error(f"Could not load data from {data_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _prepare_features(self):
        """Prepare technical indicators and features"""
        df = self.data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['ma7'] = df['close'].rolling(window=7).mean()
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma100'] = df['close'].rolling(window=100).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        
        # Exponential moving averages
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # Price difference from moving averages
        df['close_ma7_diff'] = df['close'] - df['ma7']
        df['close_ma21_diff'] = df['close'] - df['ma21']
        df['close_ma50_diff'] = df['close'] - df['ma50']
        df['close_ma100_diff'] = df['close'] - df['ma100']
        
        # Volatility indicators
        df['atr14'] = self._calculate_atr(df, period=14)
        df['atr14_pct'] = df['atr14'] / df['close']
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI
        df['rsi14'] = self._calculate_rsi(df, period=14)
        
        # MACD
        macd, signal, hist = self._calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # ADX
        df['adx'] = self._calculate_adx(df, period=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df)
        
        # Keltner Channels
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = self._calculate_keltner_channels(df)
        
        # Price position
        df['price_vs_bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['price_vs_kc'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
        # Trend indicators
        df['psar'] = self._calculate_parabolic_sar(df)
        df['above_psar'] = (df['close'] > df['psar']).astype(int)
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(14) - 1
        
        # Target variable for ML (next candle direction)
        df['next_candle_up'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        self.data = df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        df = df.copy()
        
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate directional movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = 0
        df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), 'plus_dm'] = df['up_move']
        
        df['minus_dm'] = 0
        df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), 'minus_dm'] = df['down_move']
        
        # Calculate smoothed values
        df['tr_ma'] = df['tr'].rolling(window=period).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr_ma'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr_ma'])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df['adx']
    
    def _calculate_bollinger_bands(self, df, period=20, multiplier=2):
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper = middle + multiplier * std
        lower = middle - multiplier * std
        
        return upper, middle, lower
    
    def _calculate_keltner_channels(self, df, period=20, multiplier=2):
        """Calculate Keltner Channels"""
        middle = df['close'].rolling(window=period).mean()
        atr = self._calculate_atr(df, period=period)
        
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        return upper, middle, lower
    
    def _calculate_parabolic_sar(self, df, af_start=0.02, af_increment=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        # Initialize values
        df = df.copy()
        df['psar'] = df['close'].copy()
        df['is_uptrend'] = True
        df['ep'] = df['high'].copy()  # Extreme point
        df['af'] = af_start  # Acceleration factor
        
        # Calculate PSAR
        for i in range(2, len(df)):
            # Get values for the calculation
            is_uptrend = df['is_uptrend'].iloc[i-1]
            ep_prev = df['ep'].iloc[i-1]
            psar_prev = df['psar'].iloc[i-1]
            af_prev = df['af'].iloc[i-1]
            
            # Calculate PSAR
            if is_uptrend:
                psar = psar_prev + af_prev * (ep_prev - psar_prev)
                # Adjust PSAR if needed
                if psar > df['low'].iloc[i-1]:
                    psar = df['low'].iloc[i-1]
                if psar > df['low'].iloc[i-2]:
                    psar = df['low'].iloc[i-2]
                
                # Check if trend reverses
                if psar > df['low'].iloc[i]:
                    is_uptrend = False
                    psar = ep_prev
                    ep = df['low'].iloc[i]
                    af = af_start
                else:
                    # Continue uptrend
                    is_uptrend = True
                    
                    # Update EP if needed
                    if df['high'].iloc[i] > ep_prev:
                        ep = df['high'].iloc[i]
                        af = min(af_prev + af_increment, af_max)
                    else:
                        ep = ep_prev
                        af = af_prev
            else:
                psar = psar_prev - af_prev * (psar_prev - ep_prev)
                
                # Adjust PSAR if needed
                if psar < df['high'].iloc[i-1]:
                    psar = df['high'].iloc[i-1]
                if psar < df['high'].iloc[i-2]:
                    psar = df['high'].iloc[i-2]
                
                # Check if trend reverses
                if psar < df['high'].iloc[i]:
                    is_uptrend = True
                    psar = ep_prev
                    ep = df['high'].iloc[i]
                    af = af_start
                else:
                    # Continue downtrend
                    is_uptrend = False
                    
                    # Update EP if needed
                    if df['low'].iloc[i] < ep_prev:
                        ep = df['low'].iloc[i]
                        af = min(af_prev + af_increment, af_max)
                    else:
                        ep = ep_prev
                        af = af_prev
            
            # Set values for next iteration
            df['psar'].iloc[i] = psar
            df['is_uptrend'].iloc[i] = is_uptrend
            df['ep'].iloc[i] = ep
            df['af'].iloc[i] = af
        
        return df['psar']
    
    def _detect_market_regimes(self):
        """Detect market regimes"""
        df = self.data.copy()
        
        # Calculate volatility and trend
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['trend'] = (df['ema50'] > df['ema100']).astype(int)
        
        # Define volatility threshold (75th percentile)
        volatility_threshold = df['volatility_20'].quantile(0.75)
        
        # Determine regimes
        conditions = [
            (df['volatility_20'] <= volatility_threshold) & (df['trend'] == 1),  # Normal trending up
            (df['volatility_20'] > volatility_threshold) & (df['trend'] == 1),   # Volatile trending up
            (df['volatility_20'] <= volatility_threshold) & (df['trend'] == 0),  # Normal trending down
            (df['volatility_20'] > volatility_threshold) & (df['trend'] == 0)    # Volatile trending down
        ]
        
        regimes = ['normal_trending_up', 'volatile_trending_up', 
                  'normal_trending_down', 'volatile_trending_down']
        
        df['market_regime'] = np.select(conditions, regimes, default='normal_trending_up')
        
        # Count regimes
        regime_counts = df['market_regime'].value_counts()
        logger.info(f"Market regimes: {regime_counts.to_dict()}")
        
        self.data = df
    
    def train_ml_model(self):
        """Train ML model for predictive signals"""
        from enhanced_tcn_model import EnhancedTCNModel
        
        logger.info("Training enhanced TCN model")
        
        # Prepare training data
        sequence_length = 60
        
        # Create sequences
        X, y = [], []
        features = self.data.drop(['open', 'high', 'low', 'close', 'volume', 'next_candle_up', 'market_regime'], axis=1).values
        targets = self.data['next_candle_up'].values
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # Create and train model
        model = EnhancedTCNModel(
            input_shape=X_train.shape[1:],
            output_size=1,
            filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16],
            dropout_rate=0.3,
            use_skip_connections=True,
            use_batch_norm=True,
            use_layer_norm=True,
            use_spatial_dropout=True,
            use_attention=True,
            use_transformer=True,
            use_channel_attention=True,
            l1_reg=0.0001,
            l2_reg=0.0001
        )
        
        # Train model
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            batch_size=32,
            epochs=50,
            model_prefix=f"{self.symbol}_{self.timeframe}"
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        logger.info(f"Model evaluation:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Auto-prune model
        from auto_prune_ml_models import ModelPruner
        
        logger.info("Auto-pruning ML model")
        
        pruner = ModelPruner()
        pruned_model = pruner.create_pruned_model(model.model, X_test=X_test, y_test=y_test)
        
        # Retrain pruned model
        pruned_model, _ = pruner.retrain_pruned_model(
            pruned_model, X_train, y_train, X_test, y_test
        )
        
        # Evaluate pruned model
        y_pruned_pred = pruned_model.predict(X_test)
        y_pruned_binary = (y_pruned_pred > 0.5).astype(int).flatten()
        
        pruned_accuracy = accuracy_score(y_test, y_pruned_binary)
        
        logger.info(f"Pruned model accuracy: {pruned_accuracy:.4f}")
        
        # Use the better model
        self.ml_models[f"{self.symbol}_{self.timeframe}"] = pruned_model if pruned_accuracy > accuracy else model.model
        
        # Save model
        model_path = os.path.join(RESULTS_DIR, "models", f"{self.symbol}_{self.timeframe}_enhanced_tcn.h5")
        
        if pruned_accuracy > accuracy:
            pruned_model.save(model_path)
        else:
            model.save(model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return pruned_accuracy if pruned_accuracy > accuracy else accuracy
    
    def optimize_strategy_parameters(self):
        """Optimize strategy parameters"""
        from optimize_strategy_parameters import StrategyOptimizer
        
        logger.info("Optimizing strategy parameters")
        
        try:
            # Import strategies
            from arima_strategy import ARIMAStrategy
            from integrated_strategy import IntegratedStrategy
            
            # Optimize ARIMA strategy
            logger.info("Optimizing ARIMA strategy parameters")
            
            arima_param_ranges = {
                "lookback_period": (20, 50, 5),
                "signal_threshold": (0.1, 0.3, 0.05),
                "atr_period": (10, 30, 5),
                "atr_multiplier": (1.5, 3.0, 0.5)
            }
            
            arima_optimizer = StrategyOptimizer(
                strategy_class=ARIMAStrategy,
                symbol=self.symbol,
                timeframe=self.timeframe,
                initial_capital=self.initial_capital,
                use_bayesian=True,
                n_calls=30
            )
            
            arima_results = arima_optimizer.optimize(arima_param_ranges, scoring="sharpe_ratio")
            
            # Optimize integrated strategy
            logger.info("Optimizing integrated strategy parameters")
            
            integrated_param_ranges = {
                "signal_smoothing": (2, 10, 1),
                "trend_strength_threshold": (0.1, 0.5, 0.1),
                "volatility_filter_threshold": (0.005, 0.02, 0.005),
                "min_adx_threshold": (15, 35, 5)
            }
            
            integrated_optimizer = StrategyOptimizer(
                strategy_class=IntegratedStrategy,
                symbol=self.symbol,
                timeframe=self.timeframe,
                initial_capital=self.initial_capital,
                use_bayesian=True,
                n_calls=30
            )
            
            integrated_results = integrated_optimizer.optimize(integrated_param_ranges, scoring="sharpe_ratio")
            
            # Save optimized parameters
            arima_params_path = os.path.join(RESULTS_DIR, f"{self.symbol}_{self.timeframe}_arima_params.json")
            integrated_params_path = os.path.join(RESULTS_DIR, f"{self.symbol}_{self.timeframe}_integrated_params.json")
            
            with open(arima_params_path, 'w') as f:
                json.dump(arima_results['best_params'], f, indent=2)
            
            with open(integrated_params_path, 'w') as f:
                json.dump(integrated_results['best_params'], f, indent=2)
            
            logger.info(f"Optimized ARIMA parameters: {arima_results['best_params']}")
            logger.info(f"Optimized integrated parameters: {integrated_results['best_params']}")
            
            return {
                'arima': arima_results['best_params'],
                'integrated': integrated_results['best_params']
            }
        
        except Exception as e:
            logger.error(f"Error optimizing strategy parameters: {e}")
            return None
    
    def run_backtest(self, optimized_params=None, use_ml=True):
        """Run comprehensive backtest with optimized parameters and ML"""
        logger.info(f"Running comprehensive backtest for {self.symbol} on {self.timeframe} timeframe")
        
        # Import strategies
        from arima_strategy import ARIMAStrategy
        from integrated_strategy import IntegratedStrategy
        
        # Create strategies with optimized parameters
        strategies = {}
        
        if optimized_params and 'arima' in optimized_params:
            arima_strategy = ARIMAStrategy(self.symbol, **optimized_params['arima'])
            logger.info(f"Created ARIMA strategy with optimized parameters: {optimized_params['arima']}")
        else:
            arima_strategy = ARIMAStrategy(self.symbol)
            logger.info("Created ARIMA strategy with default parameters")
        
        if optimized_params and 'integrated' in optimized_params:
            integrated_strategy = IntegratedStrategy(self.symbol, **optimized_params['integrated'])
            logger.info(f"Created integrated strategy with optimized parameters: {optimized_params['integrated']}")
        else:
            integrated_strategy = IntegratedStrategy(self.symbol)
            logger.info("Created integrated strategy with default parameters")
        
        strategies['arima'] = arima_strategy
        strategies['integrated'] = integrated_strategy
        
        # Set up ML-enhanced strategy if requested
        if use_ml and f"{self.symbol}_{self.timeframe}" in self.ml_models:
            try:
                from ml_enhanced_strategy import MLEnhancedStrategy
                
                ml_strategy = MLEnhancedStrategy(
                    base_strategy=arima_strategy,
                    trading_pair=self.symbol,
                    ml_model=self.ml_models[f"{self.symbol}_{self.timeframe}"],
                    ml_influence=0.7,
                    confidence_threshold=0.65
                )
                
                strategies['ml_enhanced'] = ml_strategy
                logger.info("Created ML-enhanced strategy")
            
            except Exception as e:
                logger.error(f"Error creating ML-enhanced strategy: {e}")
        
        # Initialize portfolio and tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}
        equity_curve = [portfolio_value]
        trades = []
        
        # Get backtesting data
        df = self.data.copy()
        
        # Performance by regime
        regime_performance = {
            'normal_trending_up': {'total_return': 0, 'trades': 0, 'wins': 0},
            'volatile_trending_up': {'total_return': 0, 'trades': 0, 'wins': 0},
            'normal_trending_down': {'total_return': 0, 'trades': 0, 'wins': 0},
            'volatile_trending_down': {'total_return': 0, 'trades': 0, 'wins': 0}
        }
        
        # Strategy signal strength settings
        strategy_strength = {
            'arima': 0.7,
            'integrated': 0.6,
            'ml_enhanced': 0.9
        }
        
        # Backtest parameters
        commission_rate = 0.0026  # 0.26% taker fee on Kraken
        slippage = 0.0005  # 0.05% slippage
        use_market_regimes = True
        
        # Run backtest
        logger.info(f"Starting backtest with {len(df)} data points")
        
        for i in range(100, len(df)):
            # Current timestamp and data
            timestamp = df.index[i]
            current_data = df.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_regime = current_data['market_regime'].iloc[-1]
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, position in positions.items():
                if position['type'] == 'long':
                    portfolio_value += position['quantity'] * current_price
                else:  # short
                    portfolio_value += position['value'] - (position['quantity'] * current_price)
            
            # Record portfolio value
            equity_curve.append(portfolio_value)
            
            # Generate signals from each strategy
            signals = {}
            
            for strategy_name, strategy in strategies.items():
                try:
                    # Generate signal
                    signal = strategy.generate_signal(current_data)
                    
                    # Adjust strength based on market regime if enabled
                    if use_market_regimes:
                        regime_modifier = {
                            'normal_trending_up': {'arima': 0.9, 'integrated': 0.7, 'ml_enhanced': 1.0},
                            'volatile_trending_up': {'arima': 0.7, 'integrated': 0.9, 'ml_enhanced': 0.9},
                            'normal_trending_down': {'arima': 0.8, 'integrated': 0.7, 'ml_enhanced': 0.9},
                            'volatile_trending_down': {'arima': 0.7, 'integrated': 0.9, 'ml_enhanced': 0.8}
                        }
                        
                        modifier = regime_modifier[current_regime].get(strategy_name, 1.0)
                        strength = strategy_strength[strategy_name] * modifier
                    else:
                        strength = strategy_strength[strategy_name]
                    
                    # Add ML confidence if available
                    if strategy_name == 'ml_enhanced' and hasattr(strategy, 'get_ml_confidence'):
                        confidence = strategy.get_ml_confidence(current_data)
                        strength = strength * (0.5 + confidence * 0.5)
                    
                    signals[strategy_name] = {
                        'signal': signal,
                        'strength': strength
                    }
                    
                except Exception as e:
                    logger.error(f"Error generating signal for {strategy_name}: {e}")
            
            # Resolve signals based on strength
            final_signal = None
            max_strength = 0
            
            for strategy_name, signal_info in signals.items():
                if signal_info['signal'] != 'neutral' and signal_info['strength'] > max_strength:
                    final_signal = signal_info['signal']
                    max_strength = signal_info['strength']
            
            # Execute trades based on final signal
            # For simplicity, we'll assume fixed position sizing
            position_size = 0.2  # 20% of portfolio
            max_positions = 1  # Maximum number of positions per symbol
            
            # Check if we already have a position
            current_position = positions.get(self.symbol, None)
            
            if final_signal == 'buy' and (current_position is None or current_position['type'] == 'short'):
                # Close existing short position if any
                if current_position is not None and current_position['type'] == 'short':
                    # Calculate profit/loss
                    entry_price = current_position['entry_price']
                    exit_price = current_price * (1 + slippage)  # Include slippage
                    quantity = current_position['quantity']
                    
                    # Calculate fees
                    entry_fee = entry_price * quantity * commission_rate
                    exit_fee = exit_price * quantity * commission_rate
                    
                    # Calculate profit/loss
                    pl = (entry_price - exit_price) * quantity - entry_fee - exit_fee
                    
                    # Update cash
                    cash += current_position['value'] + pl
                    
                    # Record trade
                    trade = {
                        'symbol': self.symbol,
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit_loss': pl,
                        'profit_loss_pct': pl / current_position['value'] * 100,
                        'market_regime': current_regime
                    }
                    
                    trades.append(trade)
                    
                    # Update regime performance
                    regime_performance[current_regime]['trades'] += 1
                    if pl > 0:
                        regime_performance[current_regime]['wins'] += 1
                    regime_performance[current_regime]['total_return'] += pl
                    
                    # Remove position
                    positions.pop(self.symbol)
                
                # Open new long position if we have capacity
                if len(positions) < max_positions:
                    # Calculate position size
                    position_value = portfolio_value * position_size
                    entry_price = current_price * (1 + slippage)  # Include slippage
                    quantity = position_value / entry_price
                    
                    # Calculate fee
                    fee = position_value * commission_rate
                    
                    # Update cash
                    cash -= position_value + fee
                    
                    # Record position
                    positions[self.symbol] = {
                        'type': 'long',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'value': position_value,
                        'entry_time': timestamp
                    }
            
            elif final_signal == 'sell' and (current_position is None or current_position['type'] == 'long'):
                # Close existing long position if any
                if current_position is not None and current_position['type'] == 'long':
                    # Calculate profit/loss
                    entry_price = current_position['entry_price']
                    exit_price = current_price * (1 - slippage)  # Include slippage
                    quantity = current_position['quantity']
                    
                    # Calculate fees
                    entry_fee = entry_price * quantity * commission_rate
                    exit_fee = exit_price * quantity * commission_rate
                    
                    # Calculate profit/loss
                    pl = (exit_price - entry_price) * quantity - entry_fee - exit_fee
                    
                    # Update cash
                    cash += exit_price * quantity - exit_fee
                    
                    # Record trade
                    trade = {
                        'symbol': self.symbol,
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'profit_loss': pl,
                        'profit_loss_pct': pl / current_position['value'] * 100,
                        'market_regime': current_regime
                    }
                    
                    trades.append(trade)
                    
                    # Update regime performance
                    regime_performance[current_regime]['trades'] += 1
                    if pl > 0:
                        regime_performance[current_regime]['wins'] += 1
                    regime_performance[current_regime]['total_return'] += pl
                    
                    # Remove position
                    positions.pop(self.symbol)
                
                # Open new short position if we have capacity
                if len(positions) < max_positions:
                    # Calculate position size
                    position_value = portfolio_value * position_size
                    entry_price = current_price * (1 - slippage)  # Include slippage
                    quantity = position_value / entry_price
                    
                    # Calculate fee
                    fee = position_value * commission_rate
                    
                    # Update cash
                    cash += position_value - fee
                    
                    # Record position
                    positions[self.symbol] = {
                        'type': 'short',
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'value': position_value,
                        'entry_time': timestamp
                    }
        
        # Close any remaining positions at the end
        final_price = df['close'].iloc[-1]
        
        for symbol, position in positions.items():
            if position['type'] == 'long':
                # Calculate profit/loss
                entry_price = position['entry_price']
                exit_price = final_price * (1 - slippage)  # Include slippage
                quantity = position['quantity']
                
                # Calculate fees
                entry_fee = entry_price * quantity * commission_rate
                exit_fee = exit_price * quantity * commission_rate
                
                # Calculate profit/loss
                pl = (exit_price - entry_price) * quantity - entry_fee - exit_fee
                
                # Update cash
                cash += exit_price * quantity - exit_fee
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[-1],
                    'type': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'profit_loss': pl,
                    'profit_loss_pct': pl / position['value'] * 100,
                    'market_regime': df['market_regime'].iloc[-1]
                }
                
                trades.append(trade)
                
                # Update regime performance
                current_regime = df['market_regime'].iloc[-1]
                regime_performance[current_regime]['trades'] += 1
                if pl > 0:
                    regime_performance[current_regime]['wins'] += 1
                regime_performance[current_regime]['total_return'] += pl
            
            else:  # short
                # Calculate profit/loss
                entry_price = position['entry_price']
                exit_price = final_price * (1 + slippage)  # Include slippage
                quantity = position['quantity']
                
                # Calculate fees
                entry_fee = entry_price * quantity * commission_rate
                exit_fee = exit_price * quantity * commission_rate
                
                # Calculate profit/loss
                pl = (entry_price - exit_price) * quantity - entry_fee - exit_fee
                
                # Update cash
                cash += position['value'] + pl
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': df.index[-1],
                    'type': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'profit_loss': pl,
                    'profit_loss_pct': pl / position['value'] * 100,
                    'market_regime': df['market_regime'].iloc[-1]
                }
                
                trades.append(trade)
                
                # Update regime performance
                current_regime = df['market_regime'].iloc[-1]
                regime_performance[current_regime]['trades'] += 1
                if pl > 0:
                    regime_performance[current_regime]['wins'] += 1
                regime_performance[current_regime]['total_return'] += pl
        
        # Calculate final portfolio value
        final_portfolio_value = cash
        
        # Calculate performance metrics
        total_return = final_portfolio_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        
        # Daily returns for Sharpe ratio
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Calculate drawdown
        equity_curve_series = pd.Series(equity_curve)
        running_max = equity_curve_series.cummax()
        drawdown = (equity_curve_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        wins = sum(1 for trade in trades if trade['profit_loss'] > 0)
        win_rate = wins / len(trades) if trades else 0
        
        # Calculate profit factor
        gross_profit = sum(trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0)
        gross_loss = sum(abs(trade['profit_loss']) for trade in trades if trade['profit_loss'] < 0)
        profit_factor = gross_profit / gross_loss if gross_loss else float('inf')
        
        # Calculate regime-specific metrics
        for regime in regime_performance:
            regime_data = regime_performance[regime]
            regime_data['win_rate'] = regime_data['wins'] / regime_data['trades'] if regime_data['trades'] > 0 else 0
            
            # Calculate profit factor for this regime
            regime_trades = [t for t in trades if t['market_regime'] == regime]
            regime_gross_profit = sum(t['profit_loss'] for t in regime_trades if t['profit_loss'] > 0)
            regime_gross_loss = sum(abs(t['profit_loss']) for t in regime_trades if t['profit_loss'] < 0)
            regime_data['profit_factor'] = regime_gross_profit / regime_gross_loss if regime_gross_loss else float('inf')
        
        # Store results
        self.performance = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': abs(max_drawdown),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'regime_performance': regime_performance
        }
        
        self.equity_curve = equity_curve
        self.trades = trades
        
        # Log results
        logger.info("Backtest results:")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        logger.info(f"Final portfolio value: ${final_portfolio_value:.2f}")
        logger.info(f"Total return: ${total_return:.2f} ({total_return_pct:.2f}%)")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {abs(max_drawdown):.2f}%")
        logger.info(f"Win rate: {win_rate*100:.2f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Total trades: {len(trades)}")
        
        # Save results
        self._save_results()
        
        return self.performance
    
    def _save_results(self):
        """Save backtest results"""
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Save performance metrics
        metrics_path = os.path.join(RESULTS_DIR, f"{self.symbol}_{self.timeframe}_metrics.json")
        
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            performance = {k: float(v) if isinstance(v, np.float64) else v 
                           for k, v in self.performance.items() if k != 'regime_performance'}
            
            # Handle regime performance separately
            regime_performance = {}
            for regime, metrics in self.performance['regime_performance'].items():
                regime_performance[regime] = {k: float(v) if isinstance(v, np.float64) else v 
                                           for k, v in metrics.items()}
            
            performance['regime_performance'] = regime_performance
            
            json.dump(performance, f, indent=2, default=str)
        
        # Save equity curve
        equity_path = os.path.join(RESULTS_DIR, f"{self.symbol}_{self.timeframe}_equity.csv")
        
        equity_df = pd.DataFrame({
            'portfolio_value': self.equity_curve
        })
        
        equity_df.to_csv(equity_path)
        
        # Save trades
        trades_path = os.path.join(RESULTS_DIR, f"{self.symbol}_{self.timeframe}_trades.csv")
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(trades_path, index=False)
        
        # Plot results
        self._plot_results()
        
        logger.info(f"Results saved to {RESULTS_DIR}")
    
    def _plot_results(self):
        """Plot backtest results"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve)
        plt.title(f"Equity Curve - {self.symbol} {self.timeframe}")
        plt.xlabel("Trade Days")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        
        # Plot drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        
        plt.subplot(2, 1, 2)
        plt.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        plt.title("Drawdown (%)")
        plt.xlabel("Trade Days")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{self.symbol}_{self.timeframe}_backtest.png"))
        plt.close()
        
        # Plot regime performance
        regimes = list(self.performance['regime_performance'].keys())
        win_rates = [self.performance['regime_performance'][r]['win_rate'] * 100 for r in regimes]
        profit_factors = [min(self.performance['regime_performance'][r]['profit_factor'], 5) for r in regimes]
        
        plt.figure(figsize=(12, 6))
        
        ax1 = plt.subplot(1, 1, 1)
        bars = ax1.bar(regimes, win_rates, color='blue', alpha=0.6)
        ax1.set_ylabel('Win Rate (%)', color='blue')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add profit factor line
        ax2 = ax1.twinx()
        ax2.plot(regimes, profit_factors, 'ro-', linewidth=2)
        ax2.set_ylabel('Profit Factor', color='red')
        ax2.set_ylim(0, 5.5)
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title("Performance by Market Regime")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{self.symbol}_{self.timeframe}_regime_performance.png"))
        plt.close()


def main():
    """Main function"""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Comprehensive Backtesting Script")
    
    parser.add_argument("--symbol", type=str, default=PRIMARY_SYMBOL,
                       help=f"Trading symbol (default: {PRIMARY_SYMBOL})")
    parser.add_argument("--timeframe", type=str, default=PRIMARY_TIMEFRAME,
                       help=f"Timeframe (default: {PRIMARY_TIMEFRAME})")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                       help=f"Initial capital (default: ${INITIAL_CAPITAL:.2f})")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-ml", action="store_true",
                       help="Skip ML model training")
    parser.add_argument("--skip-optimization", action="store_true",
                       help="Skip parameter optimization")
    
    args = parser.parse_args()
    
    # Run backtest
    backtester = ComprehensiveBacktester(
        initial_capital=args.capital,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Train ML model
    ml_accuracy = None
    if not args.skip_ml:
        try:
            ml_accuracy = backtester.train_ml_model()
            logger.info(f"ML model trained with accuracy: {ml_accuracy:.4f}")
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    # Optimize strategy parameters
    optimized_params = None
    if not args.skip_optimization:
        try:
            optimized_params = backtester.optimize_strategy_parameters()
            logger.info(f"Strategy parameters optimized")
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
    
    # Run backtest
    performance = backtester.run_backtest(
        optimized_params=optimized_params,
        use_ml=not args.skip_ml
    )
    
    # Print summary
    print("\n" + "="*80)
    print(f"COMPREHENSIVE BACKTEST RESULTS - {args.symbol} {args.timeframe}")
    print("="*80)
    print(f"Initial Capital: ${args.capital:.2f}")
    print(f"Final Portfolio Value: ${performance['final_portfolio_value']:.2f}")
    print(f"Total Return: ${performance['total_return']:.2f} ({performance['total_return_pct']:.2f}%)")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {performance['win_rate']*100:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Total Trades: {performance['total_trades']}")
    print("\nPerformance by Market Regime:")
    print("-"*80)
    print(f"{'Regime':<25} {'Trades':<10} {'Win Rate':<15} {'Profit Factor':<15}")
    print("-"*80)
    
    for regime, metrics in performance['regime_performance'].items():
        if metrics['trades'] > 0:
            print(f"{regime:<25} {metrics['trades']:<10} {metrics['win_rate']*100:<15.2f}% {metrics['profit_factor']:<15.2f}")
    
    print("\nML Model Accuracy:", f"{ml_accuracy:.2%}" if ml_accuracy else "N/A")
    print("-"*80)
    print(f"Comprehensive backtest completed. Results saved to {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()