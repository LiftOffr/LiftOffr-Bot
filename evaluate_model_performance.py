#!/usr/bin/env python3
"""
Evaluate Model Performance

This script evaluates the performance of trained ML models by:
1. Loading the trained models
2. Running a backtest with the models
3. Calculating performance metrics (profit/loss, win rate, etc.)
4. Visualizing the results

Usage:
    python evaluate_model_performance.py --pair PAIR --timeframe TIMEFRAME --start-capital CAPITAL
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIR = "SOLUSD"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_CAPITAL = 20000
MODEL_DIRS = {
    "tcn": "models/tcn",
    "lstm": "models/lstm",
    "gru": "models/gru",
    "transformer": "models/transformer",
    "cnn": "models/cnn",
    "bilstm": "models/bilstm",
    "attention": "models/attention",
    "hybrid": "models/hybrid",
    "ensemble": "models/ensemble"
}
TRAINING_DATA_DIR = "training_data"
RESULTS_DIR = "evaluation_results"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate ML model performance")
    parser.add_argument("--pair", type=str, default=DEFAULT_PAIR,
                        help=f"Trading pair (default: {DEFAULT_PAIR})")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help=f"Timeframe (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--start-capital", type=float, default=DEFAULT_CAPITAL,
                        help=f"Starting capital (default: ${DEFAULT_CAPITAL})")
    parser.add_argument("--leverage", type=float, default=5.0,
                        help="Leverage to use (default: 5.0)")
    parser.add_argument("--position-size-pct", type=float, default=20.0,
                        help="Position size as percentage of capital (default: 20%)")
    parser.add_argument("--fee-pct", type=float, default=0.25,
                        help="Fee percentage (default: 0.25%)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate performance visualization")
    parser.add_argument("--output-csv", action="store_true",
                        help="Output trade results to CSV")
    return parser.parse_args()

def load_dataset(pair, timeframe):
    """Load dataset for evaluation"""
    dataset_path = os.path.join(TRAINING_DATA_DIR, f"{pair}_{timeframe}_enhanced.csv")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def load_model(model_type, pair, timeframe):
    """Load trained model"""
    if model_type == "ensemble":
        # Load ensemble model configuration
        weights_path = os.path.join(MODEL_DIRS["ensemble"], f"{pair}_weights.json")
        if not os.path.exists(weights_path):
            logger.error(f"Ensemble weights not found: {weights_path}")
            return None
        
        try:
            with open(weights_path, 'r') as f:
                weights = json.load(f)
            
            # Load individual models
            models = {}
            for model_name in weights:
                if model_name in MODEL_DIRS:
                    model_path = os.path.join(MODEL_DIRS[model_name], f"{pair}_{model_name}.h5")
                    if os.path.exists(model_path):
                        models[model_name] = keras.models.load_model(model_path)
            
            if not models:
                logger.error("No valid models found for ensemble")
                return None
            
            logger.info(f"Loaded ensemble with {len(models)} models")
            return {"weights": weights, "models": models}
        
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            return None
    else:
        # Load individual model
        model_path = os.path.join(MODEL_DIRS[model_type], f"{pair}_{model_type}.h5")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        
        try:
            model = keras.models.load_model(model_path)
            logger.info(f"Loaded {model_type} model")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

def prepare_features(df):
    """Prepare features for prediction"""
    # Remove timestamp and other non-numeric columns
    columns_to_drop = ["timestamp"]
    
    # Add string/categorical columns to drop list
    for col in df.columns:
        if df[col].dtype == 'object':
            columns_to_drop.append(col)
    
    # Drop non-numeric columns
    df_features = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Remove target columns
    feature_cols = [col for col in df_features.columns if not col.startswith("target_")]
    
    # Handle missing values
    df_features = df_features[feature_cols].fillna(0)
    
    return df_features

def make_predictions(model, features, model_type="single"):
    """Make predictions using the model"""
    try:
        if model_type == "ensemble":
            # Ensemble prediction
            weighted_preds = np.zeros(len(features))
            weight_sum = 0
            
            for model_name, weight in model["weights"].items():
                if model_name in model["models"]:
                    # Reshape features for the model
                    X = features.values.reshape((features.shape[0], 1, features.shape[1]))
                    # Make prediction
                    preds = model["models"][model_name].predict(X, verbose=0).flatten()
                    weighted_preds += preds * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                weighted_preds /= weight_sum
                
            return weighted_preds
        else:
            # Single model prediction
            X = features.values.reshape((features.shape[0], 1, features.shape[1]))
            return model.predict(X, verbose=0).flatten()
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None

def run_backtest(df, predictions, args):
    """Run a simple backtest using the model predictions"""
    capital = args.start_capital
    position_size = capital * (args.position_size_pct / 100.0)
    leverage = args.leverage
    fee_pct = args.fee_pct / 100.0
    
    # Initialize results
    trades = []
    equity_curve = []
    in_position = False
    position_type = None
    entry_price = 0
    entry_index = 0
    
    # Iterate through the data
    for i in range(1, len(df)):
        # Skip the first few samples as they may have NaN values
        if i < 10:
            equity_curve.append(capital)
            continue
        
        current_price = df.iloc[i]["close"]
        current_time = df.iloc[i]["timestamp"] if "timestamp" in df.columns else i
        prediction = predictions[i]
        
        # Determine signal
        signal = None
        if prediction > 0.6:  # Bullish signal
            signal = "buy"
        elif prediction < 0.4:  # Bearish signal
            signal = "sell"
        
        # Execute trades
        if not in_position and signal:
            # Enter position
            position_type = signal
            entry_price = current_price
            entry_index = i
            in_position = True
            
            # Apply fees
            capital -= position_size * fee_pct
            
            logger.info(f"[{current_time}] Opening {position_type} position at ${entry_price:.2f}")
        
        elif in_position:
            # Check for exit
            exit_signal = False
            
            # Exit on opposing signal
            if (position_type == "buy" and predictions[i] < 0.4) or \
               (position_type == "sell" and predictions[i] > 0.6):
                exit_signal = True
            
            # Exit after holding for a certain period
            if i - entry_index > 8:  # Hold for ~8 hours in 1h timeframe
                exit_signal = True
            
            if exit_signal:
                # Calculate profit/loss
                if position_type == "buy":
                    pnl = (current_price - entry_price) / entry_price
                else:  # sell
                    pnl = (entry_price - current_price) / entry_price
                
                # Apply leverage
                pnl *= leverage
                
                # Calculate absolute profit/loss
                abs_pnl = position_size * pnl
                
                # Apply fees
                abs_pnl -= position_size * fee_pct
                
                # Update capital
                capital += abs_pnl
                
                # Record trade
                trades.append({
                    "entry_time": df.iloc[entry_index]["timestamp"] if "timestamp" in df.columns else entry_index,
                    "exit_time": current_time,
                    "position_type": position_type,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl * 100,
                    "pnl_abs": abs_pnl,
                    "capital_after": capital
                })
                
                logger.info(f"[{current_time}] Closing {position_type} position at ${current_price:.2f}, P&L: ${abs_pnl:.2f} ({pnl*100:.2f}%)")
                
                # Reset position
                in_position = False
                position_type = None
                
                # Update position size for next trade
                position_size = capital * (args.position_size_pct / 100.0)
        
        # Update equity curve
        equity_curve.append(capital)
    
    # Close any open position at the end
    if in_position:
        current_price = df.iloc[-1]["close"]
        current_time = df.iloc[-1]["timestamp"] if "timestamp" in df.columns else len(df)-1
        
        # Calculate profit/loss
        if position_type == "buy":
            pnl = (current_price - entry_price) / entry_price
        else:  # sell
            pnl = (entry_price - current_price) / entry_price
        
        # Apply leverage
        pnl *= leverage
        
        # Calculate absolute profit/loss
        abs_pnl = position_size * pnl
        
        # Apply fees
        abs_pnl -= position_size * fee_pct
        
        # Update capital
        capital += abs_pnl
        
        # Record trade
        trades.append({
            "entry_time": df.iloc[entry_index]["timestamp"] if "timestamp" in df.columns else entry_index,
            "exit_time": current_time,
            "position_type": position_type,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl_pct": pnl * 100,
            "pnl_abs": abs_pnl,
            "capital_after": capital
        })
        
        logger.info(f"[End] Closing {position_type} position at ${current_price:.2f}, P&L: ${abs_pnl:.2f} ({pnl*100:.2f}%)")
    
    return trades, equity_curve

def calculate_metrics(trades, start_capital, equity_curve):
    """Calculate performance metrics"""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "net_profit": 0,
            "net_profit_pct": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "avg_trade_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best_trade": 0,
            "worst_trade": 0
        }
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df["pnl_abs"] > 0]
    losing_trades = trades_df[trades_df["pnl_abs"] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    gross_profit = winning_trades["pnl_abs"].sum() if not winning_trades.empty else 0
    gross_loss = abs(losing_trades["pnl_abs"].sum()) if not losing_trades.empty else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    net_profit = trades_df["pnl_abs"].sum()
    net_profit_pct = (net_profit / start_capital) * 100
    
    avg_trade_pnl = trades_df["pnl_abs"].mean()
    avg_win = winning_trades["pnl_abs"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["pnl_abs"].mean() if not losing_trades.empty else 0
    
    best_trade = trades_df["pnl_abs"].max() if not trades_df.empty else 0
    worst_trade = trades_df["pnl_abs"].min() if not trades_df.empty else 0
    
    # Calculate drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown_pct = abs(drawdown.min())
    
    # Calculate Sharpe ratio (simplified, assuming risk-free rate = 0)
    daily_returns = pd.Series([eq/prev_eq - 1 for eq, prev_eq in zip(equity_curve[1:], equity_curve[:-1])])
    
    sharpe_ratio = 0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)  # Annualized
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate * 100,
        "profit_factor": profit_factor,
        "net_profit": net_profit,
        "net_profit_pct": net_profit_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "avg_trade_pnl": avg_trade_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade
    }

def visualize_performance(df, equity_curve, trades, metrics, pair, model_type):
    """Visualize performance results"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price and equity curve
    prices = df["close"].values
    
    # Primary axis for price
    ax1.plot(prices, label="Price", color="gray", alpha=0.5)
    ax1.set_ylabel("Price ($)", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")
    ax1.set_title(f"{pair} Performance - {model_type.upper()} Model")
    
    # Secondary axis for equity curve
    ax3 = ax1.twinx()
    ax3.plot(equity_curve, label="Equity", color="green", linewidth=2)
    ax3.set_ylabel("Equity ($)", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    
    # Add trade markers
    for trade in trades:
        entry_idx = int(trade["entry_time"]) if isinstance(trade["entry_time"], (int, float)) else 0
        exit_idx = int(trade["exit_time"]) if isinstance(trade["exit_time"], (int, float)) else 0
        
        if trade["position_type"] == "buy":
            color = "green" if trade["pnl_abs"] > 0 else "red"
            ax1.plot(entry_idx, prices[entry_idx], "^", color=color, markersize=8)
            ax1.plot(exit_idx, prices[exit_idx], "v", color=color, markersize=8)
        else:  # sell
            color = "green" if trade["pnl_abs"] > 0 else "red"
            ax1.plot(entry_idx, prices[entry_idx], "v", color=color, markersize=8)
            ax1.plot(exit_idx, prices[exit_idx], "^", color=color, markersize=8)
    
    # Plot equity curve
    ax2.plot(equity_curve, label="Equity", color="green", linewidth=2)
    ax2.set_ylabel("Equity ($)")
    ax2.set_xlabel("Time")
    
    # Add metrics as text
    metrics_text = (
        f"Total Trades: {metrics['total_trades']}\n"
        f"Win Rate: {metrics['win_rate']:.2f}%\n"
        f"Net Profit: ${metrics['net_profit']:.2f} ({metrics['net_profit_pct']:.2f}%)\n"
        f"Profit Factor: {metrics['profit_factor']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
    )
    
    plt.figtext(0.15, 0.01, metrics_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.8))
    
    # Adjust layout and save figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    file_path = os.path.join(RESULTS_DIR, f"{pair}_{model_type}_performance.png")
    plt.savefig(file_path)
    logger.info(f"Performance visualization saved to {file_path}")
    
    plt.close()

def output_trades_csv(trades, pair, model_type):
    """Output trades to CSV file"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    file_path = os.path.join(RESULTS_DIR, f"{pair}_{model_type}_trades.csv")
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(file_path, index=False)
    
    logger.info(f"Trades saved to {file_path}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Load dataset
    df = load_dataset(args.pair, args.timeframe)
    if df is None:
        return 1
    
    # Find available models for this pair
    available_models = []
    for model_type in MODEL_DIRS:
        if model_type == "ensemble":
            weights_path = os.path.join(MODEL_DIRS[model_type], f"{args.pair}_weights.json")
            if os.path.exists(weights_path):
                available_models.append(model_type)
        else:
            model_path = os.path.join(MODEL_DIRS[model_type], f"{args.pair}_{model_type}.h5")
            if os.path.exists(model_path):
                available_models.append(model_type)
    
    if not available_models:
        logger.error(f"No trained models found for {args.pair}")
        return 1
    
    logger.info(f"Found {len(available_models)} models for {args.pair}: {', '.join(available_models)}")
    
    # Prepare features
    features = prepare_features(df)
    if features is None:
        return 1
    
    # Process each model
    results = {}
    for model_type in available_models:
        # Load model
        model = load_model(model_type, args.pair, args.timeframe)
        if model is None:
            continue
        
        # Make predictions
        model_class = "ensemble" if model_type == "ensemble" else "single"
        predictions = make_predictions(model, features, model_class)
        if predictions is None:
            continue
        
        # Run backtest
        trades, equity_curve = run_backtest(df, predictions, args)
        
        # Calculate metrics
        metrics = calculate_metrics(trades, args.start_capital, equity_curve)
        results[model_type] = metrics
        
        # Log results
        logger.info(f"=== {model_type.upper()} Model Results ===")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logger.info(f"Net Profit: ${metrics['net_profit']:.2f} ({metrics['net_profit_pct']:.2f}%)")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Average Trade P&L: ${metrics['avg_trade_pnl']:.2f}")
        logger.info("===========================")
        
        # Visualize results if requested
        if args.visualize:
            visualize_performance(df, equity_curve, trades, metrics, args.pair, model_type)
        
        # Output trades to CSV if requested
        if args.output_csv:
            output_trades_csv(trades, args.pair, model_type)
    
    # Summarize results
    logger.info("=== Summary ===")
    for model_type, metrics in results.items():
        logger.info(f"{model_type.upper()}: Win Rate: {metrics['win_rate']:.2f}%, Profit: ${metrics['net_profit']:.2f} ({metrics['net_profit_pct']:.2f}%)")
    
    # Find best model by profit
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['net_profit'])
        logger.info(f"Best model by profit: {best_model[0].upper()} with ${best_model[1]['net_profit']:.2f} ({best_model[1]['net_profit_pct']:.2f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())