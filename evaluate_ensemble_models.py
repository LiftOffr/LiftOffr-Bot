#!/usr/bin/env python3
"""
Evaluate Ensemble Models for Kraken Trading Bot

This script evaluates the performance of the trained ensemble models and generates
visualizations to help understand their predictions and performance.

It includes an auto-pruning feature that identifies and removes unprofitable
components of the trading system, focusing only on what works best.

Usage:
    python evaluate_ensemble_models.py [--symbol SYMBOL] [--timeframe TIMEFRAME] 
                                     [--auto-prune] [--save-best]
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import ensemble model
from advanced_ensemble_model import DynamicWeightedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "historical_data"
MODELS_DIR = "models"
RESULTS_DIR = "model_evaluation"
PRUNED_MODELS_DIR = "pruned_models"
LOW_TIMEFRAME_DATA_DIR = "historical_data/high_resolution"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRUNED_MODELS_DIR, exist_ok=True)
os.makedirs(LOW_TIMEFRAME_DATA_DIR, exist_ok=True)

def load_historical_data(symbol="SOLUSD", timeframe="1h", use_low_timeframe=False):
    """
    Load and prepare historical data for evaluation
    
    Args:
        symbol (str): Trading symbol (e.g., "SOLUSD")
        timeframe (str): Timeframe for data (e.g., "1h")
        
    Returns:
        pd.DataFrame: Processed historical data
    """
    file_path = os.path.join(DATA_DIR, f"{symbol}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create basic features
        df['return'] = df['close'].pct_change()
        df['target'] = df['return'].shift(-1)  # Target is next period's return
        df['direction'] = (df['target'] > 0).astype(int)  # 1 if price goes up, 0 if down
        
        # Calculate indicators (same as in training)
        # Price based indicators
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Price relative to moving averages
        df['price_sma5_ratio'] = df['close'] / df['sma5']
        df['price_sma10_ratio'] = df['close'] / df['sma10']
        df['price_sma20_ratio'] = df['close'] / df['sma20']
        df['price_ema9_ratio'] = df['close'] / df['ema9']
        df['price_ema21_ratio'] = df['close'] / df['ema21']
        
        # Volatility indicators
        df['volatility'] = df['close'].rolling(window=20).std() / df['close']
        df['atr'] = df['high'] - df['low']  # Simple ATR
        
        # Volume indicators
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.00001)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.00001)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_ensemble(ensemble, market_data, test_period=30):
    """
    Evaluate ensemble model on historical data
    
    Args:
        ensemble: Trained ensemble model
        market_data (pd.DataFrame): Historical market data
        test_period (int): Number of days to evaluate
        
    Returns:
        dict: Evaluation results
    """
    # Make sure market_data is sorted by timestamp
    market_data = market_data.sort_values('timestamp')
    
    # Get test data (last test_period days)
    test_start = market_data['timestamp'].max() - timedelta(days=test_period)
    test_data = market_data[market_data['timestamp'] >= test_start].copy()
    
    # Generate predictions
    predictions = []
    confidences = []
    actual_directions = []
    regimes = []
    
    # Track performance per model type
    model_predictions = {model_type: [] for model_type in ensemble.models.keys()}
    
    for i in range(len(test_data) - 24):  # Need at least sequence_length data points
        # Get data window
        window = test_data.iloc[i:i+24]
        
        # Get actual direction (up=1, down=0)
        actual = test_data.iloc[i+24]['direction']
        actual_directions.append(actual)
        
        # Detect market regime
        regime = ensemble.detect_market_regime(window)
        regimes.append(regime)
        
        # Generate prediction
        try:
            pred, conf, details = ensemble.predict(window)
            predictions.append(pred)
            confidences.append(conf)
            
            # Store individual model predictions if available
            if 'model_predictions' in details:
                for model_type, model_pred in details['model_predictions'].items():
                    if model_type in model_predictions:
                        model_predictions[model_type].append(model_pred)
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            predictions.append(0.5)  # Neutral prediction
            confidences.append(0.0)
    
    # Convert predictions to binary (>0.5 is up prediction)
    binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
    
    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(actual_directions, binary_predictions),
        'precision': precision_score(actual_directions, binary_predictions, zero_division=0),
        'recall': recall_score(actual_directions, binary_predictions, zero_division=0),
        'f1': f1_score(actual_directions, binary_predictions, zero_division=0),
        'avg_confidence': np.mean(confidences),
        'regime_distribution': {regime: regimes.count(regime) / len(regimes) for regime in set(regimes)}
    }
    
    # Calculate per-model performance
    model_performance = {}
    for model_type, preds in model_predictions.items():
        if len(preds) > 0:
            bin_preds = [1 if p > 0.5 else 0 for p in preds]
            model_performance[model_type] = {
                'accuracy': accuracy_score(actual_directions[:len(bin_preds)], bin_preds),
                'count': len(preds)
            }
    
    metrics['model_performance'] = model_performance
    
    # Store predictions and actual values for visualization
    metrics['predictions'] = predictions
    metrics['binary_predictions'] = binary_predictions
    metrics['confidences'] = confidences
    metrics['actual_directions'] = actual_directions
    metrics['timestamps'] = test_data['timestamp'].iloc[24:24+len(predictions)].tolist()
    metrics['prices'] = test_data['close'].iloc[24:24+len(predictions)].tolist()
    metrics['regimes'] = regimes
    
    return metrics

def plot_prediction_results(results, symbol, timeframe):
    """
    Generate plots to visualize prediction results
    
    Args:
        results (dict): Evaluation results
        symbol (str): Trading symbol
        timeframe (str): Timeframe
    """
    # Prepare data for plotting
    timestamps = [pd.to_datetime(ts) for ts in results['timestamps']]
    prices = results['prices']
    predictions = results['predictions']
    confidences = results['confidences']
    actual_directions = results['actual_directions']
    regimes = results['regimes']
    
    # Create plots directory
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Price and Predictions
    plt.figure(figsize=(14, 8))
    
    # Price subplot
    ax1 = plt.subplot(211)
    ax1.plot(timestamps, prices, label='Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{symbol} {timeframe} Price and Ensemble Predictions')
    
    # Highlight different regimes
    regime_changes = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            regime_changes.append(i)
    
    for i in range(len(regime_changes)-1):
        start = regime_changes[i]
        end = regime_changes[i+1]
        regime = regimes[start]
        alpha = 0.2
        if 'volatile' in regime:
            color = 'red'
        elif 'trending' in regime:
            color = 'green'
        else:
            color = 'blue'
        ax1.axvspan(timestamps[start], timestamps[end], alpha=alpha, color=color)
    
    # Predictions subplot
    ax2 = plt.subplot(212, sharex=ax1)
    
    # Plot prediction probability
    ax2.plot(timestamps, predictions, label='Prediction', color='purple')
    ax2.axhline(y=0.5, color='gray', linestyle='--')
    
    # Color confidence levels
    for i in range(len(timestamps)):
        if predictions[i] > 0.5 and actual_directions[i] == 1:
            color = 'green'  # Correct up prediction
        elif predictions[i] < 0.5 and actual_directions[i] == 0:
            color = 'blue'   # Correct down prediction
        else:
            color = 'red'    # Incorrect prediction
        
        ax2.scatter(timestamps[i], predictions[i], color=color, alpha=0.7, s=confidences[i]*100)
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Prediction (1=Up, 0=Down)')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{symbol}_{timeframe}_predictions.png"))
    
    # Plot 2: Model Performance Comparison
    if 'model_performance' in results:
        model_performance = results['model_performance']
        
        # Sort models by accuracy
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        model_names = [m[0] for m in sorted_models]
        accuracies = [m[1]['accuracy'] * 100 for m in sorted_models]
        
        plt.figure(figsize=(12, 6))
        plt.bar(model_names, accuracies)
        plt.axhline(y=results['accuracy'] * 100, color='red', linestyle='--', label=f'Ensemble ({results["accuracy"]:.2%})')
        plt.axhline(y=50, color='gray', linestyle='--', label='Random')
        
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)')
        plt.title(f'Model Performance Comparison - {symbol} {timeframe}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{symbol}_{timeframe}_model_performance.png"))
    
    # Plot 3: Confidence vs. Accuracy
    plt.figure(figsize=(10, 6))
    
    # Group predictions by confidence level
    confidence_bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        lower = confidence_bins[i]
        upper = confidence_bins[i+1]
        
        # Get predictions in this confidence range
        mask = [(c >= lower and c < upper) for c in confidences]
        bin_preds = [predictions[i] > 0.5 for i in range(len(predictions)) if mask[i]]
        bin_actual = [actual_directions[i] for i in range(len(actual_directions)) if mask[i]]
        
        if len(bin_preds) > 0:
            bin_acc = accuracy_score(bin_actual, bin_preds)
        else:
            bin_acc = 0
        
        bin_accuracies.append(bin_acc)
        bin_counts.append(len(bin_preds))
    
    # Plot accuracy by confidence level
    ax1 = plt.subplot(111)
    ax1.bar(confidence_bins[:-1], bin_accuracies, width=0.08, alpha=0.7, color='blue')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Confidence Level')
    
    # Plot prediction count by confidence level
    ax2 = ax1.twinx()
    ax2.plot(confidence_bins[:-1], bin_counts, 'ro-', alpha=0.7)
    ax2.set_ylabel('Number of Predictions', color='red')
    
    plt.title(f'Confidence vs. Accuracy - {symbol} {timeframe}')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{symbol}_{timeframe}_confidence_analysis.png"))
    
    logger.info(f"Plots saved to {plots_dir}")

def save_evaluation_results(results, symbol, timeframe):
    """
    Save evaluation results to file
    
    Args:
        results (dict): Evaluation results
        symbol (str): Trading symbol
        timeframe (str): Timeframe
    """
    # Create a copy of results with certain fields removed for JSON serialization
    results_copy = results.copy()
    results_copy.pop('timestamps', None)
    results_copy.pop('predictions', None)
    results_copy.pop('binary_predictions', None)
    results_copy.pop('confidences', None)
    results_copy.pop('actual_directions', None)
    results_copy.pop('prices', None)
    results_copy.pop('regimes', None)
    
    # Save to JSON file
    results_file = os.path.join(RESULTS_DIR, f"{symbol}_{timeframe}_evaluation.json")
    with open(results_file, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Print summary
    print(f"\nEvaluation Results for {symbol} {timeframe}:")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1 Score: {results['f1']:.2%}")
    print(f"Average Confidence: {results['avg_confidence']:.2f}")
    print("\nRegime Distribution:")
    for regime, pct in results['regime_distribution'].items():
        print(f"  {regime}: {pct:.2%}")
    
    print("\nModel Performance:")
    for model, perf in results.get('model_performance', {}).items():
        print(f"  {model}: {perf.get('accuracy', 0):.2%}")
    
def prune_unprofitable_components(ensemble, results):
    """
    Identify and prune unprofitable components from the ensemble model
    
    Args:
        ensemble: The ensemble model to prune
        results (dict): Evaluation results
        
    Returns:
        DynamicWeightedEnsemble: Pruned ensemble model
    """
    # Check if we have model performance data
    if 'model_performance' not in results:
        logger.warning("No model performance data available, skipping pruning")
        return ensemble
    
    model_performance = results['model_performance']
    
    # Identify models that perform worse than random (50% accuracy)
    unprofitable_models = []
    for model_type, perf in model_performance.items():
        if perf['accuracy'] < 0.5:  # Worse than random
            unprofitable_models.append(model_type)
            logger.info(f"Flagging {model_type} for removal (accuracy: {perf['accuracy']:.2%})")
    
    # Create a copy of the ensemble with the underperforming models removed
    pruned_ensemble = DynamicWeightedEnsemble(
        trading_pair=ensemble.trading_pair,
        timeframe=ensemble.timeframe
    )
    
    # Copy all properties except models
    for attr in dir(ensemble):
        if not attr.startswith('_') and attr != 'models' and not callable(getattr(ensemble, attr)):
            setattr(pruned_ensemble, attr, getattr(ensemble, attr))
    
    # Copy only profitable models
    pruned_ensemble.models = {}
    for model_type, model in ensemble.models.items():
        if model_type not in unprofitable_models:
            pruned_ensemble.models[model_type] = model
            logger.info(f"Keeping model {model_type}")
        else:
            logger.info(f"Pruning unprofitable model {model_type}")
    
    # Recalculate weights
    pruned_ensemble._initialize_weights()
    
    return pruned_ensemble


def save_pruned_model(pruned_ensemble, symbol, timeframe):
    """
    Save pruned model to disk
    
    Args:
        pruned_ensemble: Pruned ensemble model
        symbol (str): Trading symbol
        timeframe (str): Timeframe
    """
    # Create pruned models directory if it doesn't exist
    os.makedirs(PRUNED_MODELS_DIR, exist_ok=True)
    
    # Save model configuration
    config = {
        'trading_pair': pruned_ensemble.trading_pair,
        'timeframe': pruned_ensemble.timeframe,
        'model_types': list(pruned_ensemble.models.keys()),
        'weights': pruned_ensemble.weights,
        'pruned_at': datetime.now().isoformat()
    }
    
    config_file = os.path.join(PRUNED_MODELS_DIR, f"{symbol}_{timeframe}_pruned_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved pruned model configuration to {config_file}")
    
    # We could also save the actual model files, but that would require additional 
    # serialization/deserialization logic for TensorFlow models


def main():
    """Main function to evaluate ensemble models"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Models for Kraken Trading Bot")
    parser.add_argument("--symbol", default="SOLUSD", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--test-period", type=int, default=30, help="Test period in days")
    parser.add_argument("--auto-prune", action="store_true", help="Automatically prune unprofitable components")
    parser.add_argument("--save-best", action="store_true", help="Save best model configuration")
    parser.add_argument("--use-low-timeframe", action="store_true", help="Use low timeframe data if available")
    args = parser.parse_args()
    
    # Load historical data
    logger.info(f"Loading historical data for {args.symbol} {args.timeframe}")
    market_data = load_historical_data(args.symbol, args.timeframe, use_low_timeframe=args.use_low_timeframe)
    
    if market_data is None:
        logger.error("Failed to load historical data. Exiting.")
        return
    
    # Initialize ensemble model
    logger.info("Initializing ensemble model")
    trading_pair = f"{args.symbol[:3]}/{args.symbol[3:]}"  # Convert SOLUSD to SOL/USD
    ensemble = DynamicWeightedEnsemble(trading_pair=trading_pair, timeframe=args.timeframe)
    
    # Evaluate ensemble
    logger.info(f"Evaluating ensemble model over {args.test_period} days")
    results = evaluate_ensemble(ensemble, market_data, test_period=args.test_period)
    
    # Save results
    save_evaluation_results(results, args.symbol, args.timeframe)
    
    # Generate plots
    logger.info("Generating visualization plots")
    plot_prediction_results(results, args.symbol, args.timeframe)
    
    # Auto-prune if requested
    if args.auto_prune:
        logger.info("Auto-pruning unprofitable model components...")
        pruned_ensemble = prune_unprofitable_components(ensemble, results)
        
        # Check if pruning made a difference
        if len(pruned_ensemble.models) < len(ensemble.models):
            logger.info(f"Pruned {len(ensemble.models) - len(pruned_ensemble.models)} unprofitable models")
            
            # Re-evaluate with pruned ensemble
            logger.info("Re-evaluating with pruned model...")
            pruned_results = evaluate_ensemble(pruned_ensemble, market_data, test_period=args.test_period)
            
            # Compare results
            old_accuracy = results['accuracy']
            new_accuracy = pruned_results['accuracy']
            
            logger.info(f"Original accuracy: {old_accuracy:.2%}")
            logger.info(f"Pruned accuracy: {new_accuracy:.2%}")
            
            if new_accuracy >= old_accuracy:
                logger.info("✅ Pruning improved or maintained model accuracy!")
                
                # Save pruned model if requested
                if args.save_best:
                    logger.info("Saving pruned model configuration...")
                    save_pruned_model(pruned_ensemble, args.symbol, args.timeframe)
            else:
                logger.warning(f"⚠️ Pruning decreased accuracy by {(old_accuracy - new_accuracy) * 100:.2f}%")
        else:
            logger.info("No unprofitable models identified for pruning")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()