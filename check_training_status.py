#!/usr/bin/env python3
"""
Check Training Status

This script checks the status of model training, showing which models
have been trained for each pair and timeframe, and displaying summary
metrics for the trained models.

Usage:
    python check_training_status.py [--pair PAIR]

Example:
    python check_training_status.py
    python check_training_status.py --pair SOL/USD
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

# Constants
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MODEL_TYPES = ['entry', 'exit', 'cancel', 'sizing', 'ensemble']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Check training status')
    parser.add_argument('--pair', type=str, choices=SUPPORTED_PAIRS,
                       help='Trading pair to check (e.g., SOL/USD)')
    return parser.parse_args()


def check_data_files(pair: Optional[str] = None) -> Dict:
    """
    Check which data files exist
    
    Args:
        pair: Trading pair to check (optional, if None checks all pairs)
        
    Returns:
        Dictionary mapping pairs to lists of available timeframes
    """
    pairs_to_check = [pair] if pair else SUPPORTED_PAIRS
    data_files = {}
    
    for p in pairs_to_check:
        p_formatted = p.replace('/', '_')
        available_timeframes = []
        
        for tf in TIMEFRAMES:
            file_path = f"historical_data/{p_formatted}_{tf}.csv"
            if os.path.exists(file_path):
                available_timeframes.append(tf)
        
        if available_timeframes:
            data_files[p] = available_timeframes
    
    return data_files


def check_model_files(pair: Optional[str] = None) -> Dict:
    """
    Check which model files exist
    
    Args:
        pair: Trading pair to check (optional, if None checks all pairs)
        
    Returns:
        Nested dictionary mapping pairs to timeframes to lists of available model types
    """
    pairs_to_check = [pair] if pair else SUPPORTED_PAIRS
    model_files = {}
    
    for p in pairs_to_check:
        p_formatted = p.replace('/', '_')
        model_files[p] = {}
        
        for tf in TIMEFRAMES:
            available_models = []
            
            for model_type in MODEL_TYPES:
                info_path = f"ml_models/{p_formatted}_{tf}_{model_type}_info.json"
                model_path = f"ml_models/{p_formatted}_{tf}_{model_type}_model.h5"
                
                # For ensemble, we only need the info file
                if model_type == 'ensemble' and os.path.exists(info_path):
                    available_models.append(model_type)
                # For other models, we need both info and model files
                elif os.path.exists(info_path) and (model_type == 'ensemble' or os.path.exists(model_path)):
                    available_models.append(model_type)
            
            if available_models:
                model_files[p][tf] = available_models
    
    return model_files


def get_model_metrics(pair: str, timeframe: str, model_type: str) -> Optional[Dict]:
    """
    Get metrics for a specific model
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        model_type: Model type
        
    Returns:
        Dictionary with metrics or None if not available
    """
    pair_formatted = pair.replace('/', '_')
    info_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_info.json"
    
    if not os.path.exists(info_path):
        return None
    
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        if 'metrics' in info:
            return info['metrics']
        return {}
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None


def print_training_progress(data_files: Dict, model_files: Dict, pair: Optional[str] = None):
    """
    Print training progress for all pairs
    
    Args:
        data_files: Dictionary mapping pairs to lists of available timeframes
        model_files: Nested dictionary mapping pairs to timeframes to lists of available model types
        pair: Trading pair to check (optional, if None checks all pairs)
    """
    pairs_to_check = [pair] if pair else SUPPORTED_PAIRS
    
    print("\n=== Training Progress ===\n")
    
    total_data_files = sum(len(tfs) for p, tfs in data_files.items() if p in pairs_to_check)
    total_expected_data_files = len(pairs_to_check) * len(TIMEFRAMES)
    data_percentage = (total_data_files / total_expected_data_files) * 100 if total_expected_data_files > 0 else 0
    
    print(f"Data Collection: {total_data_files}/{total_expected_data_files} files ({data_percentage:.1f}%)")
    
    total_models = 0
    total_expected_models = 0
    
    for p in pairs_to_check:
        if p in model_files:
            for tf in model_files[p]:
                total_models += len(model_files[p][tf])
                
        if p in data_files:
            timeframes = data_files[p]
            for tf in timeframes:
                total_expected_models += len(MODEL_TYPES)
    
    models_percentage = (total_models / total_expected_models) * 100 if total_expected_models > 0 else 0
    print(f"Model Training: {total_models}/{total_expected_models} models ({models_percentage:.1f}%)")
    
    # Print detailed progress for each pair
    for p in pairs_to_check:
        print(f"\n{p}:")
        
        if p not in data_files:
            print("  No data files found")
            continue
        
        for tf in TIMEFRAMES:
            if tf in data_files[p]:
                model_count = 0
                model_status = []
                
                if p in model_files and tf in model_files[p]:
                    model_count = len(model_files[p][tf])
                    for model_type in MODEL_TYPES:
                        status = "✓" if model_type in model_files[p][tf] else " "
                        model_status.append(f"{status} {model_type}")
                
                status_str = ", ".join(model_status)
                print(f"  {tf}: {model_count}/{len(MODEL_TYPES)} models - {status_str}")
            else:
                print(f"  {tf}: No data available")


def print_model_metrics(model_files: Dict, pair: Optional[str] = None):
    """
    Print metrics for trained models
    
    Args:
        model_files: Nested dictionary mapping pairs to timeframes to lists of available model types
        pair: Trading pair to check (optional, if None checks all pairs)
    """
    pairs_to_check = [pair] if pair else list(model_files.keys())
    
    print("\n=== Model Metrics ===\n")
    
    for p in pairs_to_check:
        if p not in model_files or not model_files[p]:
            continue
        
        print(f"{p}:")
        
        for tf in model_files[p]:
            print(f"  {tf}:")
            
            # Check if ensemble model exists
            if 'ensemble' in model_files[p][tf]:
                metrics = get_model_metrics(p, tf, 'ensemble')
                
                if metrics:
                    # Print primary metrics
                    win_rate = metrics.get('win_rate', 0) * 100
                    profit_factor = metrics.get('profit_factor', 0)
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    
                    print(f"    Ensemble: Win Rate={win_rate:.1f}%, Profit Factor={profit_factor:.2f}, Sharpe={sharpe_ratio:.2f}")
            
            # Print individual model metrics
            for model_type in ['entry', 'exit', 'cancel', 'sizing']:
                if model_type in model_files[p][tf]:
                    metrics = get_model_metrics(p, tf, model_type)
                    
                    if metrics:
                        # Different metrics for different model types
                        if model_type in ['entry', 'exit', 'cancel']:
                            precision = metrics.get('precision', 0)
                            recall = metrics.get('recall', 0)
                            f1 = metrics.get('f1_score', 0)
                            
                            print(f"    {model_type.capitalize()}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
                        elif model_type == 'sizing':
                            mae = metrics.get('test_mae', 0)
                            r2 = metrics.get('r_squared', 0)
                            
                            print(f"    {model_type.capitalize()}: MAE={mae:.3f}, R²={r2:.3f}")


def print_best_models():
    """Print best performing models based on metrics"""
    print("\n=== Best Performing Models ===\n")
    
    best_models = {}
    
    # Find all model info files
    info_files = glob.glob("ml_models/*_ensemble_info.json")
    
    for info_file in info_files:
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            pair = info.get('pair')
            timeframe = info.get('timeframe')
            
            if 'metrics' not in info or not pair or not timeframe:
                continue
            
            metrics = info['metrics']
            
            # Extract key metrics
            win_rate = metrics.get('win_rate', 0)
            profit_factor = metrics.get('profit_factor', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            
            # Calculate combined score (custom formula)
            combined_score = win_rate * 0.4 + min(profit_factor / 5, 1) * 0.4 + min(sharpe_ratio / 3, 1) * 0.2
            
            # Store model info
            model_key = f"{pair}_{timeframe}"
            best_models[model_key] = {
                'pair': pair,
                'timeframe': timeframe,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'combined_score': combined_score
            }
        except Exception as e:
            print(f"Error processing {info_file}: {e}")
    
    # Sort models by combined score
    sorted_models = sorted(
        best_models.values(),
        key=lambda x: x['combined_score'],
        reverse=True
    )
    
    # Print top 5 models
    for i, model in enumerate(sorted_models[:5], 1):
        pair = model['pair']
        timeframe = model['timeframe']
        win_rate = model['win_rate'] * 100
        profit_factor = model['profit_factor']
        sharpe_ratio = model['sharpe_ratio']
        score = model['combined_score'] * 100
        
        print(f"{i}. {pair} ({timeframe}): Win Rate={win_rate:.1f}%, Profit Factor={profit_factor:.2f}, Sharpe={sharpe_ratio:.2f}, Score={score:.1f}%")


def get_training_progress_summary():
    """Get a summary of training progress"""
    data_files = check_data_files()
    model_files = check_model_files()
    
    total_pairs = len(SUPPORTED_PAIRS)
    pairs_with_data = len(data_files)
    pairs_with_models = sum(1 for p in model_files if model_files[p])
    
    total_data_files = sum(len(tfs) for p, tfs in data_files.items())
    total_expected_data_files = total_pairs * len(TIMEFRAMES)
    data_percentage = (total_data_files / total_expected_data_files) * 100 if total_expected_data_files > 0 else 0
    
    total_models = 0
    total_expected_models = 0
    
    for p in model_files:
        for tf in model_files[p]:
            total_models += len(model_files[p][tf])
            
    for p in data_files:
        timeframes = data_files[p]
        for tf in timeframes:
            total_expected_models += len(MODEL_TYPES)
    
    models_percentage = (total_models / total_expected_models) * 100 if total_expected_models > 0 else 0
    
    return {
        'total_pairs': total_pairs,
        'pairs_with_data': pairs_with_data,
        'pairs_with_models': pairs_with_models,
        'data_percentage': data_percentage,
        'models_percentage': models_percentage
    }


def main():
    """Main function"""
    args = parse_arguments()
    
    # Check data files
    data_files = check_data_files(args.pair)
    
    # Check model files
    model_files = check_model_files(args.pair)
    
    # Print training progress
    print_training_progress(data_files, model_files, args.pair)
    
    # Print model metrics
    if model_files:
        print_model_metrics(model_files, args.pair)
    
    # Print best models
    if not args.pair:
        print_best_models()
    
    # Print overall summary
    if not args.pair:
        summary = get_training_progress_summary()
        print(f"\nOverall Progress: {summary['data_percentage']:.1f}% of data collected, {summary['models_percentage']:.1f}% of models trained")
        print(f"{summary['pairs_with_data']}/{summary['total_pairs']} pairs have data, {summary['pairs_with_models']}/{summary['total_pairs']} pairs have models")


if __name__ == "__main__":
    main()