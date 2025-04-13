#!/usr/bin/env python3
"""
Enhanced Auto-Pruning System for ML Models

This module provides an improved auto-pruning system for ML models in the trading bot.
It intelligently evaluates model performance across different market conditions
and removes consistently underperforming models while preserving model diversity.

Features:
1. Market regime-aware pruning - evaluates models in their specialized regimes
2. Adaptive performance thresholds based on model type and market conditions
3. Performance consistency metrics over different time windows
4. Diversity preservation to maintain model variety
5. Dynamic pruning frequency based on market volatility
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

from backtest_ml_ensemble import MLEnsembleBacktester
from advanced_ensemble_model import DynamicWeightedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/auto_pruning.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

class EnhancedAutoPruner:
    """
    Enhanced auto-pruning system for ML models in the trading bot
    
    This class analyzes model performance across different market regimes
    and automatically prunes underperforming models while maintaining
    model diversity for robust predictions.
    """
    
    def __init__(self, 
                 ensemble_model,
                 min_models_per_type=1,
                 min_total_models=4,
                 base_performance_threshold=0.5,
                 consistency_weight=0.3,
                 accuracy_weight=0.7,
                 min_samples=10,
                 pruning_frequency_days=7,
                 backtest_window_days=30,
                 visualization_path='optimization_results/model_pruning'):
        """
        Initialize the enhanced auto-pruner
        
        Args:
            ensemble_model: The DynamicWeightedEnsemble model to analyze
            min_models_per_type: Minimum number of models to keep per type
            min_total_models: Minimum total number of models to keep
            base_performance_threshold: Base threshold for pruning decision
            consistency_weight: Weight given to consistency in performance scoring
            accuracy_weight: Weight given to overall accuracy in performance scoring
            min_samples: Minimum prediction samples required before pruning
            pruning_frequency_days: Days between pruning operations
            backtest_window_days: Days of historical data to use for testing
            visualization_path: Path to save visualization results
        """
        self.ensemble = ensemble_model
        self.min_models_per_type = min_models_per_type
        self.min_total_models = min_total_models
        self.base_threshold = base_performance_threshold
        self.consistency_weight = consistency_weight
        self.accuracy_weight = accuracy_weight
        self.min_samples = min_samples
        self.pruning_frequency = pruning_frequency_days
        self.backtest_window = backtest_window_days
        self.viz_path = visualization_path
        
        # Create visualization directory if it doesn't exist
        if not os.path.exists(self.viz_path):
            os.makedirs(self.viz_path)
        
        # Track last pruning date
        self.last_pruning_date_file = os.path.join(self.viz_path, 'last_pruning_date.txt')
        if os.path.exists(self.last_pruning_date_file):
            with open(self.last_pruning_date_file, 'r') as f:
                self.last_pruning_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
        else:
            self.last_pruning_date = datetime.now() - timedelta(days=self.pruning_frequency + 1)
    
    def _update_last_pruning_date(self):
        """Update the last pruning date file"""
        with open(self.last_pruning_date_file, 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d'))
        self.last_pruning_date = datetime.now()
    
    def should_prune(self):
        """
        Determine if pruning should be performed now
        
        Returns:
            bool: True if pruning should be performed
        """
        days_since_last_pruning = (datetime.now() - self.last_pruning_date).days
        return days_since_last_pruning >= self.pruning_frequency
    
    def _get_regime_specific_thresholds(self):
        """
        Get performance thresholds adjusted for different market regimes
        
        Returns:
            dict: Performance thresholds for each regime
        """
        # Base threshold is modified based on regime difficulty
        return {
            'normal_trending_up': self.base_threshold - 0.05,      # Easier regime
            'normal_trending_down': self.base_threshold,           # Standard difficulty
            'volatile_trending_up': self.base_threshold + 0.05,    # More difficult
            'volatile_trending_down': self.base_threshold + 0.1,   # Most difficult
        }
    
    def _get_model_type_specific_thresholds(self):
        """
        Get performance thresholds adjusted for different model types
        
        Returns:
            dict: Performance thresholds for each model type
        """
        # Different model architectures have different baseline expectations
        return {
            'tcn': self.base_threshold,
            'cnn': self.base_threshold - 0.02,
            'lstm': self.base_threshold,
            'gru': self.base_threshold,
            'bilstm': self.base_threshold + 0.02,  # Higher expectation
            'attention': self.base_threshold + 0.03,  # Higher expectation
            'transformer': self.base_threshold + 0.05,  # Higher expectation
            'hybrid': self.base_threshold + 0.05,  # Higher expectation
        }
    
    def _analyze_model_performance(self):
        """
        Analyze detailed performance metrics for all models
        
        Returns:
            dict: Performance metrics for each model
        """
        performance_metrics = {}
        
        # Get model status from ensemble
        model_info = self.ensemble.get_loaded_models()
        model_types = model_info.get('models', [])
        
        # Get performance history from ensemble
        prediction_history = self.ensemble.prediction_history
        
        # Get current regime properties
        current_regime = self.ensemble.current_regime if hasattr(self.ensemble, 'current_regime') else 'normal_trending_up'
        
        # Analyze each model type
        for model_type in model_types:
            # Skip if not enough prediction samples
            if model_type not in prediction_history or len(prediction_history[model_type]) < self.min_samples:
                continue
            
            # Extract performance metrics
            model_predictions = prediction_history[model_type]
            
            # Overall accuracy
            correct_predictions = sum(1 for pred in model_predictions if pred['outcome'] > 0)
            total_predictions = len(model_predictions)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Recent performance (last 10 predictions)
            recent_predictions = model_predictions[-10:] if len(model_predictions) >= 10 else model_predictions
            recent_correct = sum(1 for pred in recent_predictions if pred['outcome'] > 0)
            recent_accuracy = recent_correct / len(recent_predictions) if recent_predictions else 0
            
            # Performance by regime
            regime_performance = defaultdict(list)
            for pred in model_predictions:
                if 'market_regime' in pred:
                    regime_performance[pred['market_regime']].append(pred['outcome'])
            
            regime_accuracy = {}
            for regime, outcomes in regime_performance.items():
                regime_correct = sum(1 for outcome in outcomes if outcome > 0)
                regime_accuracy[regime] = regime_correct / len(outcomes) if outcomes else 0
            
            # Consistency calculation (standard deviation of rolling accuracy)
            rolling_window = 5
            if len(model_predictions) >= rolling_window:
                rolling_accuracies = []
                for i in range(len(model_predictions) - rolling_window + 1):
                    window = model_predictions[i:i+rolling_window]
                    window_correct = sum(1 for pred in window if pred['outcome'] > 0)
                    window_accuracy = window_correct / rolling_window
                    rolling_accuracies.append(window_accuracy)
                
                consistency = 1 - np.std(rolling_accuracies)  # Higher is better
            else:
                consistency = 0.5  # Default for insufficient data
            
            # Specialized regime performance
            specialty_regime = None
            specialty_accuracy = 0
            
            for regime, acc in regime_accuracy.items():
                if acc > specialty_accuracy:
                    specialty_accuracy = acc
                    specialty_regime = regime
            
            # Combine metrics for overall score
            # Weight recent performance and consistency more
            regime_adjustment = regime_accuracy.get(current_regime, 0) - 0.5
            overall_score = (
                self.accuracy_weight * accuracy + 
                self.consistency_weight * consistency +
                0.1 * recent_accuracy +
                0.1 * regime_adjustment
            )
            
            performance_metrics[model_type] = {
                'accuracy': accuracy,
                'recent_accuracy': recent_accuracy,
                'consistency': consistency,
                'regime_accuracy': regime_accuracy,
                'specialty_regime': specialty_regime,
                'specialty_accuracy': specialty_accuracy,
                'overall_score': overall_score,
                'samples': total_predictions
            }
        
        return performance_metrics
    
    def _identify_models_to_prune(self, performance_metrics):
        """
        Identify models to prune based on performance metrics
        
        Args:
            performance_metrics: Performance metrics for each model
            
        Returns:
            tuple: (models_to_prune, kept_models, pruning_reasons)
        """
        models_to_prune = []
        kept_models = []
        pruning_reasons = {}
        
        # Get regime and model type thresholds
        regime_thresholds = self._get_regime_specific_thresholds()
        model_type_thresholds = self._get_model_type_specific_thresholds()
        
        # Group models by type for diversity preservation
        models_by_type = defaultdict(list)
        
        for model_type, metrics in performance_metrics.items():
            model_base_type = model_type.split('_')[0] if '_' in model_type else model_type
            models_by_type[model_base_type].append((model_type, metrics))
        
        # For each model type, decide which to keep/prune
        for base_type, models in models_by_type.items():
            # Sort models by overall score (descending)
            sorted_models = sorted(models, key=lambda x: x[1]['overall_score'], reverse=True)
            
            # Always keep at least min_models_per_type of each type
            to_keep = sorted_models[:self.min_models_per_type]
            to_evaluate = sorted_models[self.min_models_per_type:]
            
            # Evaluate remaining models against thresholds
            threshold = model_type_thresholds.get(base_type, self.base_threshold)
            
            for model_type, metrics in to_evaluate:
                # Get applicable regime threshold if model has specialty
                specialty_regime = metrics.get('specialty_regime')
                if specialty_regime and specialty_regime in regime_thresholds:
                    # Adjust threshold based on specialty regime
                    adjusted_threshold = threshold - 0.05
                else:
                    adjusted_threshold = threshold
                
                # Decision to prune or keep
                if metrics['overall_score'] < adjusted_threshold:
                    models_to_prune.append(model_type)
                    pruning_reasons[model_type] = (
                        f"Score {metrics['overall_score']:.4f} below threshold {adjusted_threshold:.4f}, "
                        f"Accuracy: {metrics['accuracy']:.4f}, Consistency: {metrics['consistency']:.4f}"
                    )
                else:
                    kept_models.append(model_type)
        
        # Final check: ensure we're not pruning too many models
        if len(kept_models) < self.min_total_models:
            # Sort models to prune by score (ascending)
            prune_metrics = {model: performance_metrics[model] for model in models_to_prune}
            sorted_prune = sorted(prune_metrics.items(), key=lambda x: x[1]['overall_score'])
            
            # Move models back to kept list until minimum is reached
            models_to_restore = sorted_prune[:(self.min_total_models - len(kept_models))]
            for model_type, _ in models_to_restore:
                kept_models.append(model_type)
                models_to_prune.remove(model_type)
                pruning_reasons.pop(model_type)
        
        return models_to_prune, kept_models, pruning_reasons
    
    def visualize_performance(self, performance_metrics, pruned_models=None, kept_models=None):
        """
        Create visualizations of model performance
        
        Args:
            performance_metrics: Performance metrics dict
            pruned_models: List of pruned models
            kept_models: List of kept models
        """
        if not performance_metrics:
            return
        
        # Create plot directory if it doesn't exist
        if not os.path.exists(self.viz_path):
            os.makedirs(self.viz_path)
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall Performance by Model Type
        plt.subplot(2, 2, 1)
        model_types = list(performance_metrics.keys())
        accuracies = [metrics['accuracy'] * 100 for metrics in performance_metrics.values()]
        recent_accuracies = [metrics['recent_accuracy'] * 100 for metrics in performance_metrics.values()]
        
        x = np.arange(len(model_types))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Overall Accuracy')
        plt.bar(x + width/2, recent_accuracies, width, label='Recent Accuracy')
        
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_types, rotation=45)
        plt.legend()
        
        # Plot 2: Consistency vs Accuracy
        plt.subplot(2, 2, 2)
        accuracies = [metrics['accuracy'] for metrics in performance_metrics.values()]
        consistencies = [metrics['consistency'] for metrics in performance_metrics.values()]
        
        plt.scatter(accuracies, consistencies, s=100, alpha=0.7)
        
        # Highlight pruned and kept models
        if pruned_models and kept_models:
            for i, model in enumerate(model_types):
                color = 'red' if model in pruned_models else 'green'
                marker = 'x' if model in pruned_models else 'o'
                plt.scatter(accuracies[i], consistencies[i], color=color, marker=marker, s=150)
        
        plt.xlabel('Accuracy')
        plt.ylabel('Consistency')
        plt.title('Consistency vs Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add model labels
        for i, model in enumerate(model_types):
            plt.annotate(model, (accuracies[i], consistencies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot 3: Regime Performance Heatmap
        plt.subplot(2, 2, 3)
        
        # Collect regime accuracy data
        regimes = set()
        for metrics in performance_metrics.values():
            regimes.update(metrics.get('regime_accuracy', {}).keys())
        
        regimes = sorted(list(regimes))
        
        # Create data for heatmap
        heatmap_data = []
        for model in model_types:
            model_regime_acc = performance_metrics[model].get('regime_accuracy', {})
            row = [model_regime_acc.get(regime, 0) for regime in regimes]
            heatmap_data.append(row)
        
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xticks(np.arange(len(regimes)), regimes, rotation=45)
        plt.yticks(np.arange(len(model_types)), model_types)
        plt.title('Performance by Market Regime')
        
        # Plot 4: Overall Scores with Threshold
        plt.subplot(2, 2, 4)
        scores = [metrics['overall_score'] for metrics in performance_metrics.values()]
        
        # Get thresholds
        model_type_thresholds = self._get_model_type_specific_thresholds()
        thresholds = [model_type_thresholds.get(model.split('_')[0], self.base_threshold) 
                     for model in model_types]
        
        plt.bar(x, scores, width, label='Overall Score')
        
        # Add threshold markers
        for i, threshold in enumerate(thresholds):
            plt.plot([i-width/2, i+width/2], [threshold, threshold], 'r-', linewidth=2)
        
        plt.xlabel('Model Type')
        plt.ylabel('Score')
        plt.title('Overall Performance Scores vs Thresholds')
        plt.xticks(x, model_types, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_path, f'model_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
    
    def run_pruning(self, force=False):
        """
        Run the enhanced auto-pruning process
        
        Args:
            force: Force pruning even if it's not time yet
            
        Returns:
            tuple: (pruned_models, kept_models, pruning_details)
        """
        if not force and not self.should_prune():
            logger.info(f"Skipping pruning - next scheduled pruning in {self.pruning_frequency - (datetime.now() - self.last_pruning_date).days} days")
            return [], [], {}
        
        logger.info("Starting enhanced auto-pruning process...")
        
        # Step 1: Analyze model performance
        performance_metrics = self._analyze_model_performance()
        logger.info(f"Analyzed performance for {len(performance_metrics)} models")
        
        # Step 2: Identify models to prune
        models_to_prune, kept_models, pruning_reasons = self._identify_models_to_prune(performance_metrics)
        
        pruning_details = {
            'pruned_models': len(models_to_prune),
            'kept_models': len(kept_models),
            'reasons': pruning_reasons,
            'performance_metrics': performance_metrics
        }
        
        logger.info(f"Identified {len(models_to_prune)} models to prune and {len(kept_models)} models to keep")
        
        # Step 3: Visualize results
        self.visualize_performance(performance_metrics, models_to_prune, kept_models)
        
        # Step 4: Perform the pruning if we have models to prune
        if models_to_prune:
            # Tell the ensemble to deactivate these models
            pruned_result = self.ensemble.deactivate_models(models_to_prune)
            logger.info(f"Pruning complete - deactivated {len(pruned_result)} models")
            
            # Update last pruning date
            self._update_last_pruning_date()
            
            # Save pruning report
            report_file = os.path.join(self.viz_path, f'pruning_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                report = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pruned_models': models_to_prune,
                    'kept_models': kept_models,
                    'reasons': pruning_reasons,
                    'performance_summary': {
                        model: {
                            'accuracy': metrics['accuracy'],
                            'recent_accuracy': metrics['recent_accuracy'],
                            'overall_score': metrics['overall_score']
                        }
                        for model, metrics in performance_metrics.items()
                    }
                }
                json.dump(report, f, indent=2)
        else:
            logger.info("No models to prune at this time")
        
        return models_to_prune, kept_models, pruning_details

def main():
    """Main function to run a pruning cycle"""
    # Set up the trading pair and timeframe
    trading_pair = 'SOL/USD'
    timeframe = '1h'
    
    # Create the ensemble model
    ensemble = DynamicWeightedEnsemble(trading_pair, timeframe)
    
    # Create the auto-pruner
    pruner = EnhancedAutoPruner(
        ensemble_model=ensemble,
        min_models_per_type=1,
        min_total_models=4,
        base_performance_threshold=0.55,  # Baseline 55% accuracy required
        pruning_frequency_days=7,
        backtest_window_days=30
    )
    
    # Run pruning (force=True to run regardless of schedule)
    pruned, kept, details = pruner.run_pruning(force=True)
    
    # Print summary
    print(f"\nPruning Summary:")
    print(f"  Models pruned: {len(pruned)}")
    print(f"  Models kept: {len(kept)}")
    
    if pruned:
        print("\nPruned models:")
        for model in pruned:
            reason = details['reasons'].get(model, "No specific reason")
            print(f"  - {model}: {reason}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())