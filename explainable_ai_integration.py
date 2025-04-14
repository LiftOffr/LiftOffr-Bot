#!/usr/bin/env python3
"""
Explainable AI Integration for Kraken Trading Bot

This module implements techniques to make the ML models' decision-making process
transparent and interpretable, helping to build trust in the trading system and
enabling better understanding of market conditions that drive predictions.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
EXPLANATIONS_DIR = "trading_explanations"
os.makedirs(EXPLANATIONS_DIR, exist_ok=True)
MAX_EXPLANATIONS_STORED = 100  # Maximum number of explanations to store per asset
FEATURE_GROUPS = {
    "price_action": ["close", "open", "high", "low", "volume"],
    "trend": ["ema9", "ema21", "ema50", "ema100", "ema200"],
    "momentum": ["rsi", "macd", "macd_signal", "stoch_k", "stoch_d"],
    "volatility": ["atr", "bollinger_width", "keltner_width"],
    "support_resistance": ["support_level", "resistance_level", "distance_to_support", "distance_to_resistance"],
    "volume_profile": ["volume", "obv", "volume_ma", "volume_momentum"],
    "cross_asset": []  # Will be populated dynamically for each asset pair relationship
}

class ExplainableAI:
    """
    Implements explainable AI techniques to provide insights into the
    model's decision-making process for trading signals.
    """
    def __init__(self, model_type, asset, model=None):
        """
        Initialize the explainable AI system
        
        Args:
            model_type (str): Type of model (e.g., "tcn", "lstm", "attention_gru", "tft")
            asset (str): Trading pair/asset (e.g., "SOL/USD")
            model (tf.keras.Model, optional): The model to explain
        """
        self.model_type = model_type
        self.asset = asset
        self.model = model
        self.explainer = None
        self.explanation_history = []
        self.baseline_values = {}
        
        # Create asset-specific directory
        self.asset_dir = os.path.join(EXPLANATIONS_DIR, asset.replace('/', '_'))
        os.makedirs(self.asset_dir, exist_ok=True)
        
        # Load previous explanations if available
        self._load_previous_explanations()
        
        logger.info(f"Initialized explainable AI for {model_type} model on {asset}")
    
    def set_model(self, model):
        """
        Set the model to explain
        
        Args:
            model (tf.keras.Model): The model to explain
        """
        self.model = model
        # Reset explainer since model changed
        self.explainer = None
    
    def _load_previous_explanations(self):
        """Load previous explanations if available"""
        explanations_path = os.path.join(
            self.asset_dir, 
            f"{self.model_type}_explanations.json"
        )
        
        if os.path.exists(explanations_path):
            try:
                with open(explanations_path, 'r') as f:
                    data = json.load(f)
                
                self.explanation_history = data.get('explanations', [])
                self.baseline_values = data.get('baseline_values', {})
                
                logger.info(f"Loaded {len(self.explanation_history)} previous explanations for {self.model_type} on {self.asset}")
            except Exception as e:
                logger.error(f"Error loading previous explanations: {e}")
    
    def _save_explanations(self):
        """Save explanations for future reference"""
        explanations_path = os.path.join(
            self.asset_dir, 
            f"{self.model_type}_explanations.json"
        )
        
        # Trim explanation history if too long
        if len(self.explanation_history) > MAX_EXPLANATIONS_STORED:
            # Keep most recent explanations
            self.explanation_history = self.explanation_history[-MAX_EXPLANATIONS_STORED:]
        
        data = {
            'model_type': self.model_type,
            'asset': self.asset,
            'explanations': self.explanation_history,
            'baseline_values': self.baseline_values,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(explanations_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.explanation_history)} explanations for {self.model_type} on {self.asset}")
    
    def initialize_shap_explainer(self, X_background):
        """
        Initialize SHAP explainer with background data
        
        Args:
            X_background (np.ndarray): Background data for SHAP explainer
        """
        if self.model is None:
            logger.error("No model set for explanation")
            return
        
        try:
            # For sequential models, use DeepExplainer
            self.explainer = shap.DeepExplainer(self.model, X_background)
            logger.info(f"Initialized SHAP DeepExplainer for {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            try:
                # Fall back to Kernel explainer if DeepExplainer fails
                prediction_function = lambda x: self.model.predict(x)
                self.explainer = shap.KernelExplainer(prediction_function, X_background)
                logger.info(f"Initialized SHAP KernelExplainer for {self.model_type}")
            except Exception as e2:
                logger.error(f"Error initializing fallback SHAP explainer: {e2}")
    
    def explain_prediction(self, X, feature_names=None, additional_info=None):
        """
        Explain a prediction using SHAP values
        
        Args:
            X (np.ndarray): Input data to explain
            feature_names (list, optional): Names of features
            additional_info (dict, optional): Additional information about the prediction
            
        Returns:
            dict: Explanation of the prediction
        """
        if self.model is None:
            logger.error("No model set for explanation")
            return None
        
        if self.explainer is None:
            logger.info("Initializing explainer with current sample as background")
            self.initialize_shap_explainer(X[:10] if len(X) > 10 else X)
        
        prediction = self.model.predict(X)
        
        # Extract scalar prediction value for simple cases
        if isinstance(prediction, dict):  # For TFT model
            pred_value = prediction.get('predictions', np.array([0]))[0, -1, 0]
        else:
            pred_value = prediction[0, 0] if prediction.ndim > 1 else prediction[0]
        
        # Default explanation with basic info
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'prediction': float(pred_value),
            'model_type': self.model_type,
            'feature_importance': {},
            'signal': 'BUY' if pred_value > 0 else 'SELL' if pred_value < 0 else 'NEUTRAL',
            'confidence': min(1.0, abs(float(pred_value)) * 2),  # Scale to 0-1 range
            'additional_info': additional_info or {}
        }
        
        try:
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Process SHAP values based on model type and output
            if isinstance(shap_values, list):
                # For classification models with multiple outputs
                shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
            else:
                # For regression models
                shap_vals = shap_values
            
            # Average SHAP values across samples for batch inputs
            if shap_vals.ndim > 2:
                # For sequence models, focus on final timestep
                avg_shap = np.mean(shap_vals[:, -1, :], axis=0)
            else:
                avg_shap = np.mean(shap_vals, axis=0)
            
            # Map to feature names if provided
            if feature_names and len(feature_names) == len(avg_shap):
                feature_importance = {feature_names[i]: float(avg_shap[i]) for i in range(len(feature_names))}
            else:
                feature_importance = {f"feature_{i}": float(avg_shap[i]) for i in range(len(avg_shap))}
            
            # Add SHAP values to explanation
            explanation['feature_importance'] = feature_importance
            
            # Group features by category if possible
            if feature_names:
                grouped_importance = self._group_feature_importance(feature_importance, feature_names)
                explanation['grouped_importance'] = grouped_importance
            
            # Add explanation to history
            self.explanation_history.append(explanation)
            self._save_explanations()
            
            logger.info(f"Generated explanation for {self.model_type} prediction: {pred_value:.4f}")
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            
            # Fall back to permutation importance if SHAP fails
            try:
                if feature_names and len(X[0]) == len(feature_names):
                    # Recompute prediction for comparison
                    baseline_pred = self.model.predict(X)
                    
                    # Calculate simple feature importance by zeroing out each feature
                    permutation_importance = {}
                    for i, feature in enumerate(feature_names):
                        # Create a copy of the input with the feature zeroed out
                        X_permuted = X.copy()
                        X_permuted[:, i] = 0
                        
                        # Predict with permuted feature
                        permuted_pred = self.model.predict(X_permuted)
                        
                        # Calculate impact as difference in prediction
                        if isinstance(baseline_pred, dict) and isinstance(permuted_pred, dict):
                            impact = np.mean(np.abs(baseline_pred['predictions'] - permuted_pred['predictions']))
                        else:
                            impact = np.mean(np.abs(baseline_pred - permuted_pred))
                        
                        permutation_importance[feature] = float(impact)
                    
                    # Normalize importance
                    max_impact = max(permutation_importance.values()) if permutation_importance else 1.0
                    if max_impact > 0:
                        permutation_importance = {k: v / max_impact for k, v in permutation_importance.items()}
                    
                    explanation['feature_importance'] = permutation_importance
                    explanation['importance_method'] = 'permutation'
                    
                    # Group features by category if possible
                    grouped_importance = self._group_feature_importance(permutation_importance, feature_names)
                    explanation['grouped_importance'] = grouped_importance
                    
                    # Add explanation to history
                    self.explanation_history.append(explanation)
                    self._save_explanations()
                    
                    logger.info(f"Generated fallback permutation explanation for {self.model_type}")
            except Exception as e2:
                logger.error(f"Error generating fallback explanation: {e2}")
        
        return explanation
    
    def _group_feature_importance(self, feature_importance, feature_names):
        """
        Group feature importance by feature categories
        
        Args:
            feature_importance (dict): Feature importance values
            feature_names (list): List of feature names
            
        Returns:
            dict: Grouped importance by category
        """
        grouped_importance = {}
        
        # Initialize all groups with zero importance
        for group in FEATURE_GROUPS:
            grouped_importance[group] = 0.0
        
        # Add "other" group for uncategorized features
        grouped_importance["other"] = 0.0
        
        # Match features to groups
        for feature, importance in feature_importance.items():
            assigned = False
            for group, group_features in FEATURE_GROUPS.items():
                # Check if feature belongs to this group
                if any(group_feature in feature.lower() for group_feature in group_features):
                    grouped_importance[group] += abs(importance)
                    assigned = True
                    break
            
            # If not assigned to any group, add to "other"
            if not assigned:
                grouped_importance["other"] += abs(importance)
        
        # Normalize grouped importance
        total_importance = sum(grouped_importance.values())
        if total_importance > 0:
            grouped_importance = {k: v / total_importance for k, v in grouped_importance.items()}
        
        return grouped_importance
    
    def generate_explanation_text(self, explanation, max_features=5):
        """
        Generate human-readable explanation text
        
        Args:
            explanation (dict): Explanation data
            max_features (int): Maximum number of features to include in explanation
            
        Returns:
            str: Human-readable explanation
        """
        if not explanation:
            return "No explanation available."
        
        prediction = explanation.get('prediction', 0)
        signal = explanation.get('signal', 'NEUTRAL')
        confidence = explanation.get('confidence', 0) * 100  # Convert to percentage
        
        # Start with the signal and confidence
        text = [f"Signal: {signal} (Confidence: {confidence:.2f}%)"]
        
        # Add feature importance explanation
        feature_importance = explanation.get('feature_importance', {})
        if feature_importance:
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:max_features]
            
            text.append("\nTop influencing factors:")
            for feature, importance in sorted_features:
                # Format feature name for readability
                readable_name = feature.replace('_', ' ').title()
                
                # Determine direction of influence
                if importance > 0:
                    direction = "increased"
                else:
                    direction = "decreased"
                
                text.append(f"- {readable_name}: {direction} the signal ({abs(importance):.3f})")
        
        # Add grouped importance if available
        grouped_importance = explanation.get('grouped_importance', {})
        if grouped_importance:
            # Sort groups by importance
            sorted_groups = sorted(
                grouped_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Top 3 groups
            
            text.append("\nFactor categories:")
            for group, importance in sorted_groups:
                if importance > 0.05:  # Only include significant groups
                    readable_group = group.replace('_', ' ').title()
                    text.append(f"- {readable_group}: {importance*100:.1f}% of signal")
        
        # Add market context if available
        additional_info = explanation.get('additional_info', {})
        market_regime = additional_info.get('market_regime')
        if market_regime:
            text.append(f"\nMarket Regime: {market_regime.title()}")
        
        recent_trend = additional_info.get('recent_trend')
        if recent_trend:
            text.append(f"Recent Trend: {recent_trend}")
        
        # Add any other relevant information
        signal_strength = additional_info.get('signal_strength')
        if signal_strength is not None:
            text.append(f"Signal Strength: {signal_strength:.2f}")
        
        return "\n".join(text)
    
    def compare_with_baseline(self, explanation, baseline_window=30):
        """
        Compare the current explanation with baseline to highlight deviations
        
        Args:
            explanation (dict): Current explanation
            baseline_window (int): Number of recent explanations to use as baseline
            
        Returns:
            dict: Comparison results highlighting deviations
        """
        if not explanation or not self.explanation_history:
            return {}
        
        # Get baseline explanations (excluding current one)
        baseline_explanations = self.explanation_history[:-1] if self.explanation_history[-1] == explanation else self.explanation_history
        
        # Use recent explanations for baseline
        baseline_explanations = baseline_explanations[-baseline_window:] if len(baseline_explanations) > baseline_window else baseline_explanations
        
        if not baseline_explanations:
            return {}
        
        # Calculate baseline values
        baseline = {
            'prediction': np.mean([e.get('prediction', 0) for e in baseline_explanations]),
            'confidence': np.mean([e.get('confidence', 0) for e in baseline_explanations]),
            'feature_importance': {}
        }
        
        # Calculate baseline feature importance
        all_features = set()
        for e in baseline_explanations:
            all_features.update(e.get('feature_importance', {}).keys())
        
        for feature in all_features:
            values = [e.get('feature_importance', {}).get(feature, 0) for e in baseline_explanations if feature in e.get('feature_importance', {})]
            if values:
                baseline['feature_importance'][feature] = np.mean(values)
        
        # Calculate deviations from baseline
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'prediction_change': explanation.get('prediction', 0) - baseline['prediction'],
            'confidence_change': explanation.get('confidence', 0) - baseline['confidence'],
            'feature_deviations': {}
        }
        
        # Calculate feature importance deviations
        for feature, importance in explanation.get('feature_importance', {}).items():
            baseline_importance = baseline['feature_importance'].get(feature, 0)
            deviation = importance - baseline_importance
            if abs(deviation) > 0.05:  # Only include significant deviations
                comparison['feature_deviations'][feature] = deviation
        
        # Calculate grouped importance deviations
        comparison['grouped_deviations'] = {}
        for group, importance in explanation.get('grouped_importance', {}).items():
            # Calculate baseline grouped importance
            baseline_group_importance = 0
            count = 0
            for e in baseline_explanations:
                if group in e.get('grouped_importance', {}):
                    baseline_group_importance += e['grouped_importance'][group]
                    count += 1
            
            if count > 0:
                baseline_group_importance /= count
                deviation = importance - baseline_group_importance
                if abs(deviation) > 0.05:  # Only include significant deviations
                    comparison['grouped_deviations'][group] = deviation
        
        # Update the baseline values
        self.baseline_values.update({
            'prediction': baseline['prediction'],
            'confidence': baseline['confidence'],
            'last_updated': datetime.now().isoformat()
        })
        
        return comparison
    
    def generate_comparison_text(self, comparison, max_deviations=3):
        """
        Generate human-readable comparison text
        
        Args:
            comparison (dict): Comparison data
            max_deviations (int): Maximum number of deviations to include
            
        Returns:
            str: Human-readable comparison
        """
        if not comparison:
            return "No comparison available."
        
        prediction_change = comparison.get('prediction_change', 0)
        confidence_change = comparison.get('confidence_change', 0) * 100  # Convert to percentage
        
        # Start with overall changes
        text = ["Compared to recent signals:"]
        
        if abs(prediction_change) > 0.05:
            direction = "stronger" if prediction_change > 0 else "weaker"
            text.append(f"- Signal is {direction} than usual (by {abs(prediction_change):.3f})")
        
        if abs(confidence_change) > 5:
            direction = "higher" if confidence_change > 0 else "lower"
            text.append(f"- Confidence is {direction} than usual (by {abs(confidence_change):.1f}%)")
        
        # Add notable feature deviations
        feature_deviations = comparison.get('feature_deviations', {})
        if feature_deviations:
            # Sort deviations by absolute magnitude
            sorted_deviations = sorted(
                feature_deviations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:max_deviations]
            
            text.append("\nUnusual factors:")
            for feature, deviation in sorted_deviations:
                # Format feature name for readability
                readable_name = feature.replace('_', ' ').title()
                
                # Determine direction of influence change
                if deviation > 0:
                    direction = "more important"
                else:
                    direction = "less important"
                
                text.append(f"- {readable_name}: {direction} than usual ({abs(deviation):.3f})")
        
        # Add grouped deviations if available
        grouped_deviations = comparison.get('grouped_deviations', {})
        if grouped_deviations:
            # Sort groups by absolute deviation
            sorted_groups = sorted(
                grouped_deviations.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:max_deviations]
            
            text.append("\nUnusual factor categories:")
            for group, deviation in sorted_groups:
                readable_group = group.replace('_', ' ').title()
                
                # Determine direction of influence change
                if deviation > 0:
                    direction = "more influential"
                else:
                    direction = "less influential"
                
                text.append(f"- {readable_group}: {direction} than usual ({abs(deviation)*100:.1f}%)")
        
        return "\n".join(text)
    
    def plot_feature_importance(self, explanation, output_path=None, max_features=10):
        """
        Plot feature importance from explanation
        
        Args:
            explanation (dict): Explanation data
            output_path (str, optional): Path to save the plot
            max_features (int): Maximum number of features to include
            
        Returns:
            str: Path to saved plot or None if plotting failed
        """
        if not explanation:
            logger.warning("No explanation available for plotting")
            return None
        
        feature_importance = explanation.get('feature_importance', {})
        if not feature_importance:
            logger.warning("No feature importance data available for plotting")
            return None
        
        try:
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:max_features]
            
            features, importances = zip(*sorted_features)
            
            # Clean up feature names for display
            display_names = [f.replace('_', ' ').title() for f in features]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(display_names)), importances, align='center')
            
            # Color bars based on positive/negative importance
            for i, imp in enumerate(importances):
                bars[i].set_color('green' if imp > 0 else 'red')
            
            plt.yticks(range(len(display_names)), display_names)
            plt.xlabel('Feature Importance')
            plt.title(f'{self.model_type.upper()} Model Feature Importance for {self.asset}')
            plt.tight_layout()
            
            # Save plot if output path provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Feature importance plot saved to {output_path}")
            else:
                # Generate default output path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    self.asset_dir, 
                    f"{self.model_type}_importance_{timestamp}.png"
                )
                plt.savefig(output_path)
                logger.info(f"Feature importance plot saved to {output_path}")
            
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return None
    
    def plot_grouped_importance(self, explanation, output_path=None):
        """
        Plot grouped feature importance from explanation
        
        Args:
            explanation (dict): Explanation data
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to saved plot or None if plotting failed
        """
        if not explanation:
            logger.warning("No explanation available for plotting")
            return None
        
        grouped_importance = explanation.get('grouped_importance', {})
        if not grouped_importance:
            logger.warning("No grouped importance data available for plotting")
            return None
        
        try:
            # Sort groups by importance
            sorted_groups = sorted(
                grouped_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            groups, importances = zip(*sorted_groups)
            
            # Clean up group names for display
            display_names = [g.replace('_', ' ').title() for g in groups]
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(
                importances, 
                labels=display_names,
                autopct='%1.1f%%',
                startangle=90,
                explode=[0.1 if i == 0 else 0 for i in range(len(importances))],  # Explode largest slice
                shadow=True
            )
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(f'{self.model_type.upper()} Model Factor Categories for {self.asset}')
            
            # Save plot if output path provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Grouped importance plot saved to {output_path}")
            else:
                # Generate default output path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(
                    self.asset_dir, 
                    f"{self.model_type}_grouped_importance_{timestamp}.png"
                )
                plt.savefig(output_path)
                logger.info(f"Grouped importance plot saved to {output_path}")
            
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error plotting grouped importance: {e}")
            return None
    
    def analyze_decision_boundary(self, X, feature_idx, feature_name, output_path=None, num_samples=50):
        """
        Analyze how changes in a feature affect the prediction
        
        Args:
            X (np.ndarray): Input data
            feature_idx (int): Index of feature to analyze
            feature_name (str): Name of feature
            output_path (str, optional): Path to save the plot
            num_samples (int): Number of samples to use for analysis
            
        Returns:
            str: Path to saved plot or None if analysis failed
        """
        if self.model is None:
            logger.error("No model set for decision boundary analysis")
            return None
        
        try:
            # Take first sample for analysis
            x_sample = X[0:1].copy()
            
            # Get feature range
            feature_min = np.min(X[:, feature_idx])
            feature_max = np.max(X[:, feature_idx])
            
            # Create samples with varying feature values
            feature_values = np.linspace(feature_min, feature_max, num_samples)
            predictions = []
            
            for value in feature_values:
                x_modified = x_sample.copy()
                x_modified[0, feature_idx] = value
                
                # Get prediction
                pred = self.model.predict(x_modified)
                
                # Extract prediction value
                if isinstance(pred, dict):  # For TFT model
                    pred_value = pred.get('predictions', np.array([0]))[0, -1, 0]
                else:
                    pred_value = pred[0, 0] if pred.ndim > 1 else pred[0]
                
                predictions.append(float(pred_value))
            
            # Plot decision boundary
            plt.figure(figsize=(10, 6))
            plt.plot(feature_values, predictions)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add line at y=0
            
            plt.xlabel(feature_name)
            plt.ylabel('Prediction')
            plt.title(f'Decision Boundary for {feature_name} ({self.asset})')
            plt.grid(True, alpha=0.3)
            
            # Shade regions for buy/sell signals
            plt.fill_between(feature_values, predictions, 0, where=(np.array(predictions) > 0), 
                             color='green', alpha=0.3, label='Buy Signal')
            plt.fill_between(feature_values, predictions, 0, where=(np.array(predictions) < 0), 
                             color='red', alpha=0.3, label='Sell Signal')
            
            plt.legend()
            
            # Save plot if output path provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Decision boundary plot saved to {output_path}")
            else:
                # Generate default output path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
                output_path = os.path.join(
                    self.asset_dir, 
                    f"{self.model_type}_{safe_feature_name}_decision_boundary_{timestamp}.png"
                )
                plt.savefig(output_path)
                logger.info(f"Decision boundary plot saved to {output_path}")
            
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error analyzing decision boundary: {e}")
            return None
    
    def explain_trade_decision(self, X, feature_names, market_data, trade_action, confidence):
        """
        Generate a comprehensive explanation for a trade decision
        
        Args:
            X (np.ndarray): Input data
            feature_names (list): Names of features
            market_data (pd.DataFrame): Market data used for the decision
            trade_action (str): Trade action (BUY, SELL, HOLD)
            confidence (float): Confidence in the decision
            
        Returns:
            dict: Comprehensive explanation of the trade decision
        """
        # Prepare additional information
        current_price = market_data['close'].iloc[-1] if 'close' in market_data.columns else None
        
        # Detect trend using simple EMA comparison
        recent_trend = "Unknown"
        if 'ema9' in market_data.columns and 'ema21' in market_data.columns:
            ema9 = market_data['ema9'].iloc[-1]
            ema21 = market_data['ema21'].iloc[-1]
            if ema9 > ema21:
                recent_trend = "Uptrend (EMA9 > EMA21)"
            else:
                recent_trend = "Downtrend (EMA9 < EMA21)"
        
        # Calculate volatility
        volatility = None
        if 'high' in market_data.columns and 'low' in market_data.columns and 'close' in market_data.columns:
            recent_highs = market_data['high'].iloc[-20:]
            recent_lows = market_data['low'].iloc[-20:]
            recent_closes = market_data['close'].iloc[-20:]
            
            # Calculate Average True Range as volatility measure
            true_ranges = []
            for i in range(1, len(recent_closes)):
                true_range = max(
                    recent_highs.iloc[i] - recent_lows.iloc[i],
                    abs(recent_highs.iloc[i] - recent_closes.iloc[i-1]),
                    abs(recent_lows.iloc[i] - recent_closes.iloc[i-1])
                )
                true_ranges.append(true_range)
            
            if true_ranges:
                atr = sum(true_ranges) / len(true_ranges)
                # Normalize by price
                volatility = atr / recent_closes.iloc[-1]
        
        # Prepare additional info dictionary
        additional_info = {
            'current_price': current_price,
            'recent_trend': recent_trend,
            'volatility': volatility,
            'trade_action': trade_action,
            'signal_strength': confidence
        }
        
        # Detect market regime
        if volatility is not None:
            if volatility > 0.02:  # High volatility threshold
                additional_info['market_regime'] = 'volatile'
            elif recent_trend.startswith('Uptrend') or recent_trend.startswith('Downtrend'):
                additional_info['market_regime'] = 'trending'
            else:
                additional_info['market_regime'] = 'ranging'
        
        # Generate explanation
        explanation = self.explain_prediction(X, feature_names, additional_info)
        
        # If the trade action doesn't match the model's signal, note the conflict
        if explanation and trade_action != explanation.get('signal'):
            explanation['signal_conflict'] = True
            explanation['original_signal'] = explanation.get('signal')
            explanation['final_signal'] = trade_action
            explanation['signal_override_reason'] = "Signal modified by trading rules or risk management"
        
        # Generate plain text explanation
        if explanation:
            explanation['explanation_text'] = self.generate_explanation_text(explanation)
            
            # Compare with baseline
            comparison = self.compare_with_baseline(explanation)
            if comparison:
                explanation['comparison'] = comparison
                explanation['comparison_text'] = self.generate_comparison_text(comparison)
        
        return explanation
    
    def evaluate_explanation_quality(self, explanation, actual_outcome):
        """
        Evaluate the quality of an explanation based on the actual outcome
        
        Args:
            explanation (dict): Explanation data
            actual_outcome (dict): Actual outcome with 'correct' (bool) and 'profit' (float)
            
        Returns:
            dict: Evaluation results
        """
        if not explanation:
            return {'score': 0, 'reason': 'No explanation available'}
        
        # Start with a base score
        score = 0.5
        reasons = []
        
        # Check if prediction was correct
        if actual_outcome.get('correct', False):
            score += 0.25
            reasons.append("Prediction was correct")
        else:
            score -= 0.25
            reasons.append("Prediction was incorrect")
        
        # Check if confidence matched outcome
        confidence = explanation.get('confidence', 0.5)
        profit = actual_outcome.get('profit', 0)
        
        # High confidence, high profit - good
        if confidence > 0.7 and profit > 0:
            score += 0.2
            reasons.append("High confidence correctly predicted profit")
        # High confidence, negative profit - bad
        elif confidence > 0.7 and profit < 0:
            score -= 0.3
            reasons.append("High confidence incorrectly predicted outcome")
        # Low confidence, any outcome - neutral
        elif confidence < 0.3:
            # No score adjustment for low confidence
            reasons.append("Low confidence made minimal prediction")
        
        # Check explanation completeness
        feature_importance = explanation.get('feature_importance', {})
        if len(feature_importance) > 5:
            score += 0.1
            reasons.append("Explanation included comprehensive feature analysis")
        
        # Check if grouped importance was included
        grouped_importance = explanation.get('grouped_importance', {})
        if grouped_importance:
            score += 0.05
            reasons.append("Explanation included factor categories")
        
        # Check for explanation text
        explanation_text = explanation.get('explanation_text')
        if explanation_text and len(explanation_text) > 100:
            score += 0.05
            reasons.append("Explanation included detailed text description")
        
        # Check for comparison with baseline
        comparison = explanation.get('comparison')
        if comparison:
            score += 0.05
            reasons.append("Explanation included comparison with baseline")
        
        # Cap score at 0-1 range
        score = max(0, min(1, score))
        
        return {
            'score': score,
            'reasons': reasons,
            'outcome': actual_outcome
        }
    
    def track_explanation_performance(self, explanation_id, actual_outcome):
        """
        Track the performance of a specific explanation
        
        Args:
            explanation_id (str): ID of explanation to track
            actual_outcome (dict): Actual outcome with 'correct' (bool) and 'profit' (float)
            
        Returns:
            bool: Success of tracking operation
        """
        # Find the explanation by ID
        target_explanation = None
        for exp in self.explanation_history:
            if exp.get('id') == explanation_id:
                target_explanation = exp
                break
        
        if not target_explanation:
            logger.warning(f"Explanation with ID {explanation_id} not found")
            return False
        
        # Evaluate explanation quality
        evaluation = self.evaluate_explanation_quality(target_explanation, actual_outcome)
        
        # Add evaluation to explanation
        target_explanation['evaluation'] = evaluation
        target_explanation['actual_outcome'] = actual_outcome
        
        # Update in history
        self._save_explanations()
        
        logger.info(f"Tracked performance for explanation {explanation_id}, quality score: {evaluation['score']:.2f}")
        
        return True
    
    def generate_summary_report(self, recent_count=30):
        """
        Generate a summary report of recent explanations and their performance
        
        Args:
            recent_count (int): Number of recent explanations to include
            
        Returns:
            dict: Summary report
        """
        if not self.explanation_history:
            return {'status': 'No explanations available for reporting'}
        
        # Get recent explanations
        recent_explanations = self.explanation_history[-recent_count:]
        
        # Calculate statistics
        evaluated_explanations = [e for e in recent_explanations if 'evaluation' in e]
        
        if not evaluated_explanations:
            return {'status': 'No evaluated explanations available for reporting'}
        
        # Calculate overall explanation quality
        quality_scores = [e['evaluation']['score'] for e in evaluated_explanations]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Calculate model performance
        outcomes = [e.get('actual_outcome', {}) for e in evaluated_explanations]
        correct_predictions = sum(1 for o in outcomes if o.get('correct', False))
        accuracy = correct_predictions / len(outcomes) if outcomes else 0
        
        # Calculate average profit
        profits = [o.get('profit', 0) for o in outcomes]
        avg_profit = sum(profits) / len(profits) if profits else 0
        
        # Identify most important features
        all_feature_importance = {}
        for e in recent_explanations:
            for feature, importance in e.get('feature_importance', {}).items():
                all_feature_importance[feature] = all_feature_importance.get(feature, 0) + abs(importance)
        
        # Sort by total importance
        sorted_features = sorted(
            all_feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]  # Top 10 features
        
        # Normalize importance
        total_importance = sum(i for _, i in sorted_features)
        if total_importance > 0:
            normalized_features = [(f, i/total_importance) for f, i in sorted_features]
        else:
            normalized_features = sorted_features
        
        # Group explanations by market regime
        regime_explanations = {}
        for e in recent_explanations:
            regime = e.get('additional_info', {}).get('market_regime', 'unknown')
            if regime not in regime_explanations:
                regime_explanations[regime] = []
            regime_explanations[regime].append(e)
        
        # Calculate performance by regime
        regime_performance = {}
        for regime, exps in regime_explanations.items():
            evaluated_exps = [e for e in exps if 'evaluation' in e]
            if evaluated_exps:
                correct = sum(1 for e in evaluated_exps if e.get('actual_outcome', {}).get('correct', False))
                accuracy = correct / len(evaluated_exps)
                
                profits = [e.get('actual_outcome', {}).get('profit', 0) for e in evaluated_exps]
                avg_profit = sum(profits) / len(profits) if profits else 0
                
                regime_performance[regime] = {
                    'count': len(evaluated_exps),
                    'accuracy': accuracy,
                    'avg_profit': avg_profit
                }
        
        # Create the report
        report = {
            'timestamp': datetime.now().isoformat(),
            'period': f"Last {len(recent_explanations)} explanations",
            'overall': {
                'explanation_quality': avg_quality,
                'prediction_accuracy': accuracy,
                'average_profit': avg_profit,
                'evaluated_count': len(evaluated_explanations)
            },
            'important_features': [{'name': f, 'importance': float(i)} for f, i in normalized_features],
            'market_regimes': regime_performance
        }
        
        return report

class ExplainableAIManager:
    """
    Manager class to coordinate explainable AI across multiple models and assets
    """
    def __init__(self):
        self.explainers = {}
        
        logger.info("Initialized ExplainableAIManager")
    
    def get_explainer(self, model_type, asset, model=None):
        """
        Get or create an explainer for the specified model and asset
        
        Args:
            model_type (str): Type of model to explain
            asset (str): Trading pair/asset
            model (tf.keras.Model, optional): The model to explain
            
        Returns:
            ExplainableAI: Explainer instance
        """
        key = f"{model_type}_{asset}"
        
        if key not in self.explainers:
            self.explainers[key] = ExplainableAI(model_type, asset, model)
        elif model is not None:
            # Update model if provided
            self.explainers[key].set_model(model)
        
        return self.explainers[key]
    
    def explain_all_models(self, X, feature_names, market_data, trade_action, confidence):
        """
        Generate explanations for all models
        
        Args:
            X (dict): Input data dictionary with model_type keys
            feature_names (dict): Feature names dictionary with model_type keys
            market_data (pd.DataFrame): Market data
            trade_action (str): Trade action
            confidence (float): Confidence in the decision
            
        Returns:
            dict: Explanations for all models
        """
        explanations = {}
        
        for key, explainer in self.explainers.items():
            model_type = key.split('_')[0]
            
            if model_type in X and model_type in feature_names:
                explanations[model_type] = explainer.explain_trade_decision(
                    X[model_type],
                    feature_names[model_type],
                    market_data,
                    trade_action,
                    confidence
                )
        
        return explanations
    
    def create_ensemble_explanation(self, individual_explanations, weights=None):
        """
        Create an ensemble explanation from individual model explanations
        
        Args:
            individual_explanations (dict): Dictionary of individual explanations by model_type
            weights (dict, optional): Weights for each model in the ensemble
            
        Returns:
            dict: Ensemble explanation
        """
        if not individual_explanations:
            return None
        
        # Use equal weights if not provided
        if weights is None:
            model_types = list(individual_explanations.keys())
            weights = {model_type: 1.0 / len(model_types) for model_type in model_types}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if total is zero
            model_types = list(individual_explanations.keys())
            normalized_weights = {model_type: 1.0 / len(model_types) for model_type in model_types}
        
        # Initialize ensemble explanation
        ensemble_explanation = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'ensemble',
            'feature_importance': {},
            'grouped_importance': {},
            'individual_explanations': individual_explanations,
            'weights': normalized_weights
        }
        
        # Aggregate predictions and confidences
        weighted_prediction = 0
        weighted_confidence = 0
        
        for model_type, explanation in individual_explanations.items():
            if model_type in normalized_weights:
                weight = normalized_weights[model_type]
                
                # Add weighted prediction and confidence
                if explanation:
                    weighted_prediction += explanation.get('prediction', 0) * weight
                    weighted_confidence += explanation.get('confidence', 0) * weight
                    
                    # Aggregate feature importance
                    for feature, importance in explanation.get('feature_importance', {}).items():
                        if feature not in ensemble_explanation['feature_importance']:
                            ensemble_explanation['feature_importance'][feature] = 0
                        ensemble_explanation['feature_importance'][feature] += importance * weight
                    
                    # Aggregate grouped importance
                    for group, importance in explanation.get('grouped_importance', {}).items():
                        if group not in ensemble_explanation['grouped_importance']:
                            ensemble_explanation['grouped_importance'][group] = 0
                        ensemble_explanation['grouped_importance'][group] += importance * weight
        
        # Set ensemble prediction and signal
        ensemble_explanation['prediction'] = weighted_prediction
        ensemble_explanation['confidence'] = weighted_confidence
        ensemble_explanation['signal'] = 'BUY' if weighted_prediction > 0 else 'SELL' if weighted_prediction < 0 else 'NEUTRAL'
        
        # Copy additional info from most heavily weighted explanation
        if individual_explanations:
            # Find model with highest weight
            best_model = max(normalized_weights.items(), key=lambda x: x[1])[0]
            if best_model in individual_explanations and individual_explanations[best_model]:
                ensemble_explanation['additional_info'] = individual_explanations[best_model].get('additional_info', {})
        
        # Generate plain text explanation
        ensemble_explanation['explanation_text'] = self._generate_ensemble_explanation_text(ensemble_explanation)
        
        return ensemble_explanation
    
    def _generate_ensemble_explanation_text(self, ensemble_explanation):
        """
        Generate human-readable explanation text for ensemble
        
        Args:
            ensemble_explanation (dict): Ensemble explanation data
            
        Returns:
            str: Human-readable explanation
        """
        if not ensemble_explanation:
            return "No ensemble explanation available."
        
        prediction = ensemble_explanation.get('prediction', 0)
        signal = ensemble_explanation.get('signal', 'NEUTRAL')
        confidence = ensemble_explanation.get('confidence', 0) * 100  # Convert to percentage
        
        # Start with the signal and confidence
        text = [f"Ensemble Signal: {signal} (Confidence: {confidence:.2f}%)"]
        
        # Add model contribution information
        weights = ensemble_explanation.get('weights', {})
        individual_explanations = ensemble_explanation.get('individual_explanations', {})
        
        if weights and individual_explanations:
            text.append("\nModel contributions:")
            
            # Sort models by weight
            sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            for model_type, weight in sorted_models:
                if model_type in individual_explanations and individual_explanations[model_type]:
                    model_signal = individual_explanations[model_type].get('signal', 'UNKNOWN')
                    model_confidence = individual_explanations[model_type].get('confidence', 0) * 100
                    
                    # Highlight agreement/disagreement with ensemble
                    agreement = model_signal == signal
                    agreement_text = "agrees with" if agreement else "disagrees with"
                    
                    text.append(f"- {model_type.upper()} ({weight*100:.1f}%): {model_signal} ({model_confidence:.1f}%), {agreement_text} ensemble")
        
        # Add feature importance explanation
        feature_importance = ensemble_explanation.get('feature_importance', {})
        if feature_importance:
            # Sort features by absolute importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:5]  # Top 5 features
            
            text.append("\nTop influencing factors:")
            for feature, importance in sorted_features:
                # Format feature name for readability
                readable_name = feature.replace('_', ' ').title()
                
                # Determine direction of influence
                if importance > 0:
                    direction = "increased"
                else:
                    direction = "decreased"
                
                text.append(f"- {readable_name}: {direction} the signal ({abs(importance):.3f})")
        
        # Add market context if available
        additional_info = ensemble_explanation.get('additional_info', {})
        market_regime = additional_info.get('market_regime')
        if market_regime:
            text.append(f"\nMarket Regime: {market_regime.title()}")
        
        recent_trend = additional_info.get('recent_trend')
        if recent_trend:
            text.append(f"Recent Trend: {recent_trend}")
        
        return "\n".join(text)
    
    def get_all_reports(self):
        """
        Generate reports for all explainers
        
        Returns:
            dict: Reports for all explainers
        """
        reports = {}
        
        for key, explainer in self.explainers.items():
            reports[key] = explainer.generate_summary_report()
        
        return reports

# Create a global instance for easy access
explainable_ai_manager = ExplainableAIManager()

def main():
    """Test the explainable AI integration module"""
    logger.info("Explainable AI Integration module imported")

if __name__ == "__main__":
    main()