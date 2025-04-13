# Signal Strength and Auto-Pruning Guide

This guide explains the signal strength mechanism and auto-pruning functionality in the enhanced Kraken Trading Bot.

## Signal Strength System

The signal strength system allows different strategies to override each other based on the strength of their signals, ensuring that the strongest convictions take precedence.

### Key Concepts

1. **Signal Strength Value**: Each trading signal is assigned a strength value between 0.0 and 1.0, where:
   - 0.0 indicates no conviction
   - 1.0 indicates maximum conviction

2. **Strategy Hierarchy**: When multiple strategies generate conflicting signals, the signal with the highest strength wins.

3. **Dynamic Adjustment**: Signal strengths are dynamically adjusted based on:
   - Recent strategy performance
   - Market volatility
   - Signal persistence over time
   - ML confidence scores

### Implementation

The signal strength mechanism is implemented in the following components:

1. **Bot Manager**: Coordinates signals from multiple strategies and applies the strength-based arbitration:
   ```python
   # BotManager registers signals with their strengths
   manager.register_signal("ARIMAStrategy", "BUY", 0.8)
   manager.register_signal("AdaptiveStrategy", "SELL", 0.6)
   
   # BotManager resolves conflicts based on highest strength
   final_signal = manager.resolve_signals()  # Returns "BUY" with strength 0.8
   ```

2. **Strategy Signal Generation**: Trading strategies calculate signal strength based on multiple factors:
   ```python
   # Example of dynamic signal strength calculation in a strategy
   def calculate_signal_strength(self):
       # Base strength
       strength = 0.5
       
       # Adjust based on recent accuracy
       strength += (self.recent_accuracy - 0.5) * 0.2
       
       # Adjust based on signal confidence
       strength += self.signal_confidence * 0.3
       
       # Adjust based on market conditions
       if self.market_volatility > self.volatility_threshold:
           strength -= 0.1
       
       return max(0.0, min(1.0, strength))  # Clamp to [0.0, 1.0]
   ```

3. **ML Integration**: ML models contribute to strength calculation based on prediction confidence:
   ```python
   # ML prediction confidence feeds into signal strength
   ml_prediction, ml_confidence = self.model.predict(self.features)
   signal_strength = 0.5 + (ml_confidence - 0.5) * self.ml_influence
   ```

### Configuration Options

Signal strength behavior can be customized in `config.py`:

```python
# Signal strength settings
SIGNAL_STRENGTH_SETTINGS = {
    'enable_strength_arbitration': True,      # Enable/disable feature
    'minimum_override_difference': 0.2,       # Minimum difference to override
    'dynamic_adjustment': True,               # Enable performance-based adjustment
    'performance_influence': 0.3,             # How much past performance influences strength
    'volatility_penalty': 0.1,                # Penalty in high volatility
    'minimum_strength_threshold': 0.3         # Signals below this are ignored
}
```

## Auto-Pruning System

The auto-pruning system automatically identifies and removes underperforming components from ML models and trading strategies, improving overall system performance.

### Key Concepts

1. **Performance Evaluation**: Components are evaluated based on:
   - Prediction accuracy
   - Contribution to overall performance
   - Stability across different market regimes

2. **Pruning Threshold**: Components performing below threshold (default: 55%) are candidates for removal.

3. **Component-Level Granularity**: Pruning occurs at multiple levels:
   - Entire models
   - Model branches (TCN, CNN, Attention)
   - Individual features
   - Specific timeframes

4. **Retraining**: After pruning, models are automatically retrained with optimized architectures.

### Implementation

The auto-pruning system is implemented in the following components:

1. **Model Evaluator**: Analyzes performance of model components:
   ```python
   # Evaluate component performance
   component_performance = evaluator.evaluate_model_components(model, X_test, y_test)
   
   # Identify components for pruning
   components_to_prune = [
       name for name, accuracy in component_performance.items()
       if accuracy < PERFORMANCE_THRESHOLD
   ]
   ```

2. **Pruning Engine**: Removes underperforming components and retrains:
   ```python
   # Create pruned model
   pruned_model = pruner.create_pruned_model(model, components_to_prune)
   
   # Retrain pruned model
   retrained_model = pruner.retrain_pruned_model(pruned_model, X_train, y_train)
   ```

3. **Ensemble Pruner**: Optimizes ensemble model compositions:
   ```python
   # Prune ensemble and adjust weights
   pruned_ensemble = ensemble_pruner.prune_ensemble(models, models_to_prune)
   weights = ensemble_pruner.create_ensemble_weights(model_performance)
   ```

### Benefits of Auto-Pruning

1. **Improved Accuracy**: Removing noise from the system increases prediction accuracy
2. **Better Performance**: Smaller, more efficient models provide faster execution
3. **Adaptation**: The system continuously adapts to changing market conditions
4. **Reduced Overfitting**: Pruning helps prevent models from overfitting to noise

### Usage Examples

To run the auto-pruning system on existing models:

```bash
python auto_prune_ml_models.py --symbol SOLUSD --timeframe 1h --auto-prune-all
```

To apply auto-pruning during model training:

```python
# Create model with auto-pruning enabled
model = EnhancedTCNModel(
    input_shape=(60, 64),
    enable_auto_pruning=True,
    performance_threshold=0.55
)

# Train with pruning
model.train_with_pruning(X_train, y_train, X_val, y_val)
```

## Integration of Signal Strength and Auto-Pruning

These two systems work together to create a highly adaptive trading system:

1. **Dynamic Feedback Loop**: Auto-pruning improves model accuracy, which leads to stronger signals with higher conviction.

2. **Performance-Based Weighting**: As components prove their worth through accurate predictions, they gain more influence on signal strength.

3. **Regime-Specific Optimization**: Both systems adapt to different market regimes, applying different weights and keeping different components based on the current market conditions.

4. **Continuous Improvement**: The combined system learns from its mistakes, gradually improving over time by strengthening what works and removing what doesn't.

## Advanced Configurations

For advanced users, here are some additional configuration options:

### Signal Strength Fine-Tuning

```python
# Fine-grained control over strategy-specific strengths
STRATEGY_STRENGTH_OVERRIDES = {
    'ARIMAStrategy': {
        'base_strength': 0.7,
        'market_regime_multipliers': {
            'volatile': 0.8,      # Reduce strength in volatile markets
            'trending': 1.2,      # Increase strength in trending markets
        }
    },
    'IntegratedStrategy': {
        'base_strength': 0.6,
        'market_regime_multipliers': {
            'volatile': 1.2,      # Increase strength in volatile markets
            'trending': 0.9       # Reduce strength in trending markets
        }
    }
}
```

### Auto-Pruning Advanced Settings

```python
# Advanced auto-pruning configuration
AUTO_PRUNING_SETTINGS = {
    'performance_threshold': 0.55,   # Minimum accuracy to retain component
    'high_performance_threshold': 0.7,  # High-performers get extra weight
    'retraining_epochs': 50,        # Epochs to retrain after pruning
    'prune_frequency': 100,         # Trading iterations between pruning runs
    'min_data_points': 500,         # Minimum data points before pruning
    'feature_importance_threshold': 0.02  # Min importance to keep a feature
}
```

## Conclusion

The signal strength mechanism and auto-pruning system together create a robust, self-improving trading system that can adapt to various market conditions. By continuously evaluating and optimizing its components, the system gradually improves its predictive accuracy and trading performance over time.

These systems are key to achieving the target 90%+ accuracy for directional predictions, as they allow the bot to focus on its most accurate signals while removing noise from the decision-making process.