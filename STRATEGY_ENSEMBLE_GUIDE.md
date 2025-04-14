# Strategy Ensemble Training Guide

This guide explains how to use the strategy ensemble training system to make your trading models work better together rather than competing.

## Overview

The Strategy Ensemble Training System optimizes how multiple trading strategies collaborate with each other rather than just optimizing them individually. This creates a "team" of strategies that complement each other across different market conditions.

### Key Benefits

1. **Reduced Signal Conflicts**: Strategies learn when to defer to others, eliminating contradictory signals and reducing confusion.

2. **Market Regime Specialization**: Strategies specialize in different market conditions (volatile/trending/ranging), allowing the ensemble to perform well in all environments.

3. **Improved Collective Intelligence**: The ensemble becomes smarter than any individual strategy, leading to better overall performance.

4. **Optimized Leverage Settings**: Different strategies learn to use different leverage levels appropriate to their confidence and the market conditions.

5. **Adaptive Decision Making**: The ensemble dynamically adjusts strategy weights based on recent performance.

## Components

The Strategy Ensemble System consists of three main components:

1. **Strategy Ensemble Trainer** (`strategy_ensemble_trainer.py`): Trains strategies to work together by assigning specialized roles and optimizing weights.

2. **Model Collaboration Integrator** (`model_collaboration_integrator.py`): Integrates signals from multiple strategies in real-time using the trained ensemble weights.

3. **Run Ensemble Training** (`run_ensemble_training.py`): Command-line utility to execute the ensemble training process.

## Getting Started

### Step 1: Run the Ensemble Training

```bash
python run_ensemble_training.py --assets "SOL/USD" "ETH/USD" "BTC/USD" --training-days 60 --visualize
```

This will:
- Train the ensemble using 60 days of historical data
- Optimize how strategies work together across different market regimes
- Generate visualization of ensemble performance
- Create a weights file in `models/ensemble/strategy_ensemble_weights.json`

### Step 2: Integrate with your Trading System

Modify your trading bot to use the `model_collaboration_integrator.py` for signal integration:

```python
from model_collaboration_integrator import ModelCollaborationIntegrator

# Initialize the integrator with trained weights
integrator = ModelCollaborationIntegrator()

# When you have signals from multiple strategies
signals = {
    "ARIMAStrategy": {"signal": "BUY", "strength": 0.7},
    "AdaptiveStrategy": {"signal": "NEUTRAL", "strength": 0.5},
    "MLStrategy": {"signal": "BUY", "strength": 0.8}
}

# Integrate signals using the trained weights
decision = integrator.integrate_signals(signals, market_data)

# Use the integrated decision for trading
if decision["signal"] == "BUY" and decision["confidence"] > 0.6:
    # Execute buy order with decision["parameters"]
    place_order("buy", decision["parameters"]["leverage"])
```

## Strategy Role Specialization

The system assigns specific roles to each strategy based on its strengths:

| Strategy | Specialized Roles |
|----------|------------------|
| ARIMAStrategy | Trend Following, Range-Bound Markets |
| AdaptiveStrategy | Counter-Trend, Defensive |
| IntegratedStrategy | Volatility, Multi-Timeframe |
| MLStrategy | Breakout, Aggressive |

This specialization allows each strategy to "take the lead" in market conditions where it performs best.

## Market Regime Detection

The system automatically detects five distinct market regimes:

1. **Volatile Trending Up**: Rapid upward price movement with high volatility
2. **Volatile Trending Down**: Rapid downward price movement with high volatility
3. **Normal Trending Up**: Steady upward price movement with normal volatility
4. **Normal Trending Down**: Steady downward price movement with normal volatility
5. **Neutral/Ranging**: Sideways price movement with low volatility

For each regime, different strategy weights are optimized to maximize performance.

## Configuration Options

When running the ensemble training, you can customize the following parameters:

```bash
python run_ensemble_training.py --help
```

Key parameters:
- `--assets`: Trading pairs to include (default: SOL/USD, ETH/USD, BTC/USD)
- `--strategies`: Strategies to include in the ensemble
- `--timeframes`: Timeframes to use for training
- `--training-days`: Number of days of historical data to use
- `--validation-days`: Number of days for validation
- `--visualize`: Generate performance visualizations

## Performance Monitoring

The system tracks performance metrics for both individual strategies and the ensemble:

```python
# Update performance when trade outcomes are known
integrator.update_strategy_performance(
    strategy="ARIMAStrategy",
    was_correct=True,
    signal_type="BUY",
    confidence=0.7,
    regime="volatile_trending_up"
)

# Update ensemble performance
integrator.update_ensemble_performance(
    was_correct=True,
    signal_type="BUY",
    confidence=0.8,
    regime="volatile_trending_up",
    contributing_strategies={"ARIMAStrategy": 0.4, "MLStrategy": 0.6}
)
```

This performance tracking allows the system to continuously adapt strategy weights based on recent results.

## Advanced Features

### Signal Conflict Resolution

When strategies disagree (e.g., some say BUY while others say SELL), the collaboration integrator resolves conflicts by:

1. Considering strategy specialization for the current market regime
2. Evaluating recent performance in similar conditions
3. Comparing signal strengths and confidence levels
4. Applying adaptive weighting based on historical accuracy

### Collaborative Confidence Scoring

The system calculates a "collaborative confidence" score that combines:

- Signal agreement between strategies
- Individual strategy confidence levels
- Market regime alignment with strategy specialization
- Recent performance in similar conditions

This confidence score helps determine optimal position sizing and leverage.

## Interpreting Results

After training, examine the visualization to understand:

1. Which strategies perform best in which market regimes
2. How the ensemble outperforms individual strategies
3. Strategy weight distribution across different regimes
4. Areas for potential improvement

The `strategy_ensemble_weights.json` file contains the optimized weights that will be used during live trading.

## Troubleshooting

If you encounter issues:

- Ensure you have sufficient historical data for all assets
- Check that all strategies are generating signals properly
- Verify that the `models/ensemble` directory exists and is writable
- Consider running with fewer assets or shorter training period initially

## Next Steps

1. Run the ensemble training periodically (e.g., weekly) to adapt to changing market conditions
2. Monitor the performance of the ensemble vs. individual strategies
3. Consider adding new specialized strategies to fill gaps in the ensemble's capabilities
4. Fine-tune the aggressive/conservative settings based on your risk tolerance