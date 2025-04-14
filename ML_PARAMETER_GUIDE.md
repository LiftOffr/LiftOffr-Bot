# ML Parameter Update Guide

This guide explains how to modify the ML trading parameters while continuing to trade in real-time using the Kraken Trading Bot.

## Quick Start

The simplest way to modify parameters is using the `update_ml_parameters.py` script. This tool allows you to make changes to the ML configuration without manually editing the JSON file.

```bash
python update_ml_parameters.py --asset "SOL/USD" --max-leverage 150 --restart
```

This example increases the maximum leverage for SOL/USD to 150x and restarts the trading bot.

## Common Parameter Adjustments

### 1. Adjust Leverage Settings

```bash
python update_ml_parameters.py --asset "SOL/USD" --min-leverage 25 --default-leverage 50 --max-leverage 150 --confidence-threshold 0.6 --restart
```

This command sets SOL/USD to use:
- Minimum leverage: 25x
- Default leverage: 50x
- Maximum leverage: 150x
- Confidence threshold: 0.6 (trades will only be taken when confidence is at least 0.6)

### 2. Adjust Position Sizing

```bash
python update_ml_parameters.py --asset "ETH/USD" --confidence-thresholds "0.65,0.75,0.85,0.95" --size-multipliers "0.25,0.5,0.75,1.0" --restart
```

This sets the position sizing for ETH/USD so that:
- Confidence of 0.65-0.75 → 25% position size
- Confidence of 0.75-0.85 → 50% position size
- Confidence of 0.85-0.95 → 75% position size
- Confidence of 0.95+ → 100% position size

### 3. Adjust Risk Management

```bash
python update_ml_parameters.py --asset "BTC/USD" --max-drawdown 4.5 --take-profit-multiplier 2.5 --stop-loss-multiplier 1.0 --restart
```

This adjusts the risk management for BTC/USD:
- Maximum drawdown: 4.5%
- Take profit at 2.5x the predicted movement
- Stop loss at 1.0x the predicted movement

### 4. Adjust Capital Allocation

```bash
python update_ml_parameters.py --sol-allocation 0.5 --eth-allocation 0.3 --btc-allocation 0.2 --restart
```

This changes how capital is allocated across assets:
- SOL/USD: 50%
- ETH/USD: 30%
- BTC/USD: 20%

### 5. Enable/Disable Extreme Leverage

```bash
python update_ml_parameters.py --extreme-leverage true --restart
```

This enables the extreme leverage settings. To disable:

```bash
python update_ml_parameters.py --extreme-leverage false --restart
```

## Manual Configuration

You can also directly edit the `ml_config.json` file to make changes. The configuration file is structured as follows:

```json
{
  "global_settings": {
    "extreme_leverage_enabled": true,
    "model_pruning_threshold": 0.4,
    "default_capital_allocation": {
      "SOL/USD": 0.40,
      "ETH/USD": 0.35,
      "BTC/USD": 0.25
    }
  },
  "asset_configs": {
    "SOL/USD": {
      "leverage_settings": { ... },
      "position_sizing": { ... },
      "risk_management": { ... },
      "model_weights": { ... }
    },
    "ETH/USD": { ... },
    "BTC/USD": { ... }
  },
  "training_parameters": { ... }
}
```

After manually editing, you can restart the trading bot with:

```bash
python run_optimized_ml_trading.py --reset --sandbox
```

## Best Practices

1. **Start Conservative**: Begin with lower leverage settings and gradually increase as you gain confidence in the model.

2. **Monitor Performance**: After making changes, monitor the trading performance carefully for at least 24 hours.

3. **One Change at a Time**: Make one set of changes at a time to easily identify which parameter adjustments are improving performance.

4. **Back Up Before Changes**: Before making significant changes, back up your `ml_config.json` file.

5. **Test in Sandbox**: Always test parameter changes in sandbox mode before using them in live trading.

## Advanced Configuration

For advanced users, additional parameters can be configured in `ml_config.json`:

- **Market Regime Weights**: Customize how different models are weighted in trending, ranging, and volatile markets.
- **Training Parameters**: Adjust epochs, batch size, learning rates, and other training parameters.
- **Model Pruning Settings**: Configure how and when underperforming models are automatically removed from the ensemble.

Refer to the ML_LIVE_TRADING_GUIDE.md for detailed information on all available configuration options.