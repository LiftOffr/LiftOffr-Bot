#!/usr/bin/env python3
"""
Apply Enhanced Simulation

This script upgrades your existing trading bot with ultra-realistic market conditions.
It applies the enhanced simulation features to the current sandbox trading environment.
"""

import os
import sys
import json
import logging
import datetime
import argparse
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
FEE_CONFIG_FILE = f"{CONFIG_DIR}/fee_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
MARKET_IMPACT_CONFIG_FILE = f"{CONFIG_DIR}/market_impact_config.json"
LATENCY_CONFIG_FILE = f"{CONFIG_DIR}/latency_config.json"
ADVANCED_FEE_CONFIG_FILE = f"{CONFIG_DIR}/advanced_fee_config.json"
TIER_BASED_FEES_FILE = f"{CONFIG_DIR}/tier_based_fees.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
RISK_METRICS_FILE = f"{DATA_DIR}/risk_metrics.json"

def create_market_impact_config():
    """Create market impact configuration file"""
    config = {
        "impact_factor": 0.2,  # Lower = less impact
        "decay_factor": 0.5,   # Higher = faster decay
        "impact_window": 3600,  # Impact window in seconds
        "pair_factors": {
            "BTC/USD": 0.5,    # Lower = less impact
            "ETH/USD": 0.7,
            "SOL/USD": 1.2,
            "ADA/USD": 1.5,
            "DOT/USD": 1.4,
            "LINK/USD": 1.3
        }
    }
    
    os.makedirs(os.path.dirname(MARKET_IMPACT_CONFIG_FILE), exist_ok=True)
    with open(MARKET_IMPACT_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created market impact configuration: {MARKET_IMPACT_CONFIG_FILE}")
    return config

def create_latency_config():
    """Create latency configuration file"""
    config = {
        "base_latency_ms": 150,  # Base latency in milliseconds
        "latency_variance_ms": 50,  # Variance in latency
        "market_rush_factor": 2.5,  # Factor for high volatility periods
        "peak_hours_factor": 1.5,  # Factor during peak trading hours
        "connection_failure_prob": 0.001,  # Probability of connection failure
        "timeout_prob": 0.002,  # Probability of timeout
        "retry_delay_ms": 500  # Delay before retry in milliseconds
    }
    
    os.makedirs(os.path.dirname(LATENCY_CONFIG_FILE), exist_ok=True)
    with open(LATENCY_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created latency configuration: {LATENCY_CONFIG_FILE}")
    return config

def create_advanced_fee_config():
    """Create advanced fee configuration file"""
    config = {
        "base_maker_fee": 0.0002,  # 0.02%
        "base_taker_fee": 0.0005,  # 0.05%
        "funding_fee_8h": 0.0001,  # 0.01%
        "liquidation_fee": 0.0075,  # 0.75%
        "min_margin_ratio": 0.0125,  # 1.25%
        "maintenance_margin": 0.04,  # 4%
        "withdrawal_fee": {
            "BTC": 0.0005,
            "ETH": 0.005,
            "SOL": 0.01,
            "ADA": 1,
            "DOT": 0.1,
            "LINK": 0.1,
            "USD": 5
        },
        "deposit_fee": {
            "crypto": 0,
            "bank_transfer": 0,
            "credit_card": 0.035  # 3.5%
        }
    }
    
    os.makedirs(os.path.dirname(ADVANCED_FEE_CONFIG_FILE), exist_ok=True)
    with open(ADVANCED_FEE_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created advanced fee configuration: {ADVANCED_FEE_CONFIG_FILE}")
    return config

def create_tier_based_fees_config():
    """Create tier-based fees configuration file"""
    config = {
        "tiers": [
            {
                "min_volume": 0,
                "max_volume": 50000,
                "maker_fee": 0.0002,  # 0.02%
                "taker_fee": 0.0005   # 0.05%
            },
            {
                "min_volume": 50000,
                "max_volume": 100000,
                "maker_fee": 0.00016,  # 0.016%
                "taker_fee": 0.0004    # 0.04%
            },
            {
                "min_volume": 100000,
                "max_volume": 250000,
                "maker_fee": 0.00014,  # 0.014%
                "taker_fee": 0.00035   # 0.035%
            },
            {
                "min_volume": 250000,
                "max_volume": 500000,
                "maker_fee": 0.00012,  # 0.012%
                "taker_fee": 0.0003    # 0.03%
            },
            {
                "min_volume": 500000,
                "max_volume": 1000000,
                "maker_fee": 0.0001,   # 0.01%
                "taker_fee": 0.00025   # 0.025%
            },
            {
                "min_volume": 1000000,
                "max_volume": float('inf'),
                "maker_fee": 0.00008,  # 0.008%
                "taker_fee": 0.0002    # 0.02%
            }
        ]
    }
    
    os.makedirs(os.path.dirname(TIER_BASED_FEES_FILE), exist_ok=True)
    with open(TIER_BASED_FEES_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created tier-based fees configuration: {TIER_BASED_FEES_FILE}")
    return config

def update_fee_config():
    """Update fee configuration with more realistic values"""
    # Load existing fee config if it exists
    if os.path.exists(FEE_CONFIG_FILE):
        with open(FEE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with more realistic values
    updated_config = {
        "maker_fee": 0.0002,           # 0.02%
        "taker_fee": 0.0005,           # 0.05%
        "funding_fee_8h": 0.0001,      # 0.01% per 8 hours
        "liquidation_fee": 0.0075,     # 0.75%
        "min_margin_ratio": 0.0125,    # 1.25%
        "maintenance_margin": 0.04,    # 4%
        "max_leverage": 125.0,         # Maximum 125x leverage
        "slippage": {
            "low_volume": 0.001,       # 0.1% for low volume
            "medium_volume": 0.0003,   # 0.03% for medium volume
            "high_volume": 0.0001      # 0.01% for high volume
        },
        "volume_thresholds": {
            "low": 100000,             # $100k
            "medium": 500000           # $500k
        },
        "liquidation_cascade": {
            "enabled": True,
            "threshold": 0.05,         # 5% price drop triggers cascade
            "impact_factor": 1.5       # 1.5x normal impact during cascade
        }
    }
    
    # Merge with existing config to preserve any custom settings
    for key, value in updated_config.items():
        config[key] = value
    
    # Save updated config
    os.makedirs(os.path.dirname(FEE_CONFIG_FILE), exist_ok=True)
    with open(FEE_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated fee configuration: {FEE_CONFIG_FILE}")
    return config

def enhance_risk_config():
    """Enhance risk configuration with more realistic values"""
    # Load existing risk config if it exists
    if os.path.exists(RISK_CONFIG_FILE):
        with open(RISK_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with more realistic values
    updated_config = {
        "base_leverage": 20.0,
        "max_leverage": 125.0,
        "confidence_threshold": 0.65,
        "risk_percentage": 0.20,
        "max_risk_percentage": 0.30,
        "max_drawdown_threshold": 0.25,
        "max_positions": 6,
        "max_positions_per_pair": 1,
        "max_exposure_per_asset_class": 0.4,  # Max 40% in one asset class
        "stop_loss_percentage": 0.04,
        "take_profit_percentage": 0.10,
        "trailing_stop_enabled": True,
        "trailing_stop_activation": 0.03,
        "trailing_stop_distance": 0.02,
        "max_total_leverage": 500.0,
        "max_leverage_per_pair": {
            "SOL/USD": 125.0,
            "BTC/USD": 100.0,
            "ETH/USD": 90.0,
            "ADA/USD": 75.0,
            "DOT/USD": 70.0,
            "LINK/USD": 85.0
        },
        "market_regime_adjustments": {
            "volatile": {
                "leverage_adjustment": 0.7,
                "risk_adjustment": 0.8,
                "stop_loss_adjustment": 1.2,
                "take_profit_adjustment": 1.2
            },
            "trending_up": {
                "leverage_adjustment": 1.2,
                "risk_adjustment": 1.1,
                "long_bias": 0.2,
                "short_bias": -0.2
            },
            "trending_down": {
                "leverage_adjustment": 1.0,
                "risk_adjustment": 0.9,
                "long_bias": -0.2,
                "short_bias": 0.2
            },
            "ranging": {
                "leverage_adjustment": 0.9,
                "risk_adjustment": 0.9,
                "stop_loss_adjustment": 0.8
            }
        },
        "strategy_priority": {
            "Adaptive": 1,
            "ARIMA": 2
        },
        "correlation_limits": {
            "max_correlation": 0.8,
            "risk_scaling": True
        },
        "auto_risk_adjustment": True,
        "downside_protection": {
            "enabled": True,
            "drawdown_threshold": 0.15,
            "reduction_factor": 0.5,
            "recovery_threshold": 0.05
        },
        "market_conditions": {
            "high_volatility_threshold": 0.04,
            "low_liquidity_threshold": 100000
        }
    }
    
    # Merge with existing config to preserve any custom settings
    for key, value in updated_config.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            # Deep merge dictionaries
            for sub_key, sub_value in value.items():
                config[key][sub_key] = sub_value
        else:
            config[key] = value
    
    # Save updated config
    os.makedirs(os.path.dirname(RISK_CONFIG_FILE), exist_ok=True)
    with open(RISK_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Enhanced risk configuration: {RISK_CONFIG_FILE}")
    return config

def enhance_dynamic_params():
    """Enhance dynamic parameters configuration"""
    # Load existing dynamic params config if it exists
    if os.path.exists(DYNAMIC_PARAMS_CONFIG_FILE):
        with open(DYNAMIC_PARAMS_CONFIG_FILE, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with more realistic values
    updated_config = {
        "leverage_adjustment": 1.0,
        "risk_adjustment": 1.0,
        "confidence_boost": 0.05,
        "dynamic_adjustments": {
            "high_volatility": {
                "leverage_adjustment": 0.7,
                "risk_adjustment": 0.8,
                "stop_loss_adjustment": 1.2,
                "take_profit_adjustment": 1.2
            },
            "low_volatility": {
                "leverage_adjustment": 1.2,
                "risk_adjustment": 1.1,
                "stop_loss_adjustment": 0.9,
                "take_profit_adjustment": 0.9
            },
            "strong_uptrend": {
                "leverage_adjustment": 1.1,
                "risk_adjustment": 1.1,
                "long_bias": 0.1,
                "short_bias": -0.1
            },
            "strong_downtrend": {
                "leverage_adjustment": 1.0,
                "risk_adjustment": 0.9,
                "long_bias": -0.1,
                "short_bias": 0.1
            }
        },
        "pair_specific_adjustments": {
            "SOL/USD": {
                "leverage_adjustment": 1.1,
                "risk_adjustment": 1.1
            },
            "BTC/USD": {
                "leverage_adjustment": 1.0,
                "risk_adjustment": 1.0
            },
            "ETH/USD": {
                "leverage_adjustment": 1.0,
                "risk_adjustment": 1.0
            },
            "ADA/USD": {
                "leverage_adjustment": 0.9,
                "risk_adjustment": 0.9
            },
            "DOT/USD": {
                "leverage_adjustment": 0.9,
                "risk_adjustment": 0.9
            },
            "LINK/USD": {
                "leverage_adjustment": 1.05,
                "risk_adjustment": 1.05
            }
        },
        "strategy_specific_adjustments": {
            "Adaptive": {
                "leverage_adjustment": 1.05,
                "risk_adjustment": 1.0
            },
            "ARIMA": {
                "leverage_adjustment": 0.9,
                "risk_adjustment": 0.9
            }
        },
        "time_of_day_adjustments": {
            "high_volume_hours": {
                "leverage_adjustment": 1.1,
                "slippage_adjustment": 0.9
            },
            "low_volume_hours": {
                "leverage_adjustment": 0.9,
                "slippage_adjustment": 1.2
            }
        },
        "position_sizing": {
            "scale_by_volatility": True,
            "scale_by_conviction": True,
            "conviction_scale_factor": 1.5
        },
        "execution": {
            "partial_fill_threshold": 0.5,
            "entry_price_tolerance": 0.005,
            "exit_price_tolerance": 0.01
        }
    }
    
    # Merge with existing config to preserve any custom settings
    for key, value in updated_config.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            # Deep merge dictionaries
            for sub_key, sub_value in value.items():
                config[key][sub_key] = sub_value
        else:
            config[key] = value
    
    # Save updated config
    os.makedirs(os.path.dirname(DYNAMIC_PARAMS_CONFIG_FILE), exist_ok=True)
    with open(DYNAMIC_PARAMS_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Enhanced dynamic parameters configuration: {DYNAMIC_PARAMS_CONFIG_FILE}")
    return config

def update_positions_with_liquidation_prices():
    """Update existing positions with liquidation prices"""
    # Load existing positions if they exist
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, 'r') as f:
            positions = json.load(f)
    else:
        positions = []
    
    # Load fee config for maintenance margin
    if os.path.exists(FEE_CONFIG_FILE):
        with open(FEE_CONFIG_FILE, 'r') as f:
            fee_config = json.load(f)
    else:
        fee_config = {"maintenance_margin": 0.04}
    
    maintenance_margin = fee_config.get("maintenance_margin", 0.04)
    
    # Update each position with liquidation price
    for position in positions:
        entry_price = position.get("entry_price", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        # Calculate liquidation price
        if direction == "Long":
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
        else:  # Short
            liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
        
        # Add liquidation price to position
        position["liquidation_price"] = liquidation_price
        
        # Add confidence if not present
        if "confidence" not in position:
            position["confidence"] = 0.7
    
    # Save updated positions
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)
    
    logger.info(f"Updated {len(positions)} positions with liquidation prices")
    return positions

def update_trades_with_fees():
    """Update existing trades with fees"""
    # Load existing trades if they exist
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
    else:
        trades = []
    
    # Load fee config
    if os.path.exists(FEE_CONFIG_FILE):
        with open(FEE_CONFIG_FILE, 'r') as f:
            fee_config = json.load(f)
    else:
        fee_config = {"maker_fee": 0.0002, "taker_fee": 0.0005}
    
    maker_fee = fee_config.get("maker_fee", 0.0002)
    taker_fee = fee_config.get("taker_fee", 0.0005)
    
    # Update each trade with fees
    for trade in trades:
        # Skip trades that already have fees
        if "fees" in trade:
            continue
        
        trade_type = trade.get("type", "Entry")
        size = trade.get("size", 0)
        
        if trade_type == "Entry":
            price = trade.get("entry_price", 0)
            fee_rate = taker_fee
        else:  # Exit or Liquidation
            price = trade.get("exit_price", 0)
            fee_rate = taker_fee
        
        # Calculate fee
        fee = size * price * fee_rate
        
        # Add fee to trade
        trade["fees"] = fee
        trade["fee_rate"] = fee_rate
    
    # Save updated trades
    with open(TRADES_FILE, 'w') as f:
        json.dump(trades, f, indent=2)
    
    logger.info(f"Updated {len(trades)} trades with fees")
    return trades

def main():
    """Main function"""
    logger.info("Applying enhanced simulation to trading bot...")
    
    # Create directories if they don't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create and update configurations
    update_fee_config()
    enhance_risk_config()
    enhance_dynamic_params()
    create_market_impact_config()
    create_latency_config()
    create_advanced_fee_config()
    create_tier_based_fees_config()
    
    # Update positions and trades
    update_positions_with_liquidation_prices()
    update_trades_with_fees()
    
    logger.info("Enhanced simulation has been applied to the trading bot")
    logger.info("You can now run the enhanced simulation with:")
    logger.info("  python run_enhanced_simulation.py --sandbox --flash-crash --latency --stress-test")

if __name__ == "__main__":
    main()