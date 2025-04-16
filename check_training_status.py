#!/usr/bin/env python3
"""
Check Training Status for Hybrid Models

This script checks the status of hybrid model training and displays:
1. Available model files
2. ML configuration settings
3. Risk management parameters
4. Architecture overview for each pair

Usage:
    python check_training_status.py
"""

import os
import json
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
DATA_DIR = "data"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
TRADING_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

def load_ml_config():
    """Load ML configuration"""
    try:
        if os.path.exists(ML_CONFIG_PATH):
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded ML configuration with {len(config.get('models', {}))} models")
            return config
        else:
            logger.warning(f"ML configuration file not found: {ML_CONFIG_PATH}")
            return {"models": {}, "global_settings": {}}
    except Exception as e:
        logger.error(f"Error loading ML configuration: {e}")
        return {"models": {}, "global_settings": {}}

def load_portfolio():
    """Load portfolio data"""
    try:
        if os.path.exists(PORTFOLIO_PATH):
            with open(PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            logger.info(f"Loaded portfolio with balance: ${portfolio.get('balance', 0):.2f}")
            return portfolio
        else:
            logger.warning(f"Portfolio file not found: {PORTFOLIO_PATH}")
            return {"balance": 0, "initial_balance": 0}
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return {"balance": 0, "initial_balance": 0}

def load_positions():
    """Load positions data"""
    try:
        if os.path.exists(POSITIONS_PATH):
            with open(POSITIONS_PATH, 'r') as f:
                positions = json.load(f)
            logger.info(f"Loaded {len(positions)} positions")
            return positions
        else:
            logger.warning(f"Positions file not found: {POSITIONS_PATH}")
            return {}
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return {}

def check_model_files():
    """Check for available model files"""
    if not os.path.exists(MODEL_WEIGHTS_DIR):
        logger.warning(f"Model weights directory not found: {MODEL_WEIGHTS_DIR}")
        return []
    
    model_files = []
    for file in os.listdir(MODEL_WEIGHTS_DIR):
        if file.endswith(".h5"):
            model_files.append(file)
    
    return model_files

def check_model_results():
    """Check for model training results"""
    if not os.path.exists(RESULTS_DIR):
        logger.warning(f"Results directory not found: {RESULTS_DIR}")
        return []
    
    result_files = []
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".txt") or file.endswith(".png"):
            result_files.append(file)
    
    return result_files

def format_model_status(pair, config, model_files):
    """Format status of a model for a particular pair"""
    # Get model configuration for the pair
    model_config = config.get("models", {}).get(pair, {})
    
    # Check if model file exists
    model_path = model_config.get("model_path", "")
    model_filename = os.path.basename(model_path) if model_path else ""
    model_exists = model_filename in model_files
    
    # Format model status
    status = "AVAILABLE" if model_exists else "NOT AVAILABLE"
    model_type = model_config.get("model_type", "N/A")
    accuracy = model_config.get("accuracy", 0)
    win_rate = model_config.get("win_rate", 0)
    
    # Format status string
    status_str = f"{pair}:\n"
    status_str += f"  Status: {status}\n"
    status_str += f"  Model Type: {model_type}\n"
    status_str += f"  Model Path: {model_path}\n"
    status_str += f"  Accuracy: {accuracy:.4f}\n" if accuracy else "  Accuracy: N/A\n"
    status_str += f"  Win Rate: {win_rate:.4f}\n" if win_rate else "  Win Rate: N/A\n"
    
    return status_str, model_exists

def print_hybrid_model_architecture():
    """Print overview of the hybrid model architecture"""
    architecture = "\n" + "=" * 80 + "\n"
    architecture += "HYBRID MODEL ARCHITECTURE OVERVIEW\n"
    architecture += "=" * 80 + "\n\n"
    
    architecture += "The implemented architecture combines multiple model types:\n\n"
    
    architecture += "1. CNN BRANCH - Local Price Pattern Recognition\n"
    architecture += "   ├── Conv1D Layers: Capture local patterns in price data\n"
    architecture += "   ├── Batch Normalization: Normalize activations\n"
    architecture += "   ├── Max Pooling: Extract most important features\n"
    architecture += "   └── Dense Layers: Process extracted features\n\n"
    
    architecture += "2. LSTM BRANCH - Long-Term Sequential Dependencies\n"
    architecture += "   ├── LSTM Layers: Capture long-term patterns and sequences\n"
    architecture += "   ├── Dropout: Prevent overfitting\n"
    architecture += "   └── Dense Layers: Process sequence outputs\n\n"
    
    architecture += "3. TCN BRANCH - Temporal Convolutional Network\n"
    architecture += "   ├── Dilated Causal Convolutions: See wider time horizons\n"
    architecture += "   ├── Residual Connections: Help with gradient flow\n"
    architecture += "   └── Temporal Modeling: Capture complex time dependencies\n\n"
    
    architecture += "4. ATTENTION MECHANISMS\n"
    architecture += "   ├── Self-Attention: Learn relationships between time steps\n"
    architecture += "   ├── Multi-Head Attention: Multiple attention perspectives\n"
    architecture += "   └── Highlight important parts of the sequence\n\n"
    
    architecture += "5. META-LEARNER - Combines All Branches\n"
    architecture += "   ├── Concatenates outputs from all branches\n"
    architecture += "   ├── Deep neural network to optimally combine signals\n"
    architecture += "   └── Outputs final prediction with confidence\n\n"
    
    architecture += "6. DYNAMIC RISK MANAGEMENT\n"
    architecture += "   ├── Maximum portfolio risk: 25%\n"
    architecture += "   ├── Dynamic leverage: 5x - 75x based on confidence\n"
    architecture += "   ├── Dynamic position sizing\n"
    architecture += "   └── Adjusts for market volatility\n"
    
    print(architecture)

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("HYBRID MODEL TRAINING STATUS")
    print("=" * 80 + "\n")
    
    # Check model files
    model_files = check_model_files()
    print(f"Model files found: {len(model_files)}")
    for i, file in enumerate(model_files):
        print(f"  {i+1}. {file}")
    print()
    
    # Check result files
    result_files = check_model_results()
    print(f"Result files found: {len(result_files)}")
    for i, file in enumerate(result_files[:5]):  # Show only first 5 files
        print(f"  {i+1}. {file}")
    if len(result_files) > 5:
        print(f"  ... and {len(result_files) - 5} more files")
    print()
    
    # Load ML configuration
    config = load_ml_config()
    
    # Print global settings
    global_settings = config.get("global_settings", {})
    print("Global Settings:")
    print(f"  Maximum Portfolio Risk: {global_settings.get('max_portfolio_risk', 0.25):.2%}")
    print(f"  Base Leverage: {global_settings.get('base_leverage', 5.0)}x")
    print(f"  Max Leverage: {global_settings.get('max_leverage', 75.0)}x")
    print(f"  Confidence Threshold: {global_settings.get('confidence_threshold', 0.65)}")
    print(f"  Risk Percentage: {global_settings.get('risk_percentage', 0.20):.2%}")
    print()
    
    # Load portfolio and positions
    portfolio = load_portfolio()
    positions = load_positions()
    
    # Print portfolio information
    print("Portfolio Information:")
    print(f"  Balance: ${portfolio.get('balance', 0):.2f}")
    print(f"  Initial Balance: ${portfolio.get('initial_balance', 0):.2f}")
    profit_loss = portfolio.get('balance', 0) - portfolio.get('initial_balance', 0)
    profit_loss_percentage = (profit_loss / portfolio.get('initial_balance', 1)) * 100
    print(f"  Profit/Loss: ${profit_loss:.2f} ({profit_loss_percentage:.2f}%)")
    print(f"  Open Positions: {len(positions)}")
    print()
    
    # Print model status for each pair
    print("Model Status by Trading Pair:")
    available_models = 0
    for pair in TRADING_PAIRS:
        status_str, model_exists = format_model_status(pair, config, model_files)
        print(status_str)
        if model_exists:
            available_models += 1
    
    # Print summary
    print("\nTraining Progress Summary:")
    print(f"  Trading Pairs: {len(TRADING_PAIRS)}")
    print(f"  Models Available: {available_models}/{len(TRADING_PAIRS)}")
    print(f"  Training Progress: {available_models/len(TRADING_PAIRS):.0%} complete")
    print()
    
    # Print hybrid model architecture
    print_hybrid_model_architecture()
    
    print("\n" + "=" * 80)
    print("TRAINING STATUS CHECK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error checking training status: {e}")
        import traceback
        traceback.print_exc()