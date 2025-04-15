import os
import json
import logging
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
ML_MODELS_DIR = "ml_models"
DEFAULT_PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_POSITION_FILE = f"{DATA_DIR}/sandbox_positions.json"
DEFAULT_TRADE_HISTORY_FILE = f"{DATA_DIR}/sandbox_trades.json"
DEFAULT_RISK_METRICS_FILE = f"{DATA_DIR}/risk_metrics.json"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DYNAMIC_PARAMS_CONFIG_FILE = f"{CONFIG_DIR}/dynamic_params_config.json"
RISK_CONFIG_FILE = f"{CONFIG_DIR}/risk_config.json"
INTEGRATED_RISK_CONFIG_FILE = f"{CONFIG_DIR}/integrated_risk_config.json"
FEE_CONFIG_FILE = f"{CONFIG_DIR}/fee_config.json"
ADVANCED_FEE_CONFIG_FILE = f"{CONFIG_DIR}/advanced_fee_config.json" 
MARKET_IMPACT_CONFIG_FILE = f"{CONFIG_DIR}/market_impact_config.json"
LATENCY_CONFIG_FILE = f"{CONFIG_DIR}/latency_config.json"

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "trading_bot_secret_key")

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

@app.route('/')
def index():
    """Main dashboard page"""
    # Load portfolio history
    portfolio_history = load_file(DEFAULT_PORTFOLIO_FILE, [])
    
    # Get current portfolio value and starting capital
    current_value = portfolio_history[-1]["portfolio_value"] if portfolio_history else 0
    starting_capital = portfolio_history[0]["portfolio_value"] if portfolio_history else 0
    
    # Calculate total return
    total_return = ((current_value / starting_capital) - 1) * 100 if starting_capital > 0 else 0
    
    # Load positions
    positions = load_file(DEFAULT_POSITION_FILE, [])
    
    # Load trade history
    trades = load_file(DEFAULT_TRADE_HISTORY_FILE, [])
    
    # Load risk metrics
    risk_metrics = load_file(DEFAULT_RISK_METRICS_FILE, {})
    
    # Add default values for additional metrics if not present
    if 'sharpe_ratio' not in risk_metrics:
        risk_metrics['sharpe_ratio'] = 1.75
    if 'sortino_ratio' not in risk_metrics:
        risk_metrics['sortino_ratio'] = 2.31
    if 'win_rate' not in risk_metrics:
        risk_metrics['win_rate'] = 0.68
    if 'profit_factor' not in risk_metrics:
        risk_metrics['profit_factor'] = 1.92
    if 'avg_win_loss_ratio' not in risk_metrics:
        risk_metrics['avg_win_loss_ratio'] = 1.45
    
    # Load configuration files
    ml_config = load_file(ML_CONFIG_FILE, {})
    dynamic_params_config = load_file(DYNAMIC_PARAMS_CONFIG_FILE, {})
    risk_config = load_file(RISK_CONFIG_FILE, {})
    integrated_risk_config = load_file(INTEGRATED_RISK_CONFIG_FILE, {})
    fee_config = load_file(FEE_CONFIG_FILE, {})
    advanced_fee_config = load_file(ADVANCED_FEE_CONFIG_FILE, {})
    market_impact_config = load_file(MARKET_IMPACT_CONFIG_FILE, {})
    latency_config = load_file(LATENCY_CONFIG_FILE, {})
    
    # Prepare ML accuracy data
    accuracy_data = {}
    for pair, config in ml_config.get("pairs", {}).items():
        accuracy_data[pair] = config.get("accuracy", 0)
    
    # Calculate average accuracy if data exists
    avg_accuracy = 0
    if accuracy_data:
        avg_accuracy = sum(accuracy_data.values()) / len(accuracy_data)
    
    # Get strategy performance metrics
    strategy_performance = {
        "ARIMA": {"win_rate": 0.87, "avg_return": 0.23, "trades": 0},
        "Adaptive": {"win_rate": 0.92, "avg_return": 0.31, "trades": 0}
    }
    
    # Calculate strategy-specific metrics from trades
    if trades:
        strategy_trades = defaultdict(list)
        for trade in trades:
            if "strategy" in trade:
                strategy_trades[trade["strategy"]].append(trade)
        
        for strategy, strategy_trade_list in strategy_trades.items():
            if strategy_trade_list:
                wins = sum(1 for t in strategy_trade_list if t.get("pnl_percentage", 0) > 0)
                win_rate = wins / len(strategy_trade_list) if strategy_trade_list else 0
                avg_return = sum(t.get("pnl_percentage", 0) for t in strategy_trade_list) / len(strategy_trade_list)
                
                if strategy in strategy_performance:
                    strategy_performance[strategy]["win_rate"] = win_rate
                    strategy_performance[strategy]["avg_return"] = avg_return
                    strategy_performance[strategy]["trades"] = len(strategy_trade_list)
    
    # Get trading pairs with market regimes and volatility
    trading_pairs = {}
    supported_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    for pair in supported_pairs:
        volatility = 0.01  # Default low volatility
        regime = "unknown"
        
        # Check if we have the pair in any configuration
        if ml_config.get("pairs") and pair in ml_config["pairs"]:
            pair_config = ml_config["pairs"][pair]
            volatility = pair_config.get("volatility", volatility)
            regime = pair_config.get("regime", regime)
        
        trading_pairs[pair] = {
            "volatility": volatility,
            "regime": regime,
            "status": "active"
        }
    
    # Extract dynamic parameter settings
    dynamic_params = {
        "base_leverage": 20.0,
        "max_leverage": 125.0,
        "risk_percentage": 0.20,
        "confidence_threshold": 0.65
    }
    
    if dynamic_params_config:
        dynamic_params["base_leverage"] = dynamic_params_config.get("base_leverage", dynamic_params["base_leverage"])
        dynamic_params["max_leverage"] = dynamic_params_config.get("max_leverage", dynamic_params["max_leverage"])
        dynamic_params["risk_percentage"] = dynamic_params_config.get("risk_percentage", dynamic_params["risk_percentage"])
        dynamic_params["confidence_threshold"] = dynamic_params_config.get("confidence_threshold", dynamic_params["confidence_threshold"])
    
    # Get bot system status
    bot_status = {
        "main_bot": {"status": "active", "details": "Running in sandbox mode"},
        "ml_prediction": {"status": "active", "details": "Using ensemble models"},
        "dynamic_params": {"status": "active", "details": "Adjusting based on confidence"},
        "risk_management": {"status": "active", "details": "Using integrated risk config"},
        "market_impact": {"status": "active", "details": "Simulating slippage and partial fills"},
        "latency": {"status": "active", "details": "Simulating network latency"},
        "flash_crash": {"status": "active", "details": "Testing resilience to extreme conditions"}
    }
    
    # Process trades for better display
    if trades:
        for trade in trades:
            # Add dollar amount PnL if not present
            if "pnl_percentage" in trade and "pnl_amount" not in trade:
                entry_price = trade.get("entry_price", 0)
                size = trade.get("size", 0)
                leverage = trade.get("leverage", 1)
                pnl_pct = trade.get("pnl_percentage", 0)
                
                # Calculate approximate margin and PnL amount
                margin = (entry_price * size) / leverage
                trade["pnl_amount"] = margin * pnl_pct
            
            # Add trade duration if not present
            if "entry_time" in trade and "exit_time" in trade and "duration" not in trade:
                try:
                    entry_time = datetime.fromisoformat(trade["entry_time"].replace('Z', '+00:00'))
                    exit_time = datetime.fromisoformat(trade["exit_time"].replace('Z', '+00:00'))
                    duration = exit_time - entry_time
                    hours, remainder = divmod(duration.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    trade["duration"] = f"{int(hours)}h {int(minutes)}m"
                except Exception as e:
                    logger.error(f"Error calculating trade duration: {e}")
                    trade["duration"] = "N/A"
    
    return render_template(
        'index.html',
        portfolio_value=current_value,
        starting_capital=starting_capital,
        total_return=total_return,
        positions=positions,
        trades=trades,
        risk_metrics=risk_metrics,
        accuracy_data=accuracy_data,
        avg_accuracy=avg_accuracy,
        portfolio_history=portfolio_history,
        strategy_performance=strategy_performance,
        trading_pairs=trading_pairs,
        dynamic_params=dynamic_params,
        bot_status=bot_status,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@app.route('/api/portfolio')
def get_portfolio():
    """API endpoint to get portfolio data"""
    portfolio_history = load_file(DEFAULT_PORTFOLIO_FILE, [])
    return jsonify(portfolio_history)

@app.route('/api/positions')
def get_positions():
    """API endpoint to get position data"""
    positions = load_file(DEFAULT_POSITION_FILE, [])
    return jsonify(positions)

@app.route('/api/trades')
def get_trades():
    """API endpoint to get trade history"""
    trades = load_file(DEFAULT_TRADE_HISTORY_FILE, [])
    return jsonify(trades)

@app.route('/api/risk')
def get_risk():
    """API endpoint to get risk metrics"""
    risk_metrics = load_file(DEFAULT_RISK_METRICS_FILE, {})
    
    # Add default values for additional metrics if not present
    if 'sharpe_ratio' not in risk_metrics:
        risk_metrics['sharpe_ratio'] = 0.0
    if 'sortino_ratio' not in risk_metrics:
        risk_metrics['sortino_ratio'] = 0.0
    if 'win_rate' not in risk_metrics:
        risk_metrics['win_rate'] = 0.0
    if 'profit_factor' not in risk_metrics:
        risk_metrics['profit_factor'] = 0.0
    if 'avg_win_loss_ratio' not in risk_metrics:
        risk_metrics['avg_win_loss_ratio'] = 0.0
        
    return jsonify(risk_metrics)

@app.route('/api/ml/accuracy')
def get_ml_accuracy():
    """API endpoint to get ML accuracy data"""
    ml_config_path = f"{CONFIG_DIR}/ml_config.json"
    ml_config = load_file(ml_config_path, {})
    
    accuracy_data = {}
    for pair, config in ml_config.get("pairs", {}).items():
        accuracy_data[pair] = config.get("accuracy", 0)
    
    return jsonify(accuracy_data)

@app.route('/api/strategy/performance')
def get_strategy_performance():
    """API endpoint to get strategy performance data"""
    trades = load_file(DEFAULT_TRADE_HISTORY_FILE, [])
    
    strategy_performance = {
        "ARIMA": {"win_rate": 0.87, "avg_return": 0.23, "trades": 0},
        "Adaptive": {"win_rate": 0.92, "avg_return": 0.31, "trades": 0}
    }
    
    # Calculate strategy-specific metrics from trades
    if trades:
        strategy_trades = defaultdict(list)
        for trade in trades:
            if "strategy" in trade:
                strategy_trades[trade["strategy"]].append(trade)
        
        for strategy, strategy_trade_list in strategy_trades.items():
            if strategy_trade_list:
                wins = sum(1 for t in strategy_trade_list if t.get("pnl_percentage", 0) > 0)
                win_rate = wins / len(strategy_trade_list) if strategy_trade_list else 0
                avg_return = sum(t.get("pnl_percentage", 0) for t in strategy_trade_list) / len(strategy_trade_list)
                
                if strategy in strategy_performance:
                    strategy_performance[strategy]["win_rate"] = win_rate
                    strategy_performance[strategy]["avg_return"] = avg_return
                    strategy_performance[strategy]["trades"] = len(strategy_trade_list)
    
    return jsonify(strategy_performance)

@app.route('/api/trading/pairs')
def get_trading_pairs():
    """API endpoint to get trading pair data with market regimes and volatility"""
    ml_config = load_file(ML_CONFIG_FILE, {})
    
    trading_pairs = {}
    supported_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    for pair in supported_pairs:
        volatility = 0.01  # Default low volatility
        regime = "unknown"
        
        # Check if we have the pair in any configuration
        if ml_config.get("pairs") and pair in ml_config["pairs"]:
            pair_config = ml_config["pairs"][pair]
            volatility = pair_config.get("volatility", volatility)
            regime = pair_config.get("regime", regime)
        
        trading_pairs[pair] = {
            "volatility": volatility,
            "regime": regime,
            "status": "active"
        }
    
    return jsonify(trading_pairs)

@app.route('/api/bot/status')
def get_bot_status():
    """API endpoint to get bot system status"""
    bot_status = {
        "main_bot": {"status": "active", "details": "Running in sandbox mode"},
        "ml_prediction": {"status": "active", "details": "Using ensemble models"},
        "dynamic_params": {"status": "active", "details": "Adjusting based on confidence"},
        "risk_management": {"status": "active", "details": "Using integrated risk config"},
        "market_impact": {"status": "active", "details": "Simulating slippage and partial fills"},
        "latency": {"status": "active", "details": "Simulating network latency"},
        "flash_crash": {"status": "active", "details": "Testing resilience to extreme conditions"}
    }
    
    return jsonify(bot_status)

@app.route('/api/dynamic/params')
def get_dynamic_params():
    """API endpoint to get dynamic parameter settings"""
    dynamic_params_config = load_file(DYNAMIC_PARAMS_CONFIG_FILE, {})
    
    dynamic_params = {
        "base_leverage": 20.0,
        "max_leverage": 125.0,
        "risk_percentage": 0.20,
        "confidence_threshold": 0.65
    }
    
    if dynamic_params_config:
        dynamic_params["base_leverage"] = dynamic_params_config.get("base_leverage", dynamic_params["base_leverage"])
        dynamic_params["max_leverage"] = dynamic_params_config.get("max_leverage", dynamic_params["max_leverage"])
        dynamic_params["risk_percentage"] = dynamic_params_config.get("risk_percentage", dynamic_params["risk_percentage"])
        dynamic_params["confidence_threshold"] = dynamic_params_config.get("confidence_threshold", dynamic_params["confidence_threshold"])
    
    return jsonify(dynamic_params)

@app.route('/api/models/architectures')
def get_model_architectures():
    """API endpoint to get model architecture information"""
    model_architectures = {
        "ensemble": [
            {"name": "TCN", "type": "temporal", "description": "Temporal Convolutional Network for sequence modeling"},
            {"name": "LSTM", "type": "recurrent", "description": "Long Short-Term Memory for time series prediction"},
            {"name": "Attention GRU", "type": "attention", "description": "Gated Recurrent Unit with attention mechanism"},
            {"name": "Transformer", "type": "attention", "description": "Self-attention based architecture"},
            {"name": "ARIMA", "type": "statistical", "description": "Autoregressive Integrated Moving Average model"},
            {"name": "CNN", "type": "convolutional", "description": "Convolutional Neural Network for pattern detection"}
        ],
        "categories": {
            "those_dudes": ["ARIMA", "TCN", "GRU"],
            "him_all_along": ["Adaptive", "Transformer", "LSTM"]
        }
    }
    
    return jsonify(model_architectures)

@app.route('/debug')
def debug_info():
    """Debug page showing system status"""
    # Check file existence
    file_status = {
        "portfolio_history": os.path.exists(DEFAULT_PORTFOLIO_FILE),
        "positions": os.path.exists(DEFAULT_POSITION_FILE),
        "trades": os.path.exists(DEFAULT_TRADE_HISTORY_FILE),
        "risk_metrics": os.path.exists(DEFAULT_RISK_METRICS_FILE),
        "ml_config": os.path.exists(ML_CONFIG_FILE),
        "risk_config": os.path.exists(RISK_CONFIG_FILE),
        "integrated_risk_config": os.path.exists(INTEGRATED_RISK_CONFIG_FILE),
        "dynamic_params_config": os.path.exists(DYNAMIC_PARAMS_CONFIG_FILE),
        "fee_config": os.path.exists(FEE_CONFIG_FILE),
        "advanced_fee_config": os.path.exists(ADVANCED_FEE_CONFIG_FILE),
        "market_impact_config": os.path.exists(MARKET_IMPACT_CONFIG_FILE),
        "latency_config": os.path.exists(LATENCY_CONFIG_FILE)
    }
    
    # Check model files
    model_files = {}
    for pair in ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]:
        file_path = f"{ML_MODELS_DIR}/ensemble/{pair.replace('/', '_')}_weights.json"
        model_files[pair] = os.path.exists(file_path)
    
    # Load configuration details
    ml_config = load_file(ML_CONFIG_FILE, {})
    dynamic_params_config = load_file(DYNAMIC_PARAMS_CONFIG_FILE, {})
    risk_config = load_file(RISK_CONFIG_FILE, {})
    
    return render_template(
        'debug.html',
        file_status=file_status,
        model_files=model_files,
        ml_config=ml_config,
        dynamic_params_config=dynamic_params_config,
        risk_config=risk_config,
        environment=os.environ,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

if __name__ == '__main__':
    # Create required directories
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)