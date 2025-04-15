import os
import json
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from datetime import datetime

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
    
    # Load ML configuration
    ml_config_path = f"{CONFIG_DIR}/ml_config.json"
    ml_config = load_file(ml_config_path, {})
    
    # Prepare ML accuracy data
    accuracy_data = {}
    for pair, config in ml_config.get("pairs", {}).items():
        accuracy_data[pair] = config.get("accuracy", 0)
    
    return render_template(
        'index.html',
        portfolio_value=current_value,
        starting_capital=starting_capital,
        total_return=total_return,
        positions=positions,
        trades=trades,
        risk_metrics=risk_metrics,
        accuracy_data=accuracy_data,
        portfolio_history=portfolio_history,
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

@app.route('/debug')
def debug_info():
    """Debug page showing system status"""
    # Check file existence
    file_status = {
        "portfolio_history": os.path.exists(DEFAULT_PORTFOLIO_FILE),
        "positions": os.path.exists(DEFAULT_POSITION_FILE),
        "trades": os.path.exists(DEFAULT_TRADE_HISTORY_FILE),
        "risk_metrics": os.path.exists(DEFAULT_RISK_METRICS_FILE),
        "ml_config": os.path.exists(f"{CONFIG_DIR}/ml_config.json"),
        "risk_config": os.path.exists(f"{CONFIG_DIR}/risk_config.json"),
        "integrated_risk_config": os.path.exists(f"{CONFIG_DIR}/integrated_risk_config.json"),
        "dynamic_params_config": os.path.exists(f"{CONFIG_DIR}/dynamic_params_config.json")
    }
    
    # Check model files
    model_files = {}
    for pair in ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]:
        file_path = f"{ML_MODELS_DIR}/ensemble/{pair.replace('/', '_')}_weights.json"
        model_files[pair] = os.path.exists(file_path)
    
    return render_template(
        'debug.html',
        file_status=file_status,
        model_files=model_files,
        environment=os.environ
    )

if __name__ == '__main__':
    # Create required directories
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)