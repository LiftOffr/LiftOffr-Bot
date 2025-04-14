import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "kraken_trading_bot_secret")

# Set up portfolio status data path
PORTFOLIO_STATUS_PATH = "portfolio_status.json"
TRADES_LOG_PATH = "trades_log.json"
CONFIG_PATH = "config/ml_config.json"

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """API endpoint to get the current bot status"""
    try:
        status = {}
        
        # Try to read portfolio status
        if os.path.exists(PORTFOLIO_STATUS_PATH):
            with open(PORTFOLIO_STATUS_PATH, 'r') as f:
                status['portfolio'] = json.load(f)
        else:
            status['portfolio'] = {
                "initial_value": 20000.00,
                "current_value": 20000.00,
                "profit_loss": 0.00,
                "profit_loss_percent": 0.00,
                "allocated_capital": 0.00,
                "available_funds": 20000.00,
                "total_trades": 0,
                "strategies": []
            }
        
        # Try to read trades log
        if os.path.exists(TRADES_LOG_PATH):
            with open(TRADES_LOG_PATH, 'r') as f:
                trades = json.load(f)
                status['recent_trades'] = trades[-20:] if len(trades) > 20 else trades
                
                # Calculate win rate
                if trades:
                    profitable_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
                    status['win_rate'] = (profitable_trades / len(trades)) * 100 if len(trades) > 0 else 0
                else:
                    status['win_rate'] = 0
        else:
            status['recent_trades'] = []
            status['win_rate'] = 0
        
        # Try to read ML config
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                status['ml_config'] = json.load(f)
        else:
            status['ml_config'] = {}
        
        # Add system status
        status['system'] = {
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "running": True,
            "sandbox_mode": os.environ.get("USE_SANDBOX", "True").lower() in ("true", "1", "t")
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/trades')
def get_trades():
    """API endpoint to get all trades"""
    try:
        if os.path.exists(TRADES_LOG_PATH):
            with open(TRADES_LOG_PATH, 'r') as f:
                trades = json.load(f)
                return jsonify(trades)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """API endpoint to get performance metrics"""
    try:
        if os.path.exists(TRADES_LOG_PATH):
            with open(TRADES_LOG_PATH, 'r') as f:
                trades = json.load(f)
                
            # Calculate performance metrics
            if trades:
                # Simple implementation without pandas
                # Group trades by day
                daily_performance = {}
                for trade in trades:
                    if 'timestamp' in trade and 'profit_loss' in trade:
                        # Extract just the date part from the timestamp
                        date = trade['timestamp'].split('T')[0] if 'T' in trade['timestamp'] else trade['timestamp'].split(' ')[0]
                        
                        if date not in daily_performance:
                            daily_performance[date] = 0
                        
                        daily_performance[date] += trade.get('profit_loss', 0)
                
                # Group trades by strategy
                strategy_performance = {}
                for trade in trades:
                    if 'strategy' in trade and 'profit_loss' in trade:
                        strategy = trade['strategy']
                        
                        if strategy not in strategy_performance:
                            strategy_performance[strategy] = {
                                'strategy': strategy,
                                'sum': 0,
                                'count': 0,
                                'avg_profit_per_trade': 0
                            }
                        
                        strategy_performance[strategy]['sum'] += trade.get('profit_loss', 0)
                        strategy_performance[strategy]['count'] += 1
                
                # Calculate average profit per trade for each strategy
                for strategy in strategy_performance:
                    if strategy_performance[strategy]['count'] > 0:
                        strategy_performance[strategy]['avg_profit_per_trade'] = (
                            strategy_performance[strategy]['sum'] / strategy_performance[strategy]['count']
                        )
                
                return jsonify({
                    'daily_performance': daily_performance,
                    'strategy_performance': list(strategy_performance.values())
                })
            else:
                return jsonify({
                    'daily_performance': {},
                    'strategy_performance': []
                })
        else:
            return jsonify({
                'daily_performance': {},
                'strategy_performance': []
            })
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-config', methods=['POST'])
def update_config():
    """API endpoint to update the configuration"""
    try:
        data = request.json
        
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            
            # Update configuration values
            for key, value in data.items():
                if key in config:
                    config[key] = value
            
            # Save updated configuration
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            
            return jsonify({"success": True, "message": "Configuration updated"})
        else:
            return jsonify({"error": "Configuration file not found"}), 404
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5001))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Error starting web application: {e}")