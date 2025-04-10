from flask import Flask, render_template, request, jsonify
import os
import logging
from kraken_api import KrakenAPI
from config import TRADING_PAIR, TRADE_QUANTITY, STRATEGY_TYPE

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/server-time', methods=['GET'])
def get_server_time():
    try:
        api = KrakenAPI()
        server_time = api.get_server_time()
        return jsonify({"success": True, "data": server_time})
    except Exception as e:
        logger.error(f"Error getting server time: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/asset-info', methods=['GET'])
def get_asset_info():
    try:
        api = KrakenAPI()
        asset = request.args.get('asset')
        assets = [asset] if asset else None
        asset_info = api.get_asset_info(assets)
        return jsonify({"success": True, "data": asset_info})
    except Exception as e:
        logger.error(f"Error getting asset info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/ticker', methods=['GET'])
def get_ticker():
    try:
        api = KrakenAPI()
        pair = request.args.get('pair', TRADING_PAIR)
        ticker = api.get_ticker([pair])
        return jsonify({"success": True, "data": ticker})
    except Exception as e:
        logger.error(f"Error getting ticker: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "tradingPair": TRADING_PAIR,
        "tradeQuantity": TRADE_QUANTITY,
        "strategyType": STRATEGY_TYPE
    })

# This helps Gunicorn find the Flask app
app_instance = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)