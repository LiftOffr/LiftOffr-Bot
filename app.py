import os
import logging
import sys
import argparse
from flask import Flask, render_template, request, jsonify
from kraken_api import KrakenAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "development-secret-key")

# Global bot manager instance
bot_manager = None

@app.route('/')
def index():
    """
    Main page for trading bot web interface
    """
    return """
    <!DOCTYPE html>
    <html data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kraken Trading Bot</title>
        <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .container { max-width: 1200px; }
            .stats-card { min-height: 160px; }
            .chart-container { height: 400px; }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <div class="row mb-4">
                <div class="col">
                    <h1>Kraken Multi-Strategy Trading Bot</h1>
                    <p class="lead">Advanced algorithmic trading system for cryptocurrency markets</p>
                </div>
            </div>
            
            <div class="row mb-4">
                <!-- Market Stats Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Market Data</div>
                        <div class="card-body">
                            <h5 class="card-title" id="current-pair">Trading Pair: <span id="trading-pair">--</span></h5>
                            <p class="card-text">Current Price: <span id="current-price">--</span></p>
                            <p class="card-text">24h Change: <span id="price-change">--</span></p>
                            <p class="card-text">24h Volume: <span id="volume">--</span></p>
                        </div>
                    </div>
                </div>
                
                <!-- Bot Stats Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Bot Status</div>
                        <div class="card-body">
                            <h5 class="card-title">Active Strategies: <span id="strategy-count">--</span></h5>
                            <div id="strategy-list">Loading strategies...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Performance Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Performance</div>
                        <div class="card-body">
                            <h5 class="card-title">Portfolio: <span id="portfolio-value">--</span></h5>
                            <p class="card-text">Total Profit: <span id="total-profit">--</span></p>
                            <p class="card-text">ROI: <span id="roi-percent">--</span></p>
                            <p class="card-text">Completed Trades: <span id="trade-count">--</span></p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Price Chart -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">Price Chart</div>
                        <div class="card-body chart-container">
                            <canvas id="price-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Active Strategies -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">Active Trading Strategies</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Strategy</th>
                                            <th>Pair</th>
                                            <th>Position</th>
                                            <th>Current Price</th>
                                            <th>Entry Price</th>
                                            <th>Profit/Loss</th>
                                        </tr>
                                    </thead>
                                    <tbody id="strategies-table">
                                        <tr><td colspan="6" class="text-center">Loading strategies...</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Status Messages -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">Status Messages</div>
                        <div class="card-body">
                            <div id="status-messages" class="small" style="max-height: 200px; overflow-y: scroll;">
                                <div class="text-muted">Waiting for status updates...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Set initial values
                document.getElementById('trading-pair').innerText = 'Loading...';
                document.getElementById('strategy-count').innerText = 'Loading...';
                
                // Function to update UI with bot status
                function updateBotStatus() {
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                console.error('Error fetching bot status:', data.error);
                                return;
                            }
                            
                            // Update portfolio stats
                            document.getElementById('portfolio-value').innerText = '$' + data.portfolio_value.toFixed(2);
                            document.getElementById('total-profit').innerText = '$' + data.total_profit.toFixed(2);
                            document.getElementById('roi-percent').innerText = data.total_profit_percent.toFixed(2) + '%';
                            document.getElementById('trade-count').innerText = data.trade_count;
                            
                            // Update strategies table
                            const strategiesTable = document.getElementById('strategies-table');
                            let tableHTML = '';
                            
                            const bots = data.bots || {};
                            const botIds = Object.keys(bots);
                            document.getElementById('strategy-count').innerText = botIds.length;
                            
                            if (botIds.length === 0) {
                                tableHTML = '<tr><td colspan="6" class="text-center">No active strategies</td></tr>';
                            } else {
                                for (const botId of botIds) {
                                    const bot = bots[botId];
                                    const position = bot.position || 'None';
                                    const currentPrice = bot.current_price ? '$' + bot.current_price.toFixed(2) : '--';
                                    const entryPrice = bot.entry_price ? '$' + bot.entry_price.toFixed(2) : '--';
                                    
                                    let profitLoss = '--';
                                    if (bot.position && bot.current_price && bot.entry_price) {
                                        const diff = bot.position === 'long' 
                                            ? bot.current_price - bot.entry_price 
                                            : bot.entry_price - bot.current_price;
                                        const percent = (diff / bot.entry_price) * 100;
                                        profitLoss = `${diff.toFixed(2)} (${percent.toFixed(2)}%)`;
                                    }
                                    
                                    tableHTML += `
                                    <tr>
                                        <td>${bot.strategy_type}</td>
                                        <td>${bot.trading_pair}</td>
                                        <td>${position}</td>
                                        <td>${currentPrice}</td>
                                        <td>${entryPrice}</td>
                                        <td>${profitLoss}</td>
                                    </tr>
                                    `;
                                }
                            }
                            
                            strategiesTable.innerHTML = tableHTML;
                            
                            // Update strategy list in stats card
                            let strategyListHTML = '';
                            for (const botId of botIds) {
                                const bot = bots[botId];
                                strategyListHTML += `<p>${bot.strategy_type} on ${bot.trading_pair}</p>`;
                            }
                            document.getElementById('strategy-list').innerHTML = strategyListHTML;
                            
                            // Add a status message
                            const statusMessages = document.getElementById('status-messages');
                            const timestamp = new Date().toLocaleTimeString();
                            statusMessages.innerHTML = `<div>${timestamp} - Status updated: ${botIds.length} active strategies</div>` 
                                + statusMessages.innerHTML;
                        })
                        .catch(error => {
                            console.error('Error fetching bot status:', error);
                        });
                }
                
                // Initialize price chart (would be updated via API)
                const ctx = document.getElementById('price-chart').getContext('2d');
                const priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Loading...'],
                        datasets: [{
                            label: 'Price',
                            data: [0],
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
                
                // Update bot status immediately and then every 5 seconds
                updateBotStatus();
                setInterval(updateBotStatus, 5000);
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def api_status():
    """
    API endpoint for getting bot status
    """
    global bot_manager
    
    # Check if bot_manager is set directly
    if bot_manager:
        return jsonify(bot_manager.get_status())
    
    # Check if bot_manager is set in app config
    if 'BOT_MANAGER' in app.config and app.config['BOT_MANAGER']:
        return jsonify(app.config['BOT_MANAGER'].get_status())
        
    return jsonify({'error': 'Bot manager not initialized'})

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
        from config import TRADING_PAIR
        pair = request.args.get('pair', TRADING_PAIR)
        ticker = api.get_ticker([pair])
        return jsonify({"success": True, "data": ticker})
    except Exception as e:
        logger.error(f"Error getting ticker: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    from config import TRADING_PAIR, TRADE_QUANTITY, STRATEGY_TYPE
    return jsonify({
        "tradingPair": TRADING_PAIR,
        "tradeQuantity": TRADE_QUANTITY,
        "strategyType": STRATEGY_TYPE
    })

def run_app_with_bot_manager(manager):
    """
    Run the Flask app with a bot manager
    
    Args:
        manager: Bot manager instance
    """
    global bot_manager
    bot_manager = manager
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # When running directly, just start the web server
    # The bot manager will need to be provided separately
    app.run(host='0.0.0.0', port=5000, debug=True)