import os
import logging
import time
from flask import Flask, jsonify, render_template
from bot_manager import BotManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Create a mock BotManager for when the real bot isn't running
bot_manager = BotManager()
try:
    # Try to initialize with a default strategy for web display
    bot_manager.add_bot('adaptive', 'SOLUSD')
except Exception as e:
    logger.warning(f"Could not initialize default strategy: {e}")

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

            <!-- Portfolio Monitor Section -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">Portfolio Monitor</div>
                        <div class="card-body">
                            <p>The trading bot has a built-in portfolio monitor that displays the status of your portfolio at regular intervals (every 30 seconds by default).</p>
                            <p>To adjust the interval, use the command:</p>
                            <pre class="bg-dark text-light p-2">python set_monitor_interval.py &lt;seconds&gt;</pre>
                            <p>For example:</p>
                            <pre class="bg-dark text-light p-2">python set_monitor_interval.py 60  # Update every minute</pre>
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
    
    if bot_manager:
        return jsonify(bot_manager.get_status())
    else:
        return jsonify({'error': 'Bot manager not initialized'})

if __name__ == '__main__':
    # Run Flask on port 5001 to avoid conflict with the trading bot
    app.run(host='0.0.0.0', port=5001, debug=True)