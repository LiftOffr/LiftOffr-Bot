"""
Web App for Kraken Trading Bot

This is a Flask application to provide a web interface for the Kraken Trading Bot.
It's designed to run alongside the main trading bot.
"""

import os
from flask import Flask, render_template, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "trading-bot-secret-key")

# Import the routes after app is created
from web_server import index, api_status

# Register the routes
app.route("/")(index)
app.route("/api/status")(api_status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)