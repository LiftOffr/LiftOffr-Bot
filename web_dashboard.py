"""
Simplified web dashboard for the Kraken trading bot.
This file is used to start just the web server without the bot management components.
"""
import os
import logging
from flask import Flask, render_template, jsonify
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting web dashboard for Kraken trading bot")
    
    # Run Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)