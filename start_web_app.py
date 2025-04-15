#!/usr/bin/env python3
"""
Start Web App

This script starts the Flask web application for the Kraken Trading Bot.
"""

import os
from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)