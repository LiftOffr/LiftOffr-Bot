#!/usr/bin/env python3
"""
Simple starter script for the dashboard on port 8080
"""
from dashboard_app import app

if __name__ == "__main__":
    print("Starting Trading Bot Dashboard on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=True)