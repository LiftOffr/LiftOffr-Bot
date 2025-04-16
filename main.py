#!/usr/bin/env python3
"""
ML Trading Dashboard main entry point

This file serves as the main entry point for the Flask web dashboard,
allowing it to be easily started by the Replit workflow system.
"""

from dashboard_app import app

# This allows gunicorn to import the app
# Do not remove this line

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)