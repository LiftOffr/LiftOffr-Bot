#!/usr/bin/env python3
import os
import sys
from app import app

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"Error starting web application: {e}")
        sys.exit(1)