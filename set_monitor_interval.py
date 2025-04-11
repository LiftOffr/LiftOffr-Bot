#!/usr/bin/env python3
"""
Set Monitor Interval Script

This script allows you to set the update interval for the portfolio monitor.
The interval value is stored in the .env file.

Usage:
    python set_monitor_interval.py [interval_in_seconds]
    
Example:
    python set_monitor_interval.py 30  # Set update interval to 30 seconds
"""

import os
import sys
import re
from pathlib import Path

# Define constants
MIN_INTERVAL = 10  # seconds
MAX_INTERVAL = 3600  # 1 hour
DEFAULT_INTERVAL = 60  # 1 minute
ENV_FILE = ".env"
ENV_VAR_NAME = "PORTFOLIO_MONITOR_INTERVAL"

def load_dotenv(env_file='.env'):
    """Load environment variables from .env file"""
    if not os.path.exists(env_file):
        return {}
        
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip().strip('"\'')
            
    return env_vars

def save_dotenv(env_vars, env_file='.env'):
    """Save environment variables to .env file, preserving comments and formatting"""
    if not os.path.exists(env_file):
        # Create new file with just the variable we want to set
        with open(env_file, 'w') as f:
            f.write(f"{ENV_VAR_NAME}={env_vars[ENV_VAR_NAME]}\n")
        return
        
    # Read existing file
    with open(env_file, 'r') as f:
        lines = f.readlines()
        
    # Check if the variable exists
    var_pattern = re.compile(f"^{ENV_VAR_NAME}=.*$")
    found = False
    
    for i, line in enumerate(lines):
        if var_pattern.match(line.strip()):
            lines[i] = f"{ENV_VAR_NAME}={env_vars[ENV_VAR_NAME]}\n"
            found = True
            break
            
    if not found:
        # Add to the end
        lines.append(f"{ENV_VAR_NAME}={env_vars[ENV_VAR_NAME]}\n")
        
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)

def set_interval(interval):
    """Set the portfolio monitor interval"""
    # Validate interval
    try:
        interval = int(interval)
        if interval < MIN_INTERVAL:
            print(f"Interval too small. Using minimum value: {MIN_INTERVAL} seconds.")
            interval = MIN_INTERVAL
        elif interval > MAX_INTERVAL:
            print(f"Interval too large. Using maximum value: {MAX_INTERVAL} seconds.")
            interval = MAX_INTERVAL
    except ValueError:
        print(f"Invalid interval. Using default: {DEFAULT_INTERVAL} seconds.")
        interval = DEFAULT_INTERVAL
        
    # Load environment variables
    env_vars = load_dotenv(ENV_FILE)
    
    # Update the interval
    env_vars[ENV_VAR_NAME] = str(interval)
    
    # Save back to .env file
    save_dotenv(env_vars, ENV_FILE)
    
    print(f"Portfolio monitor interval set to {interval} seconds.")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        # No argument provided, show current setting
        env_vars = load_dotenv(ENV_FILE)
        interval = env_vars.get(ENV_VAR_NAME, DEFAULT_INTERVAL)
        print(f"Current portfolio monitor interval: {interval} seconds")
        print(f"Usage: {sys.argv[0]} [interval_in_seconds]")
        print(f"Valid range: {MIN_INTERVAL}-{MAX_INTERVAL} seconds")
        return
        
    # Get interval from command line
    interval = sys.argv[1]
    set_interval(interval)

if __name__ == "__main__":
    main()