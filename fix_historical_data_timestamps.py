#!/usr/bin/env python3
"""
Script to fix the timestamps in historical data files
"""

import os
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = "historical_data"

def fix_timestamps(file_path):
    """
    Fix timestamps in a CSV file
    
    Args:
        file_path (str): Path to the CSV file
    """
    print(f"Processing {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create properly formatted dates (starting from recent dates)
    # Since we're getting historical data for the past 2 years
    end_date = datetime.now()
    
    # Determine interval based on file name
    if "_1m" in file_path:
        interval = timedelta(minutes=1)
    elif "_5m" in file_path:
        interval = timedelta(minutes=5)
    elif "_15m" in file_path:
        interval = timedelta(minutes=15)
    elif "_30m" in file_path:
        interval = timedelta(minutes=30)
    elif "_1h" in file_path:
        interval = timedelta(hours=1)
    elif "_4h" in file_path:
        interval = timedelta(hours=4)
    elif "_1d" in file_path:
        interval = timedelta(days=1)
    else:
        interval = timedelta(hours=1)  # Default to 1 hour
    
    # Generate timestamps in reverse order (newest first)
    timestamps = []
    current_date = end_date
    for _ in range(len(df)):
        timestamps.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        current_date -= interval
    
    # Reverse the list to have oldest first (to match the data order)
    timestamps.reverse()
    
    # Update timestamps in the dataframe
    df["timestamp"] = timestamps
    
    # Save the updated dataframe
    df.to_csv(file_path, index=False)
    print(f"Fixed timestamps in {file_path}")

def main():
    """Fix timestamps in all historical data files"""
    # Make sure the historical data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist")
        return
    
    # Process all CSV files in the directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            fix_timestamps(file_path)
    
    print("All timestamps fixed successfully")

if __name__ == "__main__":
    main()