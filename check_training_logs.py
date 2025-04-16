#!/usr/bin/env python3
"""
Check Training Logs

This script checks the training logs to see if the training process is 
making progress, and displays a summary of the training status including
any errors encountered.

Usage:
    python check_training_logs.py [--lines N]
"""

import argparse
import os
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Constants
LOG_FILES = [
    'logs/training_output.log',
    'training_metrics.log',
    'training.log',
    'activate_models.log'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Check training logs')
    parser.add_argument('--lines', type=int, default=100,
                      help='Number of lines to read from the end of each log file')
    return parser.parse_args()


def get_last_n_lines(file_path: str, n: int = 100) -> List[str]:
    """
    Get the last n lines from a file
    
    Args:
        file_path: Path to the file
        n: Number of lines to read
        
    Returns:
        List of lines
    """
    try:
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        return lines[-n:]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def parse_log_line(line: str) -> Tuple[Optional[datetime], Optional[str], str]:
    """
    Parse a log line into timestamp, level, and message
    
    Args:
        line: Log line
        
    Returns:
        Tuple of (timestamp, level, message)
    """
    # Try to parse timestamp and log level
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
    level_pattern = r'\[(INFO|WARNING|ERROR|DEBUG)\]'
    
    timestamp_match = re.search(timestamp_pattern, line)
    level_match = re.search(level_pattern, line)
    
    timestamp = None
    if timestamp_match:
        try:
            timestamp_str = timestamp_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        except:
            pass
    
    level = level_match.group(1) if level_match else None
    
    # Extract message
    message = line
    if timestamp_match:
        message = message[timestamp_match.end():]
    if level_match:
        message = message[level_match.end():]
    
    message = message.strip()
    
    return timestamp, level, message


def extract_training_progress(log_lines: List[str]) -> Dict:
    """
    Extract training progress information from log lines
    
    Args:
        log_lines: List of log lines
        
    Returns:
        Dictionary with progress information
    """
    progress = {
        'pairs_processed': set(),
        'models_trained': defaultdict(list),
        'errors': [],
        'last_action': None,
        'last_timestamp': None
    }
    
    # Extract progress information
    for line in log_lines:
        timestamp, level, message = parse_log_line(line)
        
        if timestamp and (not progress['last_timestamp'] or timestamp > progress['last_timestamp']):
            progress['last_timestamp'] = timestamp
        
        # Check for pair processing
        pair_match = re.search(r'Processing pair (\d+)/\d+: ([A-Z]+/USD)', message)
        if pair_match:
            pair = pair_match.group(2)
            progress['pairs_processed'].add(pair)
            progress['last_action'] = f"Processing {pair}"
        
        # Check for model training
        model_match = re.search(r'Training (entry|exit|cancel|sizing|ensemble) model for ([A-Z]+/USD) \((\w+)\)', message)
        if model_match:
            model_type = model_match.group(1)
            pair = model_match.group(2)
            timeframe = model_match.group(3)
            progress['models_trained'][(pair, timeframe)].append(model_type)
            progress['last_action'] = f"Training {model_type} model for {pair} ({timeframe})"
        
        # Check for errors
        if level == 'ERROR':
            progress['errors'].append(message)
    
    return progress


def print_training_summary(progress: Dict):
    """
    Print a summary of the training progress
    
    Args:
        progress: Dictionary with progress information
    """
    print("\n=== Training Progress Summary ===\n")
    
    # Print last action and timestamp
    if progress['last_timestamp']:
        print(f"Last activity: {progress['last_action']} at {progress['last_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate time since last activity
        time_since = datetime.now() - progress['last_timestamp']
        hours = time_since.seconds // 3600
        minutes = (time_since.seconds % 3600) // 60
        seconds = time_since.seconds % 60
        
        print(f"Time since last activity: {hours}h {minutes}m {seconds}s")
    
    # Print pairs processed
    pairs_processed = list(progress['pairs_processed'])
    if pairs_processed:
        print(f"\nPairs processed ({len(pairs_processed)}):")
        for pair in sorted(pairs_processed):
            print(f"  {pair}")
    
    # Print models trained
    if progress['models_trained']:
        print("\nModels trained:")
        
        # Count models by pair and timeframe
        for (pair, timeframe), models in sorted(progress['models_trained'].items()):
            model_set = set(models)
            model_count = len(model_set)
            total_models = 5  # entry, exit, cancel, sizing, ensemble
            
            # Format model list
            model_status = []
            for model_type in ['entry', 'exit', 'cancel', 'sizing', 'ensemble']:
                status = "✓" if model_type in model_set else " "
                model_status.append(f"{status} {model_type}")
            
            status_str = ", ".join(model_status)
            print(f"  {pair} ({timeframe}): {model_count}/{total_models} models - {status_str}")
    
    # Print errors
    if progress['errors']:
        print("\nRecent errors:")
        
        # Count error occurrences
        error_counts = Counter(progress['errors'])
        
        # Print top 5 errors
        for error, count in error_counts.most_common(5):
            print(f"  [{count}x] {error}")
    
    print("\nOverall progress:")
    pairs_total = 10  # Total number of pairs
    pairs_progress = len(progress['pairs_processed']) / pairs_total if pairs_total > 0 else 0
    print(f"  Pairs: {len(progress['pairs_processed'])}/{pairs_total} ({pairs_progress:.1%})")
    
    models_completed = sum(1 for models in progress['models_trained'].values() if len(set(models)) == 5)
    models_total = pairs_total * 6  # 6 timeframes per pair
    models_progress = models_completed / models_total if models_total > 0 else 0
    print(f"  Complete model sets: {models_completed}/{models_total} ({models_progress:.1%})")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Read log files
    all_lines = []
    for log_file in LOG_FILES:
        lines = get_last_n_lines(log_file, args.lines)
        all_lines.extend(lines)
    
    if not all_lines:
        print("No log files found or all log files are empty")
        return
    
    # Sort lines by timestamp if possible
    sorted_lines = []
    for line in all_lines:
        timestamp, _, _ = parse_log_line(line)
        sorted_lines.append((timestamp, line))
    
    # Sort by timestamp, keeping lines without timestamps at the beginning
    sorted_lines.sort(key=lambda x: x[0] if x[0] else datetime.min)
    
    # Extract training progress
    progress = extract_training_progress([line for _, line in sorted_lines])
    
    # Print summary
    print_training_summary(progress)


if __name__ == "__main__":
    main()