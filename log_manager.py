#!/usr/bin/env python3
"""
Log Management Module for Kraken Trading Bot

Provides functionality for:
1. Log rotation and archiving
2. Log parsing and analysis
3. Storage optimization
"""

import os
import sys
import time
import logging
import gzip
import shutil
import glob
from datetime import datetime, timedelta
import argparse
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('log_manager.log'),
        logging.StreamHandler()
    ]
)

# Constants
DEFAULT_MAX_LOG_SIZE_MB = 50
DEFAULT_MAX_LOG_AGE_DAYS = 30
DEFAULT_MAX_STORAGE_GB = 2
DEFAULT_RETENTION_POLICY = 'size'  # 'size', 'time', or 'hybrid'
LOG_DIR = 'logs'
ARCHIVE_DIR = os.path.join(LOG_DIR, 'archive')

# Ensure log directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Log file patterns
LOG_PATTERNS = {
    'trading': ['integrated_strategy_log.txt', 'nohup.out'],
    'analysis': ['ml_training.log', 'analyze_integrated_logs.log'],
    'notification': ['trade_notifications.log'],
    'web': ['web_server.log'],
    'all': ['*.log', '*.txt', 'nohup.out']
}


def get_file_size_mb(filepath):
    """Get file size in megabytes"""
    return os.path.getsize(filepath) / (1024 * 1024)


def get_file_age_days(filepath):
    """Get file age in days"""
    mtime = os.path.getmtime(filepath)
    age = datetime.now() - datetime.fromtimestamp(mtime)
    return age.days


def compress_file(source_path, delete_source=True):
    """
    Compress a file using gzip
    
    Args:
        source_path (str): Path to source file
        delete_source (bool): Whether to delete the source file after compression
        
    Returns:
        str: Path to compressed file
    """
    # Create archive filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = os.path.basename(source_path)
    base_name, _ = os.path.splitext(filename)
    compressed_path = os.path.join(ARCHIVE_DIR, f"{base_name}_{timestamp}.gz")
    
    # Compress file
    with open(source_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Delete source if requested
    if delete_source:
        os.remove(source_path)
    
    logging.info(f"Compressed {source_path} to {compressed_path}")
    return compressed_path


def rotate_log(log_path, max_size_mb=DEFAULT_MAX_LOG_SIZE_MB, delete_source=True):
    """
    Rotate a log file if it exceeds the maximum size
    
    Args:
        log_path (str): Path to log file
        max_size_mb (float): Maximum log size in MB
        delete_source (bool): Whether to delete the source file after rotation
        
    Returns:
        bool: Whether rotation was performed
    """
    if not os.path.exists(log_path):
        logging.warning(f"Log file does not exist: {log_path}")
        return False
    
    # Check if log file exceeds maximum size
    if get_file_size_mb(log_path) >= max_size_mb:
        # Create a copy of the log file
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = os.path.basename(log_path)
        base_name, ext = os.path.splitext(filename)
        copy_path = os.path.join(LOG_DIR, f"{base_name}_{timestamp}{ext}")
        
        shutil.copy2(log_path, copy_path)
        
        # Clear original log file
        if delete_source:
            with open(log_path, 'w') as f:
                f.write(f"Log rotated at {datetime.now()}\n")
        
        # Compress the copy
        compress_file(copy_path, delete_source=True)
        
        logging.info(f"Rotated log file: {log_path}")
        return True
    
    return False


def clean_old_logs(max_age_days=DEFAULT_MAX_LOG_AGE_DAYS):
    """
    Clean up old log archives
    
    Args:
        max_age_days (int): Maximum age of log archives in days
        
    Returns:
        int: Number of deleted archives
    """
    deleted_count = 0
    
    # Get all compressed log files
    for pattern in ['*.gz', '*.zip']:
        for archive_path in glob.glob(os.path.join(ARCHIVE_DIR, pattern)):
            # Check file age
            if get_file_age_days(archive_path) > max_age_days:
                os.remove(archive_path)
                deleted_count += 1
                logging.info(f"Deleted old archive: {archive_path}")
    
    return deleted_count


def enforce_storage_limit(max_storage_gb=DEFAULT_MAX_STORAGE_GB):
    """
    Enforce storage limit by deleting oldest log archives
    
    Args:
        max_storage_gb (float): Maximum storage in GB
        
    Returns:
        int: Number of deleted archives
    """
    deleted_count = 0
    
    # Get all compressed log files
    archives = []
    for pattern in ['*.gz', '*.zip']:
        archives.extend(glob.glob(os.path.join(ARCHIVE_DIR, pattern)))
    
    # Sort by modification time (oldest first)
    archives.sort(key=os.path.getmtime)
    
    # Calculate total size
    total_size_gb = sum(os.path.getsize(path) for path in archives) / (1024 * 1024 * 1024)
    
    # Delete oldest archives until under the limit
    while total_size_gb > max_storage_gb and archives:
        oldest_archive = archives.pop(0)
        size_gb = os.path.getsize(oldest_archive) / (1024 * 1024 * 1024)
        os.remove(oldest_archive)
        total_size_gb -= size_gb
        deleted_count += 1
        logging.info(f"Deleted oldest archive to enforce storage limit: {oldest_archive}")
    
    return deleted_count


def rotate_all_logs(log_type='all', max_size_mb=DEFAULT_MAX_LOG_SIZE_MB):
    """
    Rotate all logs of specified type
    
    Args:
        log_type (str): Type of logs to rotate
        max_size_mb (float): Maximum log size in MB
        
    Returns:
        int: Number of rotated logs
    """
    rotated_count = 0
    
    patterns = LOG_PATTERNS.get(log_type, LOG_PATTERNS['all'])
    
    for pattern in patterns:
        # Look in current directory and LOG_DIR
        for log_path in glob.glob(pattern) + glob.glob(os.path.join(LOG_DIR, pattern)):
            if rotate_log(log_path, max_size_mb=max_size_mb):
                rotated_count += 1
    
    return rotated_count


def parse_log_section(log_path, section_marker, max_entries=100):
    """
    Parse sections of a log file based on markers
    
    Args:
        log_path (str): Path to log file
        section_marker (str): Section marker (e.g., '【ANALYSIS】')
        max_entries (int): Maximum number of entries to return
        
    Returns:
        list: Parsed log entries
    """
    if not os.path.exists(log_path):
        logging.warning(f"Log file does not exist: {log_path}")
        return []
    
    entries = []
    
    with open(log_path, 'r') as f:
        for line in f:
            if section_marker in line:
                entries.append(line.strip())
                if len(entries) >= max_entries:
                    break
    
    return entries


def setup_automatic_rotation(cron=False):
    """
    Set up automatic log rotation
    
    Args:
        cron (bool): Whether to provide instructions for cron job
        
    Returns:
        None
    """
    if cron:
        # Print cron job instructions
        script_path = os.path.abspath(__file__)
        print("\nTo set up automatic log rotation with cron, add the following line to your crontab:")
        print(f"0 0 * * * python {script_path} --rotate all --clean\n")
    else:
        # Create a simple script to run log rotation
        script_path = os.path.join(os.path.dirname(__file__), 'rotate_logs.sh')
        with open(script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# Automatic log rotation script
# Created: {datetime.now()}

python {os.path.abspath(__file__)} --rotate all --clean
""")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"\nCreated automatic log rotation script: {script_path}")
        print("You can run this script manually or add it to your startup scripts.\n")


def run_maintenance(args):
    """
    Run log maintenance based on command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        None
    """
    # Rotate logs if requested
    if args.rotate:
        rotated_count = rotate_all_logs(args.rotate, args.max_size)
        print(f"Rotated {rotated_count} logs")
    
    # Clean old logs if requested
    if args.clean:
        deleted_count = clean_old_logs(args.max_age)
        print(f"Cleaned {deleted_count} old log archives")
    
    # Enforce storage limit if requested
    if args.enforce_limit:
        deleted_count = enforce_storage_limit(args.max_storage)
        print(f"Deleted {deleted_count} archives to enforce storage limit")
    
    # Set up automatic rotation if requested
    if args.setup_auto:
        setup_automatic_rotation(args.cron)


def main():
    parser = argparse.ArgumentParser(description='Log Management for Kraken Trading Bot')
    parser.add_argument('--rotate', choices=list(LOG_PATTERNS.keys()), help='Rotate logs of specified type')
    parser.add_argument('--max-size', type=float, default=DEFAULT_MAX_LOG_SIZE_MB, help='Maximum log size in MB')
    parser.add_argument('--clean', action='store_true', help='Clean old log archives')
    parser.add_argument('--max-age', type=int, default=DEFAULT_MAX_LOG_AGE_DAYS, help='Maximum age of log archives in days')
    parser.add_argument('--enforce-limit', action='store_true', help='Enforce storage limit')
    parser.add_argument('--max-storage', type=float, default=DEFAULT_MAX_STORAGE_GB, help='Maximum storage in GB')
    parser.add_argument('--setup-auto', action='store_true', help='Set up automatic log rotation')
    parser.add_argument('--cron', action='store_true', help='Show cron job instructions')
    
    args = parser.parse_args()
    
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    run_maintenance(args)


if __name__ == '__main__':
    main()