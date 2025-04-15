#!/usr/bin/env python3
"""
Fix Port Conflict

This script updates the port used by the main dashboard application
to avoid conflicts with the trading bot.
"""
import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Target files
APP_FILES = [
    'main.py',
    'app.py',
    'dashboard_app.py',
    'dashboard.py'
]

def fix_port_in_file(file_path, old_port='5000', new_port='5001'):
    """
    Fix port in a specific file
    
    Args:
        file_path: Path to the file
        old_port: Old port number
        new_port: New port number
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define patterns to find the port settings
        patterns = [
            r'port\s*=\s*' + old_port,
            r'PORT\s*=\s*' + old_port,
            r'--bind\s+0\.0\.0\.0:' + old_port,
            r'"port"\s*:\s*' + old_port,
            r'\'port\'\s*:\s*' + old_port,
            r'app\.run\([^)]*port\s*=\s*' + old_port
        ]
        
        # Check if any pattern matches
        modified = False
        for pattern in patterns:
            replacement = pattern.replace(old_port, new_port)
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
        
        if modified:
            with open(file_path, 'w') as f:
                f.write(content)
            logger.info(f"Updated port in {file_path} from {old_port} to {new_port}")
            return True
        else:
            logger.info(f"No matching port pattern found in {file_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating port in {file_path}: {e}")
        return False

def fix_gunicorn_port(old_port='5000', new_port='5001'):
    """
    Fix port in the .replit file for gunicorn
    
    Args:
        old_port: Old port number
        new_port: New port number
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        # Check workflow files
        workflow_files = [
            '.web_server_workflow.json',
            '.dashboard_workflow.json',
            '.replit'
        ]
        
        modified = False
        for file_path in workflow_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for gunicorn command with port binding
                pattern = r'gunicorn\s+--bind\s+0\.0\.0\.0:' + old_port
                if re.search(pattern, content):
                    content = re.sub(pattern, f'gunicorn --bind 0.0.0.0:{new_port}', content)
                    with open(file_path, 'w') as f:
                        f.write(content)
                    logger.info(f"Updated gunicorn port in {file_path} from {old_port} to {new_port}")
                    modified = True
        
        return modified
    
    except Exception as e:
        logger.error(f"Error updating gunicorn port: {e}")
        return False

def update_workflow_file(workflow_name='Start application', old_port='5000', new_port='5001'):
    """
    Update workflow file with new port
    
    Args:
        workflow_name: Name of the workflow
        old_port: Old port number
        new_port: New port number
    
    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        workflow_file = f'.{workflow_name.lower().replace(" ", "_")}_workflow.json'
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            # Update port in workflow file
            content = content.replace(f':{old_port}', f':{new_port}')
            
            with open(workflow_file, 'w') as f:
                f.write(content)
            
            logger.info(f"Updated port in workflow file {workflow_file}")
            return True
        else:
            logger.warning(f"Workflow file not found: {workflow_file}")
            return False
    
    except Exception as e:
        logger.error(f"Error updating workflow file: {e}")
        return False

def main():
    """Main function"""
    logger.info("Fixing port conflict...")
    
    old_port = '5000'
    new_port = '5001'
    
    # Fix port in application files
    modified_files = []
    for file_path in APP_FILES:
        if fix_port_in_file(file_path, old_port, new_port):
            modified_files.append(file_path)
    
    # Fix gunicorn port
    if fix_gunicorn_port(old_port, new_port):
        modified_files.append('gunicorn_config')
    
    # Update workflow file
    if update_workflow_file('Start application', old_port, new_port):
        modified_files.append('.start_application_workflow.json')
    
    if modified_files:
        logger.info(f"Successfully updated port from {old_port} to {new_port} in: {', '.join(modified_files)}")
        logger.info("Please restart the application workflow for changes to take effect.")
    else:
        logger.warning("No files were modified. Port conflict may still exist.")
    
    return modified_files

if __name__ == "__main__":
    main()