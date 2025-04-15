#!/usr/bin/env python3
"""
Workflow Runner

This script manages the workflows for the trading bot,
ensuring each component runs on a separate port.
"""
import os
import sys
import subprocess
import time
import logging
import signal
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables
processes = {}

def signal_handler(sig, frame):
    """Handle signals to gracefully stop processes"""
    logger.info(f"Received signal {sig}, shutting down...")
    stop_all_processes()
    sys.exit(0)

def start_process(name, command, port=None):
    """Start a process with the given command"""
    try:
        # Add PORT environment variable if specified
        env = os.environ.copy()
        if port:
            env["PORT"] = str(port)
            
        # Add identifier for the process type
        if "trading_bot" in name.lower():
            env["TRADING_BOT_PROCESS"] = "1"
        
        # Start the process
        logger.info(f"Starting process '{name}' with command: {command}")
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Store the process
        processes[name] = {
            "process": process,
            "command": command,
            "port": port
        }
        
        # Start a thread to read output
        import threading
        def read_output(proc, name):
            for line in proc.stdout:
                logger.info(f"[{name}] {line.strip()}")
        
        thread = threading.Thread(target=read_output, args=(process, name))
        thread.daemon = True
        thread.start()
        
        return True
    except Exception as e:
        logger.error(f"Error starting process '{name}': {e}")
        return False

def stop_process(name):
    """Stop a running process"""
    if name in processes:
        try:
            process = processes[name]["process"]
            logger.info(f"Stopping process '{name}'")
            
            # Try to terminate gracefully
            process.terminate()
            
            # Wait for process to terminate
            for _ in range(5):
                if process.poll() is not None:
                    break
                time.sleep(1)
            
            # Force kill if still running
            if process.poll() is None:
                logger.info(f"Force killing process '{name}'")
                process.kill()
            
            # Remove from processes
            del processes[name]
            return True
        except Exception as e:
            logger.error(f"Error stopping process '{name}': {e}")
            return False
    else:
        logger.warning(f"Process '{name}' not found")
        return False

def stop_all_processes():
    """Stop all running processes"""
    process_names = list(processes.keys())
    for name in process_names:
        stop_process(name)

def restart_process(name):
    """Restart a running process"""
    if name in processes:
        command = processes[name]["command"]
        port = processes[name]["port"]
        stop_process(name)
        time.sleep(1)
        return start_process(name, command, port)
    else:
        logger.warning(f"Process '{name}' not found")
        return False

def check_process(name):
    """Check if a process is still running"""
    if name in processes:
        process = processes[name]["process"]
        return process.poll() is None
    return False

def run_all_components():
    """Run all components of the system"""
    # Start the dashboard on port 5000
    start_process("dashboard", "gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app", 5000)
    
    # Wait a bit for the dashboard to start
    time.sleep(2)
    
    # Start the trading bot on port 8000
    start_process("trading_bot", "python run_trading_bot.py", 8000)
    
    # Monitor processes
    logger.info("All components started. Monitoring...")
    
    try:
        while True:
            for name in list(processes.keys()):
                if not check_process(name):
                    logger.warning(f"Process '{name}' has died, restarting...")
                    restart_process(name)
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_all_processes()
    
    return 0

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting workflow runner")
    run_all_components()
    logger.info("Workflow runner stopped")