#!/usr/bin/env python3
"""
Run ML Retraining and Activation

This script:
1. Retrains ML models with correct input shapes
2. Creates ensemble models
3. Updates ML config
4. Activates ML trading

Usage:
    python run_ml_retraining_and_activation.py [--pairs PAIRS] [--epochs EPOCHS]
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_retraining_activation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
MODEL_TYPES = ["lstm", "gru", "transformer", "cnn", "bilstm", "attention", "hybrid"]
FAST_MODEL_TYPES = ["lstm", "gru", "cnn"]  # Faster models for quick training

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ML retraining and activation")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help=f"Trading pairs to train (default: {' '.join(DEFAULT_PAIRS)})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--model-types", nargs="+", default=None,
                        help=f"Model types to train (default: faster models only)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training")
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(f"[STEP] {description}")
    
    cmd_str = " ".join(cmd)
    logger.info(f"Running command: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        
        # Log command output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"[OUT] {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"[ERR] {line}")
        return None

def retrain_models(args):
    """Retrain models with correct input shapes"""
    # Choose which model types to train
    model_types = args.model_types if args.model_types else FAST_MODEL_TYPES
    
    cmd = [
        "python", "retrain_models_with_correct_shape.py",
        "--pairs"]
    cmd.extend(args.pairs)
    cmd.extend(["--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--model-types"])
    cmd.extend(model_types)
    
    logger.info(f"Retraining models: {', '.join(model_types)}")
    result = run_command(cmd, "Retraining models with correct input shapes")
    
    if result:
        logger.info("Model retraining completed successfully")
        return True
    else:
        logger.error("Model retraining failed")
        return False

def activate_ml_trading(args):
    """Activate ML trading with ensemble models"""
    cmd = [
        "python", "activate_ml_with_ensembles.py",
        "--pairs"]
    cmd.extend(args.pairs)
    
    if args.sandbox:
        cmd.append("--sandbox")
    
    logger.info(f"Activating ML trading for pairs: {', '.join(args.pairs)}")
    result = run_command(cmd, "Activating ML trading with ensemble models")
    
    if result:
        logger.info("ML trading activation completed successfully")
        return True
    else:
        logger.error("ML trading activation failed")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    start_time = time.time()
    logger.info("Starting ML retraining and activation...")
    logger.info(f"Pairs: {args.pairs}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Step 1: Retrain models with correct input shapes
    if not args.skip_training:
        if not retrain_models(args):
            logger.error("Failed to retrain models")
            return 1
    else:
        logger.info("Skipping model training as requested")
    
    # Step 2: Activate ML trading with ensemble models
    if not activate_ml_trading(args):
        logger.error("Failed to activate ML trading")
        return 1
    
    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"ML retraining and activation completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())