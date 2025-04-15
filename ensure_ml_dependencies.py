#!/usr/bin/env python3

"""
Ensure ML Dependencies

This script ensures that all required ML dependencies are installed for the
trading bot's enhanced training and optimization process. It:

1. Checks for core ML libraries (TensorFlow, NumPy, Pandas, etc.)
2. Installs missing dependencies
3. Verifies proper versions are installed
4. Configures TensorFlow for optimal performance

This script should be run before starting the enhanced training process.
"""

import os
import sys
import logging
import subprocess
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "tensorflow": "2.4.0",
    "numpy": "1.19.0",
    "pandas": "1.2.0",
    "optuna": "2.4.0",
    "matplotlib": "3.4.0",
    "scikit-learn": "0.24.0",
    "statsmodels": "0.12.0",
    "keras-tcn": "3.3.0",
    "scipy": "1.6.0",
    "seaborn": "0.11.0"
}

def run_pip_command(args: List[str]) -> Optional[subprocess.CompletedProcess]:
    """Run a pip command and return the result"""
    cmd = [sys.executable, "-m", "pip"] + args
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return None

def get_installed_packages() -> Dict[str, str]:
    """Get a dictionary of installed packages and their versions"""
    installed = {}
    
    result = run_pip_command(["list", "--format=json"])
    if not result:
        logger.error("Failed to get installed packages")
        return installed
    
    try:
        import json
        packages = json.loads(result.stdout)
        for pkg in packages:
            installed[pkg["name"].lower()] = pkg["version"]
        return installed
    except Exception as e:
        logger.error(f"Error parsing pip list output: {e}")
        return installed

def check_package(package: str, min_version: str, installed: Dict[str, str]) -> Tuple[bool, bool]:
    """
    Check if a package is installed and meets the minimum version
    
    Returns:
        Tuple[bool, bool]: (is_installed, needs_upgrade)
    """
    is_installed = package.lower() in installed
    needs_upgrade = False
    
    if is_installed:
        import pkg_resources
        try:
            installed_version = installed[package.lower()]
            needs_upgrade = pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version)
        except Exception:
            # If we can't parse the version, assume an upgrade is needed
            needs_upgrade = True
    
    return is_installed, needs_upgrade

def install_package(package: str, version: str) -> bool:
    """Install or upgrade a package to the specified version"""
    package_spec = f"{package}>={version}"
    result = run_pip_command(["install", "-U", package_spec])
    return result is not None

def main():
    """Main function"""
    logger.info("Checking ML dependencies...")
    
    # Get installed packages
    installed = get_installed_packages()
    
    # Check each required package
    packages_to_install = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        is_installed, needs_upgrade = check_package(package, min_version, installed)
        
        if not is_installed:
            logger.info(f"{package} not found, will install >= {min_version}")
            packages_to_install.append((package, min_version))
        elif needs_upgrade:
            logger.info(f"{package} needs upgrade to >= {min_version}")
            packages_to_install.append((package, min_version))
        else:
            logger.info(f"{package} is already installed and up to date")
    
    # Install missing or outdated packages
    if packages_to_install:
        logger.info(f"Installing {len(packages_to_install)} packages...")
        
        for package, version in packages_to_install:
            logger.info(f"Installing {package} >= {version}")
            if not install_package(package, version):
                logger.error(f"Failed to install {package}")
                return 1
    else:
        logger.info("All required packages are already installed")
    
    # Verify TensorFlow is working
    logger.info("Verifying TensorFlow installation...")
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s)")
            
            # Configure TensorFlow to use memory growth
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Error setting memory growth: {e}")
        else:
            logger.warning("No GPU detected, training will use CPU only")
        
        # Run a simple test to confirm TensorFlow works
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[1, 2], [3, 4]])
        c = tf.matmul(a, b)
        logger.info(f"TensorFlow test result: {c.numpy()}")
        
        logger.info("TensorFlow is working correctly")
    except ImportError:
        logger.error("Failed to import TensorFlow")
        return 1
    except Exception as e:
        logger.error(f"Error verifying TensorFlow: {e}")
        return 1
    
    logger.info("All ML dependencies are installed and verified")
    return 0

if __name__ == "__main__":
    sys.exit(main())