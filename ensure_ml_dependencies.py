#!/usr/bin/env python3
"""
Ensure ML Dependencies

This script verifies that all required dependencies for the advanced ML trading system
are installed and available. It checks for both Python packages and system dependencies,
installing any that are missing.

The dependencies include:
1. Core ML libraries (TensorFlow, Scikit-learn, etc.)
2. Technical analysis libraries (TA-Lib)
3. Data processing libraries (NumPy, Pandas, etc.)
4. Visualization libraries (Matplotlib, Seaborn, etc.)
5. NLP libraries for sentiment analysis (NLTK, Transformers, etc.)
6. Specialized ML architecture libraries (TCN, Attention mechanisms, etc.)
"""

import os
import sys
import subprocess
import logging
import importlib
import pkg_resources
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("ml_dependencies.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define required packages and versions
CORE_DEPENDENCIES = {
    "numpy": "1.23.0",
    "pandas": "1.5.0",
    "scipy": "1.9.0",
    "scikit-learn": "1.0.0",
    "statsmodels": "0.13.0",
    "matplotlib": "3.5.0",
    "seaborn": "0.12.0"
}

ML_DEPENDENCIES = {
    "tensorflow": "2.10.0",
    "keras": "2.10.0",
    "keras-tcn": "3.5.0"
}

NLP_DEPENDENCIES = {
    "nltk": "3.7.0",
    "transformers": "4.21.0",
    "tokenizers": "0.12.0"
}

WEB_DEPENDENCIES = {
    "requests": "2.28.0",
    "websockets": "10.3.0",
    "trafilatura": "1.4.0"
}

def check_package_installed(package_name: str, min_version: Optional[str] = None) -> bool:
    """
    Check if a Python package is installed and its version meets requirements
    
    Args:
        package_name: Name of the package to check
        min_version: Minimum required version, if any
        
    Returns:
        bool: True if package is installed and meets version requirements
    """
    try:
        # Check if package can be imported
        module = importlib.import_module(package_name)
        
        # Check version if specified
        if min_version:
            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                    logger.warning(f"{package_name} version {installed_version} is installed, "
                                  f"but {min_version} or higher is required")
                    return False
            except Exception as e:
                logger.warning(f"Could not determine version of {package_name}: {e}")
                return False
        
        return True
    
    except ImportError:
        return False

def install_package(package_name: str, version: Optional[str] = None) -> bool:
    """
    Install a Python package using pip
    
    Args:
        package_name: Name of the package to install
        version: Specific version to install, if any
        
    Returns:
        bool: True if installation succeeded
    """
    try:
        package_spec = f"{package_name}"
        if version:
            package_spec += f">={version}"
        
        logger.info(f"Installing {package_spec}...")
        
        # Execute pip install
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            check=True,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            logger.info(f"Successfully installed {package_spec}")
            return True
        else:
            logger.error(f"Failed to install {package_spec}: {process.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False

def check_nltk_data() -> bool:
    """
    Check and download required NLTK data
    
    Returns:
        bool: True if all required NLTK data is available
    """
    try:
        import nltk
        
        # Define required NLTK data
        required_data = [
            "punkt",
            "stopwords",
            "vader_lexicon"
        ]
        
        # Download missing data
        for data in required_data:
            try:
                nltk.data.find(f"tokenizers/{data}")
                logger.debug(f"NLTK data '{data}' is already available")
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking NLTK data: {e}")
        return False

def check_talib_installed() -> bool:
    """
    Check if TA-Lib is installed
    
    Returns:
        bool: True if TA-Lib is installed
    """
    try:
        import talib
        return True
    except ImportError:
        return False

def check_transformers_models() -> bool:
    """
    Check and download required transformer models
    
    Returns:
        bool: True if all required transformer models are available
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Define required transformer models
        required_models = [
            "distilbert-base-uncased-finetuned-sst-2-english",  # Sentiment analysis
            "ProsusAI/finbert"  # Financial sentiment analysis
        ]
        
        # Check and potentially download models
        for model_name in required_models:
            try:
                logger.info(f"Checking transformer model: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                logger.info(f"Transformer model {model_name} is available")
            except Exception as e:
                logger.error(f"Error loading transformer model {model_name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking transformer models: {e}")
        return False

def ensure_all_dependencies() -> bool:
    """
    Ensure all required dependencies are installed
    
    Returns:
        bool: True if all dependencies are available
    """
    all_dependencies = {}
    all_dependencies.update(CORE_DEPENDENCIES)
    all_dependencies.update(ML_DEPENDENCIES)
    all_dependencies.update(NLP_DEPENDENCIES)
    all_dependencies.update(WEB_DEPENDENCIES)
    
    # Check and install missing packages
    missing_packages = []
    for package, version in all_dependencies.items():
        if not check_package_installed(package, version):
            missing_packages.append((package, version))
    
    # Install missing packages
    if missing_packages:
        logger.warning(f"Found {len(missing_packages)} missing packages")
        for package, version in missing_packages:
            install_package(package, version)
    else:
        logger.info("All required Python packages are installed")
    
    # Check special dependencies
    
    # 1. Check TA-Lib
    if not check_talib_installed():
        logger.warning("TA-Lib is not installed. Some technical indicators may not be available.")
        # We don't attempt to install TA-Lib automatically as it requires C library
    
    # 2. Check NLTK data
    if "nltk" in all_dependencies and check_package_installed("nltk"):
        check_nltk_data()
    
    # 3. Check transformer models
    if "transformers" in all_dependencies and check_package_installed("transformers"):
        check_transformers_models()
    
    # Verify all packages again
    all_installed = True
    for package, version in all_dependencies.items():
        if not check_package_installed(package, version):
            logger.error(f"Package {package} >= {version} could not be installed")
            all_installed = False
    
    return all_installed

def main():
    """Main function"""
    logger.info("Checking ML dependencies...")
    
    success = ensure_all_dependencies()
    
    if success:
        logger.info("All ML dependencies are available")
        return 0
    else:
        logger.error("Some ML dependencies could not be installed")
        return 1

if __name__ == "__main__":
    sys.exit(main())