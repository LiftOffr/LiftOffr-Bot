#!/usr/bin/env python3
"""
Integrate Improved Model

This script integrates the improved hybrid model with the trading system:
1. Updates model loading mechanism to support different model architectures
2. Creates a hybrid prediction function for multi-class outputs
3. Implements dynamic leverage based on prediction confidence
4. Connects the improved model to the existing trading infrastructure

Usage:
    python integrate_improved_model.py [--pairs ALL|BTC/USD,ETH/USD,...]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Dict, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("integrate_improved_model.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
MODEL_WEIGHTS_DIR = "model_weights"
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [CONFIG_DIR, MODEL_WEIGHTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrate improved model")
    parser.add_argument("--pairs", type=str, default="ALL",
                        help="Trading pairs to integrate, comma-separated (default: ALL)")
    return parser.parse_args()

class ImprovedModelIntegrator:
    """Class to handle integration of improved models with trading system"""
    
    def __init__(self, pairs: List[str]):
        """Initialize the integrator"""
        self.pairs = pairs
        self.config = self._load_ml_config()
        self.models = {}
    
    def _load_ml_config(self) -> Dict[str, Any]:
        """Load ML configuration"""
        if os.path.exists(ML_CONFIG_PATH):
            try:
                with open(ML_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded ML configuration with {len(config.get('models', {}))} models")
                return config
            except Exception as e:
                logger.error(f"Error loading ML configuration: {e}")
                return {"models": {}, "global_settings": {}}
        else:
            logger.warning(f"ML configuration not found: {ML_CONFIG_PATH}")
            return {"models": {}, "global_settings": {}}
    
    def detect_model_outputs(self, model) -> int:
        """Detect the number of output classes in a model"""
        try:
            # Get the output shape from the model
            output_shape = model.output_shape
            
            # Check if it's a tuple or list and extract the last dimension
            if isinstance(output_shape, (tuple, list)):
                if len(output_shape) > 1:
                    return output_shape[-1]
            
            # Default to 3 classes (traditional model)
            return 3
        except Exception as e:
            logger.error(f"Error detecting model outputs: {e}")
            return 3
    
    def map_prediction_to_signal(self, prediction: np.ndarray, num_classes: int) -> Tuple[int, float]:
        """Map multi-class prediction to trading signal and confidence"""
        # Get predicted class and confidence
        if num_classes == 3:
            # Traditional 3-class model: -1 (down), 0 (neutral), 1 (up)
            # Predicted class index is 0, 1, 2, so we subtract 1 to get -1, 0, 1
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            signal = pred_class - 1
        elif num_classes == 5:
            # Enhanced 5-class model: -1 (strong down), -0.5 (moderate down), 0 (neutral), 0.5 (moderate up), 1 (strong up)
            # Predicted class index is 0, 1, 2, 3, 4
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            
            # Map class to signal
            signal_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
            signal = signal_map.get(pred_class, 0.0)
        else:
            # Unknown model type, use max probability
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            
            # Scale to [-1, 1] range
            signal = 2 * (pred_class / (num_classes - 1)) - 1
        
        return signal, confidence
    
    def calculate_dynamic_leverage(self, confidence: float, base_leverage: float, max_leverage: float) -> float:
        """Calculate dynamic leverage based on prediction confidence"""
        # Linear scaling between base and max leverage based on confidence
        min_confidence = 0.5  # Minimum confidence for base leverage
        max_confidence = 0.95  # Confidence for max leverage
        
        if confidence < min_confidence:
            return base_leverage
        elif confidence > max_confidence:
            return max_leverage
        else:
            # Linear interpolation
            confidence_range = max_confidence - min_confidence
            leverage_range = max_leverage - base_leverage
            leverage_factor = (confidence - min_confidence) / confidence_range
            return base_leverage + leverage_factor * leverage_range
    
    def create_prediction_function(self, pair: str, model, num_classes: int) -> callable:
        """Create a prediction function for the pair"""
        pair_config = self.config.get("models", {}).get(pair, {})
        base_leverage = pair_config.get("base_leverage", 5.0)
        max_leverage = pair_config.get("max_leverage", 75.0)
        confidence_threshold = pair_config.get("confidence_threshold", 0.65)
        
        def predict_fn(data, threshold=confidence_threshold):
            """Prediction function for the model"""
            # Ensure data is in the right shape
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
            
            # Get prediction
            prediction = model.predict(data)
            
            # Map to signal and confidence
            signal, confidence = self.map_prediction_to_signal(prediction[0], num_classes)
            
            # Calculate leverage
            leverage = self.calculate_dynamic_leverage(confidence, base_leverage, max_leverage)
            
            # Determine if confidence meets threshold
            meets_threshold = confidence >= threshold
            
            # Return trading decision
            return {
                "signal": signal,
                "confidence": confidence,
                "leverage": leverage,
                "meets_threshold": meets_threshold
            }
        
        return predict_fn
    
    def load_models(self) -> bool:
        """Load all models for the specified pairs"""
        success = True
        
        for pair in self.pairs:
            try:
                # Skip if pair not in config
                if pair not in self.config.get("models", {}):
                    logger.warning(f"No configuration found for {pair}, skipping")
                    continue
                
                # Get model path
                model_path = self.config["models"][pair].get("model_path")
                if not model_path or not os.path.exists(model_path):
                    # Try default path
                    pair_clean = pair.replace("/", "_").lower()
                    alt_paths = [
                        f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_improved_model.h5",
                        f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_model.h5",
                        f"{MODEL_WEIGHTS_DIR}/hybrid_{pair_clean}_quick_model.h5"
                    ]
                    
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            break
                    
                    if not model_path or not os.path.exists(model_path):
                        logger.warning(f"Model file not found for {pair}, skipping")
                        continue
                
                # Load model
                logger.info(f"Loading model for {pair} from {model_path}")
                model = load_model(model_path)
                
                # Detect model outputs
                num_classes = self.detect_model_outputs(model)
                logger.info(f"Detected {num_classes} output classes for {pair} model")
                
                # Create prediction function
                predict_fn = self.create_prediction_function(pair, model, num_classes)
                
                # Store model and prediction function
                self.models[pair] = {
                    "model": model,
                    "predict": predict_fn,
                    "num_classes": num_classes
                }
                
                logger.info(f"Successfully integrated model for {pair}")
            except Exception as e:
                logger.error(f"Error loading model for {pair}: {e}")
                success = False
        
        # Log summary
        logger.info(f"Loaded {len(self.models)} models out of {len(self.pairs)} pairs")
        
        return success
    
    def update_integration_module(self) -> bool:
        """Update the model integration module for improved models"""
        integration_file = "ml_model_integration.py"
        improved_integration_file = "improved_ml_integration.py"
        
        # Create improved integration module
        try:
            with open(improved_integration_file, 'w') as f:
                f.write(f"""#!/usr/bin/env python3
\"\"\"
Improved ML Model Integration

This module handles integration of improved ML models with the trading system.
It supports both traditional 3-class models and enhanced 5-class models.

Auto-generated by integrate_improved_model.py
\"\"\"

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Dict, Any, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{{CONFIG_DIR}}/ml_config.json"
MODEL_WEIGHTS_DIR = "model_weights"

# Global variables
loaded_models = {{}}
model_configs = {{}}

def load_ml_config() -> Dict[str, Any]:
    \"\"\"Load ML configuration\"\"\"
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading ML configuration: {{e}}")
            return {{"models": {{}}, "global_settings": {{}}}}
    else:
        logger.warning(f"ML configuration not found: {{ML_CONFIG_PATH}}")
        return {{"models": {{}}, "global_settings": {{}}}}

def detect_model_outputs(model) -> int:
    \"\"\"Detect the number of output classes in a model\"\"\"
    try:
        # Get the output shape from the model
        output_shape = model.output_shape
        
        # Check if it's a tuple or list and extract the last dimension
        if isinstance(output_shape, (tuple, list)):
            if len(output_shape) > 1:
                return output_shape[-1]
        
        # Default to 3 classes (traditional model)
        return 3
    except Exception as e:
        logger.error(f"Error detecting model outputs: {{e}}")
        return 3

def map_prediction_to_signal(prediction: np.ndarray, num_classes: int) -> Tuple[float, float]:
    \"\"\"Map multi-class prediction to trading signal and confidence\"\"\"
    # Get predicted class and confidence
    if num_classes == 3:
        # Traditional 3-class model: -1 (down), 0 (neutral), 1 (up)
        # Predicted class index is 0, 1, 2, so we subtract 1 to get -1, 0, 1
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]
        signal = pred_class - 1
    elif num_classes == 5:
        # Enhanced 5-class model: -1 (strong down), -0.5 (moderate down), 0 (neutral), 0.5 (moderate up), 1 (strong up)
        # Predicted class index is 0, 1, 2, 3, 4
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]
        
        # Map class to signal
        signal_map = {{0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}}
        signal = signal_map.get(pred_class, 0.0)
    else:
        # Unknown model type, use max probability
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class]
        
        # Scale to [-1, 1] range
        signal = 2 * (pred_class / (num_classes - 1)) - 1
    
    return signal, confidence

def calculate_dynamic_leverage(confidence: float, base_leverage: float, max_leverage: float) -> float:
    \"\"\"Calculate dynamic leverage based on prediction confidence\"\"\"
    # Linear scaling between base and max leverage based on confidence
    min_confidence = 0.5  # Minimum confidence for base leverage
    max_confidence = 0.95  # Confidence for max leverage
    
    if confidence < min_confidence:
        return base_leverage
    elif confidence > max_confidence:
        return max_leverage
    else:
        # Linear interpolation
        confidence_range = max_confidence - min_confidence
        leverage_range = max_leverage - base_leverage
        leverage_factor = (confidence - min_confidence) / confidence_range
        return base_leverage + leverage_factor * leverage_range

def load_model_for_pair(pair: str) -> Optional[Dict[str, Any]]:
    \"\"\"Load model for a specific trading pair\"\"\"
    global loaded_models, model_configs
    
    # Return already loaded model
    if pair in loaded_models:
        return loaded_models[pair]
    
    # Load ML config if not loaded yet
    if not model_configs:
        model_configs = load_ml_config()
    
    # Check if pair in config
    if pair not in model_configs.get("models", {{}}):
        logger.warning(f"No configuration found for {{pair}}")
        return None
    
    # Get model path
    model_path = model_configs["models"][pair].get("model_path")
    if not model_path or not os.path.exists(model_path):
        # Try default path
        pair_clean = pair.replace("/", "_").lower()
        alt_paths = [
            f"{{MODEL_WEIGHTS_DIR}}/hybrid_{{pair_clean}}_improved_model.h5",
            f"{{MODEL_WEIGHTS_DIR}}/hybrid_{{pair_clean}}_model.h5",
            f"{{MODEL_WEIGHTS_DIR}}/hybrid_{{pair_clean}}_quick_model.h5"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        
        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found for {{pair}}")
            return None
    
    try:
        # Load model
        logger.info(f"Loading model for {{pair}} from {{model_path}}")
        model = load_model(model_path)
        
        # Detect model outputs
        num_classes = detect_model_outputs(model)
        logger.info(f"Detected {{num_classes}} output classes for {{pair}} model")
        
        # Get model parameters
        pair_config = model_configs["models"][pair]
        base_leverage = pair_config.get("base_leverage", 5.0)
        max_leverage = pair_config.get("max_leverage", 75.0)
        confidence_threshold = pair_config.get("confidence_threshold", 0.65)
        risk_percentage = pair_config.get("risk_percentage", 0.20)
        
        # Store model info
        loaded_models[pair] = {{
            "model": model,
            "num_classes": num_classes,
            "base_leverage": base_leverage,
            "max_leverage": max_leverage,
            "confidence_threshold": confidence_threshold,
            "risk_percentage": risk_percentage
        }}
        
        return loaded_models[pair]
    except Exception as e:
        logger.error(f"Error loading model for {{pair}}: {{e}}")
        return None

def get_prediction(pair: str, data, threshold: float = None) -> Dict[str, Any]:
    \"\"\"Get prediction for a specific trading pair\"\"\"
    # Load model if not loaded
    model_info = load_model_for_pair(pair)
    if not model_info:
        return {{"signal": 0, "confidence": 0, "leverage": 0, "meets_threshold": False}}
    
    # Extract model and parameters
    model = model_info["model"]
    num_classes = model_info["num_classes"]
    base_leverage = model_info["base_leverage"]
    max_leverage = model_info["max_leverage"]
    confidence_threshold = threshold if threshold is not None else model_info["confidence_threshold"]
    
    # Ensure data is in the right shape
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    
    try:
        # Get prediction
        prediction = model.predict(data)
        
        # Map to signal and confidence
        signal, confidence = map_prediction_to_signal(prediction[0], num_classes)
        
        # Calculate leverage
        leverage = calculate_dynamic_leverage(confidence, base_leverage, max_leverage)
        
        # Determine if confidence meets threshold
        meets_threshold = confidence >= confidence_threshold
        
        # Return trading decision
        return {{
            "signal": signal,
            "confidence": confidence,
            "leverage": leverage,
            "meets_threshold": meets_threshold
        }}
    except Exception as e:
        logger.error(f"Error getting prediction for {{pair}}: {{e}}")
        return {{"signal": 0, "confidence": 0, "leverage": 0, "meets_threshold": False}}

def get_model_parameters(pair: str) -> Dict[str, Any]:
    \"\"\"Get model parameters for a specific trading pair\"\"\"
    # Load model if not loaded
    model_info = load_model_for_pair(pair)
    if not model_info:
        return {{
            "base_leverage": 5.0,
            "max_leverage": 75.0,
            "confidence_threshold": 0.65,
            "risk_percentage": 0.20
        }}
    
    # Return parameters
    return {{
        "base_leverage": model_info["base_leverage"],
        "max_leverage": model_info["max_leverage"],
        "confidence_threshold": model_info["confidence_threshold"],
        "risk_percentage": model_info["risk_percentage"]
    }}

def get_all_active_pairs() -> List[str]:
    \"\"\"Get list of all active trading pairs\"\"\"
    # Load ML config if not loaded yet
    if not model_configs:
        global model_configs
        model_configs = load_ml_config()
    
    # Get active pairs
    active_pairs = []
    for pair, config in model_configs.get("models", {{}}).items():
        if config.get("active", False):
            active_pairs.append(pair)
    
    return active_pairs

# Initialize by loading config
model_configs = load_ml_config()
""")
            
            logger.info(f"Created improved ML integration module: {improved_integration_file}")
            return True
        except Exception as e:
            logger.error(f"Error creating improved ML integration module: {e}")
            return False
    
    def test_integration(self) -> bool:
        """Test the integration with a simple prediction"""
        if not self.models:
            logger.warning("No models loaded, cannot test integration")
            return False
        
        # Get a model for testing
        test_pair = next(iter(self.models.keys()))
        test_model_info = self.models[test_pair]
        
        # Create dummy data
        dummy_shape = (1, 60, 10)  # Adjust based on expected input shape
        dummy_data = np.random.random(dummy_shape)
        
        try:
            # Get prediction
            logger.info(f"Testing prediction for {test_pair} with dummy data")
            prediction = test_model_info["predict"](dummy_data)
            
            # Log results
            logger.info(f"Test prediction results:")
            logger.info(f"  Signal: {prediction['signal']}")
            logger.info(f"  Confidence: {prediction['confidence']:.4f}")
            logger.info(f"  Leverage: {prediction['leverage']:.2f}x")
            logger.info(f"  Meets threshold: {prediction['meets_threshold']}")
            
            return True
        except Exception as e:
            logger.error(f"Error testing integration: {e}")
            return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Parse pairs
    pairs = ALL_PAIRS if args.pairs == "ALL" else args.pairs.split(",")
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATING IMPROVED MODELS")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info("=" * 80 + "\n")
    
    # Create integrator
    integrator = ImprovedModelIntegrator(pairs)
    
    # Load models
    logger.info("Loading models...")
    if not integrator.load_models():
        logger.warning("Some models failed to load")
    
    # Update integration module
    logger.info("Updating model integration module...")
    if not integrator.update_integration_module():
        logger.error("Failed to update integration module")
        return False
    
    # Test integration
    logger.info("Testing integration...")
    if not integrator.test_integration():
        logger.warning("Integration test failed")
    
    logger.info("\nImproved model integration completed")
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error integrating improved models: {e}")
        import traceback
        traceback.print_exc()