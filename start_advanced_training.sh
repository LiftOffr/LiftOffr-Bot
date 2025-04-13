#!/bin/bash
#
# Start Advanced ML Model Training
#
# This script initiates the full training process for advanced machine learning models
# including TCN, CNN, LSTM, GRU, Attention Mechanisms, and Transformer architectures
#

# Configuration
TRADING_PAIR="SOL/USD"
TIMEFRAME="1h"
TRAINING_MODE=${1:-"full"}  # full, incremental, or hybrid
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/advanced_training_$(date +%Y%m%d_%H%M%S).log"

# Ensure directories exist
mkdir -p "${LOG_DIR}"
mkdir -p "models"
mkdir -p "historical_data"

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] ${message}" | tee -a "${LOG_FILE}"
}

# Function to check if data exists
check_data() {
    local count=$(ls -1 historical_data/*.csv 2>/dev/null | wc -l)
    if [ "${count}" -eq 0 ]; then
        log "No historical data found. Fetching data..."
        ./fetch_historical_data.sh
    else
        log "Found ${count} historical data files."
    fi
}

# Function to check Python environment
check_environment() {
    # Check if required packages are installed
    python3 -c "import tensorflow, numpy, pandas, sklearn" &>/dev/null
    if [ $? -ne 0 ]; then
        log "ERROR: Required Python packages are missing. Please install required packages."
        exit 1
    fi
    
    log "Environment check: OK"
}

# Main training function
run_training() {
    # Step 1: Process and integrate data
    log "Step 1: Processing and integrating historical data..."
    python3 ml_data_integrator.py
    if [ $? -ne 0 ]; then
        log "ERROR: Data integration failed."
        exit 1
    fi
    
    # Step 2: Train all model architectures
    log "Step 2: Training machine learning models..."
    
    # Create model directories if they don't exist
    mkdir -p models/tcn
    mkdir -p models/cnn
    mkdir -p models/lstm
    mkdir -p models/gru
    mkdir -p models/bilstm
    mkdir -p models/attention
    mkdir -p models/transformer
    mkdir -p models/hybrid
    
    # Train models based on training mode
    if [ "${TRAINING_MODE}" = "full" ]; then
        log "Running full training for all architectures..."
        python3 train_ml_models.py --all --trading-pair "${TRADING_PAIR}" --timeframe "${TIMEFRAME}"
    elif [ "${TRAINING_MODE}" = "incremental" ]; then
        log "Running incremental training for all architectures..."
        python3 train_ml_models.py --all --incremental --trading-pair "${TRADING_PAIR}" --timeframe "${TIMEFRAME}"
    elif [ "${TRAINING_MODE}" = "hybrid" ]; then
        log "Running hybrid training (only train missing models)..."
        python3 train_ml_models.py --hybrid --trading-pair "${TRADING_PAIR}" --timeframe "${TIMEFRAME}"
    else
        log "ERROR: Unknown training mode ${TRAINING_MODE}"
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        log "ERROR: Model training failed."
        exit 1
    fi
    
    # Step 3: Test ensemble model
    log "Step 3: Testing ensemble model..."
    python3 advanced_ensemble_model.py
    if [ $? -ne 0 ]; then
        log "ERROR: Ensemble model testing failed."
        exit 1
    fi
    
    # Step 4: Test ML model integration
    log "Step 4: Testing ML model integration..."
    python3 ml_model_integrator.py
    if [ $? -ne 0 ]; then
        log "ERROR: ML model integration testing failed."
        exit 1
    fi
    
    log "Advanced ML training completed successfully."
}

# Function to display usage
usage() {
    echo "Usage: $0 [training_mode]"
    echo "  training_mode: 'full', 'incremental', or 'hybrid'"
    echo "    - full: Train all models from scratch (default)"
    echo "    - incremental: Update existing models with new data"
    echo "    - hybrid: Only train models that don't exist yet"
    exit 1
}

# Main execution
log "Starting advanced ML training process (${TRAINING_MODE} mode)..."

# Check command arguments
if [ "$1" != "" ] && [ "$1" != "full" ] && [ "$1" != "incremental" ] && [ "$1" != "hybrid" ]; then
    usage
fi

# Check environment and data
check_environment
check_data

# Run training
run_training

log "Advanced ML training process complete."
echo "==================================================="
echo "Advanced ML training completed. See logs at: ${LOG_FILE}"
echo "==================================================="