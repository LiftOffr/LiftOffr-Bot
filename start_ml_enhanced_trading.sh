#!/bin/bash
# Start the ML-Enhanced Trading Bot for the Kraken exchange

# Set environment
export PYTHONPATH=.:$PYTHONPATH

# Check if models have been trained
if [ ! -d "models" ] || [ ! "$(ls -A models 2>/dev/null)" ]; then
    echo "Models don't appear to be trained yet. Running model training first..."
    bash start_advanced_training.sh
fi

# Parse arguments
SANDBOX="--sandbox"
CAPITAL="--capital 20000"
LEVERAGE="--leverage 25"
ML_INFLUENCE="--ml-influence 0.5"
CONFIDENCE="--confidence 0.6"

# Parse arguments
for arg in "$@"
do
    case $arg in
        --live)
        SANDBOX=""
        shift
        ;;
        --capital=*)
        CAPITAL="--capital ${arg#*=}"
        shift
        ;;
        --leverage=*)
        LEVERAGE="--leverage ${arg#*=}"
        shift
        ;;
        --ml-influence=*)
        ML_INFLUENCE="--ml-influence ${arg#*=}"
        shift
        ;;
        --confidence=*)
        CONFIDENCE="--confidence ${arg#*=}"
        shift
        ;;
    esac
done

# Print configuration
echo "Starting ML-Enhanced Trading Bot with configuration:"
echo "  Sandbox mode: ${SANDBOX:-Off (LIVE MODE)}"
echo "  Capital: ${CAPITAL#--capital }"
echo "  Leverage: ${LEVERAGE#--leverage }"
echo "  ML Influence: ${ML_INFLUENCE#--ml-influence }"
echo "  Confidence Threshold: ${CONFIDENCE#--confidence }"

# Run with arguments
python start_ml_enhanced_trading.py $SANDBOX $CAPITAL $LEVERAGE $ML_INFLUENCE $CONFIDENCE

# Check if any errors occurred
if [ $? -ne 0 ]; then
    echo "Error starting ML-Enhanced Trading Bot. Check logs for details."
    exit 1
fi