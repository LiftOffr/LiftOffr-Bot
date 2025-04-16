#!/bin/bash
# This script starts the training process for all 10 cryptocurrency pairs
# It runs in the background so the training can continue even if the session is closed

echo "Starting training process for all 10 pairs..."
echo "This will run in the background and may take several hours."
echo "Check the logs directory for progress updates."

# Create logs directory
mkdir -p logs

# Start the training process
nohup python train_all_pairs_with_metrics.py > logs/training_output.log 2>&1 &

# Save the process ID
echo $! > .training_pid

echo "Training process started with PID $(cat .training_pid)"
echo "You can check progress with: tail -f logs/training_output.log"
echo "You can also view the metrics report as it's generated: cat metrics_report.md"