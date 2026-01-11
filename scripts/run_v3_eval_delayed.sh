#!/bin/bash
#
# Run delayed evaluation for v3_allstars_controls_vary experiments
# This script waits 4 hours before evaluating all experiments
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LOG_FILE="$PROJECT_DIR/experiments/v3_allstars_controls_vary/eval_delayed.log"

echo "Starting delayed evaluation script..."
echo "Logs will be written to: $LOG_FILE"
echo "This will wait 4 hours before starting evaluation"

# Run the Python script with output redirected to log file
python "$SCRIPT_DIR/eval_v3_allstars_delayed.py" \
    --delay_hours 4.0 \
    --num_eval_trajs 100 \
    --random \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Background process started with PID: $PID"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if still running: ps -p $PID"
echo ""
echo "Evaluation will begin at approximately: $(date -d '+4 hours' '+%Y-%m-%d %H:%M:%S')"
