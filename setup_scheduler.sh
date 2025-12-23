#!/bin/bash
#
# Setup script for NBA Model Scheduled Retraining
#
# This script installs the launchd job to automatically retrain
# the model every Sunday at 3am.
#

PROJECT_DIR="/Users/sygrovefamily/NBA Betting Model"
PLIST_NAME="com.nba-betting-model.retrain.plist"
PLIST_SRC="$PROJECT_DIR/$PLIST_NAME"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "=========================================="
echo "NBA Model Scheduler Setup"
echo "=========================================="

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"
echo "Created logs directory"

# Check if already installed
if [ -f "$PLIST_DEST" ]; then
    echo "Unloading existing scheduler..."
    launchctl unload "$PLIST_DEST" 2>/dev/null
fi

# Copy plist to LaunchAgents
echo "Installing scheduler..."
cp "$PLIST_SRC" "$PLIST_DEST"

# Load the scheduler
echo "Loading scheduler..."
launchctl load "$PLIST_DEST"

# Verify
if launchctl list | grep -q "com.nba-betting-model.retrain"; then
    echo ""
    echo "SUCCESS! Scheduler installed."
    echo ""
    echo "Schedule: Every Sunday at 3:00 AM"
    echo "Logs: $PROJECT_DIR/logs/"
    echo ""
    echo "Commands:"
    echo "  Check status:   launchctl list | grep nba-betting"
    echo "  Run now:        python3 '$PROJECT_DIR/scheduled_retrain.py'"
    echo "  Quick retrain:  python3 '$PROJECT_DIR/scheduled_retrain.py' --quick"
    echo "  Check if due:   python3 '$PROJECT_DIR/scheduled_retrain.py' --check"
    echo "  Uninstall:      launchctl unload '$PLIST_DEST'"
else
    echo ""
    echo "WARNING: Scheduler may not have loaded correctly."
    echo "Try running manually: launchctl load '$PLIST_DEST'"
fi
