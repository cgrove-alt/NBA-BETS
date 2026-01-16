#!/bin/bash
# Quick script to check if backtest has completed

PROCESS_PID=63816
OUTPUT_FILE="backtest_results_2025.json"
BASELINE_FILE="backtest_results/phase2_backtest.json"

# Check if process is still running
if ps -p $PROCESS_PID > /dev/null 2>&1; then
    # Get runtime and CPU
    RUNTIME=$(ps -o etime= -p $PROCESS_PID | tr -d ' ')
    CPU=$(ps -o %cpu= -p $PROCESS_PID | tr -d ' ')
    echo "⏳ Backtest still running (PID: $PROCESS_PID)"
    echo "   Runtime: $RUNTIME"
    echo "   CPU: ${CPU}%"
    echo ""
    echo "Status: IN PROGRESS"
    exit 1
else
    echo "✅ Backtest process completed"
    echo ""

    # Check if output file was updated
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_AGE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$OUTPUT_FILE")
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "Output file: $OUTPUT_FILE"
        echo "  Last modified: $FILE_AGE"
        echo "  Size: $FILE_SIZE"
        echo ""

        # Check if baseline exists for comparison
        if [ -f "$BASELINE_FILE" ]; then
            echo "✓ Baseline file found: $BASELINE_FILE"
            echo ""
            echo "Ready to run analysis:"
            echo "  python3 analyze_task_3.1_results.py"
        else
            echo "⚠️  Baseline file not found: $BASELINE_FILE"
        fi

        echo ""
        echo "Status: COMPLETE - READY FOR ANALYSIS"
        exit 0
    else
        echo "❌ Output file not found: $OUTPUT_FILE"
        echo ""
        echo "Status: COMPLETED BUT NO OUTPUT"
        exit 2
    fi
fi
