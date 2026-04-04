#!/bin/bash
# submit.sh — Wrapper for popcorn-cli that handles TTY requirements
# Usage: ./submit.sh <leaderboard> <mode> <file> [output_file]
# Example: ./submit.sh amd-mixed-mla test mixed-mla/submission.py
# Example: ./submit.sh amd-moe-mxfp4 benchmark moe-mxfp4/submission.py /tmp/result.txt

set -e

LEADERBOARD="$1"
MODE="$2"
FILEPATH="$3"
OUTPUT_FILE="${4:-/tmp/popcorn_output_$(date +%s).txt}"

if [ -z "$LEADERBOARD" ] || [ -z "$MODE" ] || [ -z "$FILEPATH" ]; then
    echo "Usage: $0 <leaderboard> <mode> <filepath> [output_file]"
    echo "Leaderboards: amd-mxfp4-mm, amd-moe-mxfp4, amd-mixed-mla"
    echo "Modes: test, benchmark, leaderboard, profile"
    exit 1
fi

if [ ! -f "$FILEPATH" ]; then
    echo "ERROR: File not found: $FILEPATH"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting $FILEPATH to $LEADERBOARD ($MODE mode)"

# Use expect to provide pseudo-tty for popcorn-cli
expect -c "
    set timeout 600
    log_user 1
    spawn popcorn-cli submit --gpu MI355X --leaderboard $LEADERBOARD --mode $MODE $FILEPATH
    expect {
        -re {.*error.*} { puts \"ERROR detected\"; exp_continue }
        -re {.*Error.*} { puts \"ERROR detected\"; exp_continue }
        timeout { puts \"TIMEOUT after 600s\" }
        eof { puts \"SUBMISSION COMPLETE\" }
    }
" 2>&1 | strings | grep -v '^\[' | tee "$OUTPUT_FILE"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output saved to: $OUTPUT_FILE"
