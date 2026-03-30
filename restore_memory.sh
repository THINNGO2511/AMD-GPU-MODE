#!/bin/bash
# Restore Claude Code memory files from repo backup.
# Run this on a new machine after cloning the repo.
# Usage: bash restore_memory.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MEMORY_SRC="$REPO_DIR/claude-memory"

# Detect the Claude memory path for this repo
# Claude Code stores memory at ~/.claude/projects/<encoded-path>/memory/
ENCODED=$(echo "$REPO_DIR" | sed 's|/|-|g' | sed 's|^-||')
MEMORY_DST="$HOME/.claude/projects/$ENCODED/memory"

echo "Source: $MEMORY_SRC"
echo "Destination: $MEMORY_DST"

if [ ! -d "$MEMORY_SRC" ]; then
    echo "ERROR: claude-memory/ directory not found in repo"
    exit 1
fi

mkdir -p "$MEMORY_DST"
cp "$MEMORY_SRC"/*.md "$MEMORY_DST/"
echo "Copied $(ls "$MEMORY_SRC"/*.md | wc -l | tr -d ' ') memory files."
echo ""
echo "Done! Claude Code will now have access to all memory from previous work."
echo "Open Claude Code in this repo and paste PICKUP_PROMPT.md to resume."
