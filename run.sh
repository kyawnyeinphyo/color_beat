#!/usr/bin/env bash
# Beat Color Visualizer Launcher
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 beat_color_visualizer.py "$@"
