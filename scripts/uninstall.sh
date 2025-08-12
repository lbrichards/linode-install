#\!/bin/bash

# Linode Ubuntu Server Uninstall Script
# Cleans up all project dependencies and tmux sessions

set -e  # Exit on error

echo "========================================"
echo "Starting Project Dependencies Uninstall"
echo "========================================"

# Kill any running tmux sessions
echo "Cleaning up tmux sessions..."
tmux kill-session -t dataset_build 2>/dev/null || true
tmux kill-session -t hrm_verify 2>/dev/null || true
tmux kill-session -t hrm_training 2>/dev/null || true
echo "Tmux sessions cleaned"

# Remove Python virtual environment
if [ -d /opt/venv ]; then
    echo "Removing Python virtual environment..."
    sudo rm -rf /opt/venv
    echo "Virtual environment removed"
else
    echo "No virtual environment found at /opt/venv"
fi

# Remove generated datasets
if [ -d /opt/HRM/data ]; then
    echo "Removing generated datasets..."
    sudo rm -rf /opt/HRM/data
    echo "Datasets removed"
else
    echo "No datasets found at /opt/HRM/data"
fi

# Remove training logs if any
if [ -f /opt/HRM/training.log ]; then
    echo "Removing training logs..."
    sudo rm -f /opt/HRM/training.log
    echo "Training logs removed"
fi

echo "========================================"
echo "Uninstall completed\!"
echo "========================================"
echo ""
echo "Note: System packages (python3, pip, git, tmux, etc.) were not removed"
echo "Note: CUDA installation was not removed"
echo "Note: /opt/HRM source code was not removed"
