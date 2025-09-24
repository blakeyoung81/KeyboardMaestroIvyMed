#!/usr/bin/env zsh

# Complete Medical Workflow - OCR + High Yield + Images + Patient Generation

# Set environment variables
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Set explicit paths for better Keyboard Maestro compatibility
VENV_PATH="/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/km-venv"
SCRIPT_PATH="/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/Screen Setups + Image/enhanced_medical_workflow.py"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Use explicit python path from venv
"$VENV_PATH/bin/python" "$SCRIPT_PATH"
