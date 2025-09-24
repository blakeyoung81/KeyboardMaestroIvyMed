#!/usr/bin/env zsh

# Clipboard to Patient Image - Fast generation via Pollinations.AI

# Set environment variables
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
# Optional: export IMG_WIDTH=1024 IMG_HEIGHT=1024

# Set explicit paths for better Keyboard Maestro compatibility
VENV_PATH="/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/km-venv"
SCRIPT_PATH="/Users/blakeyoung/Library/Mobile Documents/com~apple~CloudDocs/Coding/Keyboard Maestro/Clipboard2Image/gen_local.py"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Use explicit python path from venv
"$VENV_PATH/bin/python" "$SCRIPT_PATH"
