#!/usr/bin/env bash
# Sets up the library environment for the engine binary and runs a command.
set -euo pipefail
VENV="$HOME/chess_bot/chess"
SP="$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")"
export LD_LIBRARY_PATH="$(find "$SP/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
exec "$@"
