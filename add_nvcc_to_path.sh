#!/bin/sh

# ========================================
# VENV CHECK - FAIL IF NOT IN VIRTUAL ENV
# ========================================
if [ -z "$VIRTUAL_ENV" ] && [ "$(basename "$0")" != "activate" ]; then
    echo "❌ ERROR: This script must be run from an activated virtual environment!"
    echo "   Run: source .venv/bin/activate  # then source config_tf_env.sh"
    return 1
fi


VENV_PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
NVCC_PATH="$PWD/.venv/lib/python${VENV_PYTHON_VERSION}/site-packages/nvidia/cu13/bin"

echo "Added nvidia/cu13/bin to path:"
echo "- $NVCC_PATH"

export PATH="$NVCC_PATH:$PATH"
nvcc --version
