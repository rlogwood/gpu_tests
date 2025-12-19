#!/bin/bash
# config_tf_env.sh - Auto-configure TensorFlow GPU (VENV ONLY)


# Instructions: 
# Activate the venv 
# # Manual cleanup only when needed:
# rm -rf ~/.nv/ComputeCache  # ONLY after CUDA_ERROR_INVALID_HANDLE crashes
# source config_tf_env.sh    # Normal use - preserves cache
# python tensorflow_cuda_test.py


# ========================================
# VENV CHECK - FAIL IF NOT IN VIRTUAL ENV
# ========================================
if [ -z "$VIRTUAL_ENV" ] && [ "$(basename "$0")" != "activate" ]; then
    echo "❌ ERROR: This script must be run from an activated virtual environment!"
    echo "   Run: source .venv/bin/activate  # then source config_tf_env.sh"
    return 1
fi

# Auto-detect Python version
VENV_PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
VENV_NVIDIA_LIBS="${VIRTUAL_ENV}/lib/python${VENV_PYTHON_VERSION}/site-packages/nvidia"

# Verify NVIDIA libs exist
if [ ! -d "$VENV_NVIDIA_LIBS" ]; then
    echo "❌ ERROR: NVIDIA libraries not found at: $VENV_NVIDIA_LIBS"
    echo "   Install: pip install tf-nightly[and-cuda] nvidia-cublas-cu12 nvidia-cudnn-cu12"
    return 1
fi

# Build LD_LIBRARY_PATH from all nvidia subdirs
export LD_LIBRARY_PATH=""
for libdir in "${VENV_NVIDIA_LIBS}"/{cublas,cudnn,cuda_runtime,cufft,curand,cusolver,cusparse,cuda_nvrtc}/lib; do
    if [ -d "$libdir" ]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
    fi
done

# Add system libs as fallback
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"

echo "✅ TensorFlow GPU environment configured"
echo "Python: ${VENV_PYTHON_VERSION}"
echo "VENV: ${VIRTUAL_ENV}"
echo "NVIDIA libs: ${VENV_NVIDIA_LIBS}"
echo "LD_LIBRARY_PATH set to: ${LD_LIBRARY_PATH//:/ | }"
