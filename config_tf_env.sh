#!/bin/bash
# config_tf_env.sh - RTX 5080 TensorFlow GPU (VENV ONLY)

# VENV CHECK
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Must activate venv first: source .venv/bin/activate"
    exit 1
fi

# AUTO-DETECT NVIDIA LIBS
VENV_PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
VENV_NVIDIA_LIBS="${VIRTUAL_ENV}/lib/python${VENV_PYTHON_VERSION}/site-packages/nvidia"

echo "Checking presence of nvidia libs:"
echo "VENV_PYTHON_VERSION={$VENV_PYTHON_VERSION}"
echo "VENV_NVIDIA_LIBS={$VENV_NVIDIA_LIBS}"

if [ ! -d "$VENV_NVIDIA_LIBS" ]; then
    echo "Nvidia labs dir not found: {$VENV_NVIDIA_LIBS}"
    echo "❌ ERROR: Install NVIDIA libs: pip install tf-nightly[and-cuda]"
    return 1
fi

# BUILD LD_LIBRARY_PATH

echo "Building LD_LIBRARY_PATH from:"
echo " -- {$VENV_NVIDIA_LIBS}"

export LD_LIBRARY_PATH=""
for libdir in "${VENV_NVIDIA_LIBS}"/{cublas,cudnn,cuda_runtime,cufft,curand,cusolver,cusparse,cuda_nvrtc}/lib; do
    [ -d "$libdir" ] && export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"

echo ""
echo "* NOTE: if you receive CUDA_ERROR_INVALID_HANDLE  # Corrupted PTX cache"
echo "        manually clear the compute cache 'rm -fr ~/.nv/ComputeCache'"

echo ""
echo "✅ TensorFlow RTX 5080 GPU configured"
echo "  Python: ${VENV_PYTHON_VERSION}"
echo "  Cache: ~/.nv/ComputeCache (preserved for speed)"
echo ""
