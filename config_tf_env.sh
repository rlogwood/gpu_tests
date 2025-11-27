#!/bin/bash
# Setup script for TensorFlow GPU with NVIDIA pip packages
# Source this before running TensorFlow: source setup_tf_gpu.sh

# Activate virtual environment
#source .venv/bin/activate

# Add NVIDIA library paths from pip packages
VENV_NVIDIA_LIBS=".venv/lib/python3.13/site-packages/nvidia"

export LD_LIBRARY_PATH="\
$VENV_NVIDIA_LIBS/cublas/lib:\
$VENV_NVIDIA_LIBS/cudnn/lib:\
$VENV_NVIDIA_LIBS/cuda_runtime/lib:\
$VENV_NVIDIA_LIBS/cufft/lib:\
$VENV_NVIDIA_LIBS/curand/lib:\
$VENV_NVIDIA_LIBS/cusolver/lib:\
$VENV_NVIDIA_LIBS/cusparse/lib:\
$VENV_NVIDIA_LIBS/cuda_nvrtc/lib:\
/usr/lib/x86_64-linux-gnu:\
$LD_LIBRARY_PATH"

echo "✅ TensorFlow GPU environment configured"
echo "LD_LIBRARY_PATH set to:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -10
