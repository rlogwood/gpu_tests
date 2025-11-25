# GPU Tests for Ubuntu 25.10 w/ Nvidia 5080 

## Summary
This repo has a couple of simple tests to confirm the GPU is being used for 
PyTorch and TensorFlow. The tests were developed with claude.ai.

NOTE: TensorFlow support for the 5080 is currently in the nightly build only on Ubuntu and requires setup found in `activate_tf.sh`.

## Running the tests

### setup the virtual environment
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements
```

### In a new shell
```
. activate_tf.sh
python tensorflow_cuda_test.py
python pytorch_cuda_test.py
```

## Background
Using the nightly build of tensorflow as of 11/24/25 resolves the following problem as summarized by Claude.ai:

> Unfortunately, TensorFlow doesn't properly support RTX 5080 (Blackwell/sm_120) yet. The exact error you're seeing (CUDA_ERROR_INVALID_PTX) is a known issue affecting all RTX 50-series cards.
> The Problem
> Your RTX 5080 has compute capability 12.0 (Blackwell architecture), which is so new that:
>
> TensorFlow wasn't built with CUDA kernel binaries compatible with compute capability 12.0, and attempts to JIT-compile from PTX fail with CUDA_ERROR_INVALID_PTX GitHub

### Error:
```
Performing matrix multiplication on GPU...
W0000 00:00:1764044088.195750   20660 gpu_device.cc:2431] TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0. CUDA kernels will be jit-compiled from PTX, which could take 30 minutes or longer.
I0000 00:00:1764044088.197440   20660 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 11463 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 5080, pci bus id: 0000:01:00.0, compute capability: 12.0
2025-11-24 23:14:48.524495: W tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc:40] 'cuModuleLoadData(&module, data)' failed with 'CUDA_ERROR_INVALID_PTX'
```

see: https://github.com/tensorflow/tensorflow/issues/90291

