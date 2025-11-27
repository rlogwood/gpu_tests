
## Setup LD_LIBRARY_PATH
```aiignore
(.venv) ~/src/repos/gpu_tests$ . config_tf_env.sh
✅ TensorFlow GPU environment configured
LD_LIBRARY_PATH set to:
.venv/lib/python3.13/site-packages/nvidia/cublas/lib
.venv/lib/python3.13/site-packages/nvidia/cudnn/lib
.venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib
.venv/lib/python3.13/site-packages/nvidia/cufft/lib
.venv/lib/python3.13/site-packages/nvidia/curand/lib
.venv/lib/python3.13/site-packages/nvidia/cusolver/lib
.venv/lib/python3.13/site-packages/nvidia/cusparse/lib
.venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib
/usr/lib/x86_64-linux-gnu
```



## Run tensorflow test
```aiignore
(.venv) dever@ubiskee:~/src/repos/gpu_tests$ python tensorflow_cuda_test.py 

======================================================================
TensorFlow CUDA/GPU Test Program
======================================================================

TensorFlow Version: 2.21.0-dev20251126
Python Version: 3.13.9
======================================================================
GPU AVAILABILITY TEST
======================================================================
✅ GPU(s) detected: 1 device(s)
======================================================================
GPU INFORMATION
======================================================================
Number of GPUs: 1

GPU 0:
  Device: /physical_device:GPU:0
  Device Type: GPU
  Compute Capability: (12, 0)
  Name: NVIDIA GeForce RTX 5080
  Memory Growth: None

TensorFlow Version: 2.21.0-dev20251126
Built with CUDA: True
GPU Support Available: True
======================================================================
GPU COMPUTATION TEST
======================================================================
Creating two 5000x5000 random matrices...

Performing matrix multiplication on GPU...
✅ GPU computation completed in 0.0382 seconds

Comparing with CPU computation...
CPU computation completed in 0.1333 seconds

🚀 GPU Speedup: 3.49x faster than CPU

Verifying correctness with smaller matrices...
Maximum difference between CPU and GPU results: 1.18e-02
✅ Results match (within tolerance for TF32 execution)!
======================================================================
TENSOR OPERATIONS TEST
======================================================================
Test 1: Basic tensor operations...
  Addition: [ 3.  5.  7.  9. 11.]
  Multiplication: [ 2.  6. 12. 20. 30.]

Test 2: Neural network operations...
  Input shape: (32, 10)
  Output shape: (32, 5)
  ✅ Neural network operations work!

Test 3: Gradient computation...
  Variable created: (5, 5)
  Gradient computed: True
  Gradient shape: (5, 5)
  ✅ Gradient computation works!
======================================================================
KERAS MODEL TEST
======================================================================
Creating a simple neural network...
Model created with 9729 parameters
Training model for 3 epochs...
✅ Training completed in 0.7144 seconds
Final loss: 0.6724
Final accuracy: 0.5940

Test predictions shape: (10, 1)
✅ Keras model works on GPU!
======================================================================
MIXED PRECISION TEST
======================================================================
Current mixed precision policy: float32
New mixed precision policy: mixed_float16
Input dtype: <dtype: 'float32'>
Output dtype: <dtype: 'float16'>
Weights dtype: float32
✅ Mixed precision works!
======================================================================
MNIST DATA TEST
Verifying Keras nightly is installed with TensorFlow nightly...
Training set shape: (60000, 28, 28)
Test set shape:     (10000, 28, 28)
✅ MNIST dataset loaded successfully!
======================================================================
TEST SUMMARY
======================================================================
✅ All tests passed!
Your TensorFlow CUDA installation is working correctly.
======================================================================
```