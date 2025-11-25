#!/usr/bin/env python3
"""
Check which CUDA libraries TensorFlow can and cannot find
"""

import tensorflow as tf
import os
import ctypes.util

print("TensorFlow version:", tf.__version__)
print("\nSearching for CUDA libraries...\n")

# Common CUDA libraries that TensorFlow needs
cuda_libs = [
    'cuda',
    'cudart',
    'cublas',
    'cublasLt',
    'cufft',
    'curand',
    'cusolver',
    'cusparse',
    'cudnn',
    'nvrtc',
]

print("=" * 70)
print("CUDA Library Status")
print("=" * 70)

for lib in cuda_libs:
    # Try different version numbers
    for version in ['', '.12', '.12.8', '.13', '.13.0', '.11']:
        lib_name = f'lib{lib}.so{version}'
        path = ctypes.util.find_library(lib + version.replace('.', ''))
        if path:
            print(f"✅ {lib_name:30} FOUND: {path}")
            break
    else:
        print(f"❌ {lib:30} NOT FOUND")

print("\n" + "=" * 70)
print("LD_LIBRARY_PATH")
print("=" * 70)
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
print(ld_path)

print("\n" + "=" * 70)
print("Checking common CUDA locations")
print("=" * 70)

cuda_paths = [
    '/usr/local/cuda/lib64',
    '/usr/local/cuda-12.8/lib64',
    '/usr/local/cuda-12/lib64',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/local/lib',
]

for path in cuda_paths:
    if os.path.exists(path):
        print(f"✅ {path} exists")
        # List CUDA-related files
        try:
            files = [f for f in os.listdir(path) if 'cuda' in f.lower() or 'cublas' in f.lower() or 'cudnn' in f.lower()]
            if files:
                print(f"   Found {len(files)} CUDA-related files")
                for f in sorted(files)[:5]:  # Show first 5
                    print(f"   - {f}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more")
        except PermissionError:
            print(f"   (Permission denied to list)")
    else:
        print(f"❌ {path} does not exist")

print("\n" + "=" * 70)
print("TensorFlow GPU Detection Attempt")
print("=" * 70)
try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
except Exception as e:
    print(f"Error: {e}")
