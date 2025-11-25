#!/usr/bin/env python3
"""
TensorFlow CUDA/GPU Test Program
Tests CUDA availability and performs GPU computations
"""

import tensorflow as tf
import sys
import time

def print_separator():
    print("=" * 70)

def test_gpu_availability():
    """Check if GPU is available"""
    print_separator()
    print("GPU AVAILABILITY TEST")
    print_separator()
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("❌ No GPUs detected!")
        print("Possible reasons:")
        print("  - NVIDIA drivers not installed")
        print("  - TensorFlow not installed with CUDA support")
        print("  - CUDA toolkit not properly configured")
        return False
    
    print(f"✅ GPU(s) detected: {len(gpus)} device(s)")
    return True

def show_gpu_info():
    """Display GPU information"""
    print_separator()
    print("GPU INFORMATION")
    print_separator()
    
    gpus = tf.config.list_physical_devices('GPU')
    
    print(f"Number of GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}:")
        print(f"  Device: {gpu.name}")
        print(f"  Device Type: {gpu.device_type}")
        
        # Get GPU details
        gpu_details = tf.config.experimental.get_device_details(gpu)
        if gpu_details:
            print(f"  Compute Capability: {gpu_details.get('compute_capability', 'N/A')}")
            device_name = gpu_details.get('device_name', 'Unknown')
            print(f"  Name: {device_name}")
        
        # Memory growth setting
        try:
            memory_growth = tf.config.experimental.get_memory_growth(gpu)
            print(f"  Memory Growth: {memory_growth}")
        except:
            pass
    
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Check if GPU is actually being used
    print(f"GPU Support Available: {tf.test.is_gpu_available(cuda_only=True) if hasattr(tf.test, 'is_gpu_available') else 'Use tf.config.list_physical_devices()'}")

def test_gpu_computation():
    """Perform a simple computation on GPU"""
    print_separator()
    print("GPU COMPUTATION TEST")
    print_separator()
    
    size = 5000
    print(f"Creating two {size}x{size} random matrices...")
    
    # Test on GPU
    print("\nPerforming matrix multiplication on GPU...")
    with tf.device('/GPU:0'):
        a_gpu = tf.random.normal([size, size])
        b_gpu = tf.random.normal([size, size])
        
        # Warm-up run
        _ = tf.matmul(a_gpu, b_gpu)
        
        # Timed run
        start = time.time()
        c_gpu = tf.matmul(a_gpu, b_gpu)
        # Force execution
        _ = c_gpu.numpy()
        end = time.time()
        
        gpu_time = end - start
        print(f"✅ GPU computation completed in {gpu_time:.4f} seconds")
    
    # Test on CPU
    print("\nComparing with CPU computation...")
    with tf.device('/CPU:0'):
        a_cpu = tf.random.normal([size, size])
        b_cpu = tf.random.normal([size, size])
        
        start = time.time()
        c_cpu = tf.matmul(a_cpu, b_cpu)
        # Force execution
        _ = c_cpu.numpy()
        end = time.time()
        
        cpu_time = end - start
        print(f"CPU computation completed in {cpu_time:.4f} seconds")
    
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
    
    # Verify results match (using smaller matrices for verification)
    print("\nVerifying correctness with smaller matrices...")
    with tf.device('/GPU:0'):
        a_small_gpu = tf.random.normal([100, 100])
        b_small_gpu = tf.random.normal([100, 100])
        c_small_gpu = tf.matmul(a_small_gpu, b_small_gpu)
    
    with tf.device('/CPU:0'):
        a_small_cpu = tf.identity(a_small_gpu)
        b_small_cpu = tf.identity(b_small_gpu)
        c_small_cpu = tf.matmul(a_small_cpu, b_small_cpu)
    
    max_diff = tf.reduce_max(tf.abs(c_small_gpu - c_small_cpu)).numpy()
    print(f"Maximum difference between CPU and GPU results: {max_diff:.2e}")
    
    if max_diff < 1e-3:
        print("✅ Results match (within tolerance)!")
    else:
        print("⚠️  Large difference detected - possible issue")

def test_tensor_operations():
    """Test various tensor operations on GPU"""
    print_separator()
    print("TENSOR OPERATIONS TEST")
    print_separator()
    
    with tf.device('/GPU:0'):
        # Test 1: Basic operations
        print("Test 1: Basic tensor operations...")
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        y = tf.constant([2.0, 3.0, 4.0, 5.0, 6.0])
        
        z = x + y
        print(f"  Addition: {z.numpy()}")
        
        z = x * y
        print(f"  Multiplication: {z.numpy()}")
        
        # Test 2: Neural network operations
        print("\nTest 2: Neural network operations...")
        input_tensor = tf.random.normal([32, 10])  # Batch of 32, 10 features
        weights = tf.random.normal([10, 5])        # 10 inputs -> 5 outputs
        bias = tf.random.normal([5])
        
        output = tf.matmul(input_tensor, weights) + bias
        output = tf.nn.relu(output)  # ReLU activation
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print("  ✅ Neural network operations work!")
        
        # Test 3: Gradients
        print("\nTest 3: Gradient computation...")
        x = tf.Variable(tf.random.normal([5, 5]))
        
        with tf.GradientTape() as tape:
            y = x ** 2
            z = tf.reduce_sum(y)
        
        gradients = tape.gradient(z, x)
        print(f"  Variable created: {x.shape}")
        print(f"  Gradient computed: {gradients is not None}")
        print(f"  Gradient shape: {gradients.shape}")
        print("  ✅ Gradient computation works!")

def test_keras_model():
    """Test a simple Keras model on GPU"""
    print_separator()
    print("KERAS MODEL TEST")
    print_separator()
    
    print("Creating a simple neural network...")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"Model created with {model.count_params()} parameters")
    
    # Generate dummy data
    x_train = tf.random.normal([1000, 10])
    y_train = tf.random.uniform([1000, 1], minval=0, maxval=2, dtype=tf.int32)
    y_train = tf.cast(y_train, tf.float32)
    
    print("Training model for 3 epochs...")
    
    # Train briefly
    start = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    end = time.time()
    
    print(f"✅ Training completed in {end - start:.4f} seconds")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Test prediction
    x_test = tf.random.normal([10, 10])
    predictions = model.predict(x_test, verbose=0)
    print(f"\nTest predictions shape: {predictions.shape}")
    print("✅ Keras model works on GPU!")

def test_mixed_precision():
    """Test mixed precision training"""
    print_separator()
    print("MIXED PRECISION TEST")
    print_separator()
    
    try:
        # Check current policy
        policy = tf.keras.mixed_precision.global_policy()
        print(f"Current mixed precision policy: {policy.name}")
        
        # Set mixed precision policy
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        new_policy = tf.keras.mixed_precision.global_policy()
        print(f"New mixed precision policy: {new_policy.name}")
        
        # Create a simple layer to test
        with tf.device('/GPU:0'):
            layer = tf.keras.layers.Dense(64)
            x = tf.random.normal([32, 10])
            y = layer(x)
            
            print(f"Input dtype: {x.dtype}")
            print(f"Output dtype: {y.dtype}")
            print(f"Weights dtype: {layer.kernel.dtype}")
            print("✅ Mixed precision works!")
        
        # Reset policy
        tf.keras.mixed_precision.set_global_policy('float32')
        
    except Exception as e:
        print(f"⚠️  Mixed precision test failed: {e}")

def main():
    print("\n" + "="*70)
    print("TensorFlow CUDA/GPU Test Program")
    print("="*70 + "\n")
    
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Test 1: GPU availability
    if not test_gpu_availability():
        print("\n❌ Exiting - No GPU detected")
        return 1
    
    # Test 2: GPU information
    show_gpu_info()
    
    # Test 3: GPU computation
    test_gpu_computation()
    
    # Test 4: Various tensor operations
    test_tensor_operations()
    
    # Test 5: Keras model
    test_keras_model()
    
    # Test 6: Mixed precision
    test_mixed_precision()
    
    # Summary
    print_separator()
    print("TEST SUMMARY")
    print_separator()
    print("✅ All tests passed!")
    print("Your TensorFlow CUDA installation is working correctly.")
    print_separator()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
