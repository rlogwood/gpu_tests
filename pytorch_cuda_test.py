#!/usr/bin/env python3
"""
PyTorch CUDA/GPU Test Program
Tests CUDA availability and performs GPU computations
"""

import torch
import sys

def print_separator():
    print("=" * 70)

def test_cuda_availability():
    """Check if CUDA is available"""
    print_separator()
    print("CUDA AVAILABILITY TEST")
    print_separator()
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA is not available!")
        print("Possible reasons:")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch not installed with CUDA support")
        print("  - CUDA toolkit not properly configured")
        return False
    
    print("✅ CUDA is available!")
    return True

def show_gpu_info():
    """Display GPU information"""
    print_separator()
    print("GPU INFORMATION")
    print_separator()
    
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory information
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Total Memory: {total_memory:.2f} GB")
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  Currently Allocated: {allocated:.2f} GB")
        print(f"  Currently Reserved: {reserved:.2f} GB")
    
    print(f"\nCurrent CUDA Device: {torch.cuda.current_device()}")
    print(f"CUDA Version (PyTorch build): {torch.version.cuda}")

def test_gpu_computation():
    """Perform a simple computation on GPU"""
    print_separator()
    print("GPU COMPUTATION TEST")
    print_separator()
    
    # Create tensors on GPU
    size = 5000
    print(f"Creating two {size}x{size} random matrices on GPU...")
    
    device = torch.device("cuda")
    
    # Generate random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    print("Performing matrix multiplication on GPU...")
    
    # Warm-up run
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Timed run
    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()
    
    gpu_time = end - start
    print(f"✅ GPU computation completed in {gpu_time:.4f} seconds")
    
    # Compare with CPU
    print(f"\nComparing with CPU computation...")
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    end = time.time()
    
    cpu_time = end - start
    print(f"CPU computation completed in {cpu_time:.4f} seconds")
    
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
    
    # Verify results match
    c_gpu_to_cpu = c.cpu()
    max_diff = torch.max(torch.abs(c_cpu - c_gpu_to_cpu)).item()
    print(f"\nMaximum difference between CPU and GPU results: {max_diff:.2e}")
    
    if max_diff < 1e-3:
        print("✅ Results match (within tolerance)!")
    else:
        print("⚠️  Large difference detected - possible issue")

def test_tensor_operations():
    """Test various tensor operations on GPU"""
    print_separator()
    print("TENSOR OPERATIONS TEST")
    print_separator()
    
    device = torch.device("cuda")
    
    # Test 1: Basic operations
    print("Test 1: Basic tensor operations...")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    y = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device=device)
    
    z = x + y
    print(f"  Addition: {z}")
    
    z = x * y
    print(f"  Multiplication: {z}")
    
    # Test 2: Neural network operations
    print("\nTest 2: Neural network operations...")
    input_tensor = torch.randn(32, 10, device=device)  # Batch of 32, 10 features
    weights = torch.randn(10, 5, device=device)        # 10 inputs -> 5 outputs
    bias = torch.randn(5, device=device)
    
    output = torch.matmul(input_tensor, weights) + bias
    output = torch.relu(output)  # ReLU activation
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print("  ✅ Neural network operations work!")
    
    # Test 3: Gradients
    print("\nTest 3: Gradient computation...")
    x = torch.randn(5, 5, device=device, requires_grad=True)
    y = x ** 2
    z = y.sum()
    z.backward()
    print(f"  Input requires_grad: {x.requires_grad}")
    print(f"  Gradient computed: {x.grad is not None}")
    print("  ✅ Gradient computation works!")

def main():
    print("\n" + "="*70)
    print("PyTorch CUDA/GPU Test Program")
    print("="*70 + "\n")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    # Test 1: CUDA availability
    if not test_cuda_availability():
        print("\n❌ Exiting - CUDA not available")
        return 1
    
    # Test 2: GPU information
    show_gpu_info()
    
    # Test 3: GPU computation
    test_gpu_computation()
    
    # Test 4: Various tensor operations
    test_tensor_operations()
    
    # Summary
    print_separator()
    print("TEST SUMMARY")
    print_separator()
    print("✅ All tests passed!")
    print("Your PyTorch CUDA installation is working correctly.")
    print_separator()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    
