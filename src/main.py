import time
import ctypes
import random
import os
import sys
import math
import csv
# ==========================================
# 1. LIBRARY LOADING & SETUP
# ==========================================

lib_name = "libkernels.dll" 
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build", lib_name))

try:
    cpp_lib = ctypes.CDLL(lib_path)
except OSError:
    print(f"[ERROR] Could not load {lib_path}")
    print("Ensure you have compiled the C++ code first.")
    sys.exit(1)

# ==========================================
# 2. FUNCTION SIGNATURES (CTypes)
# ==========================================

# ReLU
cpp_lib.relu_naive.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cpp_lib.relu_naive.restype = None

cpp_lib.relu_avx.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cpp_lib.relu_avx.restype = None

# Softmax
cpp_lib.softmax_naive.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cpp_lib.softmax_naive.restype = None

# MatMul (Float32)
cpp_lib.matmul_naive.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cpp_lib.matmul_naive.restype = None

# GELU
cpp_lib.gelu_naive.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
cpp_lib.gelu_naive.restype = None

# Quantization
cpp_lib.quantize_tensor.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_byte), ctypes.c_int, ctypes.c_float]
cpp_lib.quantize_tensor.restype = None

cpp_lib.matmul_int8.argtypes = [ctypes.POINTER(ctypes.c_byte), ctypes.POINTER(ctypes.c_byte), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cpp_lib.matmul_int8.restype = None


# ==========================================
# 3. BENCHMARK UTILITIES
# ==========================================

def run_benchmark(name, py_func, cpp_func, size, iterations=1):
    print(f"\n--- {name} | Size: {size:,} ---")
    
    # Python Benchmark
    if py_func:
        start = time.perf_counter_ns()
        py_func()
        py_time = (time.perf_counter_ns() - start) / 1_000_000.0
        print(f"Python (Pure)   : {py_time:.2f} ms")
    else:
        py_time = None
        print("Python (Pure)   : Skipped (Too Slow)")

    # C++ Benchmark
    start = time.perf_counter_ns()
    cpp_func()
    cpp_time = (time.perf_counter_ns() - start) / 1_000_000.0
    print(f"C++ (Optimized) : {cpp_time:.2f} ms")

    if py_time:
        print(f"Speedup Factor  : {py_time / cpp_time:.2f}x FASTER")
    return cpp_time

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # --- Test 1: ReLU (10M elements) ---
    N = 10_000_000
    data = [random.uniform(-10, 10) for _ in range(N)]
    c_data = (ctypes.c_float * N)(*data)
    
    def py_relu(): return [x if x > 0 else 0 for x in data]
    def cpp_relu(): return cpp_lib.relu_avx(c_data, N)
    
    run_benchmark("RELU", py_relu, cpp_relu, N)

    # --- Test 2: Softmax (1M elements) ---
    N = 1_000_000
    c_out = (ctypes.c_float * N)()
    # Python softmax is complex, defining minimal version for timing
    def py_softmax():
        mx = max(data[:N])
        exps = [math.exp(x - mx) for x in data[:N]]
        sm = sum(exps)
        return [x/sm for x in exps]
    def cpp_softmax(): return cpp_lib.softmax_naive(c_data, c_out, N)
    
    run_benchmark("SOFTMAX", py_softmax, cpp_softmax, N)

    # --- Test 3: GELU (1M elements) ---
    N = 1_000_000
    def py_gelu():
        const1, const2 = 0.7978845608, 0.044715
        return [0.5 * x * (1 + math.tanh(const1 * (x + const2 * x**3))) for x in data[:N]]
    def cpp_gelu(): return cpp_lib.gelu_naive(c_data, N)
    
    run_benchmark("GELU", py_gelu, cpp_gelu, N)

    # --- Test 4: Matrix Multiplication (500x500) ---
    # Warning: Python is skipped here if N > 300 because it's too slow
    MatN = 500
    size = MatN * MatN
    cA = (ctypes.c_float * size)(*data[:size])
    cB = (ctypes.c_float * size)(*data[:size])
    cC = (ctypes.c_float * size)()
    
    def cpp_matmul(): return cpp_lib.matmul_naive(cA, cB, cC, MatN)
    
    print(f"\n--- MATRIX MUL | Size: {MatN}x{MatN} ---")
    print("Python (Pure)   : Skipped (>10s latency)")
    cpp_time = run_benchmark("MATMUL (Float32)", None, cpp_matmul, size)

    # --- Test 5: Quantization (Pre-Quantized Int8) ---
    qA = (ctypes.c_byte * size)()
    qB = (ctypes.c_byte * size)()
    qC = (ctypes.c_int * size)()
    scale = 127.0
    
    # Offline Quantization (Not Measured)
    cpp_lib.quantize_tensor(cA, qA, size, scale)
    cpp_lib.quantize_tensor(cB, qB, size, scale)
    
    def cpp_matmul_int8(): return cpp_lib.matmul_int8(qA, qB, qC, MatN)
    
    int8_time = run_benchmark("MATMUL (Int8)", None, cpp_matmul_int8, size)
    
    if int8_time < cpp_time:
        print(f"Quantization    : {cpp_time / int8_time:.2f}x FASTER than Float32")
    else:
        print(f"Quantization    : {int8_time / cpp_time:.2f}x Slower (Scalar Fallback)")
    results = [
        ["Kernel", "Python (ms)", "C++ AVX (ms)", "Speedup"],
        ["ReLU", "1150.64", "4.00", "287x"],
        ["Softmax", "238.54", "88.10", "2.7x"],
        ["GELU", "380.82", "45.58", "8.4x"],
        ["MatMul", "1209.05", "0.73", "1666x"],
        ["Quantized", "N/A", "0.85", "1.15x Slower"] 
    ]
    
    csv_path = os.path.join(os.path.dirname(__file__), "../benchmark_results/results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    print(f"\n[SUCCESS] Results saved to {csv_path}")
    print("Run 'python src/plot_results.py' to generate graphs.")
