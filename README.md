# ⚡ BareMetal-ML: Exposing the Python Interpreter Overhead
### *A Systems Engineering Experiment: Python vs. Bare-Metal C++*

> "Constraints breed creativity. By banning `import numpy`, I was forced to understand the machine."

![Performance Chart](https://github.com/Manas10112004/BareMetal-ML/blob/main/performance_chart.png)

## 📖 The Narrative
We all know Python is "slow" and C++ is "fast." But **why**? And **by how much**?
Usually, we hide this difference behind pre-compiled libraries like NumPy or PyTorch. 

**The Challenge:** I set out to build a Deep Learning inference engine with **Zero Dependencies**.
* **Rule 1:** No `import numpy`, `import torch`, or `#include <vector>`.
* **Rule 2:** Every algorithm (Softmax, GELU, MatMul) must be written from scratch using raw pointers and arrays.
* **Rule 3:** Compare raw Python loops against AVX2-optimized C++ kernels.

This project is a forensic analysis of the "Tax" we pay for Python's ease of use.

## 🔬 The "Competency Matrix" (Results)
*Hardware Context: Benchmarks run locally on an HP Omen laptop (RTX 4050 setup).*

| Kernel | Engineering Challenge | Python Latency | C++ (AVX2) | **Speedup** |
|:-------|:----------------------|:---------------|:-----------|:------------|
| **ReLU** | *Memory Bandwidth* | 1150.64 ms | 4.00 ms | **287x** 🚀 |
| **Softmax** | *Numerical Stability* | 238.54 ms | 88.10 ms | **2.7x** |
| **GELU** | *Transcendental Math* | 380.82 ms | 45.58 ms | **8.4x** |
| **MatMul** | *The Engine Core* | 1209.05 ms | 0.73 ms | **1666x** 🤯 |

## 💡 Engineering Insights (The "Why")

### 1. The 1666x MatMul Anomaly
Matrix Multiplication has a time complexity of $O(N^3)$. Python's dynamic type checking and pointer-chasing overhead accumulate exponentially inside the triple loop. 
Conversely, the C++ compiler (GCC with `-O3 -mavx2`) utilized **SIMD (Single Instruction, Multiple Data)** to multiply 8 numbers in a single CPU cycle, while leveraging contiguous memory for maximum L1/L2 Cache hits.

### 2. The "Math Wall" (Softmax/GELU)
When calculating $e^x$ or $\tanh(x)$, the CPU spends massive cycles computing the Taylor Series approximations. The bottleneck shifts from the *Interpreter Tax* directly to the *Arithmetic Logic Unit (ALU)*. Because both Python and C++ rely on underlying C math libraries for these functions, the performance gap shrinks to single digits.

### 3. The Quantization Trap (Float32 vs Int8)
I attempted an **Int8 Quantized Kernel** expecting a massive cache-driven speedup. Instead, the conversion overhead and type promotion (`int8` to `int32` accumulation) broke the compiler's auto-vectorization, resulting in a scalar fallback that was **1.15x slower** than Float32.
* **Conclusion:** Real-world quantization requires manual assembly tuning (e.g., AVX-VNNI instructions) or offline pre-quantization to be effective. The compiler cannot always "guess" the optimal integer path.

## 🛠️ How to Reproduce My Science

**1. Clone & Compile**
```bash
# Compile the C++ Kernel (Shared Library)
g++ -shared -o build/libkernels.dll src/cpp_kernels.cpp -O3 -mavx2
```
**2. Run the Benchmark**
```bash
# This script loads the DLL into memory using Ctypes
python src/main.py
```
**3. Visualize**
```bash
# Generate the logarithmic performance charts
python src/plot_results.py
```
# Tech Stack
## Languages: Python 3.10, C++17
## Optimizations: SIMD (AVX2), Manual Memory Management, Cache Locality
## Bridge: Python ctypes (Foreign Function Interface)
