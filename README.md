# BareMetal-ML: Exposing the "Interpreter Tax"
### *A Systems Engineering Experiment: Python vs. Bare-Metal C++*

> "Constraints breed creativity. By banning `import numpy`, I was forced to understand the machine."

![Performance Chart](benchmark_results/performance_chart.png)

## 📖 The Narrative
We all know Python is "slow" and C++ is "fast." But **why**? And **by how much**?
Usually, we hide this difference behind libraries like NumPy or PyTorch (which are just C++ wrappers). 

**The Challenge:** I set out to build a Deep Learning inference engine with **Zero Dependencies**.
* **Rule 1:** No `import numpy`, `import torch`, or `#include <vector>`.
* **Rule 2:** Every algorithm (Softmax, GELU, MatMul) must be written from scratch.
* **Rule 3:** Compare raw Python loops against AVX2-optimized C++ kernels.

This project is a forensic analysis of the "Tax" we pay for Python's ease of use.

## 🔬 The "Competency Matrix" (Results)
Tested on **HP Omen (Ryzen 7 / RTX 4050 Setup)**.

| Kernel | Challenge | Python Latency | C++ (AVX2) | **Speedup** |
|:-------|:----------|:---------------|:-----------|:------------|
| **ReLU** | *Memory Bandwidth* | 1150.64 ms | 4.00 ms | **287x** 🚀 |
| **Softmax** | *Numerical Stability* | 238.54 ms | 88.10 ms | **2.7x** |
| **GELU** | *Transcendental Math* | 380.82 ms | 45.58 ms | **8.4x** |
| **MatMul** | *The O(N^3) Killer* | 1209.05 ms | 0.73 ms | **1666x** 🤯 |

### 💡 Engineering Insights (The "Why")

**1. The 1666x MatMul Anomaly**
Why was Matrix Multiplication so drastically different?
* **Python:** The interpreter checks data types *every single iteration* of the loop ($N^3$ times). It chases pointers randomly in memory.
* **C++:** The compiler (GCC -O3) utilized **SIMD (Single Instruction, Multiple Data)** to multiply 8 numbers in a single CPU cycle. It also prefetched data into the L1 Cache.

**2. The "Math Wall" (Softmax/GELU)**
When calculating $e^x$ or $tanh(x)$, the CPU spends so much time computing the Taylor Series that the "Python Tax" becomes negligible. The bottleneck shifts from the *Language* to the *ALU (Arithmetic Logic Unit)*.

**3. The Quantization Trap**
I attempted an **Int8 Quantized Kernel** expecting a 4x speedup. Instead, it was **1.15x slower** than Float32.
* **Root Cause:** The `int8` -> `int32` accumulation broke the compiler's auto-vectorization.
* **Lesson:** Quantization requires manual assembly tuning (VNNI instructions); the compiler cannot always "guess" the optimal integer path.

## 🛠️ How to Run This (Reproduce My Science)

**1. Clone & Compile**
```bash
# Compile the C++ Kernel (Shared Library)
# Requires MinGW (Windows) or GCC (Linux)
g++ -shared -o build/libkernels.dll src/cpp_kernels.cpp -O3 -mavx2
