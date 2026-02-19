#include <cmath>
#include <immintrin.h> // AVX2 Intrinsics
#include <cstdint>     // Int8 types
#include <algorithm>

extern "C" {

    // ==========================================
    // TIER 1: ACTIVATIONS (ReLU)
    // ==========================================
    
    void relu_naive(float* data, int n) {
        for (int i = 0; i < n; i++) {
            if (data[i] < 0) data[i] = 0;
        }
    }

    // AVX2 Optimized (Processes 8 floats per cycle)
    void relu_avx(float* data, int n) {
        int i = 0;
        __m256 zeros = _mm256_setzero_ps();
        
        // Main Loop (8 items at a time)
        for (; i <= n - 8; i += 8) {
            __m256 input = _mm256_loadu_ps(&data[i]);
            __m256 result = _mm256_max_ps(input, zeros);
            _mm256_storeu_ps(&data[i], result);
        }
        
        // Cleanup Loop (Remaining items)
        for (; i < n; i++) {
            if (data[i] < 0) data[i] = 0;
        }
    }

    // ==========================================
    // TIER 2: PROBABILITY (Softmax)
    // ==========================================

    float find_max(const float* data, int n) {
        float max_val = data[0];
        for (int i = 1; i < n; i++) {
            if (data[i] > max_val) max_val = data[i];
        }
        return max_val;
    }

    // Stable Softmax Implementation
    void softmax_naive(const float* input, float* output, int n) {
        float max_val = find_max(input, n);
        float sum = 0.0f;
        
        // Exponentiate and Sum
        for (int i = 0; i < n; i++) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        for (int i = 0; i < n; i++) {
            output[i] /= sum;
        }
    }

    // ==========================================
    // TIER 3: THE CORE (Matrix Multiplication)
    // ==========================================

    // Naive Implementation (O(N^3))
    void matmul_naive(const float* A, const float* B, float* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                float val_A = A[i * N + k]; 
                for (int j = 0; j < N; j++) {
                    C[i * N + j] += val_A * B[k * N + j];
                }
            }
        }
    }

    // ==========================================
    // TIER 4: MODERN ACTIVATION (GELU)
    // ==========================================

    // Tanh Approximation
    void gelu_naive(float* data, int n) {
        const float SQRT_2_OVER_PI = 0.7978845608f;
        const float COEFF = 0.044715f;

        for (int i = 0; i < n; i++) {
            float x = data[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            float tanh_val = std::tanh(inner);
            data[i] = 0.5f * x * (1.0f + tanh_val);
        }
    }

    // ==========================================
    // TIER 5: QUANTIZATION (Int8)
    // ==========================================

    void quantize_tensor(const float* input, int8_t* output, int n, float scale) {
        for (int i = 0; i < n; i++) {
            float val = input[i] * scale;
            if (val > 127.0f) val = 127.0f;
            if (val < -127.0f) val = -127.0f;
            output[i] = (int8_t)(val);
        }
    }

    void matmul_int8(const int8_t* A, const int8_t* B, int32_t* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                int8_t val_A = A[i * N + k];
                for (int j = 0; j < N; j++) {
                    C[i * N + j] += val_A * B[k * N + j];
                }
            }
        }
    }
}
