// factored_gemv.cu — fused y = U @ (C @ x) for UltraTensor factored tensors.
//
//   W (m x n) ~= U @ C
//     U : fp16 basis  [m, k]
//     C : uq4 codes   [k, n]  (value = scale * (code - 8), scale = amax/8
//                              per 32-col block; 2 codes per byte, lo nibble
//                              first). Layout identical to the
//                              ultratensor/gguf_factored.py container.
//
// Two kernels:
//   factored_c_gemv_kernel : t[k] = C @ x        (wide, memory-bound)
//   factored_u_gemv_kernel : y[m] = U @ t        (tiny, negligible)
// No intermediate in DRAM beyond the k-sized t vector.
//
// Built standalone:  nvcc -shared -Xcompiler /MD -arch=sm_89 -O3 \
//                    factored_gemv.cu -o factored_gemv.dll
// Loaded via ctypes (no torch extension needed); same entry point signature
// is intended to drop into runtime/nn/cuda_kernels.cu (Phase 3).
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef _WIN32
#define UT_EXPORT extern "C" __declspec(dllexport)
#else
#define UT_EXPORT extern "C"
#endif

__global__ void factored_c_gemv_kernel(
        const float * __restrict__ scales,
        const unsigned char * __restrict__ packed,
        int k, int n,
        const float * __restrict__ x,
        float * __restrict__ t) {
    const int row = blockIdx.x;
    if (row >= k) return;
    const int nblocks = n / 32;
    const float * srow = scales + (size_t) row * nblocks;
    const unsigned char * prow = packed + (size_t) row * (n / 2);
    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const float sc = srow[b];
        const float * xb = x + b * 32;
        const unsigned char * pb = prow + b * 16;
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
            const unsigned char byte = pb[j >> 1];
            const int c0 = (byte & 15) - 8;
            const int c1 = (byte >> 4) - 8;
            acc += xb[j] * (float) c0 * sc;
            acc += xb[j + 1] * (float) c1 * sc;
        }
    }
    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&t[row], sm[0]);
}

__global__ void factored_u_gemv_kernel(
        const __half * __restrict__ U, int m, int k,
        const float * __restrict__ t, float * __restrict__ y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    const __half * Urow = U + (size_t) row * k;
    float acc = 0.0f;
#pragma unroll 4
    for (int j = 0; j < k; j++) acc += __half2float(Urow[j]) * t[j];
    y[row] = acc;
}

UT_EXPORT int factored_gemv_uq4(
        const void * U,             // fp16 [m, k]
        const float * scales,       // fp32 [k, n/32]
        const unsigned char * packed, // [k, n/2]
        int m, int k, int n,
        const float * x,            // [n]
        float * y,                  // [m]
        float * t_scratch,          // [k] device scratch (zeroed by caller)
        cudaStream_t stream) {
    if (k <= 0 || m <= 0 || n % 32 != 0) return 1;
    factored_c_gemv_kernel<<<k, 256, 0, stream>>>(scales, packed, k, n, x, t_scratch);
    factored_u_gemv_kernel<<<(m + 255) / 256, 256, 0, stream>>>(
            (const __half *) U, m, k, t_scratch, y);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int) err;
}
