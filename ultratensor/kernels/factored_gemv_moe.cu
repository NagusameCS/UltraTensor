// factored_gemv_moe.cu — batched fused y += Σ_e U_e @ (C_e @ x) for MoE
// expert layers. Same uq4/fp16 layout as factored_gemv.cu; one grid
// dimension covers the expert batch so a routed token touches only its
// activated experts with zero extra launches.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef _WIN32
#define UT_EXPORT extern "C" __declspec(dllexport)
#else
#define UT_EXPORT extern "C"
#endif

__global__ void factored_moe_c_kernel(
        const float * __restrict__ scales,
        const unsigned char * __restrict__ packed,
        int E, int k, int n,
        const float * __restrict__ x,
        float * __restrict__ t) {           // [E*k]
    const int e = blockIdx.x;
    const int row = blockIdx.y;
    if (e >= E || row >= k) return;
    const int nblocks = n / 32;
    const float * srow = scales + ((size_t) e * k + row) * nblocks;
    const unsigned char * prow = packed + ((size_t) e * k + row) * (n / 2);
    float acc = 0.0f;
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        const float sc = srow[b];
        const float * xb = x + b * 32;
        const unsigned char * pb = prow + b * 16;
#pragma unroll
        for (int j = 0; j < 32; j += 2) {
            const unsigned char byte = pb[j >> 1];
            acc += xb[j] * (float) ((byte & 15) - 8) * sc;
            acc += xb[j + 1] * (float) ((byte >> 4) - 8) * sc;
        }
    }
    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&t[e * k + row], sm[0]);
}

__global__ void factored_moe_u_kernel(
        const __half * __restrict__ U, int E, int m, int k,
        const float * __restrict__ t, float * __restrict__ y) {
    const int e = blockIdx.x;
    if (e >= E) return;
    const __half * Ue = U + (size_t) e * m * k;
    const float * te = t + (size_t) e * k;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
        const __half * Urow = Ue + (size_t) row * k;
        float acc = 0.0f;
#pragma unroll 4
        for (int j = 0; j < k; j++) acc += __half2float(Urow[j]) * te[j];
        atomicAdd(&y[row], acc);
    }
}

UT_EXPORT int factored_gemv_moe_uq4(
        const void * U,                 // fp16 [E, m, k]
        const float * scales,           // fp32 [E, k, n/32]
        const unsigned char * packed,   // [E, k, n/2]
        int E, int m, int k, int n,
        const float * x,                // [n]
        float * y,                      // [m]  (accumulates; zero it first)
        float * t_scratch,              // [E*k] zeroed by caller
        cudaStream_t stream) {
    if (E <= 0 || k <= 0 || m <= 0 || n % 32 != 0) return 1;
    dim3 grid(E, k);
    factored_moe_c_kernel<<<grid, 256, 0, stream>>>(scales, packed, E, k, n, x,
                                                    t_scratch);
    factored_moe_u_kernel<<<E, 256, 0, stream>>>((const __half *) U, E, m, k,
                                                 t_scratch, y);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int) err;
}
