/* factored_gemv_ref.c — portable CPU twin of the CUDA fused factored GEMV.
 *
 *   y = U @ (C @ x)
 *     U : fp16 basis  [m, k]
 *     C : uq4 codes   [k, n]   value = scale * (code - 8); scale = amax/8
 *                              per 32-col block; 2 codes/byte, lo nibble first.
 *
 * Same data layout and call shape as factored_gemv_uq4 (CUDA), so the
 * geodessical runtime can select CPU/CUDA backends behind one entry point.
 * Build:  cl /O2 /LD factored_gemv_ref.c   (or compile with -DUT_REF_MAIN for
 * a self-test executable). Scalar v1; the AVX2/AVX-512 twin lands with the
 * runtime JIT work (Phase 3).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define UT_EXPORT __declspec(dllexport)
#else
#define UT_EXPORT
#endif

static inline float ut_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) return sign ? -0.0f : 0.0f;
        /* subnormal: normalize */
        int e = -14;
        while (!(man & 0x400u)) { man <<= 1; e--; }
        man &= 0x3FFu;
        bits = ((uint32_t) (e + 127) << 23) | (man << 13) | sign;
    } else if (exp == 31) {
        bits = 0x7F800000u | (man << 13) | sign; /* inf/nan */
    } else {
        bits = ((exp + 112) << 23) | (man << 13) | sign;
    }
    float f;
    uint32_t b = bits;
    memcpy(&f, &b, sizeof(f));
    return f;
}

UT_EXPORT int factored_gemv_uq4_cpu(
        const void * U,             /* fp16 [m, k] */
        const float * scales,       /* fp32 [k, n/32] */
        const uint8_t * packed,     /* [k, n/2] */
        int m, int k, int n,
        const float * x,            /* [n] */
        float * y,                  /* [m] */
        float * t_scratch)          /* [k] */
{
    const int nblocks = n / 32;
    for (int r = 0; r < k; r++) {
        const float * srow = scales + (size_t) r * nblocks;
        const uint8_t * prow = packed + (size_t) r * (n / 2);
        float acc = 0.0f;
        for (int b = 0; b < nblocks; b++) {
            const float sc = srow[b];
            const float * xb = x + b * 32;
            const uint8_t * pb = prow + b * 16;
            for (int j = 0; j < 16; j++) {
                const uint8_t byte = pb[j];
                acc += xb[2 * j] * (float) ((byte & 15) - 8) * sc;
                acc += xb[2 * j + 1] * (float) ((byte >> 4) - 8) * sc;
            }
        }
        t_scratch[r] = acc;
    }
    for (int row = 0; row < m; row++) {
        const uint16_t * Urow = (const uint16_t *) U + (size_t) row * k;
        float acc = 0.0f;
        for (int j = 0; j < k; j++) {
            acc += ut_f16_to_f32(Urow[j]) * t_scratch[j];
        }
        y[row] = acc;
    }
    return 0;
}

#ifdef UT_REF_MAIN
/* self-test: k=2, n=64, m=3 with hand-checked pattern */
int main(void) {
    uint16_t U[6] = {0x3C00, 0x4000, 0x4400, 0x3C00, 0x3C00, 0x3C00};
    /* U = [1 2] [4 1] [1 1] */
    float scales[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    uint8_t packed[64] = {0};
    float x[64], y[3], t[2];
    float exp[3];
    for (int i = 0; i < 64; i++) x[i] = 1.0f;
    /* row0: codes 10 (=2) for all 32 cols of block0, 9 (=1) for block1 */
    for (int j = 0; j < 16; j++) packed[j] = 10 | (10 << 4);
    for (int j = 16; j < 32; j++) packed[j] = 9 | (9 << 4);
    /* row1: codes 8 (=0) and 14 (=6) */
    for (int j = 0; j < 16; j++) packed[32 + j] = 8 | (8 << 4);
    for (int j = 16; j < 32; j++) packed[32 + j] = 14 | (14 << 4);
    /* t = [32*0.5*2 + 32*0.5*1, 32*0.5*0 + 32*0.5*6] = [48, 96] */
    exp[0] = 1.0f * 48.0f + 2.0f * 96.0f;   /* 240 */
    exp[1] = 4.0f * 48.0f + 1.0f * 96.0f;   /* 288 */
    exp[2] = 1.0f * 48.0f + 1.0f * 96.0f;   /* 144 */
    factored_gemv_uq4_cpu(U, scales, packed, 3, 2, 64, x, y, t);
    int ok = 1;
    for (int i = 0; i < 3; i++) {
        printf("y[%d] = %.3f (expect %.3f)\n", i, y[i], exp[i]);
        if (y[i] != exp[i]) ok = 0;
    }
    printf(ok ? "SELF-TEST OK\n" : "SELF-TEST FAILED\n");
    return ok ? 0 : 1;
}
#endif
