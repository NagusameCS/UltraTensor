/* expert_gemv.c — dispatch-aware MoE expert executor (Phase 3b).
 *
 * Opens a GGUF shard, locates a 3-D expert tensor (blk.<L>.ffn_*_exps.weight)
 * and computes per-expert GEMVs without ever materializing more than a few
 * rows: y = W_e @ x   (or W_e^T @ x). This is the compute primitive behind
 * lazy top-k serving — only the routed experts are decoded.
 *
 * Supported expert storage types: Q8_0 (id 8) and Q3_K (id 11), plus Q2_K
 * (id 10) / Q4_K (12) / Q5_K (13) / Q6_K (14) via the same row-decode
 * interface (q4_K/q5_K/q6_K/q2_K ports below); decode verified against
 * llama.cpp via the oracle files (see tests/).
 *
 * Entry points (dllexport):
 *   ut_expert_open(path, name, &store)   - parse header, locate tensor
 *   ut_expert_gemv(&store, e, x, y, transpose)
 *   ut_expert_close(&store)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define UT_EXPORT __declspec(dllexport)
#else
#define UT_EXPORT
#endif

typedef struct {
    FILE * f;
    uint64_t data_start;   /* absolute file offset of the data section */
    uint64_t tensor_off;   /* tensor offset relative to data_start     */
    uint32_t ttype;        /* gguf type id                             */
    int n, m, E;           /* dims (ne0 first)                         */
    int row_blocks;        /* blocks per row                           */
    int block_align;       /* elements per block (32 or 256)           */
    int block_bytes;       /* bytes per block                          */
} ut_expert_t;

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {                       /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (man << 13);   /* inf/nan */
    } else {
        f = sign | ((exp + 112) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static int read_exact(FILE * f, void * buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

/* Windows fseek takes a 32-bit long; expert offsets exceed 4 GB.
   _fseeki64 fixes this (and fseeko elsewhere). */
static int ut_seek(FILE * f, int64_t off) {
#ifdef _WIN32
    return _fseeki64(f, off, SEEK_SET) ? -1 : 0;
#else
    return fseeko(f, (off_t) off, SEEK_SET) ? -1 : 0;
#endif
}

static int skip_kv_value(FILE * f, uint32_t vtype) {
    if (vtype == 8) {                        /* STRING */
        uint64_t slen;
        if (read_exact(f, &slen, 8)) return -1;
        return fseek(f, (long) slen, SEEK_CUR) ? -1 : 0;
    }
    if (vtype == 9) {                        /* ARRAY */
        uint32_t etype;
        uint64_t cnt;
        if (read_exact(f, &etype, 4) || read_exact(f, &cnt, 8)) return -1;
        for (uint64_t j = 0; j < cnt; j++) {
            size_t esz = (etype == 0 || etype == 1 || etype == 7) ? 1 :
                         (etype == 2 || etype == 3) ? 2 :
                         (etype == 4 || etype == 5 || etype == 6) ? 4 : 8;
            if (etype == 8) {
                uint64_t slen;
                if (read_exact(f, &slen, 8)) return -1;
                if (fseek(f, (long) slen, SEEK_CUR)) return -1;
            } else if (fseek(f, (long) esz, SEEK_CUR)) {
                return -1;
            }
        }
        return 0;
    }
    size_t vsz = (vtype == 0 || vtype == 1 || vtype == 7) ? 1 :
                 (vtype == 2 || vtype == 3) ? 2 :
                 (vtype == 4 || vtype == 5 || vtype == 6) ? 4 : 8;
    return fseek(f, (long) vsz, SEEK_CUR) ? -1 : 0;
}

static void expert_type_params(uint32_t ttype, int * align, int * bytes) {
    switch (ttype) {
        case 8:  *align = 32;  *bytes = 34;  break;  /* Q8_0  */
        case 10: *align = 256; *bytes = 84;  break;  /* Q2_K  */
        case 11: *align = 256; *bytes = 110; break;  /* Q3_K  */
        case 12: *align = 256; *bytes = 144; break;  /* Q4_K  */
        case 13: *align = 256; *bytes = 176; break;  /* Q5_K  */
        case 14: *align = 256; *bytes = 210; break;  /* Q6_K  */
        default: *align = 0;   *bytes = 0;   break;
    }
}

UT_EXPORT int ut_expert_open(const char * path, const char * name,
                             ut_expert_t * out) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    memset(out, 0, sizeof(*out));
    int rc = -1;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (read_exact(f, &magic, 4) || magic != 0x46554747u) goto done;
    if (read_exact(f, &version, 4)) goto done;
    if (read_exact(f, &n_tensors, 8)) goto done;
    if (read_exact(f, &n_kv, 8)) goto done;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen;
        uint32_t vtype;
        if (read_exact(f, &klen, 8)) goto done;
        if (ut_seek(f, (int64_t) ftell(f) + (int64_t) klen)) goto done;
        if (read_exact(f, &vtype, 4)) goto done;
        if (skip_kv_value(f, vtype)) goto done;
    }

    uint64_t target_off = 0;
    uint64_t dims[3] = {0, 0, 0};
    uint32_t nd = 0, ttype = 0;
    int found = 0;
    uint64_t infos_start = ftell(f);
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        char tname[256];
        uint32_t tnd, tt;
        uint64_t td[8], toff;
        if (read_exact(f, &nlen, 8) || nlen >= sizeof(tname)) goto done;
        if (read_exact(f, tname, nlen)) goto done;
        tname[nlen] = 0;
        if (read_exact(f, &tnd, 4) || tnd > 8) goto done;
        for (uint32_t j = 0; j < tnd; j++) {
            if (read_exact(f, &td[j], 8)) goto done;
        }
        if (read_exact(f, &tt, 4) || read_exact(f, &toff, 8)) goto done;
        if (!found && strcmp(tname, name) == 0) {
            found = 1;
            target_off = toff;
            nd = tnd;
            ttype = tt;
            for (uint32_t j = 0; j < tnd && j < 3; j++) dims[j] = td[j];
        }
    }
    if (!found || nd != 3) goto done;
    int align, bytes;
    expert_type_params(ttype, &align, &bytes);
    if (align == 0 || (dims[0] % (uint64_t) align) != 0) goto done;

    /* recompute data start: end of header = infos_start + info_size,
       aligned up to 32 */
    if (ut_seek(f, (int64_t) infos_start)) goto done;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        uint32_t tnd;
        if (read_exact(f, &nlen, 8)) goto done;
        if (ut_seek(f, (int64_t) ftell(f) + (int64_t) nlen)) goto done;
        if (read_exact(f, &tnd, 4)) goto done;
        if (ut_seek(f, (int64_t) ftell(f) + (int64_t) (8 * tnd + 4 + 8))) goto done;
    }
    out->data_start = (ftell(f) + 31) & ~(uint64_t) 31;
    out->f = f;
    out->tensor_off = target_off;
    out->ttype = ttype;
    out->n = (int) dims[0];
    out->m = (int) dims[1];
    out->E = (int) dims[2];
    out->block_align = align;
    out->block_bytes = bytes;
    out->row_blocks = out->n / align;
    rc = 0;
done:
    if (rc != 0 && f) fclose(f);
    return rc;
}

UT_EXPORT void ut_expert_close(ut_expert_t * t) {
    if (t->f) fclose(t->f);
    memset(t, 0, sizeof(*t));
}

/* ---- block decoders: fill row_buf[n] for one row ---------------------- */

static void decode_q8_0(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 34;
        float d = fp16_to_fp32((uint16_t) (p[0] | (p[1] << 8)));
        for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float) (int8_t) p[2 + i];
    }
}

/* Q3_K scale unpack: 12 bytes -> 16 int8 (little-endian u32 shuffle) */
static void q3k_scales(const uint8_t * s12, int8_t * out16) {
    uint32_t a0, a1, a2, b0, b1, b2, b3;
    memcpy(&a0, s12, 4);
    memcpy(&a1, s12 + 4, 4);
    memcpy(&a2, s12 + 8, 4);
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    b0 = (a0 & kmask2) | ((a2 & kmask1) << 4);
    b1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4);
    b2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4);
    b3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4);
    memcpy(out16, &b0, 4);
    memcpy(out16 + 4, &b1, 4);
    memcpy(out16 + 8, &b2, 4);
    memcpy(out16 + 12, &b3, 4);
}

static void decode_q3_K(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 110;
        const uint8_t * hm = p, *qs = p + 32;
        int8_t sc[16];
        q3k_scales(p + 96, sc);
        float d = fp16_to_fp32((uint16_t) (p[108] | (p[109] << 8)));
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = ((qs[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    out[b * 256 + n + j * 32 + l] = dl * (float) q;
                }
                dl = d * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = ((qs[l + 16] >> shift) & 3)
                            - ((hm[l + 16] & m) ? 0 : 4);
                    out[b * 256 + n + j * 32 + 16 + l] = dl * (float) q;
                }
                shift += 2;
                m <<= 1;
            }
            qs += 32;
        }
    }
}

/* fused Q3_K decode + dot: y = row . x, no fp32 materialization */
static float q3k_dot(const uint8_t * blk, int nb, const float * x) {
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 110;
        const uint8_t * hm = p, *qs = p + 32;
        int8_t sc[16];
        q3k_scales(p + 96, sc);
        float d = fp16_to_fp32((uint16_t) (p[108] | (p[109] << 8)));
        const float * xb = x + b * 256;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = ((qs[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
                    acc += dl * (float) q * xb[n + j * 32 + l];
                }
                dl = d * (float) (sc[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = ((qs[l + 16] >> shift) & 3)
                            - ((hm[l + 16] & m) ? 0 : 4);
                    acc += dl * (float) q * xb[n + j * 32 + 16 + l];
                }
                shift += 2;
                m <<= 1;
            }
            qs += 32;
        }
    }
    return acc;
}

/* Q4_K/Q5_K scale/min unpack: get_scale_min_k4 from llama.cpp */
static void get_scale_min_k4(int j, const uint8_t * q, int * d, int * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/* ---- AVX2 fused decode+dot (Q3_K / Q5_K: the V4-Pro hot paths) ------- */
#if defined(__AVX2__)
#include <immintrin.h>

static inline float ut_hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* 16 elements: code = ((qs[l] >> shift) & 3) - (hm[l] & m ? 0 : 4) */
static inline __m256 q3k_half16(__m256 acc, const uint8_t * qs,
                                const uint8_t * hm, const float * x16,
                                int shift, int m, float dl) {
    __m256i q = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs));
    q = _mm256_srli_epi16(q, shift);
    q = _mm256_and_si256(q, _mm256_set1_epi16(3));
    __m256i h = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) hm));
    h = _mm256_and_si256(h, _mm256_set1_epi16(m));
    __m256i sub = _mm256_and_si256(_mm256_cmpeq_epi16(h, _mm256_setzero_si256()),
                                   _mm256_set1_epi16(4));
    q = _mm256_sub_epi16(q, sub);
    __m256i qlo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q));
    __m256i qhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q, 1));
    const __m256 dlv = _mm256_set1_ps(dl);
    __m256 vlo = _mm256_mul_ps(_mm256_cvtepi32_ps(qlo), dlv);
    __m256 vhi = _mm256_mul_ps(_mm256_cvtepi32_ps(qhi), dlv);
    acc = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(x16), acc);
    return _mm256_fmadd_ps(vhi, _mm256_loadu_ps(x16 + 8), acc);
}

static float q3k_dot_avx2(const uint8_t * blk, int nb, const float * x) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 110;
        const uint8_t * hm = p, *qs = p + 32;
        int8_t sc[16];
        q3k_scales(p + 96, sc);
        const float d = fp16_to_fp32((uint16_t) (p[108] | (p[109] << 8)));
        const float * xb = x + b * 256;
        int is = 0;
        int m = 1;
        for (int n = 0; n < 256; n += 128) {
            for (int j = 0; j < 4; j++) {
                const int shift = 2 * j;
                acc = q3k_half16(acc, qs, hm, xb + n + j * 32, shift, m,
                                 d * (float) (sc[is++] - 32));
                acc = q3k_half16(acc, qs + 16, hm + 16, xb + n + j * 32 + 16,
                                 shift, m, d * (float) (sc[is++] - 32));
                m <<= 1;
            }
            qs += 32;
        }
    }
    return ut_hsum(acc);
}

/* 32 elements from one 32-byte ql chunk: code = nibble (+16 if qh&u).
   value = dd*code - mm; hi selects the high nibble. */
static inline __m256 q5k_half32(__m256 acc, const uint8_t * ql,
                                const uint8_t * qh, const float * x32,
                                int u, float dd, float mm, int hi) {
    const __m256 ddv = _mm256_set1_ps(dd);
    const __m256 mmv = _mm256_set1_ps(-mm);
    for (int c = 0; c < 2; c++) {
        __m256i q = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((const __m128i *) (ql + 16 * c)));
        if (hi) q = _mm256_srli_epi16(q, 4);
        q = _mm256_and_si256(q, _mm256_set1_epi16(15));
        __m256i h = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((const __m128i *) (qh + 16 * c)));
        h = _mm256_and_si256(h, _mm256_set1_epi16(u));
        __m256i add16 = _mm256_and_si256(
            _mm256_cmpgt_epi16(h, _mm256_setzero_si256()),
            _mm256_set1_epi16(16));
        q = _mm256_add_epi16(q, add16);
        __m256i qlo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q));
        __m256i qhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q, 1));
        __m256 vlo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qlo), ddv, mmv);
        __m256 vhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qhi), ddv, mmv);
        acc = _mm256_fmadd_ps(vlo, _mm256_loadu_ps(x32 + 16 * c), acc);
        acc = _mm256_fmadd_ps(vhi, _mm256_loadu_ps(x32 + 16 * c + 8), acc);
    }
    return acc;
}

static float q5k_dot_avx2(const uint8_t * blk, int nb, const float * x) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 176;
        const float d = fp16_to_fp32((uint16_t) (p[0] | (p[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t) (p[2] | (p[3] << 8)));
        const uint8_t * ql = p + 48;
        const uint8_t * qh = p + 16;
        const float * xb = x + b * 256;
        int is = 0;
        int u1 = 1, u2 = 2;
        for (int n = 0; n < 256; n += 64) {
            int sc, mn;
            get_scale_min_k4(is + 0, p + 4, &sc, &mn);
            const float d1 = d * (float) sc, m1 = dmin * (float) mn;
            get_scale_min_k4(is + 1, p + 4, &sc, &mn);
            const float d2 = d * (float) sc, m2 = dmin * (float) mn;
            acc = q5k_half32(acc, ql, qh, xb + n, u1, d1, m1, 0);
            acc = q5k_half32(acc, ql, qh, xb + n + 32, u2, d2, m2, 1);
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return ut_hsum(acc);
}
#endif  /* __AVX2__ */

static void decode_q2_K(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 84;
        float d = fp16_to_fp32((uint16_t) (p[0] | (p[1] << 8)));
        float dmin = fp16_to_fp32((uint16_t) (p[2] | (p[3] << 8)));
        const uint8_t * q = p + 20;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = p[4 + is++];
                float dl = d * (float) (sc & 0xF);
                float ml = dmin * (float) (sc >> 4);
                for (int l = 0; l < 16; l++)
                    out[b * 256 + n + j * 32 + l] =
                        dl * (float) ((q[l] >> shift) & 3) - ml;
                sc = p[4 + is++];
                dl = d * (float) (sc & 0xF);
                ml = dmin * (float) (sc >> 4);
                for (int l = 0; l < 16; l++)
                    out[b * 256 + n + j * 32 + 16 + l] =
                        dl * (float) ((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

static void decode_q4_K(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 144;
        float d = fp16_to_fp32((uint16_t) (p[0] | (p[1] << 8)));
        float dmin = fp16_to_fp32((uint16_t) (p[2] | (p[3] << 8)));
        const uint8_t * q = p + 16;
        int is = 0;
        for (int n = 0; n < 256; n += 64) {
            int sc, m;
            get_scale_min_k4(is + 0, p + 4, &sc, &m);
            float d1 = d * (float) sc, m1 = dmin * (float) m;
            get_scale_min_k4(is + 1, p + 4, &sc, &m);
            float d2 = d * (float) sc, m2 = dmin * (float) m;
            for (int l = 0; l < 32; l++)
                out[b * 256 + n + l] = d1 * (float) (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++)
                out[b * 256 + n + 32 + l] = d2 * (float) (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void decode_q5_K(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 176;
        float d = fp16_to_fp32((uint16_t) (p[0] | (p[1] << 8)));
        float dmin = fp16_to_fp32((uint16_t) (p[2] | (p[3] << 8)));
        const uint8_t * ql = p + 48;
        const uint8_t * qh = p + 16;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int n = 0; n < 256; n += 64) {
            int sc, m;
            get_scale_min_k4(is + 0, p + 4, &sc, &m);
            float d1 = d * (float) sc, m1 = dmin * (float) m;
            get_scale_min_k4(is + 1, p + 4, &sc, &m);
            float d2 = d * (float) sc, m2 = dmin * (float) m;
            for (int l = 0; l < 32; l++)
                out[b * 256 + n + l] =
                    d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                out[b * 256 + n + 32 + l] =
                    d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void decode_q6_K(const uint8_t * blk, int nb, float * out) {
    for (int b = 0; b < nb; b++) {
        const uint8_t * p = blk + (size_t) b * 210;
        float d = fp16_to_fp32((uint16_t) (p[208] | (p[209] << 8)));
        const uint8_t * ql = p;
        const uint8_t * qh = p + 128;
        const int8_t * sc = (const int8_t *) (p + 160);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t) ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                int8_t q2 = (int8_t) ((ql[l + 32] & 0xF)
                                     | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t) ((ql[l] >> 4)
                                     | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t) ((ql[l + 32] >> 4)
                                     | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[b * 256 + n + l] = d * sc[is] * q1;
                out[b * 256 + n + 32 + l] = d * sc[is + 2] * q2;
                out[b * 256 + n + 64 + l] = d * sc[is + 4] * q3;
                out[b * 256 + n + 96 + l] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static int decode_row(ut_expert_t * t, const uint8_t * row_blk, float * out) {
    switch (t->ttype) {
        case 8:  decode_q8_0(row_blk, t->row_blocks, out); return 0;
        case 10: decode_q2_K(row_blk, t->row_blocks, out); return 0;
        case 11: decode_q3_K(row_blk, t->row_blocks, out); return 0;
        case 12: decode_q4_K(row_blk, t->row_blocks, out); return 0;
        case 13: decode_q5_K(row_blk, t->row_blocks, out); return 0;
        case 14: decode_q6_K(row_blk, t->row_blocks, out); return 0;
        default: return -2;   /* unsupported type */
    }
}

/* y = W_e @ x (transpose == 0: y[m]; else y[n]) with streaming decode.
   The whole expert is read in ONE fread (scattered small reads are the
   dominant cost on disk-bound boxes), then rows are decoded in parallel
   (OpenMP, compile with /openmp): per-thread scratch, per-thread
   partial sums for the transpose path. */
#ifdef _OPENMP
#include <omp.h>
#endif

/* Thread pool for the row loop. MSVC classic OpenMP (/openmp) corrupts
 * the heap when this DLL is called from a Python process with other
 * active threads (native AVs in ~50% of runs; v4pro session 2026-08-15),
 * so rows are parallelized with plain Win32 threads instead. POSIX
 * builds run the loop serially. */
#ifdef _WIN32
#include <windows.h>

static int ut_num_threads(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int) si.dwNumberOfProcessors;
    if (n < 1) n = 1;
    if (n > 16) n = 16;
    return n;
}

typedef struct {
    ut_expert_t * t;
    const uint8_t * raw;
    const float * x;
    float * y;
    float * buf;
    float * partial;
    int transpose;
    int tid;
    int r0, r1;
    int rc;
} ut_row_job_t;

static DWORD WINAPI ut_row_worker(LPVOID arg) {
    ut_row_job_t * j = (ut_row_job_t *) arg;
    ut_expert_t * t = j->t;
    int rc = 0;
    for (int r = j->r0; r < j->r1; r++) {
        const size_t row_off = (size_t) r * t->row_blocks * t->block_bytes;
        const uint8_t * rowp = j->raw + row_off;
#if defined(__AVX2__)
        if (!j->transpose) {
            if (t->ttype == 11) {
                j->y[r] = q3k_dot_avx2(rowp, t->row_blocks, j->x);
                continue;
            }
            if (t->ttype == 13) {
                j->y[r] = q5k_dot_avx2(rowp, t->row_blocks, j->x);
                continue;
            }
        }
#endif
        if (!j->transpose && t->ttype == 11) {
            j->y[r] = q3k_dot(rowp, t->row_blocks, j->x);
            continue;
        }
        float * rowbuf = j->buf + (size_t) j->tid * t->n;
        if (decode_row(t, rowp, rowbuf)) {
            rc = -2;
            continue;
        }
        if (j->transpose) {
            float * py = j->partial + (size_t) j->tid * t->n;
            for (int i = 0; i < t->n; i++) py[i] += rowbuf[i] * j->x[r];
        } else {
            float acc = 0.0f;
            for (int i = 0; i < t->n; i++) acc += rowbuf[i] * j->x[i];
            j->y[r] = acc;
        }
    }
    j->rc = rc;
    return 0;
}
#endif  /* _WIN32 */

/* core: compute y = W_e @ x from an already-loaded expert buffer.
   Rows decode in parallel across a small win32 thread pool (serial on
   POSIX); AVX2 fused dots for Q3_K/Q5_K. */
static int ut_expert_gemv_raw(ut_expert_t * t, const uint8_t * raw,
                              const float * x, float * y, int transpose) {
#ifdef _WIN32
    int nthreads = ut_num_threads();
    if (nthreads > t->m) nthreads = t->m;
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif
    float * buf = (float *) malloc((size_t) nthreads * t->n * sizeof(float));
    float * partial = (float *) calloc((size_t) nthreads * t->n, sizeof(float));
    if (!buf || !partial) {
        free(buf);
        free(partial);
        return -4;
    }
    if (transpose) {
        for (int i = 0; i < t->n; i++) y[i] = 0.0f;
    }
    int rc = 0;
#ifdef _WIN32
    if (nthreads == 1) {
        ut_row_job_t job = {t, raw, x, y, buf, partial, transpose, 0,
                            0, t->m, 0};
        ut_row_worker(&job);
        rc = job.rc;
    } else {
        ut_row_job_t jobs[16];
        HANDLE hs[16];
        const int per = (t->m + nthreads - 1) / nthreads;
        for (int tid = 0; tid < nthreads; tid++) {
            jobs[tid].t = t; jobs[tid].raw = raw; jobs[tid].x = x;
            jobs[tid].y = y; jobs[tid].buf = buf; jobs[tid].partial = partial;
            jobs[tid].transpose = transpose; jobs[tid].tid = tid;
            jobs[tid].r0 = tid * per;
            jobs[tid].r1 = tid == nthreads - 1 ? t->m : (tid + 1) * per;
            jobs[tid].rc = 0;
            hs[tid] = CreateThread(NULL, 0, ut_row_worker, &jobs[tid], 0, NULL);
            if (!hs[tid]) { jobs[tid].rc = -3; }
        }
        WaitForMultipleObjects((DWORD) nthreads, hs, TRUE, INFINITE);
        for (int tid = 0; tid < nthreads; tid++) {
            if (hs[tid]) CloseHandle(hs[tid]);
            if (jobs[tid].rc && !rc) rc = jobs[tid].rc;
        }
    }
#else
    int r;
    for (r = 0; r < t->m; r++) {
        const size_t row_off = (size_t) r * t->row_blocks * t->block_bytes;
        const uint8_t * rowp = raw + row_off;
#if defined(__AVX2__)
        if (!transpose) {
            if (t->ttype == 11) {
                y[r] = q3k_dot_avx2(rowp, t->row_blocks, x);
                continue;
            }
            if (t->ttype == 13) {
                y[r] = q5k_dot_avx2(rowp, t->row_blocks, x);
                continue;
            }
        }
#endif
        if (!transpose && t->ttype == 11) {
            y[r] = q3k_dot(rowp, t->row_blocks, x);
            continue;
        }
        float * rowbuf = buf;
        if (decode_row(t, rowp, rowbuf)) {
            rc = -2;
            continue;
        }
        if (transpose) {
            for (int i = 0; i < t->n; i++) partial[i] += rowbuf[i] * x[r];
        } else {
            float acc = 0.0f;
            for (int i = 0; i < t->n; i++) acc += rowbuf[i] * x[i];
            y[r] = acc;
        }
    }
#endif
    if (rc) {
        free(buf);
        free(partial);
        return rc;
    }
    if (transpose) {
        for (int tid = 0; tid < nthreads; tid++) {
            const float * py = partial + (size_t) tid * t->n;
            for (int i = 0; i < t->n; i++) y[i] += py[i];
        }
    }
    free(buf);
    free(partial);
    return 0;
}

UT_EXPORT int ut_expert_gemv(ut_expert_t * t, int e, const float * x,
                             float * y, int transpose) {
    if (!t->f || e < 0 || e >= t->E) return -1;
    int align, bytes;
    expert_type_params(t->ttype, &align, &bytes);
    if (align == 0) return -2;
    const uint64_t expert_bytes =
        (uint64_t) t->m * t->row_blocks * t->block_bytes;
    if (ut_seek(t->f, (int64_t) (t->data_start + t->tensor_off +
                                (uint64_t) e * expert_bytes))) {
        return -3;
    }
    uint8_t * raw = (uint8_t *) malloc((size_t) expert_bytes);
    if (!raw) return -4;
    if (read_exact(t->f, raw, (size_t) expert_bytes)) {
        free(raw);
        return -5;
    }
    int rc = ut_expert_gemv_raw(t, raw, x, y, transpose);
    free(raw);
    return rc;
}

/* decode from an already-loaded expert buffer (prefetched by the caller)
   to overlap the read with the previous expert's decode */
UT_EXPORT int ut_expert_gemv_mem(ut_expert_t * t, const uint8_t * raw,
                                const float * x, float * y, int transpose) {
    if (!t->f) return -1;
    return ut_expert_gemv_raw(t, raw, x, y, transpose);
}

/* one read, B vectors: X is [B, n] row-major (n contiguous), Y [B, m]
   (or [B, n] with transpose). Each vector's rows still decode in
   parallel; the read is amortized across the batch. */
UT_EXPORT int ut_expert_gemv_batch(ut_expert_t * t, int e, const float * X,
                                   int B, float * Y, int transpose) {
    if (!t->f || e < 0 || e >= t->E) return -1;
    int align, bytes;
    expert_type_params(t->ttype, &align, &bytes);
    if (align == 0) return -2;
    const uint64_t expert_bytes =
        (uint64_t) t->m * t->row_blocks * t->block_bytes;
    if (ut_seek(t->f, (int64_t) (t->data_start + t->tensor_off +
                                (uint64_t) e * expert_bytes))) {
        return -3;
    }
    uint8_t * raw = (uint8_t *) malloc((size_t) expert_bytes);
    if (!raw) return -4;
    if (read_exact(t->f, raw, (size_t) expert_bytes)) {
        free(raw);
        return -5;
    }
    int rc = 0;
    const int out_elems = transpose ? t->n : t->m;
    for (int b = 0; b < B; b++) {
        int rcb = ut_expert_gemv_raw(t, raw, X + (size_t) b * t->n,
                                     Y + (size_t) b * out_elems, transpose);
        if (rcb && !rc) rc = rcb;
    }
    free(raw);
    return rc;
}

/* ---------------- self test: synthetic Q8_0 shard ---------------------- */
#ifdef UT_EXPERT_MAIN
#include <math.h>

static void fp32_to_fp16(float v, uint8_t * out) {
    uint32_t f;
    memcpy(&f, &v, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((f >> 23) & 0xff) - 127 + 15;
    uint32_t man = f & 0x7fffffu;
    uint16_t h;
    if (((f >> 23) & 0xff) == 0xff) {
        h = (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0));
    } else if (exp >= 31) {
        h = (uint16_t) (sign | 0x7c00u);
    } else if (exp <= 0) {
        if (exp < -10) h = (uint16_t) sign;
        else {
            man |= 0x800000u;
            man >>= 1 - exp + 13;
            h = (uint16_t) (sign | man);
        }
    } else {
        h = (uint16_t) (sign | ((uint32_t) exp << 10) | (man >> 13));
    }
    out[0] = (uint8_t) (h & 0xff);
    out[1] = (uint8_t) (h >> 8);
}

int main(void) {
    /* build a mini GGUF in memory: one Q8_0 expert tensor (n,m,E)=(64,8,2) */
    int n = 64, m = 8, E = 2;
    float W[2][8][64];
    for (int e = 0; e < E; e++)
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
                W[e][r][c] = (float) ((e + 1) * 0.5f * ((r * 7 + c * 3) % 17 - 8));

    FILE * f = fopen("ut_expert_test.gguf", "wb");
    if (!f) return 1;
    uint32_t magic = 0x46554747u, ver = 3;
    uint64_t nt = 1, nkv = 0;
    fwrite(&magic, 4, 1, f);
    fwrite(&ver, 4, 1, f);
    fwrite(&nt, 8, 1, f);
    fwrite(&nkv, 8, 1, f);
    char name[] = "blk.0.ffn_gate_exps.weight";
    uint64_t nlen = strlen(name);
    uint32_t nd = 3, ttype = 8;
    uint64_t dims[3] = {(uint64_t) n, (uint64_t) m, (uint64_t) E};
    uint64_t off = 0;
    fwrite(&nlen, 8, 1, f);
    fwrite(name, 1, nlen, f);
    fwrite(&nd, 4, 1, f);
    fwrite(dims, 8, 3, f);
    fwrite(&ttype, 4, 1, f);
    fwrite(&off, 8, 1, f);
    /* pad header to 32 */
    uint64_t pos = 4 + 4 + 8 + 8 + 8 + nlen + 4 + 24 + 4 + 8;
    while (pos % 32) { fputc(0, f); pos++; }
    for (int e = 0; e < E; e++)
        for (int r = 0; r < m; r++)
            for (int b = 0; b < n / 32; b++) {
                float amax = 0;
                for (int i = 0; i < 32; i++) {
                    float a = fabsf(W[e][r][b * 32 + i]);
                    if (a > amax) amax = a;
                }
                float d = amax / 127.0f;
                if (d == 0) d = 1.0f;
                uint8_t h16[2];
                fp32_to_fp16(d, h16);
                fwrite(h16, 1, 2, f);
                for (int i = 0; i < 32; i++) {
                    int8_t q = (int8_t) nearbyintf(W[e][r][b * 32 + i] / d);
                    fwrite(&q, 1, 1, f);
                }
            }
    fclose(f);

    ut_expert_t st;
    if (ut_expert_open("ut_expert_test.gguf", "blk.0.ffn_gate_exps.weight",
                       &st)) {
        printf("FAIL open\n");
        return 1;
    }
    float x[64], y[8];
    for (int i = 0; i < n; i++) x[i] = (float) ((i * 5) % 11 - 5);
    int fails = 0;
    for (int e = 0; e < E; e++) {
        if (ut_expert_gemv(&st, e, x, y, 0)) {
            printf("FAIL gemv e=%d\n", e);
            return 1;
        }
        for (int r = 0; r < m; r++) {
            float ref = 0;
            for (int c = 0; c < n; c++) ref += W[e][r][c] * x[c];
            if (fabsf(y[r] - ref) > 0.15f * fabsf(ref) + 1e-3f) {
                printf("MISMATCH e=%d r=%d got %.4f ref %.4f\n",
                       e, r, (double) y[r], (double) ref);
                fails++;
            }
        }
    }
    ut_expert_close(&st);
    remove("ut_expert_test.gguf");
    printf(fails ? "FAIL %d\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
#endif
