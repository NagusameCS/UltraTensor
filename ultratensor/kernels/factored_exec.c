/* factored_exec.c — C loader + executor for the UltraTensor factored GGUF
 * container. This is the reference implementation for the geodessical
 * runtime wiring (runtime/nn/gguf.c + llm.c): it parses the container,
 * finds <name>.factored_U / <name>.factored_C, and runs the portable CPU
 * kernel (factored_gemv_ref.c). v1: 2-D tensors, plain fread (mmap lands
 * with the runtime integration).
 *
 *   y = U @ (C @ x),  U fp16 [m,k], C uq4 [k,n]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "factored_gemv_ref.c"  /* brings factored_gemv_uq4_cpu */

#ifdef _WIN32
#define UT_EXPORT __declspec(dllexport)
#else
#define UT_EXPORT
#endif

#define UT_TYPE_FACTORED_C 2048u

typedef struct {
    uint16_t * U;       /* fp16 [E, m, k] */
    float * scales;     /* fp32 [E, k, n/32] */
    uint8_t * packed;   /* [E, k, n/2] */
    int E, m, k, n;
} ut_factored_tensor;

static int ut_read_exact(FILE * f, void * buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

UT_EXPORT void ut_factored_free(ut_factored_tensor * t) {
    free(t->U);
    free(t->scales);
    free(t->packed);
    memset(t, 0, sizeof(*t));
}

UT_EXPORT int ut_factored_load(const char * path, const char * name,
                               ut_factored_tensor * out) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    int rc = -1;
    memset(out, 0, sizeof(*out));

    uint32_t magic = 0, version = 0, alignment = 32;
    uint64_t n_tensors = 0, n_kv = 0;
    if (ut_read_exact(f, &magic, 4) || magic != 0x46554747u) goto done; /* "GGUF" */
    if (ut_read_exact(f, &version, 4)) goto done;
    if (ut_read_exact(f, &n_tensors, 8)) goto done;
    if (ut_read_exact(f, &n_kv, 8)) goto done;

    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen;
        char key[128];
        uint32_t vtype;
        if (ut_read_exact(f, &klen, 8)) goto done;
        if (klen < sizeof(key) && ut_read_exact(f, key, klen) == 0) {
            key[klen] = 0;
        } else {
            if (fseek(f, (long) klen, SEEK_CUR)) goto done;
            key[0] = 0;
        }
        if (ut_read_exact(f, &vtype, 4)) goto done;
        if (vtype == 8) {                       /* STRING */
            uint64_t slen;
            if (ut_read_exact(f, &slen, 8)) goto done;
            if (fseek(f, (long) slen, SEEK_CUR)) goto done;
        } else if (vtype == 9) {                /* ARRAY */
            uint32_t etype;
            uint64_t cnt;
            if (ut_read_exact(f, &etype, 4)) goto done;
            if (ut_read_exact(f, &cnt, 8)) goto done;
            for (uint64_t j = 0; j < cnt; j++) {
                size_t esz = (etype == 0 || etype == 1 || etype == 7) ? 1 :
                             (etype == 2 || etype == 3) ? 2 :
                             (etype == 4 || etype == 5 || etype == 6) ? 4 : 8;
                if (etype == 8) {
                    uint64_t slen;
                    if (ut_read_exact(f, &slen, 8)) goto done;
                    if (fseek(f, (long) slen, SEEK_CUR)) goto done;
                } else if (fseek(f, (long) esz, SEEK_CUR)) {
                    goto done;
                }
            }
        } else {
            size_t vsz = (vtype == 0 || vtype == 1 || vtype == 7) ? 1 :
                         (vtype == 2 || vtype == 3) ? 2 :
                         (vtype == 4 || vtype == 5 || vtype == 6) ? 4 : 8;
            if (fseek(f, (long) vsz, SEEK_CUR)) goto done;
        }
        if (key[0] && strcmp(key, "general.alignment") == 0) {
            /* value already skipped; re-read would be messy — we skip
             * alignment and use 32 (our writer always pads to 32). */
        }
    }

    char u_name[256], c_name[256];
    snprintf(u_name, sizeof(u_name), "%s.factored_U", name);
    snprintf(c_name, sizeof(c_name), "%s.factored_C", name);

    int found = 0;
    uint64_t u_off = 0, c_off = 0;
    uint64_t u_dims[3] = {0, 0, 0}, c_dims[3] = {0, 0, 0};
    uint32_t u_nd = 0, c_nd = 0, u_type = 0, c_type = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen;
        char tname[256];
        uint32_t nd, ttype;
        uint64_t dims[8], off;
        if (ut_read_exact(f, &nlen, 8)) goto done;
        if (nlen >= sizeof(tname)) goto done;
        if (ut_read_exact(f, tname, nlen)) goto done;
        tname[nlen] = 0;
        if (ut_read_exact(f, &nd, 4) || nd > 8) goto done;
        for (uint32_t j = 0; j < nd; j++) {
            if (ut_read_exact(f, &dims[j], 8)) goto done;
        }
        if (ut_read_exact(f, &ttype, 4)) goto done;
        if (ut_read_exact(f, &off, 8)) goto done;
        if (strcmp(tname, u_name) == 0) {
            u_off = off; u_nd = nd; u_type = ttype; found |= 1;
            for (uint32_t j = 0; j < nd && j < 3; j++) u_dims[j] = dims[j];
        } else if (strcmp(tname, c_name) == 0) {
            c_off = off; c_nd = nd; c_type = ttype; found |= 2;
            for (uint32_t j = 0; j < nd && j < 3; j++) c_dims[j] = dims[j];
        }
    }
    if (found != 3 || u_type != 1 /*F16*/ || c_type != UT_TYPE_FACTORED_C) {
        goto done;
    }
    if (u_nd == 2) u_dims[2] = 1;
    if (c_nd == 2) c_dims[2] = 1;
    /* gguf dims: U (k, m, E), C (n, k, E) */
    const int k = (int) u_dims[0];
    const int m = (int) u_dims[1];
    const int E = (int) u_dims[2];
    const int n = (int) c_dims[0];
    if ((int) c_dims[1] != k || (int) c_dims[2] != E || n % 32 != 0) goto done;

    long data_start = ftell(f);
    data_start = (data_start + alignment - 1) / alignment * alignment;

    out->E = E; out->m = m; out->k = k; out->n = n;
    out->U = (uint16_t *) malloc((size_t) E * m * k * 2);
    out->scales = (float *) malloc((size_t) E * k * (n / 32) * 4);
    out->packed = (uint8_t *) malloc((size_t) E * k * (n / 2));
    if (!out->U || !out->scales || !out->packed) goto fail_alloc;

    if (fseek(f, data_start + (long) u_off, SEEK_SET)) goto fail_alloc;
    if (ut_read_exact(f, out->U, (size_t) E * m * k * 2)) goto fail_alloc;
    if (fseek(f, data_start + (long) c_off, SEEK_SET)) goto fail_alloc;
    {
        /* C tensor is row-interleaved: per (expert, rank-row):
         * scales fp32 [n/32] then packed uint8 [n/2]. */
        const size_t nb = (size_t)(n / 32);
        const size_t row = 4 * nb + (size_t)(n / 2);
        const size_t total = (size_t)E * k * row;
        uint8_t * raw = (uint8_t *) malloc(total);
        if (!raw) goto fail_alloc;
        if (ut_read_exact(f, raw, total)) { free(raw); goto fail_alloc; }
        for (size_t er = 0; er < (size_t)E * k; er++) {
            const uint8_t * src = raw + er * row;
            memcpy(out->scales + er * nb, src, 4 * nb);
            memcpy(out->packed + er * (size_t)(n / 2), src + 4 * nb,
                   (size_t)(n / 2));
        }
        free(raw);
    }

    rc = 0;
    goto done;

fail_alloc:
    ut_factored_free(out);
done:
    fclose(f);
    return rc;
}

UT_EXPORT int ut_factored_gemv(const ut_factored_tensor * t,
                               const float * x, float * y) {
    float * tbuf = (float *) malloc((size_t) t->k * sizeof(float));
    float * ytmp = NULL;
    if (!tbuf) return -1;
    if (t->E > 1) {
        ytmp = (float *) malloc((size_t) t->m * sizeof(float));
        if (!ytmp) { free(tbuf); return -1; }
        for (int row = 0; row < t->m; row++) y[row] = 0.0f;
    }
    int rc = 0;
    for (int e = 0; e < t->E; e++) {
        const uint16_t * Ue = t->U + (size_t) e * t->m * t->k;
        const float * se = t->scales + (size_t) e * t->k * (t->n / 32);
        const uint8_t * pe = t->packed + (size_t) e * t->k * (t->n / 2);
        float * dst = t->E > 1 ? ytmp : y;
        rc = factored_gemv_uq4_cpu(Ue, se, pe, t->m, t->k, t->n, x, dst, tbuf);
        if (rc) break;
        if (t->E > 1) {
            for (int row = 0; row < t->m; row++) y[row] += dst[row];
        }
    }
    free(tbuf);
    free(ytmp);
    return rc;
}
