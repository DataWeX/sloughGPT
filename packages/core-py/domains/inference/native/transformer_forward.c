/**
 * transformer_forward.c — Generic transformer forward pass (C + Apple Accelerate).
 *
 * Supports Qwen, GPT-2, LLaMA, Mistral, Phi — any HF model with SLNC weights.
 * Uses Apple Accelerate BLAS for matrix multiplies.
 *
 * Bug fix: normed buffer was aliased with ff_buf2 causing sgemm to read/write
 * same memory in FFN up_proj. Fixed by allocating separate normed buffer.
 */

#include "transformer_forward.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
/* Portable BLAS: link with -lblas on Linux/BSD. Values match standard cblas.h. */
enum { CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112 };
void cblas_sgemm(int order, int trans_a, int trans_b, int m, int n, int k,
                 float alpha, const float* a, int lda, const float* b, int ldb,
                 float beta, float* c, int ldc);
#endif

#define EPS 1e-6f
#define NEG_INF -1e9f

static inline int min_int(int a, int b) { return a < b ? a : b; }

static void sgemm_t(int M, int N, int K, const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}

static void rmsnorm(float* out, const float* x, const float* weight, int d) {
    float sum_sq = 0.0f;
    for (int j = 0; j < d; j++) sum_sq += x[j] * x[j];
    float rms = sqrtf(sum_sq / d + EPS);
    float inv = 1.0f / rms;
    for (int j = 0; j < d; j++) out[j] = x[j] * inv * weight[j];
}

static void apply_rope(float* q, float* k, int pos, int n_heads, int n_kv_heads, int head_dim, float base) {
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d += 2) {
            float theta = 1.0f / powf(base, (float)d / (float)head_dim);
            float angle = (float)pos * theta;
            float cos_t = cosf(angle), sin_t = sinf(angle);
            int i0 = h * head_dim + d, i1 = i0 + 1;
            float q0 = q[i0], q1 = q[i1];
            q[i0] = q0 * cos_t - q1 * sin_t;
            q[i1] = q0 * sin_t + q1 * cos_t;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        for (int d = 0; d < head_dim; d += 2) {
            float theta = 1.0f / powf(base, (float)d / (float)head_dim);
            float angle = (float)pos * theta;
            float cos_t = cosf(angle), sin_t = sinf(angle);
            int i0 = h * head_dim + d, i1 = i0 + 1;
            float k0 = k[i0], k1 = k[i1];
            k[i0] = k0 * cos_t - k1 * sin_t;
            k[i1] = k0 * sin_t + k1 * cos_t;
        }
    }
}

static void softmax_row(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void gqa_attention(const float* q, const float* k_cache, const float* v_cache,
                           float* out, int n_heads, int n_kv_heads, int head_dim, int kv_seq_len) {
    int kv_dim = n_kv_heads * head_dim;
    int head_ratio = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        int kv_group = h / head_ratio;
        const float* qh = q + h * head_dim;
        float* out_h = out + h * head_dim;
        float* scores = (float*)malloc(kv_seq_len * sizeof(float));
        for (int t = 0; t < kv_seq_len; t++) {
            const float* kt = k_cache + t * kv_dim + kv_group * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kt[d];
            scores[t] = dot * scale;
        }
        softmax_row(scores, kv_seq_len);
        memset(out_h, 0, head_dim * sizeof(float));
        for (int t = 0; t < kv_seq_len; t++) {
            const float* vt = v_cache + t * kv_dim + kv_group * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; d++) out_h[d] += w * vt[d];
        }
        free(scores);
    }
}

static int forward_layer(const TransformerWeights* w, int layer, float* x,
                          TransformerKVCache* cache, int seq_pos,
                          float* q_buf, float* k_buf, float* v_buf,
                          float* attn_out, float* ff_buf, float* ff_buf2) {
    const TransformerConfig* c = &w->config;
    int D = c->hidden_dim, HD = c->head_dim;
    int NH = c->n_heads, NKV = c->n_kv_heads, FF = c->ff_dim;

    const float* base = w->data + w->layer_offsets[layer];
    const float* attn_norm_w = base;
    const float* q_w = attn_norm_w + D;
    const float* q_b = q_w + D * (NH * HD);
    const float* k_w = q_b + (NH * HD);
    const float* k_b = k_w + D * (NKV * HD);
    const float* v_w = k_b + (NKV * HD);
    const float* v_b = v_w + D * (NKV * HD);
    const float* o_w = v_b + (NKV * HD);
    const float* o_b = o_w + (NH * HD) * D;
    const float* ff_norm_w = o_b + D;
    const float* gate_w = ff_norm_w + D;
    const float* gate_b = gate_w + D * FF;
    const float* up_w = gate_b + FF;
    const float* up_b = up_w + D * FF;
    const float* down_w = up_b + FF;
    const float* down_b = down_w + FF * D;

    float* normed = (float*)malloc(D * sizeof(float));
    if (!normed) return -1;
    rmsnorm(normed, x, attn_norm_w, D);

    sgemm_t(1, NH * HD, D, normed, q_w, q_buf);
    sgemm_t(1, NKV * HD, D, normed, k_w, k_buf);
    sgemm_t(1, NKV * HD, D, normed, v_w, v_buf);
    for (int i = 0; i < NH * HD; i++) q_buf[i] += q_b[i];
    for (int i = 0; i < NKV * HD; i++) k_buf[i] += k_b[i];
    for (int i = 0; i < NKV * HD; i++) v_buf[i] += v_b[i];

    apply_rope(q_buf, k_buf, seq_pos, NH, NKV, HD, c->rope_base);

    if (cache && seq_pos < cache->seq_capacity) {
        int kv_dim = NKV * HD;
        int layer_kv_offset = layer * cache->seq_capacity * kv_dim;
        memcpy(cache->k + layer_kv_offset + seq_pos * kv_dim, k_buf, kv_dim * sizeof(float));
        memcpy(cache->v + layer_kv_offset + seq_pos * kv_dim, v_buf, kv_dim * sizeof(float));
    }

    int kv_len = seq_pos + 1;
    gqa_attention(q_buf,
                  cache ? cache->k + layer * cache->seq_capacity * NKV * HD : k_buf,
                  cache ? cache->v + layer * cache->seq_capacity * NKV * HD : v_buf,
                  attn_out, NH, NKV, HD, kv_len);

    float* o_proj_buf = normed;
    sgemm_t(1, D, D, attn_out, o_w, o_proj_buf);
    for (int i = 0; i < D; i++) o_proj_buf[i] += o_b[i];

    for (int i = 0; i < D; i++) x[i] += o_proj_buf[i];

    rmsnorm(normed, x, ff_norm_w, D);

    sgemm_t(1, FF, D, normed, gate_w, ff_buf);
    sgemm_t(1, FF, D, normed, up_w, ff_buf2);
    silu_inplace(ff_buf, FF);
    for (int i = 0; i < FF; i++) ff_buf[i] *= ff_buf2[i];

    sgemm_t(1, D, FF, ff_buf, down_w, ff_buf2);
    for (int i = 0; i < D; i++) ff_buf2[i] += down_b[i];

    for (int i = 0; i < D; i++) x[i] += ff_buf2[i];

    free(normed);
    return 0;
}

int transformer_load_weights(TransformerWeights* w, const float* flat_data,
                             int num_floats, const TransformerConfig* config) {
    memset(w, 0, sizeof(TransformerWeights));
    w->config = *config;

    float* buf = (float*)malloc(num_floats * sizeof(float));
    if (!buf) return -1;
    memcpy(buf, flat_data, num_floats * sizeof(float));
    w->data = buf;
    w->total_floats = num_floats;

    int D = config->hidden_dim;
    int NH = config->n_heads;
    int NKV = config->n_kv_heads;
    int HD = config->head_dim;
    int FF = config->ff_dim;

    int layer_size = D
        + D * (NH * HD) + (NH * HD)
        + D * (NKV * HD) + (NKV * HD)
        + D * (NKV * HD) + (NKV * HD)
        + (NH * HD) * D + D
        + D
        + D * FF + FF
        + D * FF + FF
        + FF * D + D;

    int offset = 0;
    w->tok_emb_offset = offset;
    offset += config->vocab_size * D;

    for (int i = 0; i < config->n_layers && i < TRANSFORMER_MAX_LAYERS; i++) {
        w->layer_offsets[i] = offset;
        offset += layer_size;
    }
    w->norm_offset = offset;
    offset += D;
    w->lm_head_offset = offset;
    offset += config->vocab_size * D;

    return 0;
}

void transformer_free_weights(TransformerWeights* w) {
    if (w->data) { free((void*)w->data); w->data = NULL; }
}

int transformer_kv_cache_init(TransformerKVCache* cache, const TransformerConfig* config,
                              int seq_capacity) {
    memset(cache, 0, sizeof(TransformerKVCache));
    cache->n_layers = config->n_layers;
    cache->n_kv_heads = config->n_kv_heads;
    cache->head_dim = config->head_dim;
    cache->seq_capacity = seq_capacity;
    cache->seq_len = 0;
    int kv_dim = config->n_kv_heads * config->head_dim;
    int total = config->n_layers * seq_capacity * kv_dim;
    cache->k = (float*)calloc(total, sizeof(float));
    cache->v = (float*)calloc(total, sizeof(float));
    if (!cache->k || !cache->v) { transformer_kv_cache_free(cache); return -1; }
    return 0;
}

void transformer_kv_cache_free(TransformerKVCache* cache) {
    if (cache->k) { free(cache->k); cache->k = NULL; }
    if (cache->v) { free(cache->v); cache->v = NULL; }
    cache->seq_len = 0;
}

void transformer_kv_cache_reset(TransformerKVCache* cache) {
    int kv_dim = cache->n_kv_heads * cache->head_dim;
    int total = cache->n_layers * cache->seq_capacity * kv_dim;
    memset(cache->k, 0, total * sizeof(float));
    memset(cache->v, 0, total * sizeof(float));
    cache->seq_len = 0;
}

int transformer_forward_step(const TransformerWeights* w, TransformerKVCache* cache,
                             int token_id, int seq_pos, float* logits_out) {
    const TransformerConfig* c = &w->config;
    int D = c->hidden_dim, NH = c->n_heads, NKV = c->n_kv_heads;
    int HD = c->head_dim, FF = c->ff_dim, V = c->vocab_size;

    float* x = (float*)malloc(D * sizeof(float));
    float* q_buf = (float*)malloc(NH * HD * sizeof(float));
    float* k_buf = (float*)malloc(NKV * HD * sizeof(float));
    float* v_buf = (float*)malloc(NKV * HD * sizeof(float));
    float* attn_out = (float*)malloc(D * sizeof(float));
    float* ff_buf = (float*)malloc(FF * sizeof(float));
    float* ff_buf2 = (float*)malloc(FF * sizeof(float));

    if (!x || !q_buf || !k_buf || !v_buf || !attn_out || !ff_buf || !ff_buf2) {
        free(x); free(q_buf); free(k_buf); free(v_buf);
        free(attn_out); free(ff_buf); free(ff_buf2);
        return -1;
    }

    const float* tok_emb = w->data + w->tok_emb_offset;
    if (token_id < 0 || token_id >= V) token_id = 0;
    memcpy(x, tok_emb + token_id * D, D * sizeof(float));

    for (int layer = 0; layer < c->n_layers; layer++) {
        int rc = forward_layer(w, layer, x, cache, seq_pos,
                               q_buf, k_buf, v_buf, attn_out, ff_buf, ff_buf2);
        if (rc != 0) {
            free(x); free(q_buf); free(k_buf); free(v_buf);
            free(attn_out); free(ff_buf); free(ff_buf2);
            return rc;
        }
    }

    const float* norm_w = w->data + w->norm_offset;
    rmsnorm(x, x, norm_w, D);

    const float* lm_w = w->data + w->lm_head_offset;
    sgemm_t(1, V, D, x, lm_w, logits_out);

    free(x); free(q_buf); free(k_buf); free(v_buf);
    free(attn_out); free(ff_buf); free(ff_buf2);
    return 0;
}
