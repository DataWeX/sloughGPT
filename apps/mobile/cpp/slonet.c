#include "slonet.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

#define EPS 1e-5f
#define NEG_INF -1e9f

static inline int max_int(int a, int b) { return a > b ? a : b; }
static inline int min_int(int a, int b) { return a < b ? a : b; }

int slonet_load_weights(SloNetWeights* w, const float* flat_data, int num_floats,
                        int vocab_size, int n_embed, int n_layer, int n_head, int block_size) {
    memset(w, 0, sizeof(SloNetWeights));
    w->data = NULL;
    w->num_params = 0;
    w->param_offsets = NULL;
    w->param_sizes = NULL;
    w->total_floats = num_floats;
    w->vocab_size = vocab_size;
    w->n_embed = n_embed;
    w->n_layer = n_layer;
    w->n_head = n_head;
    w->block_size = block_size;
    w->head_dim = n_embed / n_head;
    w->dim_ff = ((int)(n_embed * 8 / 3) + 63) / 64 * 64;

    float* buf = (float*)malloc(num_floats * sizeof(float));
    if (!buf) return -1;
    memcpy(buf, flat_data, num_floats * sizeof(float));
    w->data = buf;

    int num_params = 0;
    for (int i = 0; i < n_layer; i++) {
        num_params += 9;
    }
    num_params += 3;
    w->num_params = num_params;
    w->param_offsets = (int*)malloc(num_params * sizeof(int));
    w->param_sizes = (int*)malloc(num_params * sizeof(int));
    if (!w->param_offsets || !w->param_sizes) {
        free(buf);
        w->data = NULL;
        return -1;
    }

    int offset = 0;
    int pi = 0;

    w->tok_emb_offset = offset;
    int tok_emb_size = vocab_size * n_embed;
    w->param_offsets[pi] = offset; w->param_sizes[pi] = tok_emb_size; pi++; offset += tok_emb_size;

    for (int i = 0; i < n_layer && i < SLONET_MAX_LAYERS; i++) {
        w->layer_offsets[i] = offset;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed; pi++; offset += n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * n_embed; pi++; offset += n_embed * n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * n_embed; pi++; offset += n_embed * n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * n_embed; pi++; offset += n_embed * n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * n_embed; pi++; offset += n_embed * n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed; pi++; offset += n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * w->dim_ff; pi++; offset += n_embed * w->dim_ff;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = w->dim_ff * n_embed; pi++; offset += w->dim_ff * n_embed;
        w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed * w->dim_ff; pi++; offset += n_embed * w->dim_ff;
    }

    w->norm_offset = offset;
    w->param_offsets[pi] = offset; w->param_sizes[pi] = n_embed; pi++; offset += n_embed;

    w->lm_head_offset = offset;
    w->param_offsets[pi] = offset; w->param_sizes[pi] = vocab_size * n_embed; pi++; offset += vocab_size * n_embed;

    return 0;
}

void slonet_unload_weights(SloNetWeights* w) {
    if (w->data) { free(w->data); w->data = NULL; }
    if (w->param_offsets) { free(w->param_offsets); w->param_offsets = NULL; }
    if (w->param_sizes) { free(w->param_sizes); w->param_sizes = NULL; }
    memset(w, 0, sizeof(SloNetWeights));
}

static void _sgemm(int M, int N, int K, const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
}

static void _rmsnorm(float* out, const float* x, const float* weight, int n, int d) {
    for (int i = 0; i < n; i++) {
        float sum_sq = 0.0f;
        const float* row = x + i * d;
        for (int j = 0; j < d; j++) sum_sq += row[j] * row[j];
        float rms = sqrtf(sum_sq / d + EPS);
        float inv = 1.0f / rms;
        for (int j = 0; j < d; j++) out[i * d + j] = row[j] * inv * weight[j];
    }
}

static void _rope(float* q, float* k, int seq_len, int n_head, int head_dim, int start_pos) {
    for (int s = 0; s < seq_len; s++) {
        int pos = start_pos + s;
        for (int h = 0; h < n_head; h++) {
            for (int d = 0; d < head_dim; d += 2) {
                float freq = 1.0f / powf(10000.0f, (float)d / head_dim);
                float theta = (float)pos * freq;
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);
                int idx = (s * n_head + h) * head_dim + d;
                float x0 = q[idx];
                float x1 = q[idx + 1];
                q[idx] = x0 * cos_t - x1 * sin_t;
                q[idx + 1] = x0 * sin_t + x1 * cos_t;
                float k0 = k[idx];
                float k1 = k[idx + 1];
                k[idx] = k0 * cos_t - k1 * sin_t;
                k[idx + 1] = k0 * sin_t + k1 * cos_t;
            }
        }
    }
}

static void _softmax(float* out, const float* in, int n, int d) {
    for (int i = 0; i < n; i++) {
        float max_val = in[i * d];
        for (int j = 1; j < d; j++) {
            if (in[i * d + j] > max_val) max_val = in[i * d + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            out[i * d + j] = expf(in[i * d + j] - max_val);
            sum += out[i * d + j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < d; j++) out[i * d + j] *= inv_sum;
    }
}

static void _silu(float* out, const float* in, int n) {
    for (int i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

static void _topk(float* logits, int* indices, int vocab_size, int k) {
    int* idx = (int*)malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) idx[i] = i;

    for (int i = 0; i < k && i < vocab_size; i++) {
        int best = i;
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[idx[j]] > logits[idx[best]]) best = j;
        }
        int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
        indices[i] = idx[i];
    }
    free(idx);
}

int slonet_forward(const SloNetWeights* w, const int* input_ids, int seq_len,
                   float* logits) {
    if (!w->data || seq_len > SLONET_MAX_SEQ_LEN) return -1;

    int B = 1;
    int T = seq_len;
    int D = w->n_embed;
    int H = w->n_head;
    int HD = w->head_dim;
    int V = w->vocab_size;

    float* x = (float*)malloc(B * T * D * sizeof(float));
    float* residual = NULL;
    float* attn_out = (float*)malloc(B * T * D * sizeof(float));
    float* ff_out = (float*)malloc(B * T * D * sizeof(float));
    float* buf = (float*)malloc(B * T * w->dim_ff * sizeof(float));
    float* buf2 = (float*)malloc(B * T * w->dim_ff * sizeof(float));
    float* q = (float*)malloc(B * T * H * HD * sizeof(float));
    float* k = (float*)malloc(B * T * H * HD * sizeof(float));
    float* v = (float*)malloc(B * T * H * HD * sizeof(float));
    float* scores = (float*)malloc(B * H * T * T * sizeof(float));

    if (!x || !attn_out || !ff_out || !buf || !buf2 || !q || !k || !v || !scores) {
        free(x); free(attn_out); free(ff_out); free(buf); free(buf2);
        free(q); free(k); free(v); free(scores);
        return -1;
    }

    // Embedding lookup
    const float* tok_emb = w->data + w->tok_emb_offset;
    for (int t = 0; t < T; t++) {
        int token_id = input_ids[t];
        if (token_id < 0 || token_id >= V) token_id = 0;
        memcpy(x + t * D, tok_emb + token_id * D, D * sizeof(float));
    }

    for (int layer = 0; layer < w->n_layer; layer++) {
        const float* base = w->data + w->layer_offsets[layer];
        const float* attn_norm_w = base;
        const float* q_w = base + D;
        const float* k_w = base + D + D * D;
        const float* v_w = base + D + 2 * D * D;
        const float* o_w = base + D + 3 * D * D;
        const float* ff_norm_w = base + D + 4 * D * D;
        const float* w1 = base + D + 4 * D * D + D;
        const float* w2 = base + D + 4 * D * D + D + D * w->dim_ff;
        const float* w3 = base + D + 4 * D * D + D + D * w->dim_ff + w->dim_ff * D;

        // RMSNorm before attention
        float* normed = ff_out;
        _rmsnorm(normed, x, attn_norm_w, B * T, D);

        // QKV projections
        _sgemm(B * T, H * HD, D, normed, q_w, q);
        _sgemm(B * T, H * HD, D, normed, k_w, k);
        _sgemm(B * T, H * HD, D, normed, v_w, v);

        // RoPE
        _rope(q, k, T, H, HD, 0);

        // Scaled dot-product attention
        float scale = 1.0f / sqrtf((float)HD);
        for (int h = 0; h < H; h++) {
            float* s = scores + h * T * T;
            for (int ti = 0; ti < T; ti++) {
                for (int tj = 0; tj < T; tj++) {
                    if (tj > ti) {
                        s[ti * T + tj] = NEG_INF;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < HD; d++) {
                        dot += q[(ti * H + h) * HD + d] * k[(tj * H + h) * HD + d];
                    }
                    s[ti * T + tj] = dot * scale;
                }
            }
        }

        // Softmax on scores
        for (int h = 0; h < H; h++) {
            _softmax(scores + h * T * T, scores + h * T * T, T, T);
        }

        // Attention output: scores @ v
        for (int h = 0; h < H; h++) {
            for (int ti = 0; ti < T; ti++) {
                float* out_row = attn_out + ti * D + h * HD;
                memset(out_row, 0, HD * sizeof(float));
                for (int tj = 0; tj <= ti; tj++) {
                    float attn_w = scores[h * T * T + ti * T + tj];
                    const float* v_row = v + (tj * H + h) * HD;
                    for (int d = 0; d < HD; d++) {
                        out_row[d] += attn_w * v_row[d];
                    }
                }
            }
        }

        // Output projection (o_proj)
        memcpy(buf, attn_out, B * T * D * sizeof(float));
        _sgemm(B * T, D, D, buf, o_w, attn_out);

        // Residual
        for (int i = 0; i < B * T * D; i++) attn_out[i] += x[i];

        // RMSNorm before FF
        _rmsnorm(normed, attn_out, ff_norm_w, B * T, D);

        // SwiGLU: w2(silu(w1(x)) * w3(x))
        _sgemm(B * T, w->dim_ff, D, normed, w1, buf);
        _sgemm(B * T, w->dim_ff, D, normed, w3, buf2);
        _silu(buf2, buf, B * T * w->dim_ff);
        for (int i = 0; i < B * T * w->dim_ff; i++) buf[i] *= buf2[i];
        _sgemm(B * T, D, w->dim_ff, buf, w2, ff_out);

        // Residual
        for (int i = 0; i < B * T * D; i++) x[i] = attn_out[i] + ff_out[i];
    }

    // Final RMSNorm
    const float* norm_w = w->data + w->norm_offset;
    _rmsnorm(x, x, norm_w, B * T, D);

    // LM head
    const float* lm_w = w->data + w->lm_head_offset;
    _sgemm(B * T, V, D, x, lm_w, logits);

    free(x); free(attn_out); free(ff_out); free(buf); free(buf2);
    free(q); free(k); free(v); free(scores);
    return 0;
}

int slonet_generate(const SloNetWeights* w, const int* prompt_tokens, int prompt_len,
                    int* output_tokens, int* output_len, int max_new_tokens,
                    float temperature, int top_k, float top_p) {
    if (!w->data || prompt_len <= 0 || prompt_len > SLONET_MAX_SEQ_LEN) return -1;

    int max_total = min_int(prompt_len + max_new_tokens, SLONET_MAX_SEQ_LEN);
    int* tokens = (int*)malloc(max_total * sizeof(int));
    if (!tokens) return -1;
    memcpy(tokens, prompt_tokens, prompt_len * sizeof(int));
    int current_len = prompt_len;

    for (int step = 0; step < max_new_tokens; step++) {
        int ctx_len = min_int(current_len, w->block_size);
        int* ctx = tokens + (current_len - ctx_len);

        float* logits = (float*)malloc(w->vocab_size * sizeof(float));
        if (!logits) { free(tokens); return -1; }

        int ret = slonet_forward(w, ctx, ctx_len, logits);
        if (ret != 0) { free(logits); free(tokens); return ret; }

        float* last_logits = logits;
        int v = w->vocab_size;

        if (temperature > 0.0f && temperature != 1.0f) {
            float inv_temp = 1.0f / temperature;
            for (int i = 0; i < v; i++) last_logits[i] *= inv_temp;
        }

        // Top-K filtering
        if (top_k > 0 && top_k < v) {
            float* sorted = (float*)malloc(v * sizeof(float));
            memcpy(sorted, last_logits, v * sizeof(float));
            for (int i = 0; i < top_k; i++) {
                int best = i;
                for (int j = i + 1; j < v; j++)
                    if (sorted[j] > sorted[best]) best = j;
                float tmp = sorted[i]; sorted[i] = sorted[best]; sorted[best] = tmp;
            }
            float kth = sorted[top_k - 1];
            free(sorted);
            for (int i = 0; i < v; i++)
                if (last_logits[i] < kth) last_logits[i] = NEG_INF;
        }

        // Top-P (nucleus) filtering
        if (top_p < 1.0f && top_p > 0.0f) {
            int* indices = (int*)malloc(v * sizeof(int));
            for (int i = 0; i < v; i++) indices[i] = i;
            for (int i = 0; i < v; i++) {
                int best = i;
                for (int j = i + 1; j < v; j++)
                    if (last_logits[indices[j]] > last_logits[indices[best]]) best = j;
                int tmp = indices[i]; indices[i] = indices[best]; indices[best] = tmp;
            }
            float max_l = last_logits[indices[0]];
            float sum = 0.0f;
            for (int i = 0; i < v; i++) {
                float p = expf(last_logits[indices[i]] - max_l);
                sum += p;
                if (sum > top_p) {
                    for (int j = i + 1; j < v; j++)
                        last_logits[indices[j]] = NEG_INF;
                    break;
                }
            }
            free(indices);
        }

        // Softmax
        float max_l = last_logits[0];
        for (int i = 1; i < v; i++)
            if (last_logits[i] > max_l) max_l = last_logits[i];
        float sum = 0.0f;
        for (int i = 0; i < v; i++) {
            last_logits[i] = expf(last_logits[i] - max_l);
            sum += last_logits[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < v; i++) last_logits[i] *= inv_sum;

        // Sample
        float r = (float)rand() / (float)RAND_MAX;
        float cum = 0.0f;
        int next_token = 0;
        for (int i = 0; i < v; i++) {
            cum += last_logits[i];
            if (r < cum) { next_token = i; break; }
        }

        free(logits);

        if (current_len >= max_total) break;
        tokens[current_len++] = next_token;
    }

    *output_len = current_len - prompt_len;
    memcpy(output_tokens, tokens + prompt_len, *output_len * sizeof(int));
    free(tokens);
    return 0;
}
