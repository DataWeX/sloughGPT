/**
 * transformer_forward.h — Generic transformer forward pass (C + Apple Accelerate).
 *
 * Supports ANY HF-compatible model (GPT-2, Qwen, LLaMA, Mistral, Phi).
 * Auto-detects architecture from SLNC config.
 */

#ifndef TRANSFORMER_FORWARD_H
#define TRANSFORMER_FORWARD_H

#define TRANSFORMER_MAX_LAYERS 64

typedef struct {
    int n_layers;
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int ff_dim;
    int vocab_size;
    int block_size;
    float rope_base;
    float rope_theta;
} TransformerConfig;

typedef struct {
    const float* data;
    int total_floats;
    TransformerConfig config;
    int tok_emb_offset;
    int layer_offsets[TRANSFORMER_MAX_LAYERS];
    int norm_offset;
    int lm_head_offset;
} TransformerWeights;

typedef struct {
    float* k;
    float* v;
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int seq_capacity;
    int seq_len;
} TransformerKVCache;

int transformer_load_weights(TransformerWeights* w, const float* flat_data,
                             int num_floats, const TransformerConfig* config);
void transformer_free_weights(TransformerWeights* w);

int transformer_kv_cache_init(TransformerKVCache* cache, const TransformerConfig* config,
                              int seq_capacity);
void transformer_kv_cache_free(TransformerKVCache* cache);
void transformer_kv_cache_reset(TransformerKVCache* cache);

int transformer_forward_step(const TransformerWeights* w, TransformerKVCache* cache,
                             int token_id, int seq_pos, float* logits_out);

#endif
