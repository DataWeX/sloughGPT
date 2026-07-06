#ifndef SLONET_H
#define SLONET_H

#include <stdint.h>
#include <stdbool.h>

#define SLONET_MAX_LAYERS 12
#define SLONET_MAX_SEQ_LEN 128

typedef struct {
    float* data;
    int num_params;
    int* param_offsets;
    int* param_sizes;
    int total_floats;
    int vocab_size;
    int n_embed;
    int n_layer;
    int n_head;
    int block_size;
    int head_dim;
    int dim_ff;
    int tok_emb_offset;
    int norm_offset;
    int lm_head_offset;
    int layer_offsets[SLONET_MAX_LAYERS];
} SloNetWeights;

typedef struct {
    int* tokens;
    int length;
    int capacity;
} SloNetTokenBuffer;

int slonet_load_weights(SloNetWeights* w, const float* flat_data, int num_floats,
                        int vocab_size, int n_embed, int n_layer, int n_head, int block_size);

void slonet_unload_weights(SloNetWeights* w);

int slonet_forward(const SloNetWeights* w, const int* input_ids, int seq_len,
                   float* logits);

int slonet_generate(const SloNetWeights* w, const int* prompt_tokens, int prompt_len,
                    int* output_tokens, int* output_len, int max_new_tokens,
                    float temperature, int top_k, float top_p);

#endif
