// attention.wgsl — Fused attention for inference (seq_len <= max_seq).
//
// For simplicity: uses the existing matmul + softmax shaders.
// This file is a placeholder for future fused optimization.
// Current path: QKT = matmul(Q, K^T), add mask, softmax, attn = matmul(scores, V)
//
// See: matmul.wgsl, softmax.wgsl — these compose to form attention.
