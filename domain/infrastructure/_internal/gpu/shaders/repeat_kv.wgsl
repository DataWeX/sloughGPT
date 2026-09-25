// repeat_kv.wgsl — GQA key/value head expansion.
// Copy n_kv_heads → n_heads by repeating each KV head n_reps times.
// Dispatch: (n_heads, 1, 1)

struct Params { seq_len: u32, n_kv_heads: u32, head_dim: u32, n_reps: u32 }

@group(0) @binding(0) var<storage, read>       X: array<f32>;   // (seq_len, n_kv_heads, head_dim)
@group(0) @binding(1) var<storage, read_write>  Y: array<f32>;   // (seq_len, n_heads, head_dim)
@group(0) @binding(2) var<uniform>              params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_idx = gid.x;
    if (head_idx >= params.n_kv_heads * params.n_reps) { return; }

    let src_head = head_idx / params.n_reps;
    let base_in = src_head * params.head_dim;
    let base_out = head_idx * params.head_dim;

    for (var d = 0u; d < params.head_dim; d++) {
        Y[base_out + d] = X[base_in + d];
    }
}
