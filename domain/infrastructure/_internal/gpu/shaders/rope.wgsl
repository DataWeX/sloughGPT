// rope.wgsl — Rotary position embeddings.
// Apply rotation to each (x1, x2) pair in q or k.
// X layout: (n_heads * head_dim) for seq_len=1, or (seq_len * n_heads * head_dim) general.
// cos/sin: (head_dim/2) — one per frequency, same for all heads/positions when pos=0.
// Dispatch: ((n_heads * head_dim/2 + 255) / 256, 1, 1)

struct Params { total_pairs: u32, head_dim: u32 }

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       cos_t: array<f32>;
@group(0) @binding(2) var<storage, read>       sin_t: array<f32>;
@group(0) @binding(3) var<storage, read_write>  Y: array<f32>;
@group(0) @binding(4) var<uniform>              params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    if (pair_idx >= params.total_pairs) { return; }

    let hd2 = params.head_dim / 2u;
    let head = pair_idx / hd2;
    let freq = pair_idx % hd2;
    let base = head * params.head_dim;

    let x1 = X[base + freq * 2u];
    let x2 = X[base + freq * 2u + 1u];
    let c = cos_t[freq];
    let s = sin_t[freq];

    Y[base + freq * 2u] = x1 * c - x2 * s;
    Y[base + freq * 2u + 1u] = x2 * c + x1 * s;
}
