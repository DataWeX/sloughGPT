// attn1.wgsl — Single-token attention (seq_len=1, no softmax needed).
// For each head h: out_h = (q_h · k_h / sqrt(head_dim)) * v_h
// q,k,v, out: (n_heads, head_dim) flattened
// Dispatch: (n_heads, 1, 1) — one workgroup per head

struct Params { n_heads: u32, head_dim: u32 }

@group(0) @binding(0) var<storage, read>       Q: array<f32>;
@group(0) @binding(1) var<storage, read>       K: array<f32>;
@group(0) @binding(2) var<storage, read>       V: array<f32>;
@group(0) @binding(3) var<storage, read_write>  OUT: array<f32>;
@group(0) @binding(4) var<uniform>              params: Params;

const WGS: u32 = 256u;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) hid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = hid.x;
    if (head >= params.n_heads) { return; }
    let d = params.head_dim;
    let base = head * d;

    // Parallel dot product q · k
    var dot4 = vec4<f32>(0.0);
    var i = lid.x * 4u;
    let d4 = d / 4u * 4u;
    for (; i < d4; i += WGS * 4u) {
        let q4 = vec4<f32>(Q[base+i], Q[base+i+1u], Q[base+i+2u], Q[base+i+3u]);
        let k4 = vec4<f32>(K[base+i], K[base+i+1u], K[base+i+2u], K[base+i+3u]);
        dot4 += q4 * k4;
    }
    var s = dot4.x + dot4.y + dot4.z + dot4.w;
    for (; i < d; i += WGS) {
        s += Q[base+i] * K[base+i];
    }
    sh[lid.x] = s;
    workgroupBarrier();
    for (var half = WGS / 2u; half > 0u; half >>= 1u) {
        if (lid.x < half) { sh[lid.x] += sh[lid.x + half]; }
        workgroupBarrier();
    }

    // Scale and broadcast to v
    let coeff = sh[0] / sqrt(f32(d));
    i = lid.x * 4u;
    for (; i < d4; i += WGS * 4u) {
        let v4 = vec4<f32>(V[base+i], V[base+i+1u], V[base+i+2u], V[base+i+3u]);
        let o = v4 * vec4<f32>(coeff);
        OUT[base+i] = o.x; OUT[base+i+1u] = o.y; OUT[base+i+2u] = o.z; OUT[base+i+3u] = o.w;
    }
    for (; i < d; i += WGS) {
        OUT[base+i] = V[base+i] * coeff;
    }
}
