// silu.wgsl — SiLU activation with vec4f.
// Y = X * sigmoid(X)
// Dispatch: (n/64, 1, 1)

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read_write>  Y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: u32) {
    let base = gid * 4u;
    let v = vec4<f32>(X[base], X[base+1u], X[base+2u], X[base+3u]);
    let sig = 1.0 / (1.0 + exp(-v));
    let r = v * sig;
    Y[base] = r.x; Y[base+1u] = r.y; Y[base+2u] = r.z; Y[base+3u] = r.w;
}
