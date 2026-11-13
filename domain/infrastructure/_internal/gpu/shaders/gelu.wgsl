// gelu.wgsl — GELU activation with vec4f.
// Y = 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
// Dispatch: (n/64, 1, 1)

const SQRT_2_PI: f32 = 0.7978845608;

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read_write>  Y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: u32) {
    let base = gid * 4u;
    let v = vec4<f32>(X[base], X[base+1u], X[base+2u], X[base+3u]);
    let inner = SQRT_2_PI * (v + 0.044715 * v * v * v);
    let h = tanh(inner);
    let r = 0.5 * v * (vec4<f32>(1.0) + h);
    Y[base] = r.x; Y[base+1u] = r.y; Y[base+2u] = r.z; Y[base+3u] = r.w;
}
