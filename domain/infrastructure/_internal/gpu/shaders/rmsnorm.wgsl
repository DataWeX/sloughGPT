// rmsnorm.wgsl — RMS norm with vec4f.
// Y = (X / sqrt(mean(X^2) + eps)) * W
// One workgroup per row. Dispatch: (rows, 1, 1)

struct Params { rows: u32, cols: u32, eps: f32 }

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<f32>;
@group(0) @binding(2) var<storage, read_write>  Y: array<f32>;
@group(0) @binding(3) var<uniform>              params: Params;

const WGS: u32 = 256u;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: u32,
        @builtin(workgroup_id)        wid: u32) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    let base = row * params.cols;
    let cols4 = params.cols / 4u * 4u;

    // Sum of squares — vec4 wide
    var s4 = vec4<f32>(0.0);
    var i = lid * 4u;
    for (; i < cols4; i += WGS * 4u) {
        let v = vec4<f32>(X[base+i], X[base+i+1u], X[base+i+2u], X[base+i+3u]);
        s4 += v * v;
    }
    var scalar_s = s4.x + s4.y + s4.z + s4.w;
    for (; i < params.cols; i += WGS) {
        let v = X[base+i];
        scalar_s += v * v;
    }
    sh[lid] = scalar_s;
    workgroupBarrier();
    for (var s = WGS / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { sh[lid] += sh[lid+s]; }
        workgroupBarrier();
    }

    let inv = 1.0 / sqrt(sh[0] / f32(params.cols) + params.eps);
    let w4 = vec4<f32>(inv);

    // Normalize + scale — vec4 wide
    i = lid * 4u;
    for (; i < cols4; i += WGS * 4u) {
        let x = vec4<f32>(X[base+i], X[base+i+1u], X[base+i+2u], X[base+i+3u]);
        let w = vec4<f32>(W[i], W[i+1u], W[i+2u], W[i+3u]);
        let n = x * w4 * w;
        Y[base+i] = n.x; Y[base+i+1u] = n.y; Y[base+i+2u] = n.z; Y[base+i+3u] = n.w;
    }
    for (; i < params.cols; i += WGS) {
        Y[base+i] = X[base+i] * inv * W[i];
    }
}
