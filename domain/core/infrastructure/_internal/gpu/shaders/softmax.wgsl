// softmax.wgsl — Row-wise softmax with vec4f reduction.
//
// Y[i,:] = exp(X[i,:] - max(X[i,:])) / sum(exp(X[i,:] - max(X[i,:])))
// One workgroup per row. Dispatch: (rows, 1, 1)

struct Params { rows: u32, cols: u32 }

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read_write>  Y: array<f32>;
@group(0) @binding(2) var<uniform>              params: Params;

const WGS: u32 = 256u;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: u32,
        @builtin(workgroup_id)        wid: u32) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    let base = row * params.cols;
    let cols4 = params.cols / 4u * 4u;  // round down to multiple of 4

    // Max reduction — vec4 wide
    var mx = vec4<f32>(-3.4e38);
    var i = lid * 4u;
    for (; i < cols4; i += WGS * 4u) {
        let v = vec4<f32>(X[base+i], X[base+i+1u], X[base+i+2u], X[base+i+3u]);
        mx = max(mx, v);
    }
    // Handle remainder
    var scalar_mx = max(max(mx.x, mx.y), max(mx.z, mx.w));
    for (; i < params.cols; i += WGS) {
        scalar_mx = max(scalar_mx, X[base+i]);
    }
    sh[lid] = scalar_mx;
    workgroupBarrier();
    for (var s = WGS / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { sh[lid] = max(sh[lid], sh[lid+s]); }
        workgroupBarrier();
    }
    let row_max = sh[0];

    // Exp + sum — vec4 wide
    var s4 = vec4<f32>(0.0);
    i = lid * 4u;
    for (; i < cols4; i += WGS * 4u) {
        let v = vec4<f32>(X[base+i], X[base+i+1u], X[base+i+2u], X[base+i+3u]);
        let e = exp(v - vec4<f32>(row_max));
        Y[base+i] = e.x; Y[base+i+1u] = e.y; Y[base+i+2u] = e.z; Y[base+i+3u] = e.w;
        s4 += e;
    }
    var scalar_s = s4.x + s4.y + s4.z + s4.w;
    for (; i < params.cols; i += WGS) {
        let e = exp(X[base+i] - row_max);
        Y[base+i] = e;
        scalar_s += e;
    }
    sh[lid] = scalar_s;
    workgroupBarrier();
    for (var s = WGS / 2u; s > 0u; s >>= 1u) {
        if (lid < s) { sh[lid] += sh[lid+s]; }
        workgroupBarrier();
    }

    // Normalize — vec4 wide
    let inv = 1.0 / sh[0];
    i = lid * 4u;
    for (; i < cols4; i += WGS * 4u) {
        let v = vec4<f32>(Y[base+i], Y[base+i+1u], Y[base+i+2u], Y[base+i+3u]);
        let n = v * vec4<f32>(inv);
        Y[base+i] = n.x; Y[base+i+1u] = n.y; Y[base+i+2u] = n.z; Y[base+i+3u] = n.w;
    }
    for (; i < params.cols; i += WGS) {
        Y[base+i] *= inv;
    }
}
