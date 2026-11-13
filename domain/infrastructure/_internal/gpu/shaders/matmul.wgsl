// matmul.wgsl — Tiled matmul with vec4f.
//
// C[M,N] = A[M,K] @ B[K,N]
// 16x16 workgroup, each thread accumulates via vec4f dot products.
// Shared memory: 2 * 16x16 = 512 floats = 2KB
// Dispatch: (N/16, M/16, 1)

struct Params { M: u32, N: u32, K: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write>  C: array<f32>;
@group(0) @binding(3) var<uniform>              params: Params;

const T: u32 = 16u;

var<workgroup> sA: array<array<f32, 16>, 16>;
var<workgroup> sB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_id)  l: vec3<u32>) {
    let r = g.y;
    let c = g.x;
    var sum = vec4<f32>(0.0);

    let nt = (params.K + T - 1u) / T;
    for (var t = 0u; t < nt; t++) {
        // Load tiles
        let ac = t * T + l.x;
        sA[l.y][l.x] = select(0.0, A[r * params.K + ac], r < params.M && ac < params.K);

        let br = t * T + l.y;
        sB[l.y][l.x] = select(0.0, B[br * params.N + c], br < params.K && c < params.N);

        workgroupBarrier();

        // vec4 dot product: accumulate 4 elements at a time
        for (var k = 0u; k < T; k += 4u) {
            let a = vec4<f32>(sA[l.y][k], sA[l.y][k+1u], sA[l.y][k+2u], sA[l.y][k+3u]);
            let b = vec4<f32>(sB[k][l.x], sB[k+1u][l.x], sB[k+2u][l.x], sB[k+3u][l.x]);
            sum += a * b;
        }

        workgroupBarrier();
    }

    if (r < params.M && c < params.N) {
        C[r * params.N + c] = sum.x + sum.y + sum.z + sum.w;
    }
}
