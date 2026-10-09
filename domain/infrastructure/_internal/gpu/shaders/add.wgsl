// add.wgsl — Element-wise add: C = A + B
// Dispatch: ((n+255)/256, 1, 1)

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write>  C: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&A)) { return; }
    C[i] = A[i] + B[i];
}
