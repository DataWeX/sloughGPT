// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
    uint size2;
    uint size3;
};

struct Params {
    uint total_pairs;
    uint head_dim;
};
typedef float type_2[1];
uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % metal::select(rhs, 1u, rhs == 0u);
}


struct main_Input {
};
kernel void main_(
  metal::uint3 gid [[thread_position_in_grid]]
, device type_2 const& X [[user(fake0)]]
, device type_2 const& cos_t [[user(fake0)]]
, device type_2 const& sin_t [[user(fake0)]]
, device type_2& Y [[user(fake0)]]
, constant Params& params [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    uint pair_idx = gid.x;
    uint _e4 = params.total_pairs;
    if (pair_idx >= _e4) {
        return;
    }
    uint _e8 = params.head_dim;
    uint hd2_ = naga_div(_e8, 2u);
    uint head = naga_div(pair_idx, hd2_);
    uint freq = naga_mod(pair_idx, hd2_);
    uint _e15 = params.head_dim;
    uint base = head * _e15;
    float x1_ = X[base + (freq * 2u)];
    float x2_ = X[(base + (freq * 2u)) + 1u];
    float c = cos_t[freq];
    float s = sin_t[freq];
    Y[base + (freq * 2u)] = (x1_ * c) - (x2_ * s);
    Y[(base + (freq * 2u)) + 1u] = (x2_ * c) + (x1_ * s);
    return;
}
