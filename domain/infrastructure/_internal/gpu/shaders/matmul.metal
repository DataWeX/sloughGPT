// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
    uint size2;
};

struct Params {
    uint M;
    uint N;
    uint K;
    uint _pad;
};
typedef float type_2[1];
struct type_3 {
    float inner[16];
};
struct type_4 {
    type_3 inner[16];
};
constant uint T = 16u;
uint naga_div(uint lhs, uint rhs) {
    return lhs / metal::select(rhs, 1u, rhs == 0u);
}


struct main_Input {
};
kernel void main_(
  metal::uint3 g [[thread_position_in_grid]]
, metal::uint3 l [[thread_position_in_threadgroup]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, device type_2 const& A [[user(fake0)]]
, device type_2 const& B [[user(fake0)]]
, device type_2& C [[user(fake0)]]
, constant Params& params [[user(fake0)]]
, threadgroup type_4& sA
, threadgroup type_4& sB
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    if (__local_invocation_index == 0u) {
        sA = {};
        sB = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::float4 sum = metal::float4(0.0);
    uint t = 0u;
    bool local = {};
    bool local_1 = {};
    uint k = {};
    bool local_2 = {};
    uint r = g.y;
    uint c = g.x;
    uint _e9 = params.K;
    uint nt = naga_div((_e9 + T) - 1u, T);
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e158 = t;
            t = _e158 + 1u;
        }
        loop_init = false;
        uint _e18 = t;
        if (_e18 < nt) {
        } else {
            break;
        }
        {
            uint _e20 = t;
            uint ac = (_e20 * T) + l.x;
            uint _e33 = params.K;
            float _e37 = A[(r * _e33) + ac];
            uint _e40 = params.M;
            if (r < _e40) {
                uint _e46 = params.K;
                local = ac < _e46;
            } else {
                local = false;
            }
            bool _e49 = local;
            sA.inner[l.y].inner[l.x] = _e49 ? _e37 : 0.0;
            uint _e52 = t;
            uint br = (_e52 * T) + l.y;
            uint _e65 = params.N;
            float _e69 = B[(br * _e65) + c];
            uint _e72 = params.K;
            if (br < _e72) {
                uint _e78 = params.N;
                local_1 = c < _e78;
            } else {
                local_1 = false;
            }
            bool _e81 = local_1;
            sB.inner[l.y].inner[l.x] = _e81 ? _e69 : 0.0;
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            k = 0u;
            uint2 loop_bound_1 = uint2(4294967295u);
            bool loop_init_1 = true;
            while(true) {
                if (metal::all(loop_bound_1 == uint2(0u))) { break; }
                loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                if (!loop_init_1) {
                    uint _e154 = k;
                    k = _e154 + 4u;
                }
                loop_init_1 = false;
                uint _e86 = k;
                if (_e86 < T) {
                } else {
                    break;
                }
                {
                    uint _e92 = k;
                    float _e94 = sA.inner[l.y].inner[_e92];
                    uint _e98 = k;
                    float _e102 = sA.inner[l.y].inner[_e98 + 1u];
                    uint _e106 = k;
                    float _e110 = sA.inner[l.y].inner[_e106 + 2u];
                    uint _e114 = k;
                    float _e118 = sA.inner[l.y].inner[_e114 + 3u];
                    metal::float4 a = metal::float4(_e94, _e102, _e110, _e118);
                    uint _e121 = k;
                    float _e125 = sB.inner[_e121].inner[l.x];
                    uint _e127 = k;
                    float _e133 = sB.inner[_e127 + 1u].inner[l.x];
                    uint _e135 = k;
                    float _e141 = sB.inner[_e135 + 2u].inner[l.x];
                    uint _e143 = k;
                    float _e149 = sB.inner[_e143 + 3u].inner[l.x];
                    metal::float4 b = metal::float4(_e125, _e133, _e141, _e149);
                    metal::float4 _e151 = sum;
                    sum = _e151 + (a * b);
                }
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
    uint _e162 = params.M;
    if (r < _e162) {
        uint _e168 = params.N;
        local_2 = c < _e168;
    } else {
        local_2 = false;
    }
    bool _e171 = local_2;
    if (_e171) {
        uint _e175 = params.N;
        float _e180 = sum.x;
        float _e182 = sum.y;
        float _e185 = sum.z;
        float _e188 = sum.w;
        C[(r * _e175) + c] = ((_e180 + _e182) + _e185) + _e188;
        return;
    } else {
        return;
    }
}
