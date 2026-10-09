struct Params {
    uint M;
    uint N;
    uint K;
    uint _pad;
};

static const uint T = 16u;

ByteAddressBuffer A : register(t0);
ByteAddressBuffer B : register(t1);
RWByteAddressBuffer C : register(u2);
cbuffer params : register(b3) { Params params; }
groupshared float sA[16][16];
groupshared float sB[16][16];

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

[numthreads(16, 16, 1)]
void main(uint3 g : SV_DispatchThreadID, uint3 l : SV_GroupThreadID, uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        sA = (float[16][16])0;
        sB = (float[16][16])0;
    }
    GroupMemoryBarrierWithGroupSync();
    float4 sum = (0.0).xxxx;
    uint t = 0u;
    bool local = (bool)0;
    bool local_1 = (bool)0;
    uint k = (uint)0;
    bool local_2 = (bool)0;

    uint r = g.y;
    uint c = g.x;
    uint _e9 = params.K;
    uint nt = naga_div(((_e9 + T) - 1u), T);
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e158 = t;
            t = (_e158 + 1u);
        }
        loop_init = false;
        uint _e18 = t;
        if ((_e18 < nt)) {
        } else {
            break;
        }
        {
            uint _e20 = t;
            uint ac = ((_e20 * T) + l.x);
            uint _e33 = params.K;
            float _e37 = asfloat(A.Load(((r * _e33) + ac)*4));
            uint _e40 = params.M;
            if ((r < _e40)) {
                uint _e46 = params.K;
                local = (ac < _e46);
            } else {
                local = false;
            }
            bool _e49 = local;
            sA[min(uint(l.y), 15u)][min(uint(l.x), 15u)] = (_e49 ? _e37 : 0.0);
            uint _e52 = t;
            uint br = ((_e52 * T) + l.y);
            uint _e65 = params.N;
            float _e69 = asfloat(B.Load(((br * _e65) + c)*4));
            uint _e72 = params.K;
            if ((br < _e72)) {
                uint _e78 = params.N;
                local_1 = (c < _e78);
            } else {
                local_1 = false;
            }
            bool _e81 = local_1;
            sB[min(uint(l.y), 15u)][min(uint(l.x), 15u)] = (_e81 ? _e69 : 0.0);
            GroupMemoryBarrierWithGroupSync();
            k = 0u;
            uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
            bool loop_init_1 = true;
            while(true) {
                if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
                loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                if (!loop_init_1) {
                    uint _e154 = k;
                    k = (_e154 + 4u);
                }
                loop_init_1 = false;
                uint _e86 = k;
                if ((_e86 < T)) {
                } else {
                    break;
                }
                {
                    uint _e92 = k;
                    float _e94 = sA[min(uint(l.y), 15u)][min(uint(_e92), 15u)];
                    uint _e98 = k;
                    float _e102 = sA[min(uint(l.y), 15u)][min(uint((_e98 + 1u)), 15u)];
                    uint _e106 = k;
                    float _e110 = sA[min(uint(l.y), 15u)][min(uint((_e106 + 2u)), 15u)];
                    uint _e114 = k;
                    float _e118 = sA[min(uint(l.y), 15u)][min(uint((_e114 + 3u)), 15u)];
                    float4 a = float4(_e94, _e102, _e110, _e118);
                    uint _e121 = k;
                    float _e125 = sB[min(uint(_e121), 15u)][min(uint(l.x), 15u)];
                    uint _e127 = k;
                    float _e133 = sB[min(uint((_e127 + 1u)), 15u)][min(uint(l.x), 15u)];
                    uint _e135 = k;
                    float _e141 = sB[min(uint((_e135 + 2u)), 15u)][min(uint(l.x), 15u)];
                    uint _e143 = k;
                    float _e149 = sB[min(uint((_e143 + 3u)), 15u)][min(uint(l.x), 15u)];
                    float4 b = float4(_e125, _e133, _e141, _e149);
                    float4 _e151 = sum;
                    sum = (_e151 + (a * b));
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    uint _e162 = params.M;
    if ((r < _e162)) {
        uint _e168 = params.N;
        local_2 = (c < _e168);
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
        C.Store(((r * _e175) + c)*4, asuint((((_e180 + _e182) + _e185) + _e188)));
        return;
    } else {
        return;
    }
}
