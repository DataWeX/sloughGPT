struct Params {
    uint total_pairs;
    uint head_dim;
};

ByteAddressBuffer X : register(t0);
ByteAddressBuffer cos_t : register(t1);
ByteAddressBuffer sin_t : register(t2);
RWByteAddressBuffer Y : register(u3);
cbuffer params : register(b4) { Params params; }

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint pair_idx = gid.x;
    uint _e4 = params.total_pairs;
    if ((pair_idx >= _e4)) {
        return;
    }
    uint _e8 = params.head_dim;
    uint hd2_ = naga_div(_e8, 2u);
    uint head = naga_div(pair_idx, hd2_);
    uint freq = naga_mod(pair_idx, hd2_);
    uint _e15 = params.head_dim;
    uint base = (head * _e15);
    float x1_ = asfloat(X.Load((base + (freq * 2u))*4));
    float x2_ = asfloat(X.Load(((base + (freq * 2u)) + 1u)*4));
    float c = asfloat(cos_t.Load(freq*4));
    float s = asfloat(sin_t.Load(freq*4));
    Y.Store((base + (freq * 2u))*4, asuint(((x1_ * c) - (x2_ * s))));
    Y.Store(((base + (freq * 2u)) + 1u)*4, asuint(((x2_ * c) + (x1_ * s))));
    return;
}
