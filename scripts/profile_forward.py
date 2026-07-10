"""Full forward pass profile after float32 fix."""
import time, numpy as np
from pathlib import Path
from domains.infrastructure.arch_config import build_arch
from domains.infrastructure.numpy_ops import rmsnorm, layer_norm, softmax, gelu, silu, rope
from safetensors import safe_open

snap = sorted((Path.home() / '.cache/huggingface/hub/models--gpt2/snapshots').glob('*'))[0]
weights = {}
for f in sorted(snap.glob('*.safetensors')):
    with safe_open(str(f), framework='numpy') as sf:
        for k in sf.keys():
            weights[k] = sf.get_tensor(k)

arch = build_arch('gpt2', {}, set(weights.keys()))
W = arch.weight_map
N = 20
results = {}

def w(c, l=0):
    return weights[W[c].replace('{i}', str(l))]
def wn(c, l=0):
    m = W.get(c)
    if m is None: return None
    return weights.get(m.replace('{i}', str(l)))

x = w('embed.token')[np.array([100])] + w('embed.pos')[:1]

# LayerNorm
w_an = w('layers.{i}.attn_norm.weight', 0)
b_an = wn('layers.{i}.attn_norm.bias', 0)
t0=time.perf_counter()
for _ in range(N): h = layer_norm(x, w_an, b_an)
results['layer_norm'] = (time.perf_counter()-t0)/N*1000

# QKV
W_qkv = w('layers.{i}.qkv.weight', 0)
b_qkv = wn('layers.{i}.qkv.bias', 0)
t0=time.perf_counter()
for _ in range(N): qkv = h @ W_qkv + (b_qkv if b_qkv is not None else 0)
results['qkv_matmul'] = (time.perf_counter()-t0)/N*1000

# Attention
q,k,v = np.split(qkv, 3, axis=-1)
hd=arch.head_dim; n_h=arch.n_head; scale=np.float32(np.sqrt(hd))
q2=q.reshape(1,n_h,hd).transpose(1,0,2); k2=k.reshape(1,n_h,hd).transpose(1,0,2)
mask=np.zeros((1,1),dtype=np.float32)
t0=time.perf_counter()
for _ in range(N):
    attn=(q2@k2.transpose(0,2,1))/scale+mask
    a=softmax(attn,axis=-1)
results['attn_scores']=(time.perf_counter()-t0)/N*1000

v2=v.reshape(1,n_h,hd).transpose(1,0,2)
t0=time.perf_counter()
for _ in range(N): out=(a@v2).transpose(1,0,2).reshape(1,arch.n_embed)
results['attn_output']=(time.perf_counter()-t0)/N*1000

# Output proj
W_o=w('layers.{i}.o_proj.weight',0)
t0=time.perf_counter()
for _ in range(N): x=x+out@W_o
results['output_proj']=(time.perf_counter()-t0)/N*1000

# FFN
h=layer_norm(x,w('layers.{i}.ff_norm.weight',0),wn('layers.{i}.ff_norm.bias',0))
W_up=w('layers.{i}.ffn.up.weight',0); b_up=wn('layers.{i}.ffn.up.bias',0)
t0=time.perf_counter()
for _ in range(N): h2=h@W_up+(b_up if b_up is not None else 0)
results['ffn_up']=(time.perf_counter()-t0)/N*1000

t0=time.perf_counter()
for _ in range(N): h2g=gelu(h2)
results['gelu']=(time.perf_counter()-t0)/N*1000

W_down=w('layers.{i}.ffn.down.weight',0)
t0=time.perf_counter()
for _ in range(N): x=x+h2g@W_down
results['ffn_down']=(time.perf_counter()-t0)/N*1000

# Final norm + LM head
x_norm=layer_norm(x,w('final_norm.weight'),wn('final_norm.bias'))
W_lm=w('embed.token')
t0=time.perf_counter()
for _ in range(N): logits=x_norm@W_lm.T
results['lm_head']=(time.perf_counter()-t0)/N*1000

total=sum(results.values())
print(f'\nGPT-2 Profile (1 block, seq_len=1, {N} iters) — after float32 fix')
print(f'{"Operation":<25} {"ms":>8} {"%":>6}')
print('-'*41)
for op,t in sorted(results.items(),key=lambda x:-x[1]):
    print(f'{op:<25} {t:>8.3f} {t/total*100:>5.1f}%')
print('-'*41)
print(f'{"TOTAL":<25} {total:>8.3f} {100.0:>5.1f}%')
print(f'12 blocks: {total*12:.1f} ms')
