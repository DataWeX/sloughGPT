"""Train SloRAN on Shakespeare character-level language modeling."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from domains.training.sloran import SloRAN
from domains.training.slonet import Tensor, no_grad
import numpy as np

DATA = os.path.join(os.path.dirname(__file__),'..','..','..','data','shakespeare','input.txt')
B, T, V = 8, 64, None
its = 2000

with open(DATA) as f:
    text = f.read()
chars = sorted(set(text)); V = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
n = int(0.9*len(data))
train, val = data[:n], data[n:]

def batch(split):
    d = train if split=='train' else val
    ix = np.random.randint(0,len(d)-T-1,B)
    x = np.stack([d[i:i+T] for i in ix])
    y = np.stack([d[i+1:i+T+1] for i in ix])
    return x, y

m = SloRAN(vocab_size=V, d_model=128, n_layers=3, n_slots=8, d_state=64, dropout=0.1)
params = list(m.parameters())
print(f'Params: {sum(p.data.size for p in params)}, Vocab: {V}')

for step in range(1, its+1):
    x, y = batch('train')
    _, loss = m.forward(Tensor(x), targets=Tensor(y))
    loss.backward()
    for p in params:
        if p.grad is not None:
            p.data -= 1e-2 * p.grad.data
            p.grad = None
    if step % 200 == 0 or step == its:
        xv, yv = batch('val')
        _, vl = m.forward(Tensor(xv), targets=Tensor(yv))
        with no_grad():
            gen = m.generate(np.array([[stoi['\n']]]), max_new_tokens=200, temperature=1.0)
        txt = ''.join(itos[int(i)] for i in gen)
        print(f'{step:4d} loss={float(loss.data):.4f} val={float(vl.data):.4f}')
        print(f'  gen: {txt[:80]}')
