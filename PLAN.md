# Plan: Memory-Mapped Inference (.slnc format)

## Problem

Current weight loading has 3 sources of overhead:
1. **Parse overhead**: safetensors parses JSON header, maps string keys to tensors
2. **Copy overhead**: each tensor is copied from file into numpy array
3. **Graph construction**: runtime code builds the computation pipeline by indexing into flat dict

These are all redundant — the config already defines the architecture, so the file layout should match the computation order.

## Term

**Memory-mapped inference with computation-graph-aligned layout.**

The file layout IS the inference pipeline. The OS page-fault mechanism handles loading. No loader logic needed.

## Design: `.slnc` Format (SloughGPT Neural Cache)

### File Layout

```
┌──────────────────────────────────────────────┐
│ Header (fixed size, mmap'd first)            │
│  - magic: "SLNC" (4 bytes)                  │
│  - version: uint32 (1)                      │
│  - config_json_len: uint32                  │
│  - config_json: bytes                       │
│  - block_count: uint32                      │
│  - block_size: uint32 (bytes per block)     │
│  - vocab_size: uint32                       │
│  - hidden_dim: uint32                       │
│  - tensor_count: uint32                     │
│  - tensor_offsets: [name_len, name, offset, size] × N │
├──────────────────────────────────────────────┤
│ Block 0 weights (sequential, computation order) │
│  - attn_norm.weight                         │
│  - attn_norm.bias                           │
│  - attn.q_proj.weight                       │
│  - attn.q_proj.bias                         │
│  - attn.k_proj.weight                       │
│  - attn.k_proj.bias                         │
│  - attn.v_proj.weight                       │
│  - attn.v_proj.bias                         │
│  - attn.o_proj.weight                       │
│  - attn.o_proj.bias                         │
│  - ff_norm.weight                           │
│  - ff_norm.bias                             │
│  - ff.w1.weight                             │
│  - ff.w1.bias                               │
│  - ff.w2.weight                             │
│  - ff.w2.bias                               │
│  - ff.w3.weight                             │
│  - ff.w3.bias                               │
├──────────────────────────────────────────────┤
│ Block 1 weights                              │
│ ...                                          │
├──────────────────────────────────────────────┤
│ Block N-1 weights                            │
├──────────────────────────────────────────────┤
│ Final norm weights/bias                      │
├──────────────────────────────────────────────┤
│ LM head weight                               │
└──────────────────────────────────────────────┘
```

### Key Properties

1. **Computation order**: Block 0 → Block 1 → ... → Block N-1 → norm → lm_head
   - This IS the inference pipeline order
   - Accessing weights sequentially = sequential file reads = cache-friendly

2. **Fixed block size**: each block has the same layout (same tensors, same shapes)
   - Block offset = header_size + block_index × block_size
   - No parsing needed — just pointer arithmetic

3. **Tensor offsets within block**: computed from config, not stored
   - Given hidden_dim, n_heads, ff_dim: tensor sizes are deterministic
   - offset_in_block = sum of previous tensor sizes

4. **mmap-friendly**: the file IS the memory
   - `mmap.mmap(fd, 0, access=mmap.ACCESS_READ)` maps entire file
   - numpy `frombuffer(ptr, dtype, shape)` creates view into mapped memory
   - Zero copy — numpy array points at file page

5. **Demand loading**: only accessed blocks get paged in
   - If inference only touches Block 0-5 (short prompt), Blocks 6+ stay on disk
   - OS page faults handle transparent loading

### Loader Implementation

```python
class SLNCLoader:
    """Memory-mapped loader for .slnc files."""

    def __init__(self, path: str):
        self._fd = os.open(path, os.O_RDONLY)
        self._file = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)

        # Parse header (small, always in memory)
        self._parse_header()

        # Compute tensor offsets from config (no file parsing)
        self._compute_offsets()

    def get_tensor(self, name: str) -> np.ndarray:
        """Get weight by name — returns view into mmap'd memory."""
        offset, shape, dtype = self._tensor_map[name]
        return np.frombuffer(
            self._file[offset:offset + np.prod(shape) * dtype.itemsize],
            dtype=dtype
        ).reshape(shape)

    def get_block(self, block_idx: int) -> dict:
        """Get all weights for a transformer block — sequential access."""
        base = self._block_offsets[block_idx]
        return {
            'attn_norm_w': self._view(base + self._off.an_w, ...),
            'attn_norm_b': self._view(base + self._off.an_b, ...),
            # ... all block tensors
        }
```

### Converter: safetensors → .slnc

```python
def convert_to_slnc(model_id: str, output_path: str):
    """Convert HF model to .slnc format."""
    config, weights = load_weights(model_id)
    arch = build_arch(config)

    # Compute layout from config
    block_size = compute_block_size(arch)
    header_size = compute_header_size(config)

    # Write header
    write_header(f, config, arch)

    # Write blocks in computation order
    for i in range(arch.n_layers):
        write_block(f, weights, arch, i)

    # Write final norm + lm_head
    write_final(f, weights, arch)
```

### Performance Expectations

| Metric | Current (safetensors) | .slnc (mmap) |
|--------|----------------------|--------------|
| Load time | ~200ms (parse + copy) | ~1ms (mmap) |
| Memory | 100% (full copy) | ~0% initially (demand) |
| First token | ~200ms load + ~60ms compute | ~1ms mmap + ~60ms compute |
| Subsequent tokens | ~60ms | ~60ms (pages already hot) |

### Migration Path

1. **Write converter** (`convert_to_slnc.py`)
2. **Write loader** (`slnc_loader.py`)
3. **Wire into NumpyEngine** (replace `_load_weights` with mmap)
4. **Wire into SloNetChatProvider** (replace safetensors loading)
5. **Keep safetensors fallback** (for models not yet converted)

### Files to Create/Modify

| File | Action |
|------|--------|
| `domains/infrastructure/slnc_format.py` | NEW — header spec, converter, layout math |
| `domains/infrastructure/slnc_loader.py` | NEW — mmap loader |
| `domains/infrastructure/numpy_engine.py` | MODIFY — use mmap loader |
| `domains/inference/slonet_provider.py` | MODIFY — use mmap loader |
| `tests/test_slnc_format.py` | NEW — converter + loader tests |

### Risks

1. **Read-only**: mmap'd files can't be written to (weights are immutable at inference time — OK)
2. **File size**: .slnc may be larger than .safetensors due to alignment padding (acceptable)
3. **Cross-platform**: mmap works on Linux/macOS/Windows but API differs slightly (use `mmap` stdlib)
4. **Floating-point precision**: must match exactly with current implementation (test with GPT-2)
