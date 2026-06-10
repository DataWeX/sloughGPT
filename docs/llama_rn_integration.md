# llama.rn Integration Guide

This guide explains how to export SloughGPT models for use with [llama.rn](https://github.com/nicktasios/llama.rn), a React Native library for running GGUF-format language models on mobile devices.

Run **`python3`** examples from the **repository root** after `python3 -m pip install -e ".[dev]"`.

## Quick Start

```bash
# Export a model in GGUF format optimized for mobile
python3 cli.py export models/sloughgpt.safetensors -f gguf

# Or fine-tune a HuggingFace model and export
python3 apps/api/server/training/router.py  # via POST /training/hf-start (see docs/routers.md)
```

**Note:** `cli.py export` is the legacy GGUF export path. For HuggingFace models, use the training endpoint.

## Supported Quantization Formats

| Format | Bits | Memory | Quality | Recommended For |
|--------|------|--------|---------|-----------------|
| `Q4_K_M` | 4 | ~25% | High | **Recommended default** |
| `Q5_K_M` | 5 | ~31% | Very High | Balanced quality/size |
| `Q8_0` | 8 | ~50% | Near-perfect | Quality-critical |
| `F16` | 16 | 100% | Perfect | Development/debugging |

## Memory Requirements

For a typical SloughGPT model (~7B parameters):

| Format | Memory | iPhone 15 Pro | Android (8GB) |
|--------|--------|---------------|---------------|
| Q4_K_M | ~3.5 GB | ✅ | ✅ |
| Q5_K_M | ~4.3 GB | ✅ | ⚠️ |
| Q8_0 | ~7 GB | ⚠️ | ❌ |
| F16 | ~14 GB | ❌ | ❌ |

## React Native Usage

### Installation

```bash
npm install llama.rn
```

### Basic Usage

```typescript
import { LlamaModel, LlamaContext, LlamaSession } from 'llama.rn';

// Initialize model
const modelPath = './models/sloughgpt-q4_k_m.gguf';

const model = await LlamaModel.load({
  modelPath,
  gpu: 'auto', // 'auto', 'gpu', 'cpu'
});

// Create session
const session = await model.createSession({
  contextSize: 512,
  threads: 4,
});

// Generate
const response = await session.prompt('Once upon a time');
console.log(response);
```

### Advanced Configuration

```typescript
const session = await model.createSession({
  contextSize: 512,        // Context window size
  threads: 4,              // CPU threads
  gpuLayers: 32,           // Layers offloaded to GPU
  batchSize: 512,           // Prompt batch size
  ropeFreqBase: 10000,      // RoPE base frequency
  temperature: 0.7,         // Sampling temperature
  topP: 0.95,               // Top-p sampling
  repeatPenalty: 1.1,       // Repetition penalty
});

// Streaming response
for await (const chunk of session.prompt('Tell me a story', {
  stream: true,
})) {
  process.stdout.write(chunk);
}
```

## Model Export Options

### Export CLI

```bash
# Export a model in GGUF format
python3 cli.py export models/sloughgpt.safetensors -f gguf
```

## Platform-Specific Notes

### iOS

- **Recommended**: Q4_K_M for iPhone 12 and newer
- **GPU**: Set `gpu: 'auto'` to use Metal GPU acceleration
- **Memory**: iOS limits app memory to ~3-5GB depending on device
- **Threads**: Use 4-6 threads for optimal performance

```typescript
const model = await LlamaModel.load({
  modelPath,
  gpu: 'auto',
});
```

### Android

- **Recommended**: Q4_K_M for devices with 6GB+ RAM
- **GPU**: Use Vulkan for GPU acceleration on supported devices
- **Memory**: Android background apps may reclaim memory
- **Threads**: Use 4-8 threads depending on CPU

```typescript
const model = await LlamaModel.load({
  modelPath,
  gpu: 'auto',
  cpuThreads: 6,
});
```

## Performance Tips

1. **Use Q4_K_M** - Best balance of speed and quality for mobile
2. **Smaller context** - 512 tokens is usually sufficient
3. **GPU acceleration** - Enable for 2-3x speed improvement
4. **Batch size** - Keep at 512 or lower for mobile
5. **Preload model** - Load model once, reuse sessions

## Troubleshooting

### Out of Memory

```
Error: Model too large for device memory
```

**Solution**: Use a smaller quantization (Q4_K_M instead of Q5_K_M) or reduce context size.

### Slow Generation

**Solutions**:
- Enable GPU acceleration (`gpu: 'auto'`)
- Reduce context size
- Reduce thread count if overheating
- Use Q4 quantization instead of Q5/Q8

### Model Not Loading

```
Error: Failed to load GGUF file
```

**Solutions**:
- Ensure model file is bundled in app assets
- Check file path is correct
- Verify GGUF version is compatible with llama.rn

## File Size Reference

For a 7B parameter model:

| Format | File Size |
|--------|----------|
| Q4_K_M | ~3.5 GB |
| Q5_K_M | ~4.3 GB |
| Q8_0 | ~7.0 GB |
| F16 | ~14.0 GB |

## Next Steps

- Check [llama.rn GitHub](https://github.com/nicktasios/llama.rn) for latest updates
