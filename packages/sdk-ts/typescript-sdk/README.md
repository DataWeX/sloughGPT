# SloughGPT TypeScript SDK

Type-safe JavaScript/TypeScript client for the SloughGPT API. Works in **Node.js**, **browser**, and **React Native**.

## Installation

```bash
npm install @sloughgpt/typescript-sdk
```

## Development (monorepo)

When working inside **sloughGPT**, use the **Node** version from the repo root **`.nvmrc`** (`nvm use` / `fnm use`; matches **`test-sdk-ts`** in **`.github/workflows/ci_cd.yml`**). From this package directory:

```bash
npm ci
npm run ci      # lint + build + test (same as CI job test-sdk-ts)
```

## Quick Start

```typescript
import SloughGPT from '@sloughgpt/typescript-sdk';

const client = new SloughGPT({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key',
});

// Text generation
const result = await client.generate({ prompt: 'Hello, world!' });
console.log(result.text);

// Chat completion
const chat = await client.chat({
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(chat.message.content);
```

## Configuration

```typescript
const client = new SloughGPT({
  baseUrl: 'http://localhost:8000',  // API base URL
  apiKey: 'your-api-key',             // Optional API key
  timeout: 30000,                    // Request timeout in ms
  headers: { 'X-Custom': 'value' },   // Custom headers
  onLog: (level, msg) => console.log(level, msg),  // Logging callback
});
```

## API Reference

### Generation

```typescript
// Basic generation
const result = await client.generate({
  prompt: 'Write a haiku about coding',
  max_new_tokens: 50,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.9,
  personality: 'pirate',
  model: 'gpt2',
});

// Streaming generation
for await (const token of client.generateStream({ prompt: 'Once upon a time' })) {
  process.stdout.write(token);
}

// Convenience method
const text = await client.quickGenerate('Hello world');
```

### Chat

```typescript
// Chat completion
const result = await client.chat({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is 2+2?' },
  ],
  temperature: 0.7,
  max_new_tokens: 100,
});

// Streaming chat
for await (const token of client.chatStream({ messages: [{ role: 'user', content: 'Hi' }] })) {
  process.stdout.write(token);
}

// Convenience method
const reply = await client.quickChat('How are you?');
```

### Batch Processing

```typescript
const results = await client.batchGenerate([
  'What is AI?',
  'What is ML?',
  'What is NLP?',
]);
```

### Health & Status

```typescript
const health = await client.health();
const info = await client.info();
const ready = await client.readiness();
```

### Model Registry

```typescript
// Register a model
const model = await client.registerModel({
  name: 'My Custom Model',
  model_type: 'gpt2',
  description: 'Fine-tuned GPT-2',
  config: { learning_rate: 1e-4 },
});

// List models
const models = await client.listRegisteredModels();

// Get best model by metric
const best = await client.getBestRegisteredModel({
  metric: 'accuracy',
  order: 'desc',
  model_type: 'gpt2',
});

// Record metrics
await client.recordToRegistry('model-1', {
  latency_ms: 120,
  tokens_generated: 50,
  cache_hit: false,
});

// Get registry stats
const stats = await client.getRegistryStats();
```

### Experiments

```typescript
// Create experiment
const exp = await client.createExperiment('My Experiment', 'Description');

// Log metrics
await client.logMetric('exp-1', 'accuracy', 0.95, step=100);
await client.logMetric('exp-1', 'loss', 0.05);

// List and retrieve
const experiments = await client.listExperiments();
const experiment = await client.getExperiment('exp-1');
```

### Training

Trainer **`step_*.pt`** files on the server include **`stoi` / `itos` / `chars`** for char-LM eval; formats are summarized in [`docs/policies/CONTRIBUTING.md`](../../../docs/policies/CONTRIBUTING.md) (*Checkpoint vocabulary*).

```typescript
// Canonical body matches POST /training/start (``name``, ``model``, one corpus selector).
const job = await client.startTraining({
  name: 'my-run',
  model: 'sloughgpt',
  dataset: 'openwebtext',
  epochs: 3,
  log_interval: 10,
  eval_interval: 100,
});

// Legacy aliases ``model_name`` / ``dataset_id`` are still accepted.
const jobLegacy = await client.startTraining({
  model_name: 'sloughgpt',
  dataset_id: 'openwebtext',
  epochs: 3,
});

const status = await client.getTrainingStatus(job.id);
const jobs = await client.listTrainingJobs();
// `status.checkpoint` / job entries may point at native `step_*.pt` (stoi/itos/chars — CONTRIBUTING).
```

### Inference

```typescript
const result = await client.inferenceGenerate({
  prompt: 'Hello world',
  temperature: 0.7,
});

const stats = await client.inferenceStats();
const batch = await client.inferenceBatch(['Prompt 1', 'Prompt 2']);
```

### Benchmarks

```typescript
const result = await client.runBenchmark({
  model: 'gpt2',
  num_samples: 100,
  dataset: 'wikitext',
});

const perplexity = await client.runPerplexityBenchmark({
  model: 'gpt2',
  dataset: 'wikitext',
});

const comparison = await client.compareBenchmarks(['model-1', 'model-2']);
```

### Cache

```typescript
await client.clearCache();
const stats = await client.cacheStats();
```

### Rate Limiting

```typescript
const status = await client.rateLimitStatus();
const check = await client.rateLimitCheck('generate', 1);
```

### Security

```typescript
const audit = await client.getAuditLog();
const keys = await client.getSecurityKeys();
```

### Auth

```typescript
const token = await client.getToken('username', 'password');
const refreshed = await client.refreshToken('refresh-token');
```

## React Hook

For React applications, use the `useSloughGPT` hook:

```tsx
import { useSloughGPT } from '@sloughgpt/typescript-sdk/react';

function ChatComponent() {
  const { isReady, isLoading, error, generate, chat, health } = useSloughGPT({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key',
  });

  const handleGenerate = async () => {
    const text = await generate('Hello world');
    console.log(text);
  };

  if (!isReady) return <div>Connecting...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <p>Status: {health?.status}</p>
      <button onClick={handleGenerate} disabled={isLoading}>
        {isLoading ? 'Loading...' : 'Generate'}
      </button>
    </div>
  );
}
```

## Error Handling

```typescript
import SloughGPT, { SloughGPTError } from '@sloughgpt/typescript-sdk';

try {
  const result = await client.generate({ prompt: 'Hello' });
} catch (e) {
  if (e instanceof SloughGPTError) {
    console.error(`HTTP ${e.statusCode}: ${e.message}`);
  } else {
    console.error(e);
  }
}
```

## WebSocket Streaming

For real-time streaming with WebSockets, use the built-in streaming methods. They handle SSE (Server-Sent Events) automatically:

```typescript
for await (const token of client.generateStream({ prompt: 'Tell me a story' })) {
  console.log(token); // Tokens arrive incrementally
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLOUGHGPT_BASE_URL` | API base URL | `http://localhost:8000` |
| `SLOUGHGPT_API_KEY` | API key | - |
| `SLOUGHGPT_TIMEOUT` | Request timeout (ms) | `30000` |

## TypeScript

This library is written in TypeScript and ships with full type definitions. No `@types/` packages needed.

## Compatibility

- Node.js **20+** (matches monorepo root **`.nvmrc`**; older LTS may work but is not what CI exercises)
- Modern browsers (with `fetch` support)
- React Native (with `fetch` support)
- Deno (via ESM import)

## License

MIT
