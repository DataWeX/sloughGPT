# SloughGPT TypeScript SDK

Type-safe JavaScript/TypeScript client for the SloughGPT API. Works in **Node.js**, **browser**, and **React Native**.

## Installation

```bash
npm install @sloughgpt/typescript-sdk
```

## Quick Start

```typescript
import SloughGPT from '@sloughgpt/typescript-sdk';

const client = new SloughGPT({ baseUrl: 'http://localhost:8000' });

// Text generation
const result = await client.generate({ prompt: 'Hello, world!' });
console.log(result.text);

// Chat completion
const chat = await client.chat({
  messages: [{ role: 'user', content: 'Hello!' }],
});
console.log(chat.message.content);
```

## API Reference

### Generation & Inference

```typescript
const result = await client.generate({
  prompt: 'Write a haiku',
  max_new_tokens: 50,
  temperature: 0.8,
  top_k: 50,
  top_p: 0.9,
  model: 'gpt2',
});

for await (const token of client.generateStream({ prompt: 'Once upon a time' })) {
  process.stdout.write(token);
}
```

### Chat

```typescript
const result = await client.chat({
  messages: [{ role: 'user', content: 'What is 2+2?' }],
  temperature: 0.7,
});

for await (const token of client.chatStream({ messages: [{ role: 'user', content: 'Hi' }] })) {
  process.stdout.write(token);
}
```

### Health & System

```typescript
const health = await client.health();
const info = await client.info();
const detailed = await client.detailedHealth();
const metrics = await client.getSystemMetrics();
const disk = await client.getSystemDisk();
```

### Models

```typescript
const models = await client.listModels();
await client.loadModel('gpt2');
await client.unloadModel();
const current = await client.getCurrentModel();
const hfModels = await client.listHuggingFaceModels('qwen', 20);
```

### Souls

```typescript
const souls = await client.listSouls();
const current = await client.getCurrentSoul();
await client.switchSoul('friendly');
await client.switchSoul('friendly', 'checkpoint-v2');
```

### Knowledge

```typescript
const items = await client.listKnowledge();
const item = await client.addKnowledge('Paris is the capital of France', 'geography');
await client.deleteKnowledge('k1');
const results = await client.searchKnowledge('capital');
const stats = await client.getKnowledgeStats();
const topics = await client.getKnowledgeTopics();
await client.ingestKnowledgeUrl('https://example.com/doc');
```

### Sessions

```typescript
await client.saveSessionContext('sess-1', { context: { /* ... */ } });
const messages = await client.getSessionMessages('sess-1');
for await (const token of client.regenerateStream('sess-1')) {
  process.stdout.write(token);
}
```

### Tokenizer

```typescript
const stats = await client.getTokenizerStats();
const { tokens } = await client.tokenize('Hello world');
await client.trainTokenizer('training text', 32000);
```

### Training & Auto-Train

```typescript
const job = await client.startTraining({ name: 'run-1', model: 'sloughgpt', dataset: 'shakespeare' });
const status = await client.getTrainingStatus(job.id);
const jobs = await client.listTrainingJobs();
await client.stopTraining();
await client.pauseTraining();
await client.resumeTraining();

const ckpts = await client.listAutoTrainCheckpoints();
await client.loadAutoTrainCheckpoint('best');
await client.deleteAutoTrainCheckpoint('old-ckpt');
```

### Companion / Personality

```typescript
const presets = await client.listCompanionPresets();
const prompt = await client.getCompanionPrompt();
await client.setPersonality('friendly');
```

### Feedback & Workflow

```typescript
await client.recordFeedback({ session_id: 's1', message_id: 'm1', score: 1 });
const stats = await client.getFeedbackStats();
const wf = await client.getWorkflowStatus();
```

### Experiments

```typescript
const exp = await client.createExperiment('My Exp', 'Description');
await client.logMetric('exp-1', 'accuracy', 0.95, 100);
const experiments = await client.listExperiments();
```

### Datasets

```typescript
await client.importDatasetLocal('/path/to/data', 'my-dataset');
await client.importDatasetGitHub('user/repo', 'repo-data');
await client.importDatasetUrl('https://example.com/data.txt', 'url-data');
```

### Metrics, Rate Limit & Security

```typescript
const m = await client.metrics();
const rl = await client.rateLimitStatus();
const audit = await client.getAuditLog();
const keys = await client.getSecurityKeys();
```

## React Hook

```tsx
import { useSloughGPT } from '@sloughgpt/typescript-sdk/react';

function ChatComponent() {
  const { isReady, isLoading, error, generate, chat, health } = useSloughGPT({
    baseUrl: 'http://localhost:8000',
  });

  const handleGenerate = async () => {
    const text = await generate('Hello world');
    console.log(text);
  };

  return <button onClick={handleGenerate} disabled={isLoading}>Generate</button>;
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
  }
}
```

## License

MIT
