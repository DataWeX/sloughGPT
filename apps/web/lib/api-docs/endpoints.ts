import type { ApiDocEndpoint } from './types'
import { TRAIN_CORPUS_BODY_FIELDS, TRAIN_HYPERPARAM_BODY_FIELDS } from './training-body-fields'
import {
  BENCHMARK_PERPLEXITY_VOCAB_NOTE,
  MODEL_EXPORT_VOCAB_NOTE,
  TRAIN_RESOLVE_VOCAB_NOTE,
  TRAINING_CHECKPOINT_VOCAB_NOTE,
  TRAINING_JOBS_VOCAB_NOTE,
} from './vocab-notes'

export const API_DOC_ENDPOINTS: ApiDocEndpoint[] = [
  {
    method: 'GET',
    path: '/',
    description: 'API root with server info and available endpoints',
  },
  {
    method: 'GET',
    path: '/health',
    description: 'Health check endpoint',
  },
  {
    method: 'GET',
    path: '/info',
    description:
      'Detailed system and model information. When psutil is installed on the API host, includes a `host` object (CPU %, RAM bytes/%, optional `process_rss_bytes` for the API process).',
  },
  {
    method: 'POST',
    path: '/generate',
    description: 'Text generation (non-streaming)',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'personality', type: 'string', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/v1/infer',
    description:
      'Man Standard v1 inference envelope. Optional header X-Man-Standard: 1. See standards/MAN_STANDARD_V1.md.',
    body: [
      { field: 'mode', type: "'generate' | 'chat' | 'structured'", required: true },
      { field: 'task_type', type: 'string', required: false },
      { field: 'model_id', type: 'string', required: false },
      { field: 'input', type: 'object (prompt?, messages?, context?)', required: true },
      { field: 'generation', type: 'object', required: false },
      { field: 'retrieval', type: 'object', required: false },
      { field: 'output_schema_ref', type: 'string', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/generate/stream',
    description: 'Streaming text generation via Server-Sent Events',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/chat',
    description: 'Chat completion (JSON body, non-streaming)',
    body: [
      { field: 'messages', type: 'array', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'top_k', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/chat/stream',
    description: 'Streaming chat completion (SSE)',
    body: [
      { field: 'messages', type: 'array', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'top_k', type: 'number', required: false },
    ],
  },
  {
    method: 'WS',
    path: '/ws/generate',
    description: 'WebSocket endpoint for real-time generation',
  },
  {
    method: 'GET',
    path: '/models',
    description: 'List available models (local + HuggingFace)',
  },
  {
    method: 'POST',
    path: '/models/load',
    description: 'Load a HuggingFace model',
    body: [
      { field: 'model_id', type: 'string', required: true },
      { field: 'mode', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/models/hf',
    description: 'List HuggingFace models',
  },
  {
    method: 'POST',
    path: '/load',
    description: 'Load inference model on-demand',
  },
  {
    method: 'GET',
    path: '/datasets',
    description: 'List available datasets',
  },
  {
    method: 'GET',
    path: '/datasets/{dataset_id}',
    description: 'Get dataset details by ID',
  },
  {
    method: 'GET',
    path: '/datasets/{dataset_id}/preview',
    description: 'Preview dataset content and statistics',
  },
  {
    method: 'POST',
    path: '/datasets/{dataset_id}/versions',
    description: 'Create a version snapshot of the dataset',
  },
  {
    method: 'GET',
    path: '/datasets/{dataset_id}/versions',
    description: 'List all versions of a dataset',
  },
  {
    method: 'POST',
    path: '/datasets/{dataset_id}/rollback/{version_id}',
    description: 'Rollback dataset to a previous version',
  },
  {
    method: 'POST',
    path: '/datasets/{dataset_id}/export',
    description: 'Export dataset to format (json, jsonl, csv, txt)',
  },
  {
    method: 'POST',
    path: '/datasets/import/github',
    description: 'Import dataset from GitHub',
  },
  {
    method: 'POST',
    path: '/datasets/import/huggingface',
    description: 'Import dataset from HuggingFace Hub',
  },
  {
    method: 'POST',
    path: '/datasets/import/url',
    description: 'Import dataset from URL',
  },
  {
    method: 'POST',
    path: '/datasets/import/local',
    description: 'Import local dataset files',
  },
  {
    method: 'POST',
    path: '/train',
    description:
      'Start char-level training in the background (TrainRequest). Same hyperparameters as POST /training/start except name/model. Exactly one corpus selector.' +
      TRAINING_CHECKPOINT_VOCAB_NOTE,
    body: [...TRAIN_CORPUS_BODY_FIELDS, ...TRAIN_HYPERPARAM_BODY_FIELDS],
  },
  {
    method: 'POST',
    path: '/train/resolve',
    description:
      'Resolve corpus to data_path and checkpoint stem (dry run; no training). Same mutually exclusive source fields as POST /train.' +
      TRAIN_RESOLVE_VOCAB_NOTE,
    body: [
      { field: 'dataset', type: 'string', required: false },
      { field: 'manifest_uri', type: 'string', required: false },
      { field: 'dataset_ref.dataset_id', type: 'string', required: false },
      { field: 'dataset_ref.version', type: 'string', required: false },
      { field: 'dataset_ref.manifest_uri', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/train/status',
    description: 'Get training status',
  },
  {
    method: 'GET',
    path: '/metrics/prometheus',
    description: 'Prometheus text exposition format (HTTP + Man metrics when enabled)',
  },
  {
    method: 'GET',
    path: '/personalities',
    description: 'List available AI personalities',
  },
  {
    method: 'POST',
    path: '/inference/generate',
    description: 'Production inference engine - Generate text',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
      { field: 'top_p', type: 'number', required: false },
      { field: 'top_k', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/inference/generate/stream',
    description: 'Production inference - Streaming text (SSE)',
    body: [
      { field: 'prompt', type: 'string', required: true },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'temperature', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/inference/quantize',
    description: 'Quantize loaded model (fp16, int8, etc.)',
    body: [{ field: 'quantization_type', type: 'string', required: true }],
  },
  {
    method: 'GET',
    path: '/inference/stats',
    description: 'Get inference engine statistics',
  },
  {
    method: 'POST',
    path: '/benchmark/run',
    description: 'Run inference benchmark',
    body: [
      { field: 'prompt', type: 'string', required: false },
      { field: 'max_new_tokens', type: 'number', required: false },
      { field: 'num_runs', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/benchmark/perplexity',
    description: 'Calculate model perplexity on inline text.' + BENCHMARK_PERPLEXITY_VOCAB_NOTE,
    body: [{ field: 'text', type: 'string', required: true }],
  },
  {
    method: 'POST',
    path: '/model/export',
    description: 'Export current model to file.' + MODEL_EXPORT_VOCAB_NOTE,
    body: [
      { field: 'output_path', type: 'string', required: true },
      { field: 'format', type: 'string', required: false },
      { field: 'include_tokenizer', type: 'boolean', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/model/export/formats',
    description: 'List supported export formats',
  },
  {
    method: 'GET',
    path: '/experiments',
    description: 'List all experiments',
  },
  {
    method: 'POST',
    path: '/experiments',
    description: 'Create new experiment',
    body: [
      { field: 'name', type: 'string', required: true },
      { field: 'description', type: 'string', required: false },
      { field: 'parameters', type: 'string', required: false },
    ],
  },
  {
    method: 'GET',
    path: '/experiments/{id}',
    description: 'Get specific experiment',
  },
  {
    method: 'POST',
    path: '/experiments/{id}/log_metric',
    description: 'Log a metric to experiment',
    body: [
      { field: 'metric_name', type: 'string', required: true },
      { field: 'value', type: 'number', required: true },
      { field: 'step', type: 'number', required: false },
    ],
  },
  {
    method: 'POST',
    path: '/experiments/{id}/log_param',
    description: 'Log a parameter to experiment',
    body: [
      { field: 'param_name', type: 'string', required: true },
      { field: 'value', type: 'any', required: true },
    ],
  },
  {
    method: 'POST',
    path: '/training/start',
    description:
      'Start a tracked training job (web UI). Exactly one of: dataset, manifest_uri, or nested dataset_ref.' +
      TRAINING_CHECKPOINT_VOCAB_NOTE,
    body: [
      { field: 'name', type: 'string', required: true },
      { field: 'model', type: 'string', required: true },
      ...TRAIN_CORPUS_BODY_FIELDS,
      ...TRAIN_HYPERPARAM_BODY_FIELDS,
    ],
  },
  {
    method: 'GET',
    path: '/training/jobs',
    description: 'List all training jobs.' + TRAINING_JOBS_VOCAB_NOTE,
  },
]
