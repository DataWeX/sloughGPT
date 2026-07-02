/**
 * Global API intercepts for e2e tests
 * This file is loaded in cypress/support/e2e.ts
 */

const api = 'http://localhost:8000'

Cypress.Commands.add('mockHealth', (overrides = {}) => {
  cy.intercept('GET', `${api}/health`, {
    statusCode: 200,
    body: {
      status: 'healthy',
      model_type: 'gpt2',
      model_loaded: false,
      ...overrides,
    },
  }).as('health')
})

Cypress.Commands.add('mockModels', () => {
  cy.intercept('GET', `${api}/models`, {
    statusCode: 200,
    body: { models: [] },
  }).as('models')
})

Cypress.Commands.add('mockDatasets', () => {
  cy.intercept('GET', `${api}/datasets`, {
    statusCode: 200,
    body: { datasets: [] },
  }).as('datasets')
})

Cypress.Commands.add('mockSystem', () => {
  cy.intercept('GET', `${api}/system/metrics`, {
    statusCode: 200,
    body: { cpu_percent: 45.2, memory_percent: 62.1, memory_used_gb: 8.2, memory_total_gb: 16 },
  }).as('systemMetrics')
  cy.intercept('GET', `${api}/system/info`, {
    statusCode: 200,
    body: { platform: 'Darwin', platform_release: '24.0.0', architecture: 'arm64', processor: 'Apple M3', cpu_count: 12 },
  }).as('systemInfo')
  cy.intercept('GET', `${api}/system/disk`, {
    statusCode: 200,
    body: { total_gb: 256, used_gb: 120, free_gb: 136, percent: 46.9 },
  }).as('systemDisk')
  cy.intercept('GET', `${api}/health/detailed`, {
    statusCode: 200,
    body: {
      status: 'healthy', uptime_seconds: 3600, timestamp: new Date().toISOString(),
      system: { cpu_percent: 45.2, memory_percent: 62.1, memory_available_mb: 6144 },
      gpu: { backend: 'mps', device_type: 'gpu', vram_gb: 18, tier: 'high', memory_hint: '18 GB unified' },
      model_loaded: true, model_type: 'gpt2',
      inference: { inference_count: 42 },
    },
  }).as('detailedHealth')
})

Cypress.Commands.add('mockTokenizer', () => {
  cy.intercept('GET', `${api}/tokenizer/stats`, {
    statusCode: 200,
    body: { vocab_size: 50257, merges_count: 50000, pad_token: '<|endoftext|>', unk_token: '<|endoftext|>', max_length: 1024, base_chars: 128, merged_subwords: 50129, total_merges: 50000, special_tokens: 3 },
  }).as('tokenizerStats')
  cy.intercept('GET', `${api}/tokenizer/sample`, {
    statusCode: 200,
    body: { samples: [
      { word: 'the', tokens: ['the'], ids: [0], count: 523 },
      { word: 'quick', tokens: ['quick'], ids: [1], count: 87 },
      { word: 'brown', tokens: ['brown'], ids: [2], count: 42 },
    ]},
  }).as('tokenizerSamples')
})

Cypress.Commands.add('mockKnowledge', (items: string[] = []) => {
  const knowledgeItems = items.map((content, i) => ({
    id: `k${i + 1}`,
    content,
    topic: 'general',
    source: 'test',
    importance: 1,
    score: 1,
    created_at: new Date().toISOString(),
  }))
  cy.intercept('GET', `${api}/knowledge`, {
    statusCode: 200,
    body: knowledgeItems,
  }).as('mockKnowledgeList')
  cy.intercept('POST', `${api}/knowledge`, {
    statusCode: 200,
    body: { id: 'k-new', content: '', topic: 'general', source: 'manual', importance: 1, score: 1, created_at: new Date().toISOString() },
  }).as('mockKnowledgeAdd')
  cy.intercept('POST', `${api}/knowledge/batch-delete`, {
    statusCode: 200,
    body: { deleted: items.length },
  }).as('mockKnowledgeDelete')
})

Cypress.Commands.add('mockAll', () => {
  cy.mockHealth()
  cy.mockModels()
  cy.mockDatasets()
  cy.mockSystem()
  cy.mockTokenizer()
  cy.mockVisual()
})

Cypress.Commands.add('mockVisual', () => {
  cy.intercept('GET', `${api}/multimodal/checkpoints`, {
    statusCode: 200,
    body: {
      checkpoints: [
        { name: 'visual-v1', path: '/models/visual-checkpoints/visual-v1', size_mb: 256, created_at: '2026-06-22T10:00:00Z', soul_name: 'vision-v1', lineage: '', llm: 'Qwen2.5-0.5B-Instruct', final_loss: 0.85, total_steps: 120, mean_accuracy: 72.5, description: 'First visual checkpoint' },
        { name: 'visual-v2', path: '/models/visual-checkpoints/visual-v2', size_mb: 258, created_at: '2026-06-23T10:00:00Z', soul_name: 'vision-v2', lineage: 'visual-v1', llm: 'Qwen2.5-0.5B-Instruct', final_loss: 0.42, total_steps: 200, mean_accuracy: 88.3, description: 'Improved visual checkpoint' },
      ],
    },
  }).as('visualCheckpoints')
  cy.intercept('GET', `${api}/multimodal/status`, {
    statusCode: 200,
    body: { visual_loaded: true, model_dir: 'models/visual-finetuned', total_checkpoints: 2 },
  }).as('visualStatus')
  cy.intercept('GET', `${api}/multimodal/dpo/status`, {
    statusCode: 200,
    body: { dpo_running: false, dpo_completed: true, dpo_delta_ppl: 0.15, dpo_delta_bleu: 3.2 },
  }).as('dpoStatus')
  cy.intercept('GET', `${api}/multimodal/train/status`, {
    statusCode: 200,
    body: { training: false, completed: false, progress: 0, epoch: 0, loss: null, error: null },
  }).as('visualTrainStatus')
  cy.intercept('POST', `${api}/multimodal/checkpoints/*/load`, {
    statusCode: 200,
    body: { status: 'staged', name: 'visual-v2', path: 'models/visual-finetuned' },
  }).as('visualLoad')
  cy.intercept('DELETE', `${api}/multimodal/checkpoints/*`, {
    statusCode: 200,
    body: { status: 'deleted', name: 'visual-v1' },
  }).as('visualDelete')
  cy.intercept('POST', `${api}/multimodal/load`, {
    statusCode: 200,
    body: { status: 'loaded', model_dir: 'models/visual-finetuned' },
  }).as('visualProviderLoad')
})

Cypress.Commands.add('mockMultimodal', () => {
  cy.intercept('GET', `${api}/multimodal/capabilities`, {
    statusCode: 200,
    body: {
      vision_enabled: true, speech_enabled: false,
      vision_model: 'slonet', learning_progress: 0.65,
      images_learned: 42, vocab_size: 128, caption_history: [],
      accuracy_history: [], mean_accuracy: 64.2, last_accuracy: 58.0,
      trained: true, replay_buffer_size: 500,
    },
  }).as('multimodalCaps')
  cy.intercept('GET', `${api}/multimodal/training-report`, {
    statusCode: 200,
    body: {
      images_learned: 42, vocab_size: 128,
      caption_history: ['a cat sitting on a chair', 'a red car', 'a dog in the park'],
      accuracy_history: [40, 50, 60, 70, 65, 72, 68, 75, 80, 78],
      mean_accuracy: 64.2, last_accuracy: 78,
    },
  }).as('multimodalReport')
  cy.intercept('GET', `${api}/multimodal/training-status`, {
    statusCode: 200,
    body: { running: false, completed: 0, errors: 0 },
  }).as('multimodalTrainStatus')
  cy.intercept('POST', `${api}/multimodal/reset`, {
    statusCode: 200,
    body: { status: 'success', message: 'Model reset' },
  }).as('multimodalReset')
})

Cypress.Commands.add('mockAgents', (overrides: any[] = []) => {
  const defaultAgents = [
    { id: 'assistant', name: 'Assistant', description: 'General purpose AI assistant', instructions: 'You are a helpful AI assistant.', tools: ['memory', 'file_search'], avatar: 'A' },
    { id: 'coder', name: 'Coder', description: 'Programming and code execution', instructions: 'You are an expert programmer.', tools: ['code_execution', 'file_read'], avatar: 'C' },
  ]
  const agents = overrides.length > 0 ? overrides : defaultAgents
  cy.intercept('GET', `${api}/agents`, { statusCode: 200, body: agents }).as('agentsList')
  cy.intercept('POST', `${api}/agents`, { statusCode: 201, body: agents[0] }).as('agentsCreate')
  cy.intercept('PUT', `${api}/agents/*`, { statusCode: 200, body: agents[0] }).as('agentsUpdate')
  cy.intercept('DELETE', `${api}/agents/*`, { statusCode: 200, body: { status: 'deleted' } }).as('agentsDelete')
  cy.intercept('POST', `${api}/agents/*/execute`, { statusCode: 200, body: { response: 'This is a simulated agent response.', tools_used: [] } }).as('agentsExecute')
})

Cypress.Commands.add('mockExport', () => {
  cy.intercept('GET', `${api}/models`, { statusCode: 200, body: [
    { id: 'gpt2', name: 'gpt2', loaded: true, size_gb: 0.5 },
    { id: 'qwen', name: 'Qwen2.5-0.5B-Instruct', loaded: false, size_gb: 1.2 },
  ]}).as('exportModels')
  cy.intercept('GET', `${api}/health`, { statusCode: 200, body: { status: 'healthy', model_loaded: true, model_type: 'gpt2' } }).as('exportHealth')
  cy.intercept('GET', `${api}/models/export/formats`, { statusCode: 200, body: { formats: ['sou', 'onnx', 'gguf'] } }).as('exportFormats')
  cy.intercept('POST', `${api}/models/export`, { statusCode: 200, body: { status: 'exported', format: 'sou', files: ['models/exported/model.sou'] } }).as('modelExport')
})
