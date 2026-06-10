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

Cypress.Commands.add('mockExport', () => {
  cy.intercept('GET', `${api}/models/export/formats`, {
    statusCode: 200,
    body: { formats: ['sou', 'pytorch', 'onnx', 'gguf'] },
  }).as('exportFormats')
})

Cypress.Commands.add('mockAgents', () => {
  cy.intercept('GET', `${api}/agents`, {
    statusCode: 200,
    body: [
      { id: 'agent-1', name: 'Helper', description: 'General assistant', instructions: 'Be helpful', tools: ['code_execution'], avatar: '' },
    ],
  }).as('agents')
})

Cypress.Commands.add('mockAll', () => {
  cy.mockHealth()
  cy.mockModels()
  cy.mockDatasets()
  cy.mockSystem()
  cy.mockTokenizer()
  cy.mockExport()
  cy.mockAgents()
})
