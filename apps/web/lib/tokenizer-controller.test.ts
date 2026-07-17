import { beforeEach, describe, expect, it, vi } from 'vitest'

// Mock auth store – token not needed for tokenizer endpoints
vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

// Mock config – avoid real network calls
vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { tokenizerController } from './tokenizer-controller'

describe('tokenizerController.getStats', () => {
  beforeEach(() => vi.clearAllMocks())
  it('GETs /tokenizer/stats and returns typed data', async () => {
    const mock = { vocab_size: 32000, base_chars: 256, merged_subwords: 31744, special_tokens: 6, total_merges: 1000 }
    apiClient.apiGet.mockResolvedValue(mock)
    const result = await tokenizerController.getStats()
    expect(result.vocab_size).toBe(32000)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/tokenizer/stats')
  })
})

describe('tokenizerController.tokenize', () => {
  beforeEach(() => vi.clearAllMocks())
  it('POSTs /tokenizer/tokenize with text', async () => {
    const mock = { tokens: ['Hello'], ids: [123] }
    apiClient.apiPost.mockResolvedValue(mock)
    const result = await tokenizerController.tokenize('Hello')
    expect(result.ids).toEqual([123])
    expect(apiClient.apiPost).toHaveBeenCalledWith('/tokenizer/tokenize', { text: 'Hello' })
  })
})

describe('tokenizerController.detokenize', () => {
  beforeEach(() => vi.clearAllMocks())
  it('POSTs /tokenizer/detokenize with ids', async () => {
    const mock = { text: 'World' }
    apiClient.apiPost.mockResolvedValue(mock)
    const result = await tokenizerController.detokenize([456])
    expect(result.text).toBe('World')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/tokenizer/detokenize', { ids: [456] })
  })
})

describe('tokenizerController.getVocab', () => {
  beforeEach(() => vi.clearAllMocks())
  it('GETs /tokenizer/vocab with query params', async () => {
    const mock = { entries: [], total: 0, offset: 0, limit: 20 }
    apiClient.apiGet.mockResolvedValue(mock)
    const result = await tokenizerController.getVocab(20, 5)
    expect(result.limit).toBe(20)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/tokenizer/vocab?limit=20&offset=5')
  })
})

describe('tokenizerController.getMerges', () => {
  beforeEach(() => vi.clearAllMocks())
  it('GETs /tokenizer/merges with default limit', async () => {
    const mock = { merges: [], total: 0 }
    apiClient.apiGet.mockResolvedValue(mock)
    const result = await tokenizerController.getMerges()
    expect(result.total).toBe(0)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/tokenizer/merges?limit=30')
  })
})

describe('tokenizerController.getSamples', () => {
  beforeEach(() => vi.clearAllMocks())
  it('GETs /tokenizer/sample', async () => {
    const mock = { samples: [{ word: 'test', ids: [1], tokens: ['t'], count: 10 }] }
    apiClient.apiGet.mockResolvedValue(mock)
    const result = await tokenizerController.getSamples()
    expect(result.samples[0].word).toBe('test')
    expect(apiClient.apiGet).toHaveBeenCalledWith('/tokenizer/sample')
  })
})

describe('tokenizerController.trainTokenizer', () => {
  beforeEach(() => vi.clearAllMocks())
  it('POSTs to /tokenizer/train with vocab size', async () => {
    const mock = { status: 'ok' }
    apiClient.apiPost.mockResolvedValue(mock)
    const result = await tokenizerController.trainTokenizer(1024)
    expect(result.status).toBe('ok')
    expect(apiClient.apiPost).toHaveBeenCalledWith('/tokenizer/train', { vocab_size: 1024, texts: undefined })
  })
})
