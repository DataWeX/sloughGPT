import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./auth', () => ({
  useAuthStore: {
    getState: () => ({ token: null as string | null }),
  },
}))

vi.mock('./config', () => ({
  PUBLIC_API_URL: 'http://127.0.0.1:9',
}))

import { setupApiMocks, apiClient } from './__test-helper'
setupApiMocks()

import { inferController } from './infer-controller'

describe('inferController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  // ── generate ──────────────────────────────────────────────────

  describe('generate', () => {
    it('returns text from /infer', async () => {
      apiClient.apiPost.mockResolvedValue({
        text: 'Hello world',
        model: 'gpt2',
        tokens_generated: 2,
        elapsed_ms: 100,
      })

      const result = await inferController.generate({ prompt: 'hello' })
      expect(result.text).toBe('Hello world')
      expect(result.model).toBe('gpt2')
      expect(result.tokens_generated).toBe(2)
    })

    it('passes all parameters', async () => {
      apiClient.apiPost.mockResolvedValue({ text: '', model: '', tokens_generated: 0, elapsed_ms: 0 })

      await inferController.generate({
        prompt: 'test',
        max_new_tokens: 50,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 30,
      })

      expect(apiClient.apiPost).toHaveBeenCalledWith('/infer', {
        prompt: 'test',
        max_new_tokens: 50,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 30,
      })
    })
  })

  // ── generateStream ────────────────────────────────────────────

  describe('generateStream', () => {
    it('calls onToken for each token from SSE', async () => {
      const tokens: string[] = []
      const encoder = new TextEncoder()

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"stream":"infer","phase":"STREAMING","status":"working","data":{"token":"Hello"}}\n\n'))
          controller.enqueue(encoder.encode('data: {"stream":"infer","phase":"STREAMING","status":"working","data":{"token":" world"}}\n\n'))
          controller.enqueue(encoder.encode('data: {"stream":"infer","phase":"STREAMING","status":"complete","data":{},"meta":{"tokens":2}}\n\n'))
          controller.close()
        },
      })

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        body: stream,
      })

      await inferController.generateStream(
        { prompt: 'hello' },
        (t) => tokens.push(t),
        () => {},
      )

      expect(tokens).toEqual(['Hello', ' world'])
    })

    it('calls onError on error event', async () => {
      let errorMsg = ''
      const encoder = new TextEncoder()

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"stream":"infer","phase":"STREAMING","status":"error","data":{"error":"boom"}}\n\n'))
          controller.close()
        },
      })

      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        body: stream,
      })

      await inferController.generateStream(
        { prompt: 'hello' },
        () => {},
        () => {},
        (e) => { errorMsg = e },
      )

      expect(errorMsg).toBe('boom')
    })

    it('calls onError on HTTP failure', async () => {
      let errorMsg = ''
      global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 503 })

      await inferController.generateStream(
        { prompt: 'hello' },
        () => {},
        () => {},
        (e) => { errorMsg = e },
      )

      expect(errorMsg).toBe('HTTP 503')
    })
  })

  // ── embed ─────────────────────────────────────────────────────

  describe('embed', () => {
    it('returns embedding from /infer/embed', async () => {
      apiClient.apiPost.mockResolvedValue({
        embedding: [0.1, 0.2, 0.3],
        dimensions: 3,
        model: 'test',
      })

      const result = await inferController.embed({ text: 'hello' })
      expect(result.embedding).toEqual([0.1, 0.2, 0.3])
      expect(result.dimensions).toBe(3)
    })
  })

  // ── tokenize ──────────────────────────────────────────────────

  describe('tokenize', () => {
    it('returns tokens from /infer/tokenize', async () => {
      apiClient.apiPost.mockResolvedValue({
        tokens: ['hello', 'world'],
        ids: [1, 2],
        count: 2,
      })

      const result = await inferController.tokenize({ text: 'hello world' })
      expect(result.tokens).toEqual(['hello', 'world'])
      expect(result.ids).toEqual([1, 2])
      expect(result.count).toBe(2)
    })
  })

  // ── detokenize ────────────────────────────────────────────────

  describe('detokenize', () => {
    it('returns text from /infer/detokenize', async () => {
      apiClient.apiPost.mockResolvedValue({
        text: 'hello world',
        count: 2,
      })

      const result = await inferController.detokenize({ ids: [1, 2] })
      expect(result.text).toBe('hello world')
      expect(result.count).toBe(2)
    })
  })

  // ── health ────────────────────────────────────────────────────

  describe('health', () => {
    it('returns health from /infer/health', async () => {
      apiClient.apiGet.mockResolvedValue({
        status: 'ready',
        model_loaded: true,
        model_id: 'gpt2',
        engine_type: 'NumpyEngine',
        has_streaming: true,
        has_embedding: false,
      })

      const result = await inferController.health()
      expect(result.status).toBe('ready')
      expect(result.model_loaded).toBe(true)
      expect(result.model_id).toBe('gpt2')
    })
  })

  // ── info ──────────────────────────────────────────────────────

  describe('info', () => {
    it('returns model info from /infer/info', async () => {
      apiClient.apiGet.mockResolvedValue({
        model_id: 'gpt2',
        model_type: 'NumpyEngine',
        num_parameters: 124000000,
        vocab_size: 50257,
        max_context: 1024,
        num_layers: 12,
        has_tokenizer: true,
        has_streaming: true,
        has_embedding: false,
        extra: {},
      })

      const result = await inferController.info()
      expect(result.model_id).toBe('gpt2')
      expect(result.num_parameters).toBe(124000000)
    })
  })

  // ── isReady ───────────────────────────────────────────────────

  describe('isReady', () => {
    it('returns true when model loaded and ready', async () => {
      apiClient.apiGet.mockResolvedValue({
        status: 'ready',
        model_loaded: true,
      })

      expect(await inferController.isReady()).toBe(true)
    })

    it('returns false when no model', async () => {
      apiClient.apiGet.mockResolvedValue({
        status: 'no_model',
        model_loaded: false,
      })

      expect(await inferController.isReady()).toBe(false)
    })

    it('returns false on error', async () => {
      apiClient.apiGet.mockRejectedValue(new Error('conn refused'))

      expect(await inferController.isReady()).toBe(false)
    })
  })
})
