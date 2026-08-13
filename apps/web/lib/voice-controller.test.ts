import { describe, it, expect, vi, beforeEach } from 'vitest'
import { voiceController } from './voice-controller'
import * as http from './http-client'

vi.mock('./http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

const apiGet = vi.mocked(http.apiGet)
const apiPost = vi.mocked(http.apiPost)

beforeEach(() => { vi.clearAllMocks() })

describe('voiceController', () => {
  it('getStatus unwraps {data} envelope', async () => {
    apiGet.mockResolvedValue({ data: { server_tts: true, model: 'bark', error: null } })
    const result = await voiceController.getStatus()
    expect(result).toEqual({ server_tts: true, model: 'bark', error: null })
  })

  it('getStatus handles flat response', async () => {
    apiGet.mockResolvedValue({ server_tts: false, model: null, error: 'no model' })
    const result = await voiceController.getStatus()
    expect(result).toEqual({ server_tts: false, model: null, error: 'no model' })
  })

  it('tts calls apiPost with text', async () => {
    apiPost.mockResolvedValue({ audio: 'base64data', duration_ms: 100, backend: 'hf-model', sample_rate: 22050 })
    const result = await voiceController.tts('hello')
    expect(result).toEqual({ audio: 'base64data', duration_ms: 100, backend: 'hf-model', sample_rate: 22050 })
    expect(apiPost).toHaveBeenCalledWith('/voice/tts', { text: 'hello' })
  })

  it('tts sends correct text', async () => {
    apiPost.mockResolvedValue({ audio: 'data', duration_ms: 50 })
    await voiceController.tts('test message')
    expect(apiPost).toHaveBeenCalledWith('/voice/tts', { text: 'test message' })
  })

  it('getStatus calls apiGet', async () => {
    apiGet.mockResolvedValue({ server_tts: true, model: 'bark', error: null })
    await voiceController.getStatus()
    expect(apiGet).toHaveBeenCalledWith('/voice/status')
  })

  it('tts returns audio field', async () => {
    apiPost.mockResolvedValue({ audio: 'base64audio', duration_ms: 200, backend: 'model', sample_rate: 16000 })
    const result = await voiceController.tts('hi')
    expect(result.audio).toBe('base64audio')
  })

  it('getStatus throws on network error', async () => {
    apiGet.mockRejectedValue(new Error('timeout'))
    await expect(voiceController.getStatus()).rejects.toThrow('timeout')
  })

  it('tts throws on network error', async () => {
    apiPost.mockRejectedValue(new Error('timeout'))
    await expect(voiceController.tts('hi')).rejects.toThrow('timeout')
  })
})
