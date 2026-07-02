// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

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

import { voiceController } from './voice-controller'

describe('voiceController', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('tts POSTs /voice/tts', async () => {
    apiClient.apiPost.mockResolvedValue({ audio: 'base64...', sample_rate: 24000, duration_ms: 1000, backend: 'hf-model' })
    const result = await voiceController.tts('Hello world')
    expect(result.backend).toBe('hf-model')
    expect(result.duration_ms).toBe(1000)
    expect(apiClient.apiPost).toHaveBeenCalledWith('/voice/tts', { text: 'Hello world' })
  })

  it('status GETs /voice/status', async () => {
    apiClient.apiGet.mockResolvedValue({ server_tts: true, model: 'gpt2', error: null })
    const result = await voiceController.status()
    expect(result.server_tts).toBe(true)
    expect(apiClient.apiGet).toHaveBeenCalledWith('/voice/status')
  })

  it('handles tts error', async () => {
    apiClient.apiPost.mockRejectedValue(new Error('TTS failed'))
    await expect(voiceController.tts('test')).rejects.toThrow('TTS failed')
  })

  describe('playAudio', () => {
    beforeEach(() => {
      vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:url'), revokeObjectURL: vi.fn() })
    })
    afterEach(() => { vi.unstubAllGlobals() })

    it('calls Audio with decoded base64 wav', async () => {
      const mockPlay = vi.fn().mockResolvedValue(undefined)
      const mockAudio = vi.fn()
      mockAudio.mockReturnValue({ play: mockPlay, onended: null, onerror: null })
      vi.stubGlobal('Audio', mockAudio)

      const base64 = btoa('fakewavdata')
      const promise = voiceController.playAudio(base64, 44100)

      const audioInstance = mockAudio.mock.results[0].value
      audioInstance.onended()

      await promise
      expect(mockAudio).toHaveBeenCalled()
      expect(mockPlay).toHaveBeenCalled()
      expect(URL.createObjectURL).toHaveBeenCalled()
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:url')
    })

    it('rejects on audio error', async () => {
      const mockPlay = vi.fn()
      const mockAudio = vi.fn()
      mockAudio.mockReturnValue({ play: mockPlay, onended: null, onerror: null })
      vi.stubGlobal('Audio', mockAudio)

      const base64 = btoa('data')
      const promise = voiceController.playAudio(base64, 44100)
      const err = new Error('Playback failed')
      const audioInstance = mockAudio.mock.results[0].value
      audioInstance.onerror(err)

      await expect(promise).rejects.toThrow('Playback failed')
    })

    it('rejects on malformed base64', async () => {
      await expect(voiceController.playAudio('!!!invalid!!!', 44100)).rejects.toThrow()
    })
  })
})
