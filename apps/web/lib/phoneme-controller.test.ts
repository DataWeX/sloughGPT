import { describe, it, expect, vi, beforeEach } from 'vitest'
import { phonemeController, PHONEME_LANGUAGES } from './phoneme-controller'

vi.mock('./http-client', () => ({
  apiPost: vi.fn(),
  apiGet: vi.fn(),
}))

import { apiPost } from './http-client'

const mockApiPost = vi.mocked(apiPost)

describe('phonemeController', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('encode', () => {
    it('encodes text to phonemes', async () => {
      mockApiPost.mockResolvedValueOnce({
        text: 'hello',
        language: 'en',
        phonemes: ['HH', 'EH', 'L', 'OW'],
        ids: [25, 15, 23, 32],
        decoded: 'hello',
      })

      const result = await phonemeController.encode('hello', 'en')

      expect(result.phonemes).toEqual(['HH', 'EH', 'L', 'OW'])
      expect(result.language).toBe('en')
      expect(mockApiPost).toHaveBeenCalledWith('/multimodal/encode-phonemes', { text: 'hello', language: 'en' })
    })

    it('handles auto-detect language', async () => {
      mockApiPost.mockResolvedValueOnce({
        text: 'hallo',
        language: 'de',
        phonemes: ['HH', 'AA', 'L', 'OW'],
        ids: [25, 10, 23, 32],
        decoded: 'hallo',
      })

      const result = await phonemeController.encode('hallo')

      expect(result.language).toBe('de')
      expect(mockApiPost).toHaveBeenCalledWith('/multimodal/encode-phonemes', { text: 'hallo', language: undefined })
    })
  })

  describe('score', () => {
    it('scores pronunciation', async () => {
      mockApiPost.mockResolvedValueOnce({
        target: 'hello',
        spoken: 'helo',
        language: 'en',
        score: 0.75,
        precision: 0.8,
        recall: 0.7,
        target_phonemes: ['HH', 'EH', 'L', 'OW'],
        spoken_phonemes: ['HH', 'EH', 'L', 'OW'],
      })

      const result = await phonemeController.score('hello', 'helo', 'en')

      expect(result.score).toBe(0.75)
      expect(result.precision).toBe(0.8)
      expect(result.recall).toBe(0.7)
    })
  })

  describe('synthesize', () => {
    it('returns spectrogram with audio', async () => {
      mockApiPost.mockResolvedValueOnce({
        audio: 'data:audio/wav;base64,abc123',
        text: 'hello',
        duration_sec: 1.5,
        elapsed_ms: 120,
        spectrogram: {
          data: [[0.1, 0.2], [0.3, 0.4]],
          n_mels: 2,
          n_frames: 2,
        },
      })

      const result = await phonemeController.synthesize('hello')

      expect(result.audio).toBe('data:audio/wav;base64,abc123')
      expect(result.duration_sec).toBe(1.5)
      expect(result.spectrogram).toEqual({
        data: [[0.1, 0.2], [0.3, 0.4]],
        n_mels: 2,
        n_frames: 2,
      })
    })
  })

  describe('detectLanguage', () => {
    it('detects language', async () => {
      mockApiPost.mockResolvedValueOnce({
        text: 'hello',
        language: 'en',
        supported_languages: ['en', 'de', 'fr', 'es', 'it', 'pt'],
      })

      const result = await phonemeController.detectLanguage('hello')

      expect(result.language).toBe('en')
      expect(result.supported_languages).toContain('en')
    })
  })

  describe('PHONEME_LANGUAGES', () => {
    it('contains all supported languages', () => {
      expect(PHONEME_LANGUAGES).toHaveLength(6)
      expect(PHONEME_LANGUAGES.map(l => l.value)).toEqual(['en', 'de', 'fr', 'es', 'it', 'pt'])
    })
  })
})
