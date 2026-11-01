import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

let mockApiGet: ReturnType<typeof vi.fn>
let mockApiPost: ReturnType<typeof vi.fn>
let mockApiPatch: ReturnType<typeof vi.fn>
let mockApiDelete: ReturnType<typeof vi.fn>

vi.mock('@/lib/http-client', () => ({
  apiGet: (...args: unknown[]) => mockApiGet(...args),
  apiPost: (...args: unknown[]) => mockApiPost(...args),
  apiPatch: (...args: unknown[]) => mockApiPatch(...args),
  apiDelete: (...args: unknown[]) => mockApiDelete(...args),
}))

const { companionController } = await import('@/lib/companion-controller')

describe('companionController', () => {
  beforeEach(() => {
    mockApiGet = vi.fn()
    mockApiPost = vi.fn()
    mockApiPatch = vi.fn()
    mockApiDelete = vi.fn()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getInfo', () => {
    it('fetches companion info', async () => {
      mockApiGet.mockResolvedValue({ traits: { warmth: 0.5 } })
      const result = await companionController.getInfo()
      expect(mockApiGet).toHaveBeenCalledWith('/companion/')
      expect(result).toEqual({ traits: { warmth: 0.5 } })
    })

    it('propagates errors', async () => {
      mockApiGet.mockRejectedValue(new Error('fail'))
      await expect(companionController.getInfo()).rejects.toThrow('fail')
    })
  })

  describe('setPersonality', () => {
    it('posts full traits', async () => {
      const traits = { name: 'Test', warmth: 0.8, curiosity: 0.5, creativity: 0.5, confidence: 0.5, humor: 0.5 }
      mockApiPost.mockResolvedValue({ status: 'ok', traits })
      const result = await companionController.setPersonality(traits)
      expect(mockApiPost).toHaveBeenCalledWith('/companion/personality', traits)
      expect(result).toEqual({ status: 'ok', traits })
    })
  })

  describe('patchPersonality', () => {
    it('patches partial traits', async () => {
      mockApiPatch.mockResolvedValue({ status: 'ok', traits: { warmth: 0.9 } })
      const result = await companionController.patchPersonality({ warmth: 0.9 })
      expect(mockApiPatch).toHaveBeenCalledWith('/companion/personality', { warmth: 0.9 })
      expect(result).toEqual({ status: 'ok', traits: { warmth: 0.9 } })
    })
  })

  describe('setPreset', () => {
    it('posts preset with default name', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', preset: 'warm' })
      await companionController.setPreset('warm')
      expect(mockApiPost).toHaveBeenCalledWith('/companion/preset', { preset: 'warm', name: 'Friend' })
    })

    it('posts preset with custom name', async () => {
      mockApiPost.mockResolvedValue({ status: 'ok', preset: 'curious' })
      await companionController.setPreset('curious', 'Alice')
      expect(mockApiPost).toHaveBeenCalledWith('/companion/preset', { preset: 'curious', name: 'Alice' })
    })
  })

  describe('getPrompt', () => {
    it('fetches system prompt', async () => {
      mockApiGet.mockResolvedValue({ system_prompt: 'You are helpful.' })
      const result = await companionController.getPrompt()
      expect(mockApiGet).toHaveBeenCalledWith('/companion/prompt')
      expect(result).toEqual({ system_prompt: 'You are helpful.' })
    })
  })

  describe('listPresets', () => {
    it('fetches presets list', async () => {
      mockApiGet.mockResolvedValue({ presets: [{ id: '1', name: 'Warm', description: 'Friendly' }] })
      const result = await companionController.listPresets()
      expect(mockApiGet).toHaveBeenCalledWith('/companion/presets')
      expect(result.presets).toHaveLength(1)
      expect(result.presets[0].name).toBe('Warm')
    })
  })

  describe('reset', () => {
    it('resets personality', async () => {
      mockApiDelete.mockResolvedValue({ status: 'ok', traits: { warmth: 0.5 } })
      const result = await companionController.reset()
      expect(mockApiDelete).toHaveBeenCalledWith('/companion/')
      expect(result.status).toBe('ok')
    })
  })

  describe('chat', () => {
    it('sends message', async () => {
      mockApiPost.mockResolvedValue({ response: 'Hello!', system_prompt: 'You are friendly.' })
      const result = await companionController.chat('Hi there')
      expect(mockApiPost).toHaveBeenCalledWith('/companion/chat', {
        message: 'Hi there',
        user_name: undefined,
        user_mood: undefined,
        include_system_prompt: true,
      })
      expect(result.response).toBe('Hello!')
    })

    it('sends message with opts', async () => {
      mockApiPost.mockResolvedValue({ response: 'Hi!', system_prompt: '...' })
      await companionController.chat('Hey', { user_name: 'Bob', user_mood: 'happy' })
      expect(mockApiPost).toHaveBeenCalledWith('/companion/chat', {
        message: 'Hey',
        user_name: 'Bob',
        user_mood: 'happy',
        include_system_prompt: true,
      })
    })
  })
})
