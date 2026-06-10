import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useModelStore } from '@/stores/model-store'

// Mock API client
vi.mock('@/lib/api-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))

import { apiGet, apiPost } from '@/lib/api-client'

describe('Model Store', () => {
  beforeEach(() => {
    useModelStore.setState({
      models: [],
      currentModel: null,
      souls: [],
      currentSoul: null,
      checkpoints: [],
      currentCheckpoint: null,
      health: null,
      loading: false,
      loadingModelId: null,
      error: null,
    })
    vi.clearAllMocks()
  })

  it('should initialize with default state', () => {
    const state = useModelStore.getState()
    expect(state.models).toEqual([])
    expect(state.currentModel).toBeNull()
    expect(state.souls).toEqual([])
    expect(state.currentSoul).toBeNull()
    expect(state.loading).toBe(false)
  })

  it('should refresh models and souls', async () => {
    const mockModels = [
      { id: 'model-1', name: 'GPT-2', loaded: true, sizeGb: 0.5 },
      { id: 'model-2', name: 'Qwen', loaded: false, sizeGb: 1.2 },
    ]
    const mockSouls = [
      { name: 'Default', description: 'Default personality', traits: [] },
      { name: 'Creative', description: 'Creative mode', traits: ['creative'] },
    ]
    const mockHealth = {
      status: 'healthy',
      model_loaded: true,
      model_type: 'GPT-2',
      is_inferencing: false,
      inference_count: 100,
    }

    vi.mocked(apiGet).mockImplementation((path: string) => {
      if (path === '/models') return Promise.resolve(mockModels)
      if (path === '/souls') return Promise.resolve(mockSouls)
      if (path === '/souls/current') return Promise.resolve({ name: 'Default' })
      if (path === '/auto-train/checkpoints') return Promise.resolve([])
      if (path === '/health') return Promise.resolve(mockHealth)
      return Promise.resolve(null)
    })

    await useModelStore.getState().refresh()

    const state = useModelStore.getState()
    expect(state.models).toEqual(mockModels)
    expect(state.souls).toEqual(mockSouls)
    expect(state.currentSoul).toBe('Default')
    expect(state.health).toEqual(mockHealth)
    expect(state.currentModel).toBe('model-1')
    expect(state.loading).toBe(false)
  })

  it('should load model successfully', async () => {
    vi.mocked(apiPost).mockResolvedValue({ status: 'loaded' })
    vi.mocked(apiGet).mockResolvedValue([])

    const result = await useModelStore.getState().loadModel('model-1')

    expect(result).toBe(true)
    expect(apiPost).toHaveBeenCalledWith('/models/load', { model_id: 'model-1' })
    expect(useModelStore.getState().loadingModelId).toBeNull()
  })

  it('should handle load model error', async () => {
    vi.mocked(apiPost).mockRejectedValue(new Error('Load failed'))

    const result = await useModelStore.getState().loadModel('model-1')

    expect(result).toBe(false)
    expect(useModelStore.getState().error).toBe('Load failed')
    expect(useModelStore.getState().loadingModelId).toBeNull()
  })

  it('should switch soul successfully', async () => {
    vi.mocked(apiPost).mockResolvedValue({ status: 'ok' })
    vi.mocked(apiGet).mockResolvedValue([])

    const result = await useModelStore.getState().switchSoul('Creative')

    expect(result).toBe(true)
    expect(apiPost).toHaveBeenCalledWith('/souls/switch', { soul: 'Creative' })
  })

  it('should switch soul with checkpoint', async () => {
    vi.mocked(apiPost).mockResolvedValue({ status: 'ok' })
    vi.mocked(apiGet).mockResolvedValue([])

    await useModelStore.getState().switchSoul('Creative', 'checkpoint-1')

    expect(apiPost).toHaveBeenCalledWith('/souls/switch', {
      soul: 'Creative',
      checkpoint_name: 'checkpoint-1',
    })
  })

  it('should clear error', () => {
    useModelStore.setState({ error: 'Test error' })
    useModelStore.getState().clearError()
    expect(useModelStore.getState().error).toBeNull()
  })
})
