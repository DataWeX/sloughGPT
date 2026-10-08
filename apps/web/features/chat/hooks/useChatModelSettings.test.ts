import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useAppStore, initStore, DEFAULT_SETTINGS } from '@/lib/store'

const mockList = vi.fn()
const mockLoad = vi.fn()
const mockUnloadModel = vi.fn()
const mockGet = vi.fn()
const mockSoulsList = vi.fn()
const mockSoulsSwitch = vi.fn()
const mockSoulsListCheckpoints = vi.fn()
const mockStartDownload = vi.fn()
const mockGetDownloadStatus = vi.fn()
const mockIsApproved = vi.fn()
const mockListFineTuned = vi.fn()
const mockLoadFineTuned = vi.fn()

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    list: (...args: any[]) => mockList(...args),
    load: (...args: any[]) => mockLoad(...args),
    unloadModel: (...args: any[]) => mockUnloadModel(...args),
  },
}))

vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: { get: (...args: any[]) => mockGet(...args) },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: (...args: any[]) => mockSoulsList(...args),
    switch: (...args: any[]) => mockSoulsSwitch(...args),
    listCheckpoints: (...args: any[]) => mockSoulsListCheckpoints(...args),
  },
}))

vi.mock('@/lib/download-controller', () => ({
  startDownload: (...args: any[]) => mockStartDownload(...args),
  getDownloadStatus: (...args: any[]) => mockGetDownloadStatus(...args),
}))

vi.mock('@/lib/session-store', () => ({
  sessionStore: { isApproved: (...args: any[]) => mockIsApproved(...args) },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingJobsController: {
    listFineTuned: (...args: any[]) => mockListFineTuned(...args),
    loadFineTuned: (...args: any[]) => mockLoadFineTuned(...args),
  },
}))

// The hook writes fetched settings back through the store, and the real store
// pushes them to the backend (updateSettings → _pushToBackend →
// settingsController.updateGeneration). Unmocked, every run of this file PATCHed
// whatever backend happened to be on localhost:8000 — and the settings KV is
// shared across test files, so it also fed the cross-file leak this card fixes.
vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    getAll: vi.fn().mockResolvedValue(null),
    getGeneration: vi.fn().mockResolvedValue(null),
    updateGeneration: vi.fn().mockResolvedValue(null),
  },
}))

import { useChatModelSettings } from './useChatModelSettings'

describe('useChatModelSettings', () => {
  const showToast = vi.fn()
  const refreshHealth = vi.fn()

  // Hermetic guard: the real store pushes settings to the backend
  // (updateSettings → _pushToBackend → settingsController.updateGeneration).
  // Any fetch reaching past the mocks below means the test is talking to — and
  // writing — whatever backend happens to be running. See card b015795b.
  const fetchSpy =
    typeof globalThis.fetch === 'function' ? vi.spyOn(globalThis, 'fetch') : null
  const fetchUrls: string[] = []
  if (fetchSpy) {
    fetchSpy.mockImplementation(((input: any, init?: any) => {
      fetchUrls.push(String(typeof input === 'string' ? input : input?.url))
      return Promise.reject(new Error('network disabled in test'))
    }) as any)
  }

  // Hard hermetic guard: no test in this file may reach the network. The store
  // pushes settings to the backend and the controllers below are the hook's only
  // sanctioned I/O, so a fetch here means a mock went missing (card b015795b).
  afterEach(() => {
    expect(
      fetchUrls,
      `useChatModelSettings test made a network call — not hermetic: ${fetchUrls.join(', ')}`,
    ).toEqual([])
  })

  // initStore() hydrates settings from the chatDB KV — which the probe showed is
  // SHARED across test files — and it is kicked off at store-module import. Left
  // unawaited it can land AFTER the reset below, re-poisoning defaultTemp with
  // another file's value (0.8), which the hook's store-sync effect then applies
  // over the value this test just fetched. Awaiting it first makes the reset
  // final. See card b015795b.
  beforeEach(async () => {
    await initStore()
    vi.clearAllMocks()
    fetchUrls.length = 0
    useAppStore.setState({ settings: { ...DEFAULT_SETTINGS } })
    mockList.mockResolvedValue([])
    mockGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 200 })
    mockSoulsList.mockResolvedValue({ souls: [], current_soul: null })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [] })
    mockListFineTuned.mockResolvedValue([])
    mockLoadFineTuned.mockResolvedValue({ status: 'loaded' })
  })

  it('returns default state', () => {
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    expect(result.current.model).toBe('')
    expect(result.current.temperature).toBe(0.7)
    expect(result.current.maxTokens).toBe(300)
    expect(result.current.loadingModel).toBeNull()
    expect(result.current.souls).toEqual([])
    expect(result.current.currentSoul).toBeNull()
    expect(result.current.availableModels).toEqual([])
    expect(result.current.modelInfoMap).toEqual({})
  })

  it('fetchInitialData populates state', async () => {
    mockList.mockResolvedValue([
      { id: 'gpt2', cached: true, size_gb: 0.5 },
      { id: 'gpt2-medium', cached: false, size_gb: 1.5 },
    ])
    mockGet.mockResolvedValue({ temperature: 0.7, max_new_tokens: 500 })
    mockSoulsList.mockResolvedValue({
      souls: [{ name: 'friendly', description: 'Nice' }],
      current_soul: 'friendly',
    })
    mockSoulsListCheckpoints.mockResolvedValue({
      checkpoints: [
        { name: 'ckpt1', loss: 0.5, traits: { warmth: 0.8 }, is_loaded: true, verdict: 'Good' },
      ],
    })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData('gpt2')
    })

    expect(result.current.availableModels).toEqual(['gpt2', 'gpt2-medium'])
    expect(result.current.modelInfoMap.gpt2).toEqual({ cached: true, size_gb: 0.5 })
    expect(result.current.temperature).toBe(0.7)
    expect(result.current.maxTokens).toBe(500)
    expect(result.current.souls).toHaveLength(1)
    expect(result.current.currentSoul?.name).toBe('friendly')
    expect(result.current.checkpoints).toHaveLength(1)
    expect(result.current.checkpoints[0].name).toBe('ckpt1')
    expect(result.current.model).toBe('gpt2')
  })

  // fetchInitialData runs five independent requests. One slow/failed endpoint
  // (e.g. the 30s model-list timeout) must not take the whole initial load down.
  it('fetchInitialData applies the endpoints that succeed when one rejects', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockGet.mockResolvedValue({ temperature: 0.5, max_new_tokens: 123 })
    mockSoulsList.mockRejectedValue(new Error('souls endpoint down'))

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData('gpt2')
    })

    expect(result.current.availableModels).toEqual(['gpt2'])
    expect(result.current.temperature).toBe(0.5)
    expect(result.current.maxTokens).toBe(123)
    expect(result.current.souls).toEqual([])
    expect(result.current.model).toBe('gpt2')
  })

  it('fetchInitialData keeps the rest of the UI working when the model list fails', async () => {
    mockList.mockRejectedValue(new Error('Request timed out after 30s'))
    mockGet.mockResolvedValue({ temperature: 0.9, max_new_tokens: 64 })
    mockSoulsList.mockResolvedValue({
      souls: [{ name: 'friendly', description: 'Nice' }],
      current_soul: 'friendly',
    })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData('health-model')
    })

    expect(result.current.availableModels).toEqual([])
    expect(result.current.modelInfoMap).toEqual({})
    expect(result.current.temperature).toBe(0.9)
    expect(result.current.maxTokens).toBe(64)
    expect(result.current.souls).toHaveLength(1)
    expect(result.current.currentSoul?.name).toBe('friendly')
    expect(result.current.model).toBe('health-model')
  })

  it('fetchInitialData does not filter models when the fine-tuned list fails', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2' }, { id: 'gpt2__dataset_1' }])
    mockListFineTuned.mockRejectedValue(new Error('training store down'))

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })

    expect(result.current.availableModels).toEqual(['gpt2', 'gpt2__dataset_1'])
    expect(result.current.fineTuned).toEqual([])
  })

  it('handleSelectModel loads cached model', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockLoad.mockResolvedValue({ device: 'cpu' })
    refreshHealth.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })
    await act(async () => {
      await result.current.handleSelectModel('gpt2')
    })

    expect(mockLoad).toHaveBeenCalledWith('gpt2')
    expect(refreshHealth).toHaveBeenCalled()
    expect(result.current.model).toBe('gpt2')
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('Model ready'), 'success')
  })

  it('handleSelectModel does nothing if already loading', async () => {
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    act(() => {
      result.current.setLoadingModel('gpt2')
    })
    await act(async () => {
      await result.current.handleSelectModel('gpt2')
    })
    expect(mockLoad).not.toHaveBeenCalled()
  })

  it('handleSelectModel surfaces backend load error instead of false success', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockLoad.mockResolvedValue({ status: 'error', error: 'No .slnc file for gpt2' })
    refreshHealth.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })
    await act(async () => {
      await result.current.handleSelectModel('gpt2')
    })

    expect(mockLoad).toHaveBeenCalledWith('gpt2')
    expect(showToast).toHaveBeenCalledWith(
      expect.stringContaining('No .slnc file for gpt2'),
      'error',
    )
    expect(showToast).not.toHaveBeenCalledWith(expect.stringContaining('Model ready'), 'success')
    expect(result.current.model).toBe('')
    expect(refreshHealth).not.toHaveBeenCalled()
    expect(result.current.loadingModel).toBeNull()
  })

  it('handleUnloadModel unloads current model', async () => {
    mockUnloadModel.mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    act(() => {
      result.current.setModel('gpt2')
    })
    await act(async () => {
      await result.current.handleUnloadModel()
    })
    expect(mockUnloadModel).toHaveBeenCalled()
    expect(refreshHealth).toHaveBeenCalled()
    expect(result.current.model).toBe('')
  })

  it('handleSelectSoul switches soul', async () => {
    mockSoulsSwitch.mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    const soul = { name: 'friendly', description: 'Nice' } as any
    await act(async () => {
      result.current.handleSelectSoul(soul)
    })
    expect(mockSoulsSwitch).toHaveBeenCalledWith('friendly')
    expect(result.current.currentSoul).toBe(soul)
  })

  it('handleSelectModel sets pendingDownload when not cached and not approved', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: false, size_gb: 0.5 }])
    mockIsApproved.mockReturnValue(false)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })
    await act(async () => {
      await result.current.handleSelectModel('gpt2')
    })
    expect(result.current.pendingDownload).toBe('gpt2')
  })

  it('download flow calls startDownload + polls', async () => {
    vi.useFakeTimers()
    mockStartDownload.mockResolvedValue(undefined)
    mockGetDownloadStatus.mockResolvedValue({ percentage: 100, status: 'complete' })
    mockLoad.mockResolvedValue({ device: 'cpu' })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    // Directly test startDownloadFlow which populates the ref and runs the download flow
    await act(async () => {
      await result.current.startDownloadFlow('gpt2', 0.5)
    })
    expect(mockStartDownload).toHaveBeenCalledWith('gpt2', expect.any(Number))
    vi.useRealTimers()
  })

  it('fetchInitialData sets model from health even when not loaded', async () => {
    mockList.mockResolvedValue([{ id: 'Qwen/Qwen2.5-0.5B-Instruct', cached: false, size_gb: 1.0 }])
    mockGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 200 })
    mockSoulsList.mockResolvedValue({ souls: [], current_soul: null })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [] })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData('Qwen/Qwen2.5-0.5B-Instruct')
    })

    expect(result.current.model).toBe('Qwen/Qwen2.5-0.5B-Instruct')
  })

  it('fetchInitialData sets model when model_type exists but not loaded', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 200 })
    mockSoulsList.mockResolvedValue({ souls: [], current_soul: null })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [] })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData('gpt2')
    })

    expect(result.current.model).toBe('gpt2')
  })

  it('fetchInitialData does not set model when healthModel is undefined', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 200 })
    mockSoulsList.mockResolvedValue({ souls: [], current_soul: null })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [] })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })

    expect(result.current.model).toBe('')
  })

  it('fetchInitialData loads fine-tuned models', async () => {
    mockListFineTuned.mockResolvedValue([{ name: 'gpt2__dataset_1', model: 'gpt2' }])
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })
    expect(mockListFineTuned).toHaveBeenCalled()
    expect(result.current.fineTuned).toEqual([{ name: 'gpt2__dataset_1', model: 'gpt2' }])
  })

  it('fetchInitialData excludes fine-tuned dir names from availableModels', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2' }, { id: 'gpt2__dataset_1' }])
    mockListFineTuned.mockResolvedValue([{ name: 'gpt2__dataset_1', model: 'gpt2' }])
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.fetchInitialData()
    })
    expect(result.current.availableModels).toEqual(['gpt2'])
    expect(result.current.fineTuned).toHaveLength(1)
  })

  it('handleLoadFineTuned loads, refreshes list, sets model from response, and toasts', async () => {
    mockListFineTuned.mockResolvedValue([{ name: 'gpt2__dataset_1', model: 'gpt2' }])
    mockLoadFineTuned.mockResolvedValue({
      status: 'loaded',
      name: 'gpt2__dataset_1',
      model_path: '/tmp/x',
      model_id: 'gpt2__dataset_1',
    })
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.handleLoadFineTuned('gpt2__dataset_1')
    })
    expect(mockLoadFineTuned).toHaveBeenCalledWith('gpt2__dataset_1')
    expect(mockListFineTuned).toHaveBeenCalled()
    expect(result.current.model).toBe('gpt2__dataset_1')
    expect(refreshHealth).toHaveBeenCalled()
    expect(showToast).toHaveBeenCalledWith(
      expect.stringContaining('Fine-tuned model loaded'),
      'success',
    )
  })

  it('handleLoadFineTuned toasts error on failure', async () => {
    mockLoadFineTuned.mockRejectedValue(new Error('load failed'))
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => {
      await result.current.handleLoadFineTuned('gpt2__dataset_1')
    })
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('load failed'), 'error')
    expect(result.current.loadingModel).toBeNull()
  })
})
