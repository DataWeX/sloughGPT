// @vitest-environment jsdom
import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

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

vi.mock('@/lib/model-controller', () => ({
  modelController: { list: (...args: any[]) => mockList(...args), load: (...args: any[]) => mockLoad(...args), unloadModel: (...args: any[]) => mockUnloadModel(...args) },
}))

vi.mock('@/lib/generation-config-controller', () => ({
  generationConfigController: { get: (...args: any[]) => mockGet(...args) },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { list: (...args: any[]) => mockSoulsList(...args), switch: (...args: any[]) => mockSoulsSwitch(...args), listCheckpoints: (...args: any[]) => mockSoulsListCheckpoints(...args) },
}))

vi.mock('@/lib/download-controller', () => ({
  startDownload: (...args: any[]) => mockStartDownload(...args),
  getDownloadStatus: (...args: any[]) => mockGetDownloadStatus(...args),
}))

vi.mock('@/lib/session-store', () => ({
  sessionStore: { isApproved: (...args: any[]) => mockIsApproved(...args) },
}))

import { useChatModelSettings } from './useChatModelSettings'

describe('useChatModelSettings', () => {
  const showToast = vi.fn()
  const refreshHealth = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    mockList.mockResolvedValue([])
    mockGet.mockResolvedValue({ temperature: 0.8, max_new_tokens: 200 })
    mockSoulsList.mockResolvedValue({ souls: [], current_soul: null })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [] })
  })

  it('returns default state', () => {
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    expect(result.current.model).toBe('')
    expect(result.current.temperature).toBe(0.8)
    expect(result.current.maxTokens).toBe(200)
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
    mockSoulsList.mockResolvedValue({ souls: [{ name: 'friendly', description: 'Nice' }], current_soul: 'friendly' })
    mockSoulsListCheckpoints.mockResolvedValue({ checkpoints: [{ name: 'ckpt1', loss: 0.5, traits: { warmth: 0.8 }, is_loaded: true, verdict: 'Good' }] })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => { await result.current.fetchInitialData('gpt2') })

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

  it('handleSelectModel loads cached model', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: true, size_gb: 0.5 }])
    mockLoad.mockResolvedValue({ device: 'cpu' })
    refreshHealth.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => { await result.current.fetchInitialData() })
    await act(async () => { await result.current.handleSelectModel('gpt2') })

    expect(mockLoad).toHaveBeenCalledWith('gpt2')
    expect(refreshHealth).toHaveBeenCalled()
    expect(result.current.model).toBe('gpt2')
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('Model ready'), 'success')
  })

  it('handleSelectModel does nothing if already loading', async () => {
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    act(() => { result.current.setLoadingModel('gpt2') })
    await act(async () => { await result.current.handleSelectModel('gpt2') })
    expect(mockLoad).not.toHaveBeenCalled()
  })

  it('handleUnloadModel unloads current model', async () => {
    mockUnloadModel.mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    act(() => { result.current.setModel('gpt2') })
    await act(async () => { await result.current.handleUnloadModel() })
    expect(mockUnloadModel).toHaveBeenCalledWith('gpt2')
    expect(refreshHealth).toHaveBeenCalled()
    expect(result.current.model).toBe('')
  })

  it('handleSelectSoul switches soul', async () => {
    mockSoulsSwitch.mockResolvedValue(undefined)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    const soul = { name: 'friendly', description: 'Nice' } as any
    await act(async () => { result.current.handleSelectSoul(soul) })
    expect(mockSoulsSwitch).toHaveBeenCalledWith('friendly')
    expect(result.current.currentSoul).toBe(soul)
  })

  it('handleSelectModel sets pendingDownload when not cached and not approved', async () => {
    mockList.mockResolvedValue([{ id: 'gpt2', cached: false, size_gb: 0.5 }])
    mockIsApproved.mockReturnValue(false)
    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    await act(async () => { await result.current.fetchInitialData() })
    await act(async () => { await result.current.handleSelectModel('gpt2') })
    expect(result.current.pendingDownload).toBe('gpt2')
  })

  it('download flow calls startDownload + polls', async () => {
    vi.useFakeTimers()
    mockStartDownload.mockResolvedValue(undefined)
    mockGetDownloadStatus.mockResolvedValue({ percentage: 100, status: 'complete' })
    mockLoad.mockResolvedValue({ device: 'cpu' })

    const { result } = renderHook(() => useChatModelSettings(showToast, refreshHealth))
    // Directly test startDownloadFlow which populates the ref and runs the download flow
    await act(async () => { await result.current.startDownloadFlow('gpt2', 0.5) })
    expect(mockStartDownload).toHaveBeenCalledWith('gpt2', expect.any(Number))
    vi.useRealTimers()
  })
})
