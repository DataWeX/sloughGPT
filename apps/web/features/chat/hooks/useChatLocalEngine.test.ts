import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockInferArch = vi.fn()
const mockSoulNetInit = vi.fn()
const mockSoulNetLoad = vi.fn()
const mockTransformerInit = vi.fn()
const mockTransformerLoad = vi.fn()

vi.mock('@/lib/soulnet-webgpu', () => ({
  inferArch: (...args: any[]) => mockInferArch(...args),
  SoulNetWebGPU: vi.fn().mockImplementation(() => ({
    init: mockSoulNetInit,
    load: mockSoulNetLoad,
    generate: vi.fn(),
    destroy: vi.fn(),
  })),
  SoulTransformerWebGPU: vi.fn().mockImplementation(() => ({
    init: mockTransformerInit,
    load: mockTransformerLoad,
    generate: vi.fn(),
    destroy: vi.fn(),
  })),
}))

import { useChatLocalEngine } from './useChatLocalEngine'

describe('useChatLocalEngine', () => {
  const showToast = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    delete (navigator as any).gpu
  })

  it('returns default state', () => {
    const { result } = renderHook(() => useChatLocalEngine(showToast))
    expect(result.current.useLocalEngine).toBe(false)
    expect(result.current.localEngineLoading).toBe(false)
    expect(result.current.localArchInfo).toBeNull()
    expect(result.current.localModelUrl).toBe('')
  })

  it('initLocalEngine returns false when no model URL', async () => {
    (navigator as any).gpu = {}
    const { result } = renderHook(() => useChatLocalEngine(showToast))
    const ok = await act(async () => result.current.initLocalEngine())
    expect(ok).toBe(false)
    expect(showToast).toHaveBeenCalledWith('Could not load local AI: No .soul file URL configured', 'error')
  })

  it('initLocalEngine loads LSTM model', async () => {
    (navigator as any).gpu = {}
    const buf = new ArrayBuffer(100)
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => buf })
    mockInferArch.mockReturnValue({ archType: 'lstm', embedDim: 128, hiddenDim: 256 })
    mockSoulNetInit.mockResolvedValue(undefined)
    mockSoulNetLoad.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatLocalEngine(showToast))
    act(() => { result.current.setLocalModelUrl('/sou/my-model.soul') })
    const ok = await act(async () => result.current.initLocalEngine())
    expect(ok).toBe(true)
    expect(mockInferArch).toHaveBeenCalledWith(buf)
    expect(result.current.localArchInfo).toContain('LSTM')
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('Local AI ready'))
  })

  it('initLocalEngine loads Transformer model', async () => {
    (navigator as any).gpu = {}
    const buf = new ArrayBuffer(100)
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => buf })
    mockInferArch.mockReturnValue({
      archType: 'transformer', embedDim: 256, numLayers: 6, vocabSize: 32000,
      hiddenDim: 256, numHeads: 8, numKVHeads: 8, dimFF: 1024, maxSeqLen: 2048, eps: 1e-5,
    })
    mockTransformerInit.mockResolvedValue(undefined)
    mockTransformerLoad.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatLocalEngine(showToast))
    act(() => { result.current.setLocalModelUrl('/auto-train/ckpt.soul') })
    const ok = await act(async () => result.current.initLocalEngine())
    expect(ok).toBe(true)
    expect(result.current.localArchInfo).toContain('Transformer')
    expect(showToast).toHaveBeenCalledWith(expect.stringContaining('On-device AI ready'))
  })

  it('handleToggleLocalEngine toggles to server mode', async () => {
    const { result } = renderHook(() => useChatLocalEngine(showToast))
    act(() => { result.current.setUseLocalEngine(true) })
    await act(async () => { result.current.handleToggleLocalEngine() })
    expect(result.current.useLocalEngine).toBe(false)
    expect(result.current.localArchInfo).toBeNull()
    expect(showToast).toHaveBeenCalledWith('Switched to server mode', 'info')
  })

  it('handleToggleLocalEngine toggles to local mode from existing engine', async () => {
    (navigator as any).gpu = {}
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => new ArrayBuffer(100) })
    mockInferArch.mockReturnValue({ archType: 'lstm', embedDim: 64, hiddenDim: 128 })
    mockSoulNetInit.mockResolvedValue(undefined)
    mockSoulNetLoad.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatLocalEngine(showToast))
    act(() => { result.current.setLocalModelUrl('/sou/m.soul') })
    await act(async () => { await result.current.initLocalEngine() })
    await act(async () => { result.current.handleToggleLocalEngine() })
    expect(result.current.useLocalEngine).toBe(true)
  })

  it('destroyEngine cleans up engine and resets state', async () => {
    (navigator as any).gpu = {}
    const buf = new ArrayBuffer(100)
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => buf })
    mockInferArch.mockReturnValue({ archType: 'lstm', embedDim: 64, hiddenDim: 128 })
    mockSoulNetInit.mockResolvedValue(undefined)
    mockSoulNetLoad.mockResolvedValue(undefined)

    const { result } = renderHook(() => useChatLocalEngine(showToast))
    act(() => { result.current.setLocalModelUrl('/sou/m.soul') })
    await act(async () => { await result.current.initLocalEngine() })
    expect(result.current.localArchInfo).toContain('LSTM')
    act(() => { result.current.destroyEngine() })
    expect(result.current.useLocalEngine).toBe(false)
    expect(result.current.localArchInfo).toBeNull()
    expect(result.current.engineRef.current).toBeNull()
  })
})
