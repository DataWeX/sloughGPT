import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { loadCheckpoint: vi.fn().mockResolvedValue(undefined) },
}))

import { useChatHealthValue, useChatUIValue, useChatModelValue } from './useChatContextValue'
import { soulsController } from '@/lib/souls-controller'

describe('useChatModelValue.onLoadCheckpoint', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  const opts = () => ({
    model: { checkpoints: [], setCurrentCheckpoint: vi.fn() } as any,
    agents: {} as any,
    vision: {} as any,
    chat: {} as any,
    showToast: vi.fn(),
  })

  it('passes the row address through to the controller', async () => {
    const o = opts()
    const { result } = renderHook(() => useChatModelValue(o))
    await act(async () => {
      await result.current.onLoadCheckpoint('cp.soul', 'models/auto-training/cp.soul')
    })
    expect(soulsController.loadCheckpoint).toHaveBeenCalledWith(
      'cp.soul',
      'models/auto-training/cp.soul',
    )
    expect(o.model.setCurrentCheckpoint).toHaveBeenCalledWith('cp.soul')
  })

  it('keeps the legacy name load when no address is given', async () => {
    const o = opts()
    const { result } = renderHook(() => useChatModelValue(o))
    await act(async () => {
      await result.current.onLoadCheckpoint('cp.soul')
    })
    expect(soulsController.loadCheckpoint).toHaveBeenCalledWith('cp.soul', undefined)
  })
})

describe('useChatHealthValue', () => {
  it('returns health and refreshHealth', () => {
    const health = { model_loaded: true, model_type: 'gpt2' } as any
    const refreshHealth = vi.fn()
    const { result } = renderHook(() => useChatHealthValue({ health, refreshHealth }))
    expect(result.current.health).toBe(health)
    expect(result.current.refreshHealth).toBe(refreshHealth)
  })

  it('memoizes when deps unchanged', () => {
    const health = {} as any
    const refreshHealth = vi.fn()
    const { result, rerender } = renderHook(
      ({ h, r }) => useChatHealthValue({ health: h, refreshHealth: r }),
      { initialProps: { h: health, r: refreshHealth } },
    )
    const first = result.current
    rerender({ h: health, r: refreshHealth })
    expect(result.current).toBe(first)
  })

  it('returns new value when health changes', () => {
    const refreshHealth = vi.fn()
    const { result, rerender } = renderHook(
      ({ h }) => useChatHealthValue({ health: h, refreshHealth }),
      { initialProps: { h: { model_loaded: true } as any } },
    )
    const first = result.current
    rerender({ h: { model_loaded: false } as any })
    expect(result.current).not.toBe(first)
  })
})

describe('useChatUIValue', () => {
  const toggleSettings = vi.fn()
  const setShowConversationViewer = vi.fn()
  const showToast = vi.fn()
  const ui = {
    toggleSettings,
    setShowConversationViewer,
    sidebarOpen: false,
    setSidebarOpen: vi.fn(),
    toolPanelOpen: false,
    setToolPanelOpen: vi.fn(),
    voiceMode: false,
    setVoiceMode: vi.fn(),
    searchQuery: '',
    setSearchQuery: vi.fn(),
    handleSearchChange: vi.fn(),
    handleSearchClear: vi.fn(),
    matchIndex: 0,
    setMatchIndex: vi.fn(),
    showMobileSearch: false,
    setShowMobileSearch: vi.fn(),
  } as any

  it('returns onOpenSettings that calls toggleSettings', () => {
    const { result } = renderHook(() => useChatUIValue({ ui, showToast }))
    expect(result.current.onOpenSettings).toBe(toggleSettings)
  })

  it('returns onOpenConversationViewer that sets showConversationViewer', () => {
    const { result } = renderHook(() => useChatUIValue({ ui, showToast }))
    result.current.onOpenConversationViewer()
    expect(setShowConversationViewer).toHaveBeenCalledWith(true)
  })

  it('returns showToast', () => {
    const { result } = renderHook(() => useChatUIValue({ ui, showToast }))
    expect(result.current.showToast).toBe(showToast)
  })

  it('dispatches toggle-shortcuts event for onOpenShortcuts', () => {
    const dispatchSpy = vi.spyOn(window, 'dispatchEvent')
    const { result } = renderHook(() => useChatUIValue({ ui, showToast }))
    result.current.onOpenShortcuts()
    expect(dispatchSpy).toHaveBeenCalled()
    dispatchSpy.mockRestore()
  })
})
