/**
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import { useChatVision } from './useChatVision'

const mockGetCapabilities = vi.fn()
const mockGetTrainingReport = vi.fn()
vi.mock('@/lib/multimodal-controller', () => ({
  multimodalController: {
    getCapabilities: (...args: unknown[]) => mockGetCapabilities(...args),
    getTrainingReport: (...args: unknown[]) => mockGetTrainingReport(...args),
  },
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('useChatVision', () => {
  it('returns default state', () => {
    mockGetCapabilities.mockResolvedValue({ model_loaded: false, learning: false, trained_steps: 0 })
    mockGetTrainingReport.mockResolvedValue({ caption_history: [], vocab_size: undefined })
    const { result } = renderHook(() => useChatVision())
    expect(result.current.visionCaps).toBeNull()
    expect(result.current.visionCaptionHistory).toEqual([])
    expect(result.current.visionVocabSize).toBeUndefined()
  })

  it('fetches capabilities and report on mount', async () => {
    mockGetCapabilities.mockResolvedValue({ model_loaded: true, learning: false, trained_steps: 5 })
    mockGetTrainingReport.mockResolvedValue({ caption_history: ['cap1', 'cap2'], vocab_size: 100 })
    const { result } = renderHook(() => useChatVision())
    await vi.waitFor(() => {
      expect(result.current.visionCaps).toEqual({ model_loaded: true, learning: false, trained_steps: 5 })
    })
    await vi.waitFor(() => {
      expect(result.current.visionCaptionHistory).toEqual(['cap1', 'cap2'])
      expect(result.current.visionVocabSize).toBe(100)
    })
  })

  it('refreshVision re-fetches capabilities and report', async () => {
    mockGetCapabilities.mockResolvedValue({ model_loaded: false, learning: false, trained_steps: 0 })
    mockGetTrainingReport.mockResolvedValue({ caption_history: [], vocab_size: undefined })
    const { result } = renderHook(() => useChatVision())
    mockGetCapabilities.mockResolvedValue({ model_loaded: true, learning: true, trained_steps: 10 })
    mockGetTrainingReport.mockResolvedValue({ caption_history: ['new_cap'], vocab_size: 50 })
    await act(async () => { await result.current.refreshVision() })
    expect(result.current.visionCaps).toEqual({ model_loaded: true, learning: true, trained_steps: 10 })
    expect(result.current.visionCaptionHistory).toEqual(['new_cap'])
    expect(result.current.visionVocabSize).toBe(50)
  })

  it('silently handles API errors on mount', () => {
    mockGetCapabilities.mockRejectedValue(new Error('fail'))
    mockGetTrainingReport.mockRejectedValue(new Error('fail'))
    expect(() => renderHook(() => useChatVision())).not.toThrow()
  })
})
