/**
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, act, cleanup } from '@testing-library/react'
import React from 'react'
import { useHomePageData } from './useHomePageData'

const mockModelStatus = vi.fn().mockResolvedValue({ loaded: false, model_type: null })
const mockModelList = vi.fn().mockResolvedValue([])
const mockSoulsList = vi.fn().mockResolvedValue({ souls: [], current_soul: '' })
const mockSessionList = vi.fn().mockResolvedValue([])
const mockTrainingList = vi.fn().mockResolvedValue([])
const mockKnowledgeStats = vi.fn().mockResolvedValue({ total_items: 0 })

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    status: (...args: unknown[]) => mockModelStatus(...args),
    list: (...args: unknown[]) => mockModelList(...args),
  },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: {
    list: (...args: unknown[]) => mockSoulsList(...args),
  },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: {
    list: (...args: unknown[]) => mockSessionList(...args),
  },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: {
    list: (...args: unknown[]) => mockTrainingList(...args),
  },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: {
    stats: (...args: unknown[]) => mockKnowledgeStats(...args),
  },
}))

const mockGetFeedbackStats = vi.fn().mockResolvedValue({ total: 0, positive: 0, negative: 0 })
vi.mock('@/lib/feedback-controller', () => ({
  feedbackController: {
    getFeedbackStats: (...args: unknown[]) => mockGetFeedbackStats(...args),
  },
}))

const mockDatasetList = vi.fn().mockResolvedValue([])
vi.mock('@/lib/dataset-controller', () => ({
  datasetController: {
    list: (...args: unknown[]) => mockDatasetList(...args),
  },
}))

vi.mock('@/lib/query', () => ({
  useQuery: (_key: unknown, fetcher: () => Promise<unknown>) => {
    const [data, setData] = React.useState<unknown>(undefined)
    React.useEffect(() => { fetcher().then(setData).catch(() => {}) }, [fetcher])
    return { data }
  },
  useMutation: (mutate: (...a: unknown[]) => Promise<unknown>) => ({
    mutate: (...a: unknown[]) => mutate(...a),
    isLoading: false,
  }),
  useInvalidate: () => vi.fn(),
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  liveStatusStore: { setState: vi.fn(), getState: vi.fn(() => ({ ready: true })) },
  useApiReady: () => true,
  useLiveStatus: () => ({ health: null }),
}))

beforeEach(() => {
  // useLiveStatus is mocked — no store setup needed
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const ONLINE_HEALTH = { status: 'ok', model_loaded: true, model_type: 'gpt2', summary: 'ready', inference_count: 42 }

describe('useHomePageData', () => {
  it('returns default state with null health', () => {
    const { result } = renderHook(() => useHomePageData(null))
    expect(result.current.modelCount).toBeNull()
    expect(result.current.currentSoul).toBeNull()
    expect(result.current.recentSessions).toEqual([])
    expect(result.current.knowledgeCount).toBe(0)
    expect(result.current.inferenceCount).toBeNull()
  })

  it('returns default state with offline health', () => {
    const { result } = renderHook(() => useHomePageData('offline'))
    expect(result.current.modelCount).toBeNull()
    expect(result.current.modelStatus).toEqual({ loaded: false, model: null })
  })

  it('fetches model status and soul on online health', async () => {
    mockModelStatus.mockResolvedValue({ loaded: true, model_type: 'gpt2' })
    mockSoulsList.mockResolvedValue({ souls: [{ name: 'friendly', description: 'A friendly soul', traits: ['warm'] }], current_soul: 'friendly' })
    renderHook(() => useHomePageData(ONLINE_HEALTH))
    await vi.waitFor(() => {
      expect(mockModelStatus).toHaveBeenCalled()
      expect(mockSoulsList).toHaveBeenCalled()
    })
  })

  it('fetches model list, sessions, training jobs, knowledge stats on health', async () => {
    mockModelList.mockResolvedValue([{ id: 'gpt2' }, { id: 'tinyllama' }])
    mockSessionList.mockResolvedValue([
      { id: 's1', name: 'Chat A', updated_at: '2024-06-01T00:00:00Z' },
    ])
    mockTrainingList.mockResolvedValue([{ id: 'j1', name: 'Job 1', status: 'completed', created_at: '2024-06-01T00:00:00Z' }])
    mockKnowledgeStats.mockResolvedValue({ total_items: 5 })
    renderHook(() => useHomePageData(ONLINE_HEALTH))
    await vi.waitFor(() => {
      expect(mockModelList).toHaveBeenCalled()
      expect(mockSessionList).toHaveBeenCalled()
      expect(mockTrainingList).toHaveBeenCalled()
      expect(mockKnowledgeStats).toHaveBeenCalled()
    })
  })

  it('extracts inferenceCount from health', () => {
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    expect(result.current.inferenceCount).toBe(42)
  })

  it('handles null inferenceCount', () => {
    const { result } = renderHook(() => useHomePageData({ ...ONLINE_HEALTH, inference_count: undefined } as any))
    expect(result.current.inferenceCount).toBe(0)
  })
})
