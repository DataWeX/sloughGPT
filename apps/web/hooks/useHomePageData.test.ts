import { describe, expect, it, vi, afterEach } from 'vitest'
import { renderHook, cleanup } from '@testing-library/react'
import { useHomePageData } from './useHomePageData'

const mockSessionList = vi.fn().mockResolvedValue([])
const mockTrainingList = vi.fn().mockResolvedValue([])
const mockKnowledgeStats = vi.fn().mockResolvedValue({ total_items: 0 })

vi.mock('@/lib/dev-log', () => ({
  logger: { child: () => ({ warning: vi.fn(), error: vi.fn(), info: vi.fn(), debug: vi.fn() }) },
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: { status: vi.fn().mockResolvedValue({ loaded: false, model_type: null }), list: vi.fn().mockResolvedValue([]) },
}))

vi.mock('@/lib/souls-controller', () => ({
  soulsController: { list: vi.fn().mockResolvedValue({ souls: [], current_soul: '' }) },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: { list: (...args: unknown[]) => mockSessionList(...args) },
}))

vi.mock('@/lib/training-controller', () => ({
  trainingController: { list: (...args: unknown[]) => mockTrainingList(...args) },
}))

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: { stats: (...args: unknown[]) => mockKnowledgeStats(...args) },
}))

vi.mock('@/lib/feedback-controller', () => ({
  feedbackController: { getFeedbackStats: vi.fn().mockResolvedValue({ total: 0 }) },
}))

vi.mock('@/lib/dataset-controller', () => ({
  datasetController: { list: vi.fn().mockResolvedValue([]) },
}))

vi.mock('@/lib/query', () => ({
  useQuery: () => ({ data: undefined }),
  useMutation: () => ({ mutate: vi.fn(), isLoading: false }),
  useInvalidate: () => vi.fn(),
  invalidateQuery: vi.fn(),
  fetchQuery: vi.fn(),
  getQueryState: vi.fn(),
}))

vi.mock('@/lib/query/hooks', () => ({
  useQuery: () => ({ data: undefined }),
  useMutation: () => ({ mutate: vi.fn(), isLoading: false }),
  useInvalidate: () => vi.fn(),
  useIsFetching: () => 0,
}))

vi.mock('@/lib/query/api-hooks', () => ({
  useModels: () => ({ data: undefined }),
  useSouls: () => ({ data: undefined }),
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  liveStatusStore: { setState: vi.fn(), getState: () => ({ ready: true }) },
  useApiReady: () => true,
  useLiveStatus: () => ({ health: null }),
}))

afterEach(() => { cleanup(); vi.clearAllMocks() })

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

  it('sets model status from health parameter', () => {
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    expect(result.current.modelStatus).toEqual({ loaded: true, model: 'gpt2' })
  })

  it('extracts inferenceCount from health', () => {
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    expect(result.current.inferenceCount).toBe(42)
  })

  it('handles null inferenceCount', () => {
    const { result } = renderHook(() => useHomePageData({ ...ONLINE_HEALTH, inference_count: undefined } as any))
    expect(result.current.inferenceCount).toBe(0)
  })

  it('healthSummary returns model_type from health', () => {
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    expect(result.current.healthSummary).toBe('gpt2')
  })

  it('healthSummary is null when offline', () => {
    const { result } = renderHook(() => useHomePageData('offline'))
    expect(result.current.healthSummary).toBeNull()
  })
})
