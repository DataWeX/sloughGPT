import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, cleanup, act } from '@testing-library/react'

const bootstrapState = vi.hoisted(() => ({
  payload: null as any,
  error: null as string | null,
}))

vi.mock('@/contexts/BootstrapContext', () => ({
  useBootstrap: () => ({
    data: bootstrapState.payload,
    loading: bootstrapState.payload === null,
    error: bootstrapState.error,
    refresh: vi.fn(),
  }),
}))

import { useHomePageData } from './useHomePageData'

afterEach(() => cleanup())

beforeEach(() => {
  bootstrapState.payload = null
  bootstrapState.error = null
})

const ONLINE_HEALTH = {
  status: 'ok',
  model_loaded: true,
  model_type: 'gpt2',
  summary: 'ready',
  inference_count: 42,
}
const NO_INFERENCE_HEALTH = {
  status: 'ok',
  model_loaded: true,
  model_type: 'gpt2',
  summary: 'ready',
} as const

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
    const { result } = renderHook(() => useHomePageData(NO_INFERENCE_HEALTH as any))
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

  // ── Bootstrap-slice derivations (the /bootstrap landing swap) ──────────

  /** All-resolved members baseline; override per test. */
  function bootstrapPayload(
    members: Record<string, unknown> = {},
    errors: Record<string, string> = {},
  ) {
    return {
      members: {
        health: null,
        startup_progress: { phase: 'ready', step: 9, total: 9, message: 'Ready' },
        models: [],
        consciousness_status: null,
        souls: null,
        sessions: [],
        training_jobs: [],
        turbo_status: null,
        knowledge_stats: { total_items: 0 },
        feedback_stats: null,
        meta_weights_stats: null,
        datasets: { datasets: [], count: 0 },
        ...members,
      },
      errors,
    }
  }

  it('derives every landing section from the bootstrap payload', async () => {
    bootstrapState.payload = bootstrapPayload({
      models: [{ id: 'm1' }, { id: 'm2' }, { id: 'm3' }],
      souls: {
        souls: [{ name: 'Ada', description: 'curious', traits: ['warm'] }],
        current_soul: 'Ada',
      },
      sessions: [
        {
          id: 's1',
          name: 'Newest',
          updated_at: '2026-06-03T00:00:00Z',
          messages: [
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'yo' },
          ],
          pinned: true,
        },
        { id: 's2', name: 'Older', updated_at: '2026-06-01T00:00:00Z' },
      ],
      training_jobs: [
        {
          id: 'j1',
          name: 'run-a',
          status: 'completed',
          created_at: '2026-06-02T00:00:00Z',
          status_message: 'done',
        },
        {
          id: 'j2',
          name: 'run-b',
          status: 'running',
          created_at: '2026-06-03T00:00:00Z',
          status_message: 'Training...',
        },
      ],
      knowledge_stats: { total_items: 7 },
      feedback_stats: { thumbs_up: 10, thumbs_down: 2, total: 12, up_ratio: 0.8333 },
      meta_weights_stats: {
        current_weights: { temperature: 0.5, repetition_penalty: 1.2 },
        history_length: 4,
      },
      datasets: {
        datasets: [
          { id: 'd1', name: 'set', created_at: '2026-06-01T00:00:00Z', size: 100, samples: 10 },
        ],
        count: 1,
      },
    })
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    await act(async () => {})

    expect(result.current.modelCount).toBe(3)
    expect(result.current.currentSoul).toEqual({
      name: 'Ada',
      description: 'curious',
      traits: ['warm'],
    })
    expect(result.current.recentSessions).toEqual([
      {
        id: 's1',
        name: 'Newest',
        updated_at: '2026-06-03T00:00:00Z',
        message_count: 2,
        pinned: true,
        starred: undefined,
      },
      {
        id: 's2',
        name: 'Older',
        updated_at: '2026-06-01T00:00:00Z',
        message_count: undefined,
        pinned: undefined,
        starred: undefined,
      },
    ])
    expect(result.current.runningTraining).toEqual({ name: 'run-b', status_message: 'Training...' })
    expect(result.current.recentJobs.map((j) => j.id)).toEqual(['j2', 'j1'])
    expect(result.current.knowledgeCount).toBe(7)
    expect(result.current.recentDatasets).toEqual([
      { id: 'd1', name: 'set', updated_at: '2026-06-01T00:00:00Z', size: 100, samples: 10 },
    ])

    // FeedbackStats is merged from the feedback_stats + meta_weights_stats
    // members — the same buildFeedbackStats the controller path uses.
    const fb = result.current.feedbackStats as any
    expect(fb.db_stats.feedback_total).toBe(12)
    expect(fb.db_stats.thumbs_up).toBe(10)
    expect(fb.current_weights).toEqual({ temperature: 0.5, repetition_penalty: 1.2 })
    expect(fb.history_length).toBe(4)

    expect(result.current.errors).toEqual({
      models: false,
      soul: false,
      sessions: false,
      training: false,
      knowledge: false,
      feedback: false,
      datasets: false,
    })
  })

  it('treats not_ready members as pending, never as errors', async () => {
    bootstrapState.payload = bootstrapPayload(
      {
        models: null,
        souls: null,
        sessions: null,
        training_jobs: null,
        knowledge_stats: null,
        feedback_stats: null,
        datasets: null,
      },
      {
        models: 'not_ready',
        souls: 'not_ready',
        sessions: 'not_ready',
        training_jobs: 'not_ready',
        knowledge_stats: 'not_ready',
        feedback_stats: 'not_ready',
        datasets: 'not_ready',
      },
    )
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    await act(async () => {})

    expect(result.current.errors).toEqual({
      models: false,
      soul: false,
      sessions: false,
      training: false,
      knowledge: false,
      feedback: false,
      datasets: false,
    })
    // Pending sections stay empty — consumers render skeletons, not errors.
    expect(result.current.recentSessions).toEqual([])
    expect(result.current.knowledgeCount).toBe(0)
    expect(result.current.feedbackStats).toBeNull()
  })

  it('raises only the failing section for a real member error', async () => {
    bootstrapState.payload = bootstrapPayload(
      { sessions: null },
      { sessions: '500 Internal Server Error' },
    )
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    await act(async () => {})

    expect(result.current.errors.sessions).toBe(true)
    expect(result.current.errors.datasets).toBe(false)
    expect(result.current.errors.training).toBe(false)
    expect(result.current.errors.knowledge).toBe(false)
    expect(result.current.errors.feedback).toBe(false)
  })

  it('raises every section flag when the whole payload fails', async () => {
    bootstrapState.payload = null
    bootstrapState.error = 'offline'
    const { result } = renderHook(() => useHomePageData(ONLINE_HEALTH))
    await act(async () => {})

    expect(result.current.errors).toEqual({
      models: true,
      soul: true,
      sessions: true,
      training: true,
      knowledge: true,
      feedback: true,
      datasets: true,
    })
    expect(result.current.loading).toBe(true)
  })
})
