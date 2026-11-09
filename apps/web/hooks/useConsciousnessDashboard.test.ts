/**
 * Tests for useConsciousnessDashboard — the stateful core of the dashboard.
 *
 * The page suite covers wiring end-to-end but mocks `useConsciousnessLive`
 * with `latestEvent: null`, so the live-stream fold (and the beliefs-history
 * bookkeeping strip) never runs there. Those live here.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'

const { addToast } = vi.hoisted(() => ({ addToast: vi.fn() }))

vi.mock('@/lib/notify', () => ({
  // The hook writes through `notify.toast`; bind it to the spy the
  // assertions below check (it previously read useToastStore.addToast).
  notify: { toast: addToast, banner: vi.fn() },

  useToastStore: (selector: (s: { addToast: typeof addToast }) => unknown) =>
    selector({ addToast }),
}))

const liveState = {
  isLive: false,
  lastUpdate: null as number | null,
  toggleLive: vi.fn(),
  latestEvent: null as unknown,
}

vi.mock('@/hooks/useConsciousnessLive', () => ({
  useConsciousnessLive: () => liveState,
}))

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getEpisodeHistory: vi.fn(),
    getQualiaHistory: vi.fn(),
    getBeliefsHistory: vi.fn(),
    evaluate: vi.fn(),
    getSelfModel: vi.fn(),
    getQualia: vi.fn(),
    getStatus: vi.fn(),
    seedData: vi.fn(),
    reflect: vi.fn(),
    submitFeedback: vi.fn(),
    updateConfig: vi.fn(),
  },
}))

import { consciousnessController } from '@/lib/consciousness-controller'
import { useConsciousnessDashboard } from './useConsciousnessDashboard'

const mockCtrl = vi.mocked(consciousnessController)

const episodes = [
  {
    timestamp: 1_700_000_000,
    input_text: 'input',
    self_insight: 'insight',
    growth_delta: 0.05,
    qualia: { valence: 0.5 },
  },
]

const beliefPoint = { timestamp: 1_700_000_010, step: 7, competence: 0.4, helpfulness: 0.6 }

const wireQualia = {
  current: { valence: 0.1, arousal: -0.2, dominance: 0, novelty: 0.4, coherence: 0.6, beauty: 0.3 },
  narrative: 'calm',
  history_count: 0,
}

const wireSelfModel = {
  identity: { name: 'SloughGPT', capabilities: [], limitations: [], values: [] },
  beliefs: { competence: 0.7, helpfulness: 0.8 },
  doubts: [],
  episode_count: 1,
}

describe('useConsciousnessDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    liveState.isLive = false
    liveState.lastUpdate = null
    liveState.latestEvent = null

    mockCtrl.getEpisodeHistory.mockResolvedValue({ episodes: episodes as never, total: 1 })
    mockCtrl.getQualiaHistory.mockResolvedValue({ history: [] })
    mockCtrl.getBeliefsHistory.mockResolvedValue({ beliefs: [] })
    mockCtrl.evaluate.mockResolvedValue({ overall_score: 75, diagnostics: [] } as never)
    mockCtrl.getSelfModel.mockResolvedValue(wireSelfModel as never)
    mockCtrl.getQualia.mockResolvedValue(wireQualia as never)
    mockCtrl.getStatus.mockResolvedValue({ level: 2, enabled: true } as never)
  })

  it('folds all seven endpoints into state and clears loading', async () => {
    const { result } = renderHook(() => useConsciousnessDashboard())

    expect(result.current.loading).toBe(true)

    await waitFor(() => expect(result.current.loading).toBe(false))

    expect(result.current.episodes).toEqual(episodes)
    expect(result.current.totalEpisodes).toBe(1)
    expect(result.current.evalReport?.overall_score).toBe(75)
    // Both consumers of the corrected contracts: `.current` from the qualia
    // envelope, `.beliefs` from the self model.
    expect(result.current.currentQualia.valence).toBe(0.1)
    expect(result.current.currentQualia.beauty).toBe(0.3)
    expect(result.current.currentBeliefs).toEqual({ competence: 0.7, helpfulness: 0.8 })
    expect(result.current.consciousnessLevel).toBe(2)
  })

  it('takes beliefs from history when the self model has none, minus bookkeeping columns', async () => {
    mockCtrl.getSelfModel.mockResolvedValue({} as never)
    mockCtrl.getBeliefsHistory.mockResolvedValue({ beliefs: [beliefPoint] })

    const { result } = renderHook(() => useConsciousnessDashboard())
    await waitFor(() => expect(result.current.loading).toBe(false))

    // timestamp/step are chart bookkeeping, not beliefs — they must not land
    // in the Current State bars, or the panel would render "timestamp: …%".
    expect(result.current.currentBeliefs).toEqual({ competence: 0.4, helpfulness: 0.6 })
  })

  it('folds a live event into current state and appends to both histories', async () => {
    liveState.isLive = true

    const { result, rerender } = renderHook(() => useConsciousnessDashboard())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.qualiaHistory).toHaveLength(0)
    expect(result.current.beliefsHistory).toHaveLength(0)

    act(() => {
      liveState.latestEvent = {
        level: 3,
        qualia: { valence: 0.9, arousal: 0.2, novelty: 0.1, coherence: 0.5 },
        beliefs: { competence: 0.55, helpfulness: 0.65 },
        growth_delta: 0.01,
        self_insight: 'noted',
      }
      rerender()
    })

    await waitFor(() => expect(result.current.qualiaHistory).toHaveLength(1))
    expect(result.current.beliefsHistory).toHaveLength(1)
    expect(result.current.beliefsHistory[0].step).toBe(1)
    expect(result.current.currentQualia.valence).toBe(0.9)
    expect(result.current.currentBeliefs.competence).toBe(0.55)
    expect(result.current.consciousnessLevel).toBe(3)
    expect(addToast).toHaveBeenCalledWith('Consciousness state updated', { type: 'success' })
  })

  it('ignores live events while the stream is off', async () => {
    liveState.isLive = false

    const { result, rerender } = renderHook(() => useConsciousnessDashboard())
    await waitFor(() => expect(result.current.loading).toBe(false))

    act(() => {
      liveState.latestEvent = {
        level: 3,
        qualia: { valence: 0.9 },
        beliefs: { competence: 0.55 },
        growth_delta: 0.01,
        self_insight: 'noted',
      }
      rerender()
    })

    expect(result.current.qualiaHistory).toHaveLength(0)
    expect(result.current.consciousnessLevel).toBe(2)
    expect(addToast).not.toHaveBeenCalledWith('Consciousness state updated', { type: 'success' })
  })

  it('surfaces a seed failure as an error toast and resets the busy flag', async () => {
    mockCtrl.seedData.mockRejectedValue(new Error('backend down'))

    const { result } = renderHook(() => useConsciousnessDashboard())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.handleSeed()
    })

    expect(addToast).toHaveBeenCalledWith('backend down', { type: 'error' })
    expect(result.current.seeding).toBe(false)
  })

  it('reports the consciousness level change and rolls back nothing on success', async () => {
    mockCtrl.updateConfig.mockResolvedValue({ updated: true } as never)

    const { result } = renderHook(() => useConsciousnessDashboard())
    await waitFor(() => expect(result.current.loading).toBe(false))

    await act(async () => {
      await result.current.handleLevelChange(0)
    })

    expect(mockCtrl.updateConfig).toHaveBeenCalledWith({ level: 0 })
    expect(result.current.consciousnessLevel).toBe(0)
    expect(addToast).toHaveBeenCalledWith('Level set to 0', { type: 'success' })
  })
})
