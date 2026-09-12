import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import {
  useConsciousnessStatus,
  getConsciousnessLevelLabel,
  getQualiaMood,
} from './useConsciousnessStatus'

vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: {
    getStatus: vi.fn(),
  },
}))

import { consciousnessController } from '@/lib/consciousness-controller'
const mockCtrl = vi.mocked(consciousnessController)

describe('getConsciousnessLevelLabel', () => {
  it('returns Off for level 0', () => {
    expect(getConsciousnessLevelLabel(0)).toBe('Off')
  })
  it('returns Basic for level 1', () => {
    expect(getConsciousnessLevelLabel(1)).toBe('Basic')
  })
  it('returns Full for level 2', () => {
    expect(getConsciousnessLevelLabel(2)).toBe('Full')
  })
  it('returns Deep for level 3', () => {
    expect(getConsciousnessLevelLabel(3)).toBe('Deep')
  })
  it('returns Off for unknown level', () => {
    expect(getConsciousnessLevelLabel(99)).toBe('Off')
  })
})

describe('getQualiaMood', () => {
  it('returns engaged for high valence + high arousal', () => {
    expect(getQualiaMood({ valence: 0.5, arousal: 0.7, novelty: 0.3, coherence: 0.5, salience: 0.5 })).toBe('engaged')
  })
  it('returns positive for high valence', () => {
    expect(getQualiaMood({ valence: 0.5, arousal: 0.3, novelty: 0.3, coherence: 0.5, salience: 0.5 })).toBe('positive')
  })
  it('returns uneasy for negative valence', () => {
    expect(getQualiaMood({ valence: -0.5, arousal: 0.3, novelty: 0.3, coherence: 0.5, salience: 0.5 })).toBe('uneasy')
  })
  it('returns curious for high novelty', () => {
    expect(getQualiaMood({ valence: 0.0, arousal: 0.3, novelty: 0.8, coherence: 0.5, salience: 0.5 })).toBe('curious')
  })
  it('returns calm for low arousal', () => {
    expect(getQualiaMood({ valence: 0.0, arousal: 0.1, novelty: 0.3, coherence: 0.5, salience: 0.5 })).toBe('calm')
  })
  it('returns neutral for default', () => {
    expect(getQualiaMood({ valence: 0.0, arousal: 0.3, novelty: 0.3, coherence: 0.5, salience: 0.5 })).toBe('neutral')
  })
  it('returns empty for undefined', () => {
    expect(getQualiaMood(undefined)).toBe('')
  })
})

describe('useConsciousnessStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns loading state initially', () => {
    mockCtrl.getStatus.mockRejectedValue(new Error('not ready'))
    const { result } = renderHook(() => useConsciousnessStatus())
    expect(result.current.loading).toBe(true)
    expect(result.current.status).toBeNull()
  })

  it('fetches and returns status', async () => {
    const mockStatus = {
      status: 'ok',
      uptime: 3600,
      enabled: true,
      level: 2,
      episodes: 10,
      beliefs_count: 1,
      qualia_count: 1,
      narrative: 'A narrative.',
      beliefs: { competence: 0.7 },
      current_qualia: { valence: 0.5, arousal: 0.3, novelty: 0.6, coherence: 0.7, salience: 0.4 },
      training: { is_training: false, total_pairs: 0, current_epoch: 0, loss: 0 },
    }
    mockCtrl.getStatus.mockResolvedValue(mockStatus)

    const { result } = renderHook(() => useConsciousnessStatus())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.status?.enabled).toBe(true)
    expect(result.current.status?.level).toBe(2)
    expect(result.current.levelLabel).toBe('Full')
  })

  it('degrades gracefully on fetch failure', async () => {
    mockCtrl.getStatus.mockRejectedValue(new Error('network'))

    const { result } = renderHook(() => useConsciousnessStatus())
    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })
    expect(result.current.status).toBeNull()
    expect(result.current.levelLabel).toBe('Off')
  })
})
