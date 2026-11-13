import { describe, expect, it } from 'vitest'
import { experimentalFeatures, isTrackedExperimental, isFresh } from './experimental'

describe('experimental tracking', () => {
  it('lists the three experimental surfaces', () => {
    const list = experimentalFeatures()
    expect(list).toContain('consciousness')
    expect(list).toContain('voice')
    expect(list).toContain('tokenizer')
    // 'phoneme' removed 2026-11-08 (card 20261108_152007): the learning-UI
    // surface is deleted; phoneme is model capability, not a feature flag.
    expect(list).not.toContain('phoneme')
  })

  it('tracks experimental status', () => {
    expect(isTrackedExperimental('consciousness')).toBe(true)
    expect(isTrackedExperimental('chat')).toBe(false)
    expect(isTrackedExperimental('nope')).toBe(false)
  })

  it('freshness is boolean', () => {
    expect(typeof isFresh('consciousness')).toBe('boolean')
    expect(isFresh('nope')).toBe(false)
  })
})
