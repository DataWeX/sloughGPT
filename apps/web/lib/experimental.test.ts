import { describe, expect, it } from 'vitest'
import { experimentalFeatures, isTrackedExperimental, isFresh } from './experimental'

describe('experimental tracking', () => {
  it('lists the four experimental surfaces', () => {
    const list = experimentalFeatures()
    expect(list).toContain('consciousness')
    expect(list).toContain('voice')
    expect(list).toContain('phoneme')
    expect(list).toContain('tokenizer')
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
