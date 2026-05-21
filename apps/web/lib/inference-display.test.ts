import { describe, expect, it } from 'vitest'

import { catalogIdMatchesRuntime } from './inference-display'

describe('catalogIdMatchesRuntime', () => {
  it('matches exact ids', () => {
    expect(catalogIdMatchesRuntime('gpt2', 'gpt2')).toBe(true)
    expect(catalogIdMatchesRuntime('local/foo', 'local/foo')).toBe(true)
  })

  it('matches when catalog path ends with runtime stem', () => {
    expect(catalogIdMatchesRuntime('hf/gpt2', 'gpt2')).toBe(true)
    expect(catalogIdMatchesRuntime('some/nested/gpt2', 'gpt2')).toBe(true)
  })

  it('matches last path segment', () => {
    expect(catalogIdMatchesRuntime('org/model-name', 'model-name')).toBe(true)
  })

  it('returns false for unrelated ids', () => {
    expect(catalogIdMatchesRuntime('gpt2', 'gpt2-xl')).toBe(false)
    expect(catalogIdMatchesRuntime('a', 'b')).toBe(false)
  })
})
