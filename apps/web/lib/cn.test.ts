import { describe, expect, it } from 'vitest'
import { cn } from '@sloughgpt/strui'

describe('cn', () => {
  it('joins single class', () => {
    expect(cn('text-red-500')).toBe('text-red-500')
  })

  it('joins multiple classes', () => {
    expect(cn('text-red-500', 'bg-blue-100')).toBe('text-red-500 bg-blue-100')
  })

  it('handles conditional object', () => {
    const result = cn('base', { active: true, hidden: false })
    expect(result).toContain('base')
    expect(result).toContain('active')
    expect(result).not.toContain('hidden')
  })

  it('handles arrays', () => {
    expect(cn(['a', 'b'], 'c')).toContain('a')
    expect(cn(['a', 'b'], 'c')).toContain('b')
    expect(cn(['a', 'b'], 'c')).toContain('c')
  })

  it('handles conditional via booleans', () => {
    const result = cn('base', false && 'hidden', true && 'visible')
    expect(result).toContain('base')
    expect(result).toContain('visible')
    expect(result).not.toContain('hidden')
  })

  it('returns empty string for empty input', () => {
    expect(cn()).toBe('')
  })
})
