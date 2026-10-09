import { describe, expect, it } from 'vitest'
import { cn } from './utils'

describe('cn', () => {
  it('joins multiple classes', () => {
    expect(cn('text-red-500', 'bg-blue-100')).toBe('text-red-500 bg-blue-100')
  })

  it('merges conflicting tailwind classes (last wins)', () => {
    expect(cn('p-4', 'p-2')).toBe('p-2')
  })

  it('handles conditional object args', () => {
    const result = cn('base', { active: true, hidden: false })
    expect(result).toContain('base')
    expect(result).toContain('active')
    expect(result).not.toContain('hidden')
  })

  it('handles falsy and array inputs', () => {
    expect(cn(['a', 'b'], false && 'c', true && 'd')).toBe('a b d')
  })

  it('returns empty for empty input', () => {
    expect(cn()).toBe('')
  })
})