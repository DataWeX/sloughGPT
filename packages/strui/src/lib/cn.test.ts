import { describe, expect, it } from 'vitest'

import { cn } from './cn'

describe('cn', () => {
  it('merges tailwind conflicts (last wins)', () => {
    expect(cn('p-4', 'p-2')).toBe('p-2')
  })

  it('handles conditional classes', () => {
    // The idiom under test is the JSX conditional form (`cond && 'class'`).
    // eslint 9's no-constant-binary-expression — new in v9's eslint:recommended,
    // this package used to lint under v8 — rejects a *literal* left operand,
    // so bind the conditions; cn() receives identical booleans either way.
    const shouldHide = false
    const shouldShow = true
    expect(cn('base', shouldHide && 'hidden', shouldShow && 'block')).toBe('base block')
  })
})
