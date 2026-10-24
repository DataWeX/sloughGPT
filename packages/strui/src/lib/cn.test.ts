import { describe, expect, it } from 'vitest'

import { cn } from './cn'

describe('cn', () => {
  it('merges tailwind conflicts (last wins)', () => {
    expect(cn('p-4', 'p-2')).toBe('p-2')
  })

  it('handles conditional classes', () => {
    // Boolean() wrappers keep the operands dynamically falsy/truthy — eslint 9's
    // no-constant-binary-expression flags literal `false && x` / `true && x`.
    const hidden = Boolean(0)
    const shown = Boolean(1)
    expect(cn('base', hidden && 'hidden', shown && 'block')).toBe('base block')
  })
})
