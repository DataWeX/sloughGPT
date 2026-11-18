import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { StreamingCursor } from './StreamingCursor'

describe('StreamingCursor', () => {
  it('renders the pulsing caret', () => {
    const { container } = render(<StreamingCursor />)
    expect(container.querySelector('.animate-pulse')).toBeTruthy()
  })

  it('is hidden from assistive tech (the frame carries aria-live)', () => {
    const { container } = render(<StreamingCursor />)
    expect(container.firstElementChild).toHaveAttribute('aria-hidden', 'true')
  })

  it('respects reduced motion', () => {
    const { container } = render(<StreamingCursor />)
    expect(container.querySelector('.motion-reduce\\:animate-none')).toBeTruthy()
  })

  it('merges a className override', () => {
    const { container } = render(<StreamingCursor className="w-2" />)
    expect(container.firstElementChild?.className).toContain('w-2')
  })
})
