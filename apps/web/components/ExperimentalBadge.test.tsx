import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ExperimentalBadge } from './ExperimentalBadge'

describe('ExperimentalBadge', () => {
  it('renders for tracked experimental features', () => {
    render(<ExperimentalBadge feature="voice" />)
    expect(screen.getByText('Experimental')).toBeTruthy()
  })

  it('renders nothing for stable features', () => {
    const { container } = render(<ExperimentalBadge feature="chat" />)
    expect(container.textContent).toBe('')
  })

  it('renders nothing for unknown features', () => {
    const { container } = render(<ExperimentalBadge feature="nope" />)
    expect(container.textContent).toBe('')
  })
})
