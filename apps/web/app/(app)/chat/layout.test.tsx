import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import ChatLayout from './layout'

afterEach(() => {
  cleanup()
})

describe('ChatLayout', () => {
  it('renders children inside the full-height flex container', () => {
    const { container } = render(
      <ChatLayout>
        <p>chat body</p>
      </ChatLayout>,
    )
    expect(screen.getByText('chat body')).toBeInTheDocument()
    expect(container.firstElementChild?.className).toContain('flex')
    expect(container.firstElementChild?.className).toContain('overflow-hidden')
  })
})
