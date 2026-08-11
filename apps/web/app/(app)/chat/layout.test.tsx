import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
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

  it('renders multiple children', () => {
    render(
      <ChatLayout>
        <p>first</p>
        <p>second</p>
      </ChatLayout>,
    )
    expect(screen.getByText('first')).toBeInTheDocument()
    expect(screen.getByText('second')).toBeInTheDocument()
  })

  it('applies h-full class for full height', () => {
    const { container } = render(
      <ChatLayout>
        <p>content</p>
      </ChatLayout>,
    )
    expect(container.firstElementChild?.className).toContain('h-full')
  })

  it('applies w-full class for full width', () => {
    const { container } = render(
      <ChatLayout>
        <p>content</p>
      </ChatLayout>,
    )
    expect(container.firstElementChild?.className).toContain('w-full')
  })

  it('renders nested components', () => {
    render(
      <ChatLayout>
        <div>
          <span>nested content</span>
        </div>
      </ChatLayout>,
    )
    expect(screen.getByText('nested content')).toBeInTheDocument()
  })

  it('renders empty children', () => {
    const { container } = render(<ChatLayout>{null}</ChatLayout>)
    expect(container.firstElementChild).toBeTruthy()
  })
})
