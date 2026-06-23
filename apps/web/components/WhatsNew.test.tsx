// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach, beforeAll } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

const { mockWhatsNewItems } = vi.hoisted(() => ({
  mockWhatsNewItems: [
    { id: 'v2', icon: '🚀', title: 'Major Update', date: '2026-06', description: 'Big changes', tags: ['feature'], href: '/chat' },
    { id: 'v1', icon: '🐛', title: 'Bug Fixes', date: '2026-05', description: 'Fixed things', tags: ['fix'] },
  ],
}))

vi.mock('@/lib/whats-new-data', () => ({
  whatsNewItems: mockWhatsNewItems,
}))

import { WhatsNewTrigger, WhatsNewDialog } from './WhatsNew'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

describe('WhatsNewTrigger', () => {
  beforeEach(() => {
    localStorage.clear()
  })
  afterEach(cleanup)

  it('renders trigger button', () => {
    render(<WhatsNewTrigger />)
    expect(screen.getByLabelText("What's new")).toBeDefined()
  })

  it('shows what\'s new badge when unseen', () => {
    render(<WhatsNewTrigger />)
    expect(screen.getByText("What's new")).toBeDefined()
  })

  it('opens dialog on click', () => {
    render(<WhatsNewTrigger />)
    fireEvent.click(screen.getByLabelText("What's new"))
    expect(screen.getByText("What's New")).toBeDefined()
  })

  it('shows Updates text when seen', () => {
    localStorage.setItem('man_whats_new_seen', 'v2')
    render(<WhatsNewTrigger />)
    expect(screen.getByText('Updates')).toBeDefined()
  })

  it('starts tour mode when Take guided tour clicked', () => {
    render(<WhatsNewTrigger />)
    fireEvent.click(screen.getByLabelText("What's new"))
    fireEvent.click(screen.getByText(/guided tour/i))
    expect(screen.getByText(/Tour \(1\/2\)/)).toBeDefined()
  })
})

describe('WhatsNewDialog', () => {
  afterEach(cleanup)

  it('renders nothing when closed', () => {
    const { container } = render(<WhatsNewDialog open={false} onClose={vi.fn()} tourMode={false} onStartTour={vi.fn()} />)
    expect(container.innerHTML).toBe('')
  })

  it('shows title when open', () => {
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={false} onStartTour={vi.fn()} />)
    expect(screen.getByText("What's New")).toBeDefined()
  })

  it('shows feature items', () => {
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={false} onStartTour={vi.fn()} />)
    expect(screen.getByText('Major Update')).toBeDefined()
    expect(screen.getByText('Bug Fixes')).toBeDefined()
  })

  it('shows Got it and Take tour buttons', () => {
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={false} onStartTour={vi.fn()} />)
    expect(screen.getByText('Got it')).toBeDefined()
    expect(screen.getByText(/guided tour/i)).toBeDefined()
  })

  it('calls onClose on Got it click', () => {
    vi.useFakeTimers()
    const onClose = vi.fn()
    render(<WhatsNewDialog open={true} onClose={onClose} tourMode={false} onStartTour={vi.fn()} />)
    fireEvent.click(screen.getByText('Got it'))
    vi.advanceTimersByTime(200)
    expect(onClose).toHaveBeenCalled()
    vi.useRealTimers()
  })

  it('calls onStartTour on Take tour click', () => {
    const onStartTour = vi.fn()
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={false} onStartTour={onStartTour} />)
    fireEvent.click(screen.getByText(/guided tour/i))
    expect(onStartTour).toHaveBeenCalled()
  })

  it('navigates tour steps', () => {
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={true} onStartTour={vi.fn()} />)
    expect(screen.getByText('Major Update')).toBeDefined()
    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText('Bug Fixes')).toBeDefined()
    fireEvent.click(screen.getByText('Back'))
    expect(screen.getByText('Major Update')).toBeDefined()
  })

  it('shows Done on last tour step', () => {
    render(<WhatsNewDialog open={true} onClose={vi.fn()} tourMode={true} onStartTour={vi.fn()} />)
    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText('Done')).toBeDefined()
  })
})
