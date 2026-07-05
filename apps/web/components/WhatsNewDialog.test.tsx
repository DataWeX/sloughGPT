
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { WhatsNewDialog, getUnseenCount, markAllSeen } from './WhatsNewDialog'
import { whatsNewItems } from '@/lib/whats-new-data'

describe('WhatsNewDialog', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
    vi.unstubAllGlobals()
  })

  it('renders all changelog entries when open', () => {
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    whatsNewItems.forEach(item => {
      expect(screen.getAllByText(item.title).length).toBeGreaterThanOrEqual(1)
      expect(screen.getByText(item.description)).toBeInTheDocument()
    })
  })

  it('marks entries as seen on open', async () => {
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(getUnseenCount()).toBe(0)
    })
  })

  it('getUnseenCount returns number of unseen items', () => {
    expect(getUnseenCount()).toBe(whatsNewItems.length)
    localStorage.setItem('whatsnew_seen', JSON.stringify([whatsNewItems[0].id]))
    expect(getUnseenCount()).toBe(whatsNewItems.length - 1)
  })

  it('markAllSeen persists all ids and dispatches update event', () => {
    const listener = vi.fn()
    window.addEventListener('whatsnew-updated', listener)
    markAllSeen()
    expect(listener).toHaveBeenCalled()
    expect(getUnseenCount()).toBe(0)
    window.removeEventListener('whatsnew-updated', listener)
  })

  it('calls onOpenChange(false) when Escape is pressed', () => {
    const onOpenChange = vi.fn()
    const { container } = render(<WhatsNewDialog open={true} onOpenChange={onOpenChange} />)
    fireEvent.keyDown(container, { key: 'Escape' })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('renders linked titles with href when item has href', () => {
    const linked = whatsNewItems.find(i => i.href)
    if (!linked) return
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    const links = screen.getAllByRole('link')
    const link = links.find(l => l.textContent === linked.title)
    expect(link).toBeDefined()
    expect(link).toHaveAttribute('href', linked.href)
  })
})
