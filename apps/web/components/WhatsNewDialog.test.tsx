
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { whatsNewItems } from '@/lib/whats-new-data'

const kvStore = new Map<string, unknown>()

const { mockChatDB } = vi.hoisted(() => {
  const kvStore = new Map<string, unknown>()
  const mockChatDB = {
    getKV: vi.fn(async (key: string) => kvStore.get(key)),
    setKV: vi.fn(async (key: string, value: unknown) => { kvStore.set(key, value) }),
  }
  return { mockChatDB, kvStore }
})

vi.mock('@/lib/db', () => ({
  chatDB: mockChatDB,
}))

const { getUnseenCount, markAllSeen, WhatsNewDialog } = await import('./WhatsNewDialog')

describe('WhatsNewDialog', () => {
  beforeEach(() => {
    kvStore.clear()
    mockChatDB.getKV.mockImplementation(async (key: string) => kvStore.get(key))
    mockChatDB.setKV.mockImplementation(async (key: string, value: unknown) => { kvStore.set(key, value) })
  })

  it('renders all changelog entries when open', async () => {
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      whatsNewItems.forEach(item => {
        expect(screen.getAllByText(item.title).length).toBeGreaterThanOrEqual(1)
        expect(screen.getByText(item.description)).toBeInTheDocument()
      })
    })
  })

  it('marks entries as seen on open', async () => {
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      expect(mockChatDB.setKV).toHaveBeenCalled()
    })
  })

  it('getUnseenCount returns number of unseen items', async () => {
    const count = await getUnseenCount()
    expect(count).toBe(whatsNewItems.length)
    await mockChatDB.setKV('whatsnew_seen', [whatsNewItems[0].id])
    const countAfter = await getUnseenCount()
    expect(countAfter).toBe(whatsNewItems.length - 1)
  })

  it('markAllSeen persists all ids and dispatches update event', async () => {
    const listener = vi.fn()
    window.addEventListener('whatsnew-updated', listener)
    await markAllSeen()
    expect(listener).toHaveBeenCalled()
    const count = await getUnseenCount()
    expect(count).toBe(0)
    window.removeEventListener('whatsnew-updated', listener)
  })

  it('calls onOpenChange(false) when Escape is pressed', () => {
    const onOpenChange = vi.fn()
    const { container } = render(<WhatsNewDialog open={true} onOpenChange={onOpenChange} />)
    fireEvent.keyDown(container, { key: 'Escape' })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it('renders linked titles with href when item has href', async () => {
    const linked = whatsNewItems.find(i => i.href)
    if (!linked) return
    render(<WhatsNewDialog open={true} onOpenChange={() => {}} />)
    await waitFor(() => {
      const links = screen.getAllByRole('link')
      const link = links.find(l => l.textContent === linked.title)
      expect(link).toBeDefined()
      expect(link).toHaveAttribute('href', linked.href)
    })
  })
})
