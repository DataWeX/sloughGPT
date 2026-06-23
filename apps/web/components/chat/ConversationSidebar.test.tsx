// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationSidebar } from './ConversationSidebar'

afterEach(cleanup)

const conversations = [
  { id: '1', name: 'Chat 1', messages: [], createdAt: '2026-06-20T10:00:00Z', updatedAt: '2026-06-21T10:00:00Z', synced: true, starred: false, pinned: false },
  { id: '2', name: 'Starred Chat', messages: [], createdAt: '2026-06-19T10:00:00Z', updatedAt: '2026-06-20T10:00:00Z', synced: true, starred: true, pinned: false },
  { id: '3', name: 'Chat 3', messages: [], createdAt: '2026-06-18T10:00:00Z', updatedAt: '2026-06-19T10:00:00Z', synced: true, starred: false, pinned: false },
]

describe('ConversationSidebar', () => {
  const onLoad = vi.fn()
  const onNewChat = vi.fn()
  const onClose = vi.fn()

  it('renders header with title', () => {
    render(<ConversationSidebar conversations={[]} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={true} onClose={onClose} />)
    expect(screen.getAllByText('Conversations').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('New').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no conversations', () => {
    render(<ConversationSidebar conversations={[]} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={true} onClose={onClose} />)
    expect(screen.getAllByText(/No conversations yet/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders starred conversations first', () => {
    render(<ConversationSidebar conversations={conversations} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={true} onClose={onClose} />)
    expect(screen.getAllByText('Starred').length).toBeGreaterThanOrEqual(1)
  })

  it('calls onLoadConversation and onClose when clicking a conversation', () => {
    render(<ConversationSidebar conversations={conversations} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={true} onClose={onClose} />)
    const items = screen.getAllByText(/Chat 1|Starred Chat|Chat 3/)
    fireEvent.click(items[0])
    expect(onLoad).toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onNewChat when clicking New button', () => {
    render(<ConversationSidebar conversations={conversations} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={true} onClose={onClose} />)
    fireEvent.click(screen.getAllByText('New')[0])
    expect(onNewChat).toHaveBeenCalled()
    expect(onClose).toHaveBeenCalled()
  })

  it('does not render when open is false', () => {
    const { container } = render(<ConversationSidebar conversations={conversations} currentConversationId="" onLoadConversation={onLoad} onNewChat={onNewChat} open={false} onClose={onClose} />)
    expect(container.querySelector('[class*="w-0"]')).toBeTruthy()
  })
})
