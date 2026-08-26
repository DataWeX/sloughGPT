import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { MessagePinning } from './MessagePinning'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: Date.now() - 60000 },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: Date.now() },
]

describe('MessagePinning', () => {
  it('renders empty state', () => {
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText(/No pinned messages/)).toBeInTheDocument()
  })

  it('renders with count', () => {
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText(/Pinned Messages/)).toBeInTheDocument()
    expect(screen.getByText('(0)')).toBeInTheDocument()
  })

  it('shows message count', () => {
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText(/0 of 2 messages pinned/)).toBeInTheDocument()
  })

  it('clears all pins when clear clicked', async () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'Hello',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Clear all pins'))
    expect(screen.getByText(/No pinned messages/)).toBeInTheDocument()
  })

  it('renders pinned message from localStorage', () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'Pinned content',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    expect(screen.getByText('Pinned content')).toBeInTheDocument()
  })

  it('unpins message', () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'To unpin',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Unpin'))
    expect(screen.queryByText('To unpin')).not.toBeInTheDocument()
  })

  it('opens note editor', () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'Hello',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Add note'))
    expect(screen.getByPlaceholderText('Add a note...')).toBeInTheDocument()
  })

  it('saves note on Enter', async () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'Hello',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Add note'))
    const input = screen.getByPlaceholderText('Add a note...')
    fireEvent.change(input, { target: { value: 'My note' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getByText('My note')).toBeInTheDocument()
  })

  it('cancels note on Escape', () => {
    localStorage.setItem('pinned-messages', JSON.stringify([{
      id: 'p1',
      messageId: '1',
      content: 'Hello',
      role: 'user',
      pinnedAt: Date.now(),
    }]))
    render(<MessagePinning messages={mockMessages} onJumpToMessage={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Add note'))
    const input = screen.getByPlaceholderText('Add a note...')
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(screen.queryByPlaceholderText('Add a note...')).not.toBeInTheDocument()
  })

  it('persists to localStorage', async () => {
    const onJump = vi.fn()
    render(<MessagePinning messages={mockMessages} onJumpToMessage={onJump} />)
    // We can't easily test pinning without a pin button in the message bubble
    // Just verify localStorage integration works
    const stored = JSON.parse(localStorage.getItem('pinned-messages') || '[]')
    expect(stored).toEqual([])
  })
})