import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatStatsPanel } from './ChatStatsPanel'
import type { ChatMessage } from '../types'

afterEach(cleanup)

const makeMessages = (overrides?: Partial<ChatMessage>[]): ChatMessage[] => {
  const base: ChatMessage[] = [
    { id: '1', role: 'user', content: 'Hello world', timestamp: new Date('2024-01-01T10:00:00').toISOString(), reactions: {} },
    { id: '2', role: 'assistant', content: 'Hi there! How can I help?', timestamp: new Date('2024-01-01T10:01:00').toISOString(), reactions: { '👍': 1 } },
    { id: '3', role: 'user', content: 'Tell me about testing', timestamp: new Date('2024-01-01T10:02:00').toISOString(), pinned: true, reactions: {} },
  ]
  return overrides ? base.map((m, i) => ({ ...m, ...overrides[i] })) : base
}

describe('ChatStatsPanel', () => {
  it('returns null when closed', () => {
    const { container } = render(<ChatStatsPanel open={false} onClose={vi.fn()} messages={makeMessages()} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders stats when open', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={makeMessages()} />)
    expect(screen.getByText('Conversation Statistics')).toBeInTheDocument()
    expect(screen.getByText('Total messages')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('calls onClose when clicking close button', () => {
    const onClose = vi.fn()
    render(<ChatStatsPanel open={true} onClose={onClose} messages={makeMessages()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose when clicking backdrop', () => {
    const onClose = vi.fn()
    const { container } = render(<ChatStatsPanel open={true} onClose={onClose} messages={makeMessages()} />)
    const backdrop = container.querySelector('.fixed')!
    fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalled()
  })

  it('computes stats correctly', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={makeMessages()} />)
    expect(screen.getByText('Your messages')).toBeInTheDocument()
    expect(screen.getByText('Assistant messages')).toBeInTheDocument()
  })

  it('shows pinned message count', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={makeMessages()} />)
    expect(screen.getByText('Pinned messages')).toBeInTheDocument()
  })

  it('shows messages with reactions', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={makeMessages()} />)
    expect(screen.getByText('Messages with reactions')).toBeInTheDocument()
  })

  it('handles empty messages', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={[]} />)
    expect(screen.getByText('Conversation Statistics')).toBeInTheDocument()
  })

  it('shows conversation duration', () => {
    render(<ChatStatsPanel open={true} onClose={vi.fn()} messages={makeMessages()} />)
    expect(screen.getByText('Conversation duration')).toBeInTheDocument()
  })

  it('does not propagate click on panel', () => {
    const onClose = vi.fn()
    render(<ChatStatsPanel open={true} onClose={onClose} messages={makeMessages()} />)
    const panel = screen.getByText('Conversation Statistics').closest('.bg-background')!
    fireEvent.click(panel)
    expect(onClose).not.toHaveBeenCalled()
  })
})
