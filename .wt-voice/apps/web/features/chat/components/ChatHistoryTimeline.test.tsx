import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatHistoryTimeline } from './ChatHistoryTimeline'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: new Date(Date.now() - 60000) },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: new Date(Date.now() - 30000) },
  { id: '3', role: 'user', content: 'How are you?', timestamp: new Date() },
]

describe('ChatHistoryTimeline', () => {
  it('renders message count', () => {
    render(<ChatHistoryTimeline messages={mockMessages} onNavigate={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('(3 messages)')).toBeInTheDocument()
  })

  it('renders empty state', () => {
    render(<ChatHistoryTimeline messages={[]} onNavigate={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('No messages')).toBeInTheDocument()
  })

  it('shows message previews', () => {
    render(<ChatHistoryTimeline messages={mockMessages} onNavigate={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('Hello world')).toBeInTheDocument()
    expect(screen.getByText('Hi there!')).toBeInTheDocument()
    expect(screen.getByText('How are you?')).toBeInTheDocument()
  })

  it('calls onNavigate when message clicked', () => {
    const onNavigate = vi.fn()
    render(<ChatHistoryTimeline messages={mockMessages} onNavigate={onNavigate} onClose={vi.fn()} />)
    fireEvent.click(screen.getByText('Hello world'))
    expect(onNavigate).toHaveBeenCalledWith('1')
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<ChatHistoryTimeline messages={mockMessages} onNavigate={vi.fn()} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onClose).toHaveBeenCalled()
  })

  it('truncates long messages', () => {
    const longMsg: ChatMessage = {
      id: '4',
      role: 'user',
      content: 'A'.repeat(100),
      timestamp: new Date(),
    }
    render(<ChatHistoryTimeline messages={[longMsg]} onNavigate={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getByText('A'.repeat(60) + '…')).toBeInTheDocument()
  })

  it('groups messages by time and role', () => {
    render(<ChatHistoryTimeline messages={mockMessages} onNavigate={vi.fn()} onClose={vi.fn()} />)
    expect(screen.getAllByText(/You ·/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Assistant ·/)).toBeInTheDocument()
  })
})