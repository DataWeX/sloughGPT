import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { ConversationStats } from './ConversationStats'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: new Date() },
  { id: '2', role: 'assistant', content: 'Hi there! How can I help you?', timestamp: new Date() },
  { id: '3', role: 'user', content: 'Tell me about TypeScript', timestamp: new Date() },
]

describe('ConversationStats', () => {
  it('renders empty state when no messages', () => {
    render(<ConversationStats messages={[]} />)
    expect(screen.getByText('No messages yet')).toBeInTheDocument()
  })

  it('shows message count', () => {
    render(<ConversationStats messages={mockMessages} />)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('shows word count', () => {
    render(<ConversationStats messages={mockMessages} />)
    expect(screen.getByText('13')).toBeInTheDocument()
  })

  it('shows user and assistant counts', () => {
    render(<ConversationStats messages={mockMessages} />)
    // User: 2 messages, Assistant: 1 message
    const userStats = screen.getAllByText('2')
    expect(userStats.length).toBeGreaterThanOrEqual(1)
  })

  it('shows character count', () => {
    render(<ConversationStats messages={mockMessages} />)
    expect(screen.getByText('64')).toBeInTheDocument()
  })

  it('shows tool call count when present', () => {
    const messagesWithTools: ChatMessage[] = [
      { id: '1', role: 'user', content: 'Hello', timestamp: new Date(), toolCalls: [{ id: 't1', name: 'test' }] },
    ]
    render(<ConversationStats messages={messagesWithTools} />)
    expect(screen.getByText('Tool Calls')).toBeInTheDocument()
  })
})