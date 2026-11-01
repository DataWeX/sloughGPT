import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationExport } from './ConversationExport'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: new Date() },
  { id: '2', role: 'assistant', content: 'Hi there! How can I help you?', timestamp: new Date() },
]

describe('ConversationExport', () => {
  it('renders empty state when no messages', () => {
    render(<ConversationExport messages={[]} />)
    expect(screen.getByText('No messages to export')).toBeInTheDocument()
  })

  it('shows message count and word count', () => {
    render(<ConversationExport messages={mockMessages} />)
    expect(screen.getByText(/2 messages/)).toBeInTheDocument()
    expect(screen.getByText(/words/)).toBeInTheDocument()
  })

  it('renders format selector', () => {
    render(<ConversationExport messages={mockMessages} />)
    expect(screen.getByDisplayValue('Markdown')).toBeInTheDocument()
  })

  it('renders copy button', () => {
    render(<ConversationExport messages={mockMessages} />)
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument()
  })

  it('renders download button', () => {
    render(<ConversationExport messages={mockMessages} />)
    expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument()
  })

  it('changes format on select', () => {
    render(<ConversationExport messages={mockMessages} />)
    const select = screen.getByDisplayValue('Markdown')
    fireEvent.change(select, { target: { value: 'json' } })
    expect(screen.getByDisplayValue('JSON')).toBeInTheDocument()
  })
})