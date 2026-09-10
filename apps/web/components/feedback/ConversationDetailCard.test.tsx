// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
}))

import { ConversationDetailCard } from './ConversationDetailCard'

afterEach(() => { cleanup() })

describe('ConversationDetailCard', () => {
  it('renders conversation name', () => {
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText('Test Chat')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText('No messages in this conversation.')).toBeTruthy()
  })

  it('renders messages', () => {
    const messages = [
      { role: 'user' as const, content: 'Hello' },
      { role: 'assistant' as const, content: 'Hi there!' },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText('Hello')).toBeTruthy()
    expect(screen.getByText('Hi there!')).toBeTruthy()
    expect(screen.getByText(/2 messages/)).toBeTruthy()
  })

  it('shows message count', () => {
    const messages = [
      { role: 'user' as const, content: 'One' },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText(/1 message/)).toBeTruthy()
  })

  it('truncates long messages', () => {
    const longContent = 'A'.repeat(250)
    const messages = [
      { role: 'user' as const, content: longContent },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    expect(screen.getByText(/Show more/)).toBeTruthy()
  })

  it('expands long message on click', () => {
    const longContent = 'A'.repeat(250)
    const messages = [
      { role: 'user' as const, content: longContent },
    ]
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={messages} />)
    fireEvent.click(screen.getByText(/Show more/))
    expect(screen.getByText(/Show less/)).toBeTruthy()
    expect(screen.getByText(longContent)).toBeTruthy()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<ConversationDetailCard conversationId="abc123" conversationName="Test Chat" messages={[]} onClose={onClose} />)
    fireEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows truncated conversation id', () => {
    render(<ConversationDetailCard conversationId="abcdef1234567890" conversationName="Test Chat" messages={[]} />)
    expect(screen.getByText(/abcdef12/)).toBeTruthy()
  })
})
