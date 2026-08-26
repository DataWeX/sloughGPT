import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationShare } from './ConversationShare'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello world', timestamp: Date.now() },
  { id: '2', role: 'assistant', content: 'Hi there!', timestamp: Date.now() },
]

describe('ConversationShare', () => {
  it('renders empty state when no messages', () => {
    render(<ConversationShare messages={[]} />)
    expect(screen.getByText('No messages to share')).toBeInTheDocument()
  })

  it('shows message count', () => {
    render(<ConversationShare messages={mockMessages} />)
    expect(screen.getByText('2 messages')).toBeInTheDocument()
  })

  it('renders copy button', () => {
    render(<ConversationShare messages={mockMessages} />)
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument()
  })

  it('renders link button when sessionId provided', () => {
    render(<ConversationShare messages={mockMessages} sessionId="test-session" />)
    expect(screen.getByRole('button', { name: /link/i })).toBeInTheDocument()
  })

  it('does not render link button when no sessionId', () => {
    render(<ConversationShare messages={mockMessages} />)
    expect(screen.queryByRole('button', { name: /link/i })).not.toBeInTheDocument()
  })

  it('calls clipboard API on copy', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    
    render(<ConversationShare messages={mockMessages} />)
    fireEvent.click(screen.getByRole('button', { name: /copy/i }))
    
    expect(writeText).toHaveBeenCalled()
  })

  it('shows copied state after copy', async () => {
    vi.useFakeTimers()
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    
    render(<ConversationShare messages={mockMessages} />)
    const copyButton = screen.getByRole('button', { name: /copy/i })
    fireEvent.click(copyButton)
    
    // Wait for state update
    await vi.waitFor(() => {
      expect(screen.getByText(/Copied!/)).toBeInTheDocument()
    })
    
    vi.useRealTimers()
  })
})