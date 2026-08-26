import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationBranching } from './ConversationBranching'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'First message', timestamp: Date.now() - 60000 },
  { id: '2', role: 'assistant', content: 'First response', timestamp: Date.now() - 30000 },
  { id: '3', role: 'user', content: 'Second message', timestamp: Date.now() },
]

describe('ConversationBranching', () => {
  it('does not render for first message', () => {
    const { container } = render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="1"
        onBranch={vi.fn()}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders branch button for later messages', () => {
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="2"
        onBranch={vi.fn()}
      />
    )
    expect(screen.getByText('Branch from here')).toBeInTheDocument()
  })

  it('opens branching UI when clicked', () => {
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="2"
        onBranch={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('Branch from here'))
    expect(screen.getByText(/Branch from message 2/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/Enter a different prompt/)).toBeInTheDocument()
  })

  it('shows message preview', () => {
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="3"
        onBranch={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('Branch from here'))
    expect(screen.getByText('First message')).toBeInTheDocument()
    expect(screen.getByText('First response')).toBeInTheDocument()
  })

  it('calls onBranch with new messages', () => {
    const onBranch = vi.fn()
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="2"
        onBranch={onBranch}
      />
    )
    fireEvent.click(screen.getByText('Branch from here'))
    fireEvent.change(screen.getByPlaceholderText(/Enter a different prompt/), {
      target: { value: 'Alternative prompt' },
    })
    fireEvent.click(screen.getByText('Create Branch'))
    expect(onBranch).toHaveBeenCalledWith(
      '2',
      expect.arrayContaining([
        expect.objectContaining({ content: 'First message' }),
        expect.objectContaining({ content: 'First response' }),
        expect.objectContaining({ content: 'Alternative prompt', role: 'user' }),
      ])
    )
  })

  it('cancels branching', () => {
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="2"
        onBranch={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('Branch from here'))
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.getByText('Branch from here')).toBeInTheDocument()
  })

  it('disables create when prompt empty', () => {
    render(
      <ConversationBranching
        messages={mockMessages}
        currentMessageId="2"
        onBranch={vi.fn()}
      />
    )
    fireEvent.click(screen.getByText('Branch from here'))
    const createBtn = screen.getByText('Create Branch').closest('button')!
    expect(createBtn).toBeDisabled()
  })
})