import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ThreadPanel } from './ThreadPanel'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

vi.mock('@/features/chat/components/messages/MessageBubble', () => ({
  MessageBubble: ({ content, role }: any) => (
    <div data-testid={`message-${role}`}>{content}</div>
  ),
}))

const mockParentMessage: ChatMessage = {
  id: '1',
  role: 'user',
  content: 'Original message',
  timestamp: Date.now(),
}

const mockThreadMessages: ChatMessage[] = [
  { id: '2', role: 'assistant', content: 'Reply 1', timestamp: Date.now() },
  { id: '3', role: 'user', content: 'Reply 2', timestamp: Date.now() },
]

describe('ThreadPanel', () => {
  it('renders thread header', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('Thread')).toBeInTheDocument()
  })

  it('renders reply count', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={mockThreadMessages}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('2 replies')).toBeInTheDocument()
  })

  it('renders singular reply count', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[mockThreadMessages[0]]}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('1 reply')).toBeInTheDocument()
  })

  it('renders parent message', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('Original message')).toBeInTheDocument()
  })

  it('renders empty state', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('No replies yet. Start the thread below.')).toBeInTheDocument()
  })

  it('renders thread messages', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={mockThreadMessages}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByText('Reply 1')).toBeInTheDocument()
    expect(screen.getByText('Reply 2')).toBeInTheDocument()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={vi.fn()}
        onClose={onClose}
      />
    )
    fireEvent.click(screen.getByLabelText('Close thread'))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onSend when send button clicked', () => {
    const onSend = vi.fn()
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={onSend}
        onClose={vi.fn()}
      />
    )
    const input = screen.getByPlaceholderText('Reply in thread...')
    fireEvent.change(input, { target: { value: 'New reply' } })
    fireEvent.click(screen.getByLabelText('Send reply'))
    expect(onSend).toHaveBeenCalledWith('New reply')
  })

  it('calls onSend on Enter key', () => {
    const onSend = vi.fn()
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={onSend}
        onClose={vi.fn()}
      />
    )
    const input = screen.getByPlaceholderText('Reply in thread...')
    fireEvent.change(input, { target: { value: 'New reply' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('New reply')
  })

  it('disables send button when input is empty', () => {
    render(
      <ThreadPanel
        parentMessage={mockParentMessage}
        threadMessages={[]}
        onSend={vi.fn()}
        onClose={vi.fn()}
      />
    )
    expect(screen.getByLabelText('Send reply')).toBeDisabled()
  })
})