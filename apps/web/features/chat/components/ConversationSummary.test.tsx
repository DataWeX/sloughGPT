import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ConversationSummary } from './ConversationSummary'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

vi.mock('@/features/chat/hooks/useChatSummary', () => ({
  useChatSummary: vi.fn(() => ({
    summary: null,
    isGenerating: false,
    error: null,
    generateSummary: vi.fn(),
    clearSummary: vi.fn(),
  })),
}))

const mockMessages: ChatMessage[] = [
  { id: '1', role: 'user', content: 'Hello', timestamp: new Date(), reactions: {} },
  { id: '2', role: 'assistant', content: 'Hi there', timestamp: new Date(), reactions: {} },
]

describe('ConversationSummary', () => {
  it('renders generate button', () => {
    render(<ConversationSummary messages={mockMessages} />)
    expect(screen.getByText('Generate')).toBeInTheDocument()
  })

  it('renders title', () => {
    render(<ConversationSummary messages={mockMessages} />)
    expect(screen.getByText('Conversation Summary')).toBeInTheDocument()
  })

  it('disables generate when no messages', () => {
    render(<ConversationSummary messages={[]} />)
    const btn = screen.getByText('Generate').closest('button')!
    expect(btn).toBeDisabled()
  })

  it('shows placeholder when no summary', () => {
    render(<ConversationSummary messages={mockMessages} />)
    expect(screen.getByText(/Generate.*create a summary/)).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<ConversationSummary messages={mockMessages} className="custom-class" />)
    expect(container.firstChild).toHaveClass('custom-class')
  })
})
