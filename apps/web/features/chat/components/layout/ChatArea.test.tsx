import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('./../input/ChatInput', () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}))

vi.mock('./ChatScreen', () => ({
  ChatScreen: vi.fn().mockImplementation(({ messages, loading }) => (
    <div data-testid="chat-screen">
      {messages.map((m: { id: string; content: string }) => (
        <div key={m.id} data-testid="message">{m.content}</div>
      ))}
      {loading && <div data-testid="loading-indicator">Loading...</div>}
    </div>
  )),
}))

vi.mock('./ImageDropZone', () => ({
  ImageDropZone: ({ children, onImageDropped }: { children: React.ReactNode; onImageDropped: (file: File) => void }) => (
    <div data-testid="image-drop-zone" onDrop={(e) => {
      const file = (e.dataTransfer?.files?.[0] || new File([''], 'test.png'))
      onImageDropped(file)
    }}>
      {children}
    </div>
  ),
}))

import { ChatArea } from './ChatArea'

const mockHealth = {
  status: 'healthy',
  model_loaded: true,
  model_type: 'gpt2',
  summary: 'gpt2 loaded',
  uptime_seconds: 100,
  request_count: 10,
  error_count: 0,
  inference_count: 5,
  total_tokens: 1000,
  tokens_per_sec: 10,
  avg_tokens_per_request: 200,
  avg_latency_ms: 100,
  requests_per_minute: 5,
}

const createMessage = (id: string, content: string) => ({
  id,
  content,
  role: 'user' as const,
  timestamp: new Date(),
})

describe('ChatArea', () => {
  const defaultProps = {
    messages: [],
    loading: false,
    health: mockHealth,
    onRefreshHealth: vi.fn(),
    onCopy: vi.fn(),
    value: '',
    onChange: vi.fn(),
    onSend: vi.fn(),
  }

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('renders chat screen with messages', () => {
    const messages = [createMessage('1', 'Hello'), createMessage('2', 'World')]
    render(<ChatArea {...defaultProps} messages={messages} />)
    expect(screen.getByText('Hello')).toBeDefined()
    expect(screen.getByText('World')).toBeDefined()
  })

  it('renders empty state when no messages', () => {
    render(<ChatArea {...defaultProps} messages={[]} />)
    const screenEl = screen.getByTestId('chat-screen')
    expect(screenEl).toBeDefined()
  })

  it('renders loading indicator when loading', () => {
    render(<ChatArea {...defaultProps} messages={[]} loading={true} />)
    expect(screen.getByTestId('loading-indicator')).toBeDefined()
  })

  it('renders chat input', () => {
    render(<ChatArea {...defaultProps} />)
    expect(screen.getByTestId('chat-input')).toBeDefined()
  })

  it('shows jump-to-bottom button when scrolled up with messages', () => {
    const messages = [createMessage('1', 'Hello'), createMessage('2', 'World')]
    render(<ChatArea {...defaultProps} messages={messages} />)
    const container = screen.getByRole('region', { name: 'Chat messages' })
    Object.defineProperty(container, 'scrollHeight', { value: 1000 })
    Object.defineProperty(container, 'scrollTop', { value: 0 })
    Object.defineProperty(container, 'clientHeight', { value: 200 })
    fireEvent.scroll(container)
    expect(screen.getByLabelText('Jump to latest messages')).toBeDefined()
  })

  it('shows message count on jump-to-bottom button', () => {
    const messages = [createMessage('1', 'A'), createMessage('2', 'B'), createMessage('3', 'C')]
    render(<ChatArea {...defaultProps} messages={messages} />)
    const container = screen.getByRole('region', { name: 'Chat messages' })
    Object.defineProperty(container, 'scrollHeight', { value: 1000 })
    Object.defineProperty(container, 'scrollTop', { value: 0 })
    Object.defineProperty(container, 'clientHeight', { value: 200 })
    fireEvent.scroll(container)
    expect(screen.getByText('3')).toBeDefined()
  })

  it('filters messages by search query', () => {
    const messages = [
      createMessage('1', 'Hello world'),
      createMessage('2', 'Goodbye'),
    ]
    render(<ChatArea {...defaultProps} messages={messages} searchQuery="Hello" />)
    expect(screen.getByText('Hello world')).toBeDefined()
    expect(screen.queryByText('Goodbye')).toBeNull()
  })

  it('forwards imperative scrollToBottom', () => {
    const ref = React.createRef<{ scrollToBottom: () => void }>()
    render(<ChatArea {...defaultProps} ref={ref} />)
    expect(ref.current).not.toBeNull()
    expect(typeof ref.current?.scrollToBottom).toBe('function')
  })
})
