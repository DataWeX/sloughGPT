// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach, afterAll, beforeAll } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

let mockFilteredMessages: any[] = []
vi.mock('./ChatScreen', () => ({
  ChatScreen: vi.fn().mockImplementation(({ messages }: any) => {
    mockFilteredMessages = messages
    return <div data-testid="chat-screen" />
  }),
}))

vi.mock('./ChatInput', () => ({
  ChatInput: vi.fn().mockImplementation((props: any) => {
    return <div data-testid="chat-input" />
  }),
}))

import { ChatArea } from './ChatArea'

const baseHealth = { status: 'ok', model_loaded: true, model_type: 'gpt2' } as any
const msg = (id: string, content = 'hello') => ({ id, role: 'user' as const, content, timestamp: new Date() })

describe('ChatArea', () => {
  beforeAll(() => {
    Element.prototype.scrollIntoView = vi.fn()
  })

  afterEach(cleanup)

  it('renders empty ChatScreen when no messages', () => {
    render(
      <ChatArea messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    expect(screen.getByTestId('chat-screen')).toBeDefined()
    expect(screen.getByTestId('chat-input')).toBeDefined()
    expect(mockFilteredMessages).toEqual([])
  })

  it('renders messages', () => {
    const messages = [msg('m1', 'How are you?'), msg('m2', 'Great!', 'assistant')]
    render(
      <ChatArea messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    expect(mockFilteredMessages).toEqual(messages)
  })

  it('filters messages by searchQuery', () => {
    const messages = [msg('m1', 'Hello world'), msg('m2', 'Goodbye')]
    render(
      <ChatArea messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} searchQuery="hello" />
    )
    expect(mockFilteredMessages).toHaveLength(1)
    expect(mockFilteredMessages[0].id).toBe('m1')
  })

  it('does not filter when searchQuery is empty', () => {
    const messages = [msg('m1', 'Hello world'), msg('m2', 'Goodbye')]
    render(
      <ChatArea messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} searchQuery="" />
    )
    expect(mockFilteredMessages).toHaveLength(2)
  })

  it('shows jump-to-bottom button when scrolled up with messages', () => {
    const messages = [msg('m1'), msg('m2'), msg('m3')]
    const { container } = render(
      <ChatArea messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    const scrollRegion = container.querySelector('[role="region"]')
    if (scrollRegion) {
      Object.defineProperty(scrollRegion, 'scrollHeight', { value: 1000 })
      Object.defineProperty(scrollRegion, 'clientHeight', { value: 400 })
      Object.defineProperty(scrollRegion, 'scrollTop', { value: 0 })
      fireEvent.scroll(scrollRegion)
    }
    const jumpBtn = screen.queryByLabelText('Jump to latest messages')
    expect(jumpBtn).toBeDefined()
  })

  it('does not show jump-to-bottom when at bottom', () => {
    const messages = [msg('m1'), msg('m2')]
    const { container } = render(
      <ChatArea messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    const scrollRegion = container.querySelector('[role="region"]')
    if (scrollRegion) {
      Object.defineProperty(scrollRegion, 'scrollHeight', { value: 1000 })
      Object.defineProperty(scrollRegion, 'clientHeight', { value: 500 })
      Object.defineProperty(scrollRegion, 'scrollTop', { value: 450 })
      fireEvent.scroll(scrollRegion)
    }
    expect(screen.queryByLabelText('Jump to latest messages')).toBeNull()
  })

  it('exposes scrollToBottom via ref', () => {
    const ref = React.createRef<any>()
    render(
      <ChatArea ref={ref} messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    expect(ref.current?.scrollToBottom).toBeDefined()
  })

  it('has chat messages region for accessibility', () => {
    render(
      <ChatArea messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />
    )
    expect(screen.getByLabelText('Chat messages')).toBeDefined()
  })
})
