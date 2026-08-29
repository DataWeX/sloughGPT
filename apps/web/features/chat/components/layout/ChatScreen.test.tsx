import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('./../messages/MessageBubble', () => ({ MessageBubble: ({ content, role }: any) => <div data-testid="msg-bubble" data-role={role}>{content}</div> }))
vi.mock('./../messages/EmptyState', () => ({ EmptyState: ({ hasModel, onSuggestionClick }: any) => <div data-testid="empty-state" data-has-model={hasModel} /> }))
vi.mock('./../messages/SystemBanner', () => ({ SystemBanner: ({ type, title }: any) => <div data-testid="sys-banner" data-type={type}>{title}</div> }))

import { ChatScreen } from './ChatScreen'

const baseHealth = { status: 'ok', model_loaded: true, model_type: 'gpt2' } as any
const msg = (id: string, role: 'user' | 'assistant' = 'user', content = 'hello') => ({ id, role, content, timestamp: new Date() })

describe('ChatScreen', () => {
  afterEach(cleanup)

  it('renders empty state when no messages', () => {
    render(<ChatScreen messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByTestId('empty-state')).toBeDefined()
  })

  it('renders message bubbles', () => {
    const messages = [msg('m1', 'user', 'Hello'), msg('m2', 'assistant', 'Hi there')]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    const bubbles = screen.getAllByTestId('msg-bubble')
    expect(bubbles).toHaveLength(2)
    expect(bubbles[0].textContent).toContain('Hello')
    expect(bubbles[1].textContent).toContain('Hi there')
  })

  it('renders system banner when offline', () => {
    render(<ChatScreen messages={[]} loading={false} health={'offline' as any} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByTestId('sys-banner')).toBeDefined()
    expect(screen.getByText('Service Unavailable')).toBeDefined()
  })

  it('does not show empty state when offline', () => {
    render(<ChatScreen messages={[]} loading={false} health={'offline' as any} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.queryByTestId('empty-state')).toBeNull()
  })

  it('shows session loading skeleton', () => {
    render(<ChatScreen messages={[]} loading={false} sessionLoading={true} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByLabelText('Loading messages')).toBeDefined()
  })

  it('does not show empty state during session loading', () => {
    render(<ChatScreen messages={[]} loading={false} sessionLoading={true} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.queryByTestId('empty-state')).toBeNull()
  })

  it('shows thinking indicator when loading and last message is user', () => {
    const messages = [msg('m1', 'user', 'Tell me something')]
    render(<ChatScreen messages={messages} loading={true} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByText('Reasoning')).toBeDefined()
  })

  it('shows suggestion buttons after assistant message', () => {
    const messages = [msg('m1', 'user', 'What is Python?'), msg('m2', 'assistant', 'A language')]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} onSuggestionClick={vi.fn()} />)
    expect(screen.getByText('Tell me more')).toBeDefined()
    expect(screen.getByText('Give an example')).toBeDefined()
    expect(screen.getByText("Why is that?")).toBeDefined()
  })

  it('shows code-related suggestions when last message contains code block', () => {
    const messages = [msg('m1', 'user', 'How do I sort?'), msg('m2', 'assistant', 'Use sorted()\n```python\nsorted(list)\n```')]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} onSuggestionClick={vi.fn()} />)
    expect(screen.getByText('Explain this code')).toBeDefined()
  })

  it('shows summarize suggestions for long messages', () => {
    const longContent = 'a'.repeat(250)
    const messages = [msg('m1', 'user', 'What?'), msg('m2', 'assistant', longContent)]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} onSuggestionClick={vi.fn()} />)
    expect(screen.getByText('Summarize this')).toBeDefined()
  })

  it('has message feed role for accessibility', () => {
    render(<ChatScreen messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByRole('feed')).toBeDefined()
  })

  it('passes hasModel to empty state', () => {
    render(<ChatScreen messages={[]} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByTestId('empty-state').getAttribute('data-has-model')).toBe('true')
  })

  it('renders date separator between same-day messages', () => {
    const now = new Date()
    const messages = [
      { id: 'm1', role: 'user' as const, content: 'Hi', timestamp: now },
      { id: 'm2', role: 'assistant' as const, content: 'Hello', timestamp: now },
    ]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    expect(screen.getByText('Today')).toBeDefined()
  })

  it('renders date separator only once per day', () => {
    const yesterday = new Date(Date.now() - 86400000)
    const today = new Date()
    const messages = [
      { id: 'm1', role: 'user' as const, content: 'Old', timestamp: yesterday },
      { id: 'm2', role: 'assistant' as const, content: 'Reply', timestamp: yesterday },
      { id: 'm3', role: 'user' as const, content: 'New', timestamp: today },
    ]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    const separators = screen.getAllByRole('separator')
    expect(separators).toHaveLength(2)
    expect(separators[0]).toHaveAttribute('aria-label', 'Yesterday')
    expect(separators[1]).toHaveAttribute('aria-label', 'Today')
  })

  it('renders all messages without duplicate key issues', () => {
    const messages = Array.from({ length: 20 }, (_, i) => msg(`m${i}`, i % 2 === 0 ? 'user' : 'assistant', `Message ${i}`))
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    const bubbles = screen.getAllByTestId('msg-bubble')
    expect(bubbles).toHaveLength(20)
  })

  it('renders interleaved user/assistant messages in correct order', () => {
    const messages = [
      msg('m1', 'user', 'Hello'),
      msg('m2', 'assistant', 'Hi there'),
      msg('m3', 'user', 'How are you?'),
      msg('m4', 'assistant', 'Good, thanks!'),
    ]
    render(<ChatScreen messages={messages} loading={false} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    const bubbles = screen.getAllByTestId('msg-bubble')
    expect(bubbles[0].textContent).toContain('Hello')
    expect(bubbles[1].textContent).toContain('Hi there')
    expect(bubbles[2].textContent).toContain('How are you?')
    expect(bubbles[3].textContent).toContain('Good, thanks!')
  })

  it('shows streaming indicator on last assistant message during loading', () => {
    const messages = [msg('m1', 'user', 'Hi'), msg('m2', 'assistant', 'Hello')]
    render(<ChatScreen messages={messages} loading={true} health={baseHealth} onRefreshHealth={vi.fn()} onCopy={vi.fn()} />)
    const bubbles = screen.getAllByTestId('msg-bubble')
    expect(bubbles).toHaveLength(2)
  })
})
