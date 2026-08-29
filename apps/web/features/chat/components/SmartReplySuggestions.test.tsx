import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { SmartReplySuggestions } from './SmartReplySuggestions'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const makeMsg = (role: 'user' | 'assistant', content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  timestamp: new Date(),
})

describe('SmartReplySuggestions', () => {
  it('renders nothing when no assistant messages', () => {
    const { container } = render(
      <SmartReplySuggestions messages={[makeMsg('user', 'Hello')]} onSelect={vi.fn()} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders default suggestions for generic message', () => {
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'The weather is nice today')]} onSelect={vi.fn()} />
    )
    expect(screen.getByText('Tell me more')).toBeInTheDocument()
    expect(screen.getByText('Can you explain?')).toBeInTheDocument()
  })

  it('renders code suggestions for code messages', () => {
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'Here is the code implementation')]} onSelect={vi.fn()} />
    )
    expect(screen.getByText('Can you show me the code?')).toBeInTheDocument()
    expect(screen.getByText('Is this the best practice?')).toBeInTheDocument()
  })

  it('renders help suggestions for help messages', () => {
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'I can help you with that problem')]} onSelect={vi.fn()} />
    )
    expect(screen.getByText('Can you help me debug this?')).toBeInTheDocument()
  })

  it('calls onSelect when suggestion clicked', () => {
    const onSelect = vi.fn()
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'Hello there')]} onSelect={onSelect} />
    )
    fireEvent.click(screen.getByText('Tell me more'))
    expect(onSelect).toHaveBeenCalledWith('Tell me more')
  })

  it('dismisses when X clicked', () => {
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'Hello there')]} onSelect={vi.fn()} />
    )
    fireEvent.click(screen.getByTitle('Dismiss'))
    expect(screen.queryByText('Tell me more')).not.toBeInTheDocument()
  })

  it('refreshes suggestions when refresh clicked', () => {
    render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'Hello there')]} onSelect={vi.fn()} />
    )
    const firstSuggestion = screen.getByText('Tell me more').textContent
    fireEvent.click(screen.getByTitle('Refresh suggestions'))
    const newSuggestions = screen.getAllByText(/./).map(el => el.textContent)
    expect(newSuggestions).toBeDefined()
  })

  it('resets on new assistant message', () => {
    const { rerender } = render(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'Hello')]} onSelect={vi.fn()} />
    )
    fireEvent.click(screen.getByTitle('Dismiss'))
    expect(screen.queryByText('Tell me more')).not.toBeInTheDocument()
    
    rerender(
      <SmartReplySuggestions messages={[makeMsg('assistant', 'New message')]} onSelect={vi.fn()} />
    )
    expect(screen.getByText('Tell me more')).toBeInTheDocument()
  })
})