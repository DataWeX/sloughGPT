import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { SmartContextSuggestions } from './SmartContextSuggestions'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const makeMsg = (role: 'user' | 'assistant', content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  timestamp: new Date(),
})

describe('SmartContextSuggestions', () => {
  it('renders nothing when input too short', () => {
    const { container } = render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'Hello react world')]}
        currentInput="he"
        onSelect={vi.fn()}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders nothing when no input', () => {
    const { container } = render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'Hello react world')]}
        currentInput=""
        onSelect={vi.fn()}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('shows concept suggestions', () => {
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'How do I use react hooks?')]}
        currentInput="Tell me about react"
        onSelect={vi.fn()}
      />
    )
    expect(screen.getByText('react')).toBeInTheDocument()
  })

  it('shows file suggestions', () => {
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'I edited file.ts and it works now')]}
        currentInput="What about the file"
        onSelect={vi.fn()}
      />
    )
    expect(screen.getByText('Context:')).toBeInTheDocument()
  })

  it('shows URL suggestions', () => {
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'Visit https://example.com for documentation')]}
        currentInput="Tell me about example.com site"
        onSelect={vi.fn()}
      />
    )
    expect(screen.getByText(/example/)).toBeInTheDocument()
  })

  it('calls onSelect when suggestion clicked', () => {
    const onSelect = vi.fn()
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'How do I use react hooks?')]}
        currentInput="Tell me about react"
        onSelect={onSelect}
      />
    )
    fireEvent.click(screen.getByText('react'))
    expect(onSelect).toHaveBeenCalledWith('react')
  })

  it('dismisses when X clicked', () => {
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'How do I use react hooks?')]}
        currentInput="Tell me about react"
        onSelect={vi.fn()}
      />
    )
    fireEvent.click(screen.getByTitle('Dismiss'))
    expect(screen.queryByText('react')).not.toBeInTheDocument()
  })

  it('shows context label', () => {
    render(
      <SmartContextSuggestions
        messages={[makeMsg('user', 'How do I use react hooks?')]}
        currentInput="Tell me about react"
        onSelect={vi.fn()}
      />
    )
    expect(screen.getByText('Context:')).toBeInTheDocument()
  })

  it('limits suggestions to 5', () => {
    const messages = [
      makeMsg('user', 'react javascript typescript python api database function component state hook'),
    ]
    render(
      <SmartContextSuggestions
        messages={messages}
        currentInput="Tell me about react javascript typescript python api"
        onSelect={vi.fn()}
      />
    )
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeLessThanOrEqual(6)
  })
})