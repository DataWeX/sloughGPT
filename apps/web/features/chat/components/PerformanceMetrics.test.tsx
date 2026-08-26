import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { PerformanceMetrics } from './PerformanceMetrics'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const makeMsg = (role: 'user' | 'assistant', content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  timestamp: Date.now(),
})

describe('PerformanceMetrics', () => {
  it('renders empty state', () => {
    render(<PerformanceMetrics messages={[]} />)
    expect(screen.getByText('No messages yet')).toBeInTheDocument()
  })

  it('calculates total messages', () => {
    const messages = [
      makeMsg('user', 'Hello'),
      makeMsg('assistant', 'Hi there'),
      makeMsg('user', 'How are you?'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('calculates user messages', () => {
    const messages = [
      makeMsg('user', 'Hello'),
      makeMsg('assistant', 'Hi'),
      makeMsg('user', 'Test'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('User Messages')).toBeInTheDocument()
  })

  it('calculates assistant messages', () => {
    const messages = [
      makeMsg('user', 'Hello'),
      makeMsg('assistant', 'Hi there'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('Assistant Messages')).toBeInTheDocument()
  })

  it('calculates average response length', () => {
    const messages = [
      makeMsg('user', 'Hello'),
      makeMsg('assistant', 'Hi'),  // 2 chars
      makeMsg('user', 'Test'),
      makeMsg('assistant', 'Hello world'),  // 11 chars
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('Avg Length')).toBeInTheDocument()
    expect(screen.getByText('7')).toBeInTheDocument() // (2+11)/2 = 6.5, rounded to 7
  })

  it('calculates longest response', () => {
    const messages = [
      makeMsg('assistant', 'Short'),
      makeMsg('assistant', 'This is a longer response'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('Longest')).toBeInTheDocument()
    expect(screen.getByText('25')).toBeInTheDocument()
  })

  it('calculates shortest response', () => {
    const messages = [
      makeMsg('assistant', 'Short'),
      makeMsg('assistant', 'This is a longer response'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('Shortest')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('calculates total characters', () => {
    const messages = [
      makeMsg('user', 'Hello'),  // 5
      makeMsg('assistant', 'World'),  // 5
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText('Total Characters')).toBeInTheDocument()
    expect(screen.getByText('10')).toBeInTheDocument()
  })

  it('calculates response ratio', () => {
    const messages = [
      makeMsg('user', 'Hello'),
      makeMsg('assistant', 'Hi'),
      makeMsg('assistant', 'There'),
    ]
    render(<PerformanceMetrics messages={messages} />)
    expect(screen.getByText(/Response ratio/)).toBeInTheDocument()
    expect(screen.getByText(/2\.0/)).toBeInTheDocument()
  })
})