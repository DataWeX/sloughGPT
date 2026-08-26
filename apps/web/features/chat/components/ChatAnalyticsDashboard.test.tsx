import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ChatAnalyticsDashboard } from './ChatAnalyticsDashboard'
import type { ChatMessage } from '@/lib/chat-utils'

afterEach(cleanup)

const makeMsg = (role: 'user' | 'assistant', content: string, hoursAgo: number = 0): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  timestamp: Date.now() - hoursAgo * 3600000,
})

const mockMessages = [
  makeMsg('user', 'Hello world', 2),
  makeMsg('assistant', 'Hi there! How can I help?', 1),
  makeMsg('user', 'Tell me about react hooks', 0),
  makeMsg('assistant', 'React hooks are functions that let you use state and other React features without writing a class', 0),
]

describe('ChatAnalyticsDashboard', () => {
  it('renders empty state', () => {
    render(<ChatAnalyticsDashboard messages={[]} />)
    expect(screen.getByText('No messages to analyze')).toBeInTheDocument()
  })

  it('renders analytics title', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument()
  })

  it('shows message count', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('4')).toBeInTheDocument()
  })

  it('shows time range filters', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('All')).toBeInTheDocument()
    expect(screen.getByText('24h')).toBeInTheDocument()
    expect(screen.getByText('7d')).toBeInTheDocument()
  })

  it('filters by 24h', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    fireEvent.click(screen.getByText('24h'))
    expect(screen.getByText('User Messages')).toBeInTheDocument()
    expect(screen.getByText('AI Messages')).toBeInTheDocument()
  })

  it('filters by 7d', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    fireEvent.click(screen.getByText('7d'))
    expect(screen.getByText('User Messages')).toBeInTheDocument()
    expect(screen.getByText('AI Messages')).toBeInTheDocument()
  })

  it('shows word count', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Words')).toBeInTheDocument()
  })

  it('shows character count', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Characters')).toBeInTheDocument()
  })

  it('shows user and assistant counts', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('User Messages')).toBeInTheDocument()
    expect(screen.getByText('AI Messages')).toBeInTheDocument()
  })

  it('shows top words section', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Top Words')).toBeInTheDocument()
  })

  it('shows hourly chart', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Messages by Hour')).toBeInTheDocument()
  })

  it('shows daily chart', () => {
    render(<ChatAnalyticsDashboard messages={mockMessages} />)
    expect(screen.getByText('Messages by Day')).toBeInTheDocument()
  })
})