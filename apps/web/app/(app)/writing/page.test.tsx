import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import WritingAssistantPage from './page'

vi.mock('@/lib/chat-controller', () => ({
  chatController: {
    stream: vi.fn(),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: vi.fn(() => ({
    addToast: vi.fn(),
  })),
}))

describe('WritingAssistantPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Writing Assistant')).toBeInTheDocument()
  })

  it('renders tone buttons', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Friendly')).toBeInTheDocument()
    expect(screen.getByText('Professional')).toBeInTheDocument()
    expect(screen.getByText('Funny')).toBeInTheDocument()
    expect(screen.getByText('Short')).toBeInTheDocument()
    expect(screen.getByText('Detailed')).toBeInTheDocument()
  })

  it('renders type buttons', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Email')).toBeInTheDocument()
    expect(screen.getByText('Social Post')).toBeInTheDocument()
    expect(screen.getByText('Story')).toBeInTheDocument()
    expect(screen.getByText('Poem')).toBeInTheDocument()
    expect(screen.getByText('Letter')).toBeInTheDocument()
    expect(screen.getByText('Note')).toBeInTheDocument()
  })

  it('renders the input textarea', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByPlaceholderText('Tell me what you want to write about...')).toBeInTheDocument()
  })

  it('renders the Write button', () => {
    render(<WritingAssistantPage />)
    expect(screen.getByText('Write')).toBeInTheDocument()
  })

  it('selects tone when clicked', () => {
    render(<WritingAssistantPage />)
    const friendlyButton = screen.getByText('Friendly')
    fireEvent.click(friendlyButton)
    expect(friendlyButton.closest('button')).toHaveClass('bg-primary')
  })

  it('selects type when clicked', () => {
    render(<WritingAssistantPage />)
    const emailButton = screen.getByText('Email')
    fireEvent.click(emailButton)
    expect(emailButton.closest('button')).toHaveClass('bg-primary')
  })
})
