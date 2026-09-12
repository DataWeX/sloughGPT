import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import BrainstormPage from './page'

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

describe('BrainstormPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Brainstorm')).toBeInTheDocument()
  })

  it('renders welcome message', () => {
    render(<BrainstormPage />)
    expect(screen.getByText("Let's think together. What's on your mind?")).toBeInTheDocument()
  })

  it('renders suggestion chips', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Name ideas')).toBeInTheDocument()
    expect(screen.getByText('Weekend plans')).toBeInTheDocument()
    expect(screen.getByText('Gift ideas')).toBeInTheDocument()
    expect(screen.getByText('Solve a problem')).toBeInTheDocument()
    expect(screen.getByText('Plan an event')).toBeInTheDocument()
  })

  it('renders input textarea', () => {
    render(<BrainstormPage />)
    expect(screen.getByPlaceholderText("What's on your mind?")).toBeInTheDocument()
  })

  it('renders send button', () => {
    render(<BrainstormPage />)
    expect(screen.getByText('Send')).toBeInTheDocument()
  })

  it('send button is disabled when no input', () => {
    render(<BrainstormPage />)
    const sendButton = screen.getByText('Send')
    expect(sendButton).toBeDisabled()
  })
})
