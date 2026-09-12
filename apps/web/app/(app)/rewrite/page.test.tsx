import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import RewritePage from './page'

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

describe('RewritePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<RewritePage />)
    expect(screen.getByText('Rewrite & Polish')).toBeInTheDocument()
  })

  it('renders original and rewritten labels', () => {
    render(<RewritePage />)
    expect(screen.getByText('Original')).toBeInTheDocument()
    expect(screen.getByText('Rewritten')).toBeInTheDocument()
  })

  it('renders original textarea', () => {
    render(<RewritePage />)
    expect(screen.getByPlaceholderText('Paste what you wrote...')).toBeInTheDocument()
  })

  it('renders rewritten textarea', () => {
    render(<RewritePage />)
    expect(screen.getByPlaceholderText('Rewritten version will appear here...')).toBeInTheDocument()
  })

  it('renders action buttons', () => {
    render(<RewritePage />)
    expect(screen.getByText('Fix Grammar')).toBeInTheDocument()
    expect(screen.getByText('Make Shorter')).toBeInTheDocument()
    expect(screen.getByText('Make Friendlier')).toBeInTheDocument()
    expect(screen.getByText('Make Professional')).toBeInTheDocument()
    expect(screen.getByText('Sound Like Me')).toBeInTheDocument()
  })

  it('action buttons are disabled when no input', () => {
    render(<RewritePage />)
    const fixGrammarButton = screen.getByText('Fix Grammar')
    expect(fixGrammarButton).toBeDisabled()
  })
})
