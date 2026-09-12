import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ExplainPage from './page'

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

describe('ExplainPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<ExplainPage />)
    expect(screen.getByText('Explain Things Simply')).toBeInTheDocument()
  })

  it('renders the topic input', () => {
    render(<ExplainPage />)
    expect(screen.getByPlaceholderText('e.g. How does the internet work?')).toBeInTheDocument()
  })

  it('renders difficulty buttons', () => {
    render(<ExplainPage />)
    expect(screen.getByText('Simple')).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
    expect(screen.getByText('Detailed')).toBeInTheDocument()
  })

  it('renders the explain button', () => {
    render(<ExplainPage />)
    expect(screen.getByText('Explain')).toBeInTheDocument()
  })

  it('explain button is disabled when no input', () => {
    render(<ExplainPage />)
    const explainButton = screen.getByText('Explain')
    expect(explainButton).toBeDisabled()
  })

  it('selects difficulty when clicked', () => {
    render(<ExplainPage />)
    const simpleButton = screen.getByText('Simple')
    fireEvent.click(simpleButton)
    expect(simpleButton.closest('button')).toHaveClass('bg-primary')
  })
})
