import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import WellnessPage from './page'

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

describe('WellnessPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<WellnessPage />)
    expect(screen.getByText('Make Me Well')).toBeInTheDocument()
  })

  it('renders welcome message', () => {
    render(<WellnessPage />)
    expect(screen.getByText('Take a moment for yourself. What would feel good right now?')).toBeInTheDocument()
  })

  it('renders wellness options', () => {
    render(<WellnessPage />)
    expect(screen.getByText('Sleep Story')).toBeInTheDocument()
    expect(screen.getByText('Meditation')).toBeInTheDocument()
    expect(screen.getByText('Journal Prompt')).toBeInTheDocument()
    expect(screen.getByText('Breathing Exercise')).toBeInTheDocument()
    expect(screen.getByText('Positive Affirmation')).toBeInTheDocument()
  })

  it('selects option when clicked', () => {
    render(<WellnessPage />)
    const sleepButton = screen.getByText('Sleep Story')
    fireEvent.click(sleepButton)
    expect(sleepButton.closest('.cursor-pointer')).toHaveClass('border-primary')
  })

  it('shows preferences input after selection', () => {
    render(<WellnessPage />)
    const sleepButton = screen.getByText('Sleep Story')
    fireEvent.click(sleepButton)
    expect(screen.getByPlaceholderText(/A story about the ocean/)).toBeInTheDocument()
  })

  it('renders begin button', () => {
    render(<WellnessPage />)
    const sleepButton = screen.getByText('Sleep Story')
    fireEvent.click(sleepButton)
    expect(screen.getByText('Begin')).toBeInTheDocument()
  })
})
