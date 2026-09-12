import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DecidePage from './page'

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

describe('DecidePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<DecidePage />)
    expect(screen.getByRole('heading', { name: 'Help Me Decide' })).toBeInTheDocument()
  })

  it('renders the question input', () => {
    render(<DecidePage />)
    expect(screen.getByPlaceholderText('e.g. Should I take the job in New York or stay?')).toBeInTheDocument()
  })

  it('renders option inputs', () => {
    render(<DecidePage />)
    expect(screen.getByPlaceholderText('First option')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Second option')).toBeInTheDocument()
  })

  it('renders notes textareas', () => {
    render(<DecidePage />)
    expect(screen.getAllByPlaceholderText('Notes (optional)').length).toBe(2)
  })

  it('renders the decide button', () => {
    render(<DecidePage />)
    expect(screen.getAllByText('Help Me Decide').length).toBeGreaterThan(0)
  })

  it('decide button is disabled when fields are empty', () => {
    render(<DecidePage />)
    const decideButtons = screen.getAllByText('Help Me Decide')
    const decideButton = decideButtons[decideButtons.length - 1]
    expect(decideButton).toBeDisabled()
  })
})
