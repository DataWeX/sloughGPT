import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import TranslatePage from './page'

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

describe('TranslatePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the page title', () => {
    render(<TranslatePage />)
    expect(screen.getByRole('heading', { name: /translate/i })).toBeInTheDocument()
  })

  it('renders source and target labels', () => {
    render(<TranslatePage />)
    expect(screen.getByText('Source')).toBeInTheDocument()
    expect(screen.getByText('Translation')).toBeInTheDocument()
  })

  it('renders language selector', () => {
    render(<TranslatePage />)
    expect(screen.getByText('Spanish')).toBeInTheDocument()
  })

  it('renders source textarea', () => {
    render(<TranslatePage />)
    expect(screen.getByPlaceholderText('Type or paste text to translate...')).toBeInTheDocument()
  })

  it('renders target textarea', () => {
    render(<TranslatePage />)
    expect(screen.getByPlaceholderText('Translation will appear here...')).toBeInTheDocument()
  })

  it('renders action buttons', () => {
    render(<TranslatePage />)
    expect(screen.getByText('Swap')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /translate/i })).toBeInTheDocument()
    expect(screen.getByText('Copy')).toBeInTheDocument()
  })

  it('swap button swaps source and target', () => {
    render(<TranslatePage />)
    const sourceTextarea = screen.getByPlaceholderText('Type or paste text to translate...')
    const swapButton = screen.getByText('Swap')
    
    fireEvent.change(sourceTextarea, { target: { value: 'Hello' } })
    fireEvent.click(swapButton)
    
    expect(screen.getByDisplayValue('Hello')).toBeInTheDocument()
  })
})
