import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import { MessageTranslation } from './MessageTranslation'

afterEach(cleanup)

const mockTranslate = vi.fn().mockResolvedValue('Hola mundo')

describe('MessageTranslation', () => {
  it('renders translate button', () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    expect(screen.getByText('Translate')).toBeInTheDocument()
  })

  it('opens translation panel when clicked', () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    expect(screen.getByText('Translate to')).toBeInTheDocument()
    expect(screen.getByText('Spanish')).toBeInTheDocument()
    expect(screen.getByText('English')).toBeInTheDocument()
  })

  it('calls onTranslate with correct args', async () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByText('Translate', { selector: 'button' }))
    await waitFor(() => {
      expect(mockTranslate).toHaveBeenCalledWith('Hello world', 'es')
    })
  })

  it('displays translated text', async () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByText('Translate', { selector: 'button' }))
    expect(await screen.findByText('Hola mundo')).toBeInTheDocument()
  })

  it('allows language selection', () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByText('French'))
    expect(screen.getByText('French')).toHaveClass('bg-primary/20')
  })

  it('shows error on failure', async () => {
    const failingTranslate = vi.fn().mockRejectedValue(new Error('Failed'))
    render(<MessageTranslation content="Hello world" onTranslate={failingTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByText('Translate', { selector: 'button' }))
    expect(await screen.findByText('Translation failed')).toBeInTheDocument()
  })

  it('closes panel and resets', () => {
    render(<MessageTranslation content="Hello world" onTranslate={mockTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(screen.getByText('Translate')).toBeInTheDocument()
    expect(screen.queryByText('Translate to')).not.toBeInTheDocument()
  })

  it('shows loading state', async () => {
    let resolveTranslate: (v: string) => void
    const slowTranslate = vi.fn().mockImplementation(
      () => new Promise(resolve => { resolveTranslate = resolve })
    )
    render(<MessageTranslation content="Hello world" onTranslate={slowTranslate} />)
    fireEvent.click(screen.getByText('Translate'))
    fireEvent.click(screen.getByText('Translate', { selector: 'button' }))
    expect(screen.getByText('Translating...')).toBeInTheDocument()
    resolveTranslate!('Done')
    await waitFor(() => {
      expect(screen.queryByText('Translating...')).not.toBeInTheDocument()
    })
  })
})