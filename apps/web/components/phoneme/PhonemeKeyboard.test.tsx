import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import PhonemeKeyboard from './PhonemeKeyboard'

describe('PhonemeKeyboard', () => {
  const mockOnInsert = vi.fn()
  const mockOnBackspace = vi.fn()
  const mockOnClear = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without errors', () => {
    render(<PhonemeKeyboard onInsert={mockOnInsert} onBackspace={mockOnBackspace} onClear={mockOnClear} />)
    expect(screen.getByText('Phoneme Keyboard')).toBeDefined()
  })

  it('calls onInsert when phoneme is clicked', () => {
    render(<PhonemeKeyboard onInsert={mockOnInsert} onBackspace={mockOnBackspace} onClear={mockOnClear} />)
    const buttons = screen.getAllByText('P')
    fireEvent.click(buttons[0])
    expect(mockOnInsert).toHaveBeenCalledWith('P')
  })

  it('calls onBackspace when backspace is clicked', () => {
    render(<PhonemeKeyboard onInsert={mockOnInsert} onBackspace={mockOnBackspace} onClear={mockOnClear} />)
    const buttons = screen.getAllByText('⌫')
    fireEvent.click(buttons[0])
    expect(mockOnBackspace).toHaveBeenCalled()
  })

  it('calls onClear when clear is clicked', () => {
    render(<PhonemeKeyboard onInsert={mockOnInsert} onBackspace={mockOnBackspace} onClear={mockOnClear} />)
    const buttons = screen.getAllByText('Clear')
    fireEvent.click(buttons[0])
    expect(mockOnClear).toHaveBeenCalled()
  })
})
