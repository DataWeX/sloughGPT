import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockSearchAllSessions = vi.hoisted(() => vi.fn())

vi.mock('@/lib/db', () => ({
  chatDB: { searchAllSessions: mockSearchAllSessions },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: { search: vi.fn().mockRejectedValue(new Error('no server')) },
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Input: (props: any) => <input data-testid="search-input" {...props} />,
    Button: ({ children, onClick, variant, size, ...rest }: any) => (
      <button onClick={onClick} data-variant={variant} data-size={size} {...rest}>{children}</button>
    ),
    IconSearch: iconMock('search'),
    IconX: iconMock('x'),
    IconMessage: iconMock('message'),
  }
})

import { ConversationSearch } from './ConversationSearch'

describe('ConversationSearch', () => {
  const onClose = vi.fn()
  const onNavigate = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('returns null when not open', () => {
    const { container } = render(
      <ConversationSearch open={false} onClose={onClose} onNavigate={onNavigate} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders search dialog when open', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByLabelText('Search all conversations')).toBeDefined()
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('shows placeholder when query is empty', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByText('Type to search across all conversations')).toBeDefined()
  })

  it('shows Esc hint', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByText('Esc')).toBeDefined()
  })

  it('shows loading spinner while searching', () => {
    mockSearchAllSessions.mockReturnValue(new Promise(() => {}))
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeDefined()
  })

  it('shows no results message', async () => {
    mockSearchAllSessions.mockResolvedValue([])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText(/No results for/)).toBeDefined()
    })
  })

  it('renders search results', async () => {
    mockSearchAllSessions.mockResolvedValue([
      {
        session: { id: 's1', name: 'Test Chat', updated_at: 0, message_count: 2 },
        matches: [{ content: 'hello world' }],
      },
    ])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText('1 conversation found')).toBeDefined()
      expect(screen.getByText('Test Chat')).toBeDefined()
      expect(screen.getByText('1 match')).toBeDefined()
    })
  })

  it('navigates and closes on result click', async () => {
    mockSearchAllSessions.mockResolvedValue([
      {
        session: { id: 's1', name: 'Test Chat', updated_at: 0, message_count: 2 },
        matches: [{ content: 'hello world' }],
      },
    ])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText('Test Chat')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Test Chat'))
    expect(onNavigate).toHaveBeenCalledWith('s1')
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape key', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose when backdrop clicked', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    fireEvent.click(screen.getByLabelText('Search all conversations').parentElement!)
    expect(onClose).toHaveBeenCalled()
  })

  it('shows clear button when query exists', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    expect(screen.getByLabelText('Clear')).toBeDefined()
  })

  it('clears query on clear button click', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hello' } })
    fireEvent.click(screen.getByLabelText('Clear'))
    expect(input.value).toBe('')
  })

  it('resets on close', () => {
    const { rerender } = render(
      <ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />
    )
    const input = screen.getByTestId('search-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hello' } })
    rerender(<ConversationSearch open={false} onClose={onClose} onNavigate={onNavigate} />)
    const { container } = render(
      <ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />
    )
    const newInput = container.querySelector('[data-testid="search-input"]') as HTMLInputElement
    expect(newInput.value).toBe('')
  })
})
