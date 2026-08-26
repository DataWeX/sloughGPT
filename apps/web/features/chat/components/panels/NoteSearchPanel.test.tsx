import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

const mockSearchMessageNotes = vi.fn()

vi.mock('@/lib/db', () => ({
  chatDB: { searchMessageNotes: (...args: any[]) => mockSearchMessageNotes(...args) },
  type MessageNote: {},
}))

vi.mock('@/lib/conversations-utils', () => ({
  truncateMessage: (s: string) => s?.slice(0, 60) || 'Empty conversation',
}))

vi.mock('@/lib/dev-log', () => ({
  logger: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  IconSearch: (p: any) => <svg {...p} />,
  IconX: (p: any) => <svg {...p} />,
}))

import { NoteSearchPanel } from './NoteSearchPanel'

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  onNavigateToNote: vi.fn(),
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  cleanup()
})

describe('NoteSearchPanel', () => {
  it('renders nothing when closed', () => {
    render(<NoteSearchPanel {...defaultProps} open={false} />)
    expect(screen.queryByPlaceholderText(/Search notes/)).toBeNull()
  })

  it('renders when open', () => {
    render(<NoteSearchPanel {...defaultProps} />)
    expect(screen.getByPlaceholderText(/Search notes across all conversations/)).toBeDefined()
  })

  it('shows empty state initially', () => {
    render(<NoteSearchPanel {...defaultProps} />)
    expect(screen.getByText('Type to search notes across all conversations')).toBeDefined()
  })

  it('debounces search and fetches results', async () => {
    mockSearchMessageNotes.mockResolvedValue([
      { sessionId: 'sess-1', messageId: 'msg-1', content: 'Important note', createdAt: Date.now(), updatedAt: Date.now() },
    ])
    render(<NoteSearchPanel {...defaultProps} />)

    const input = screen.getByPlaceholderText(/Search notes/)
    await act(async () => {
      fireEvent.change(input, { target: { value: 'note' } })
      vi.advanceTimersByTime(300)
    })

    await waitFor(() => {
      expect(mockSearchMessageNotes).toHaveBeenCalledWith('note')
    })
    expect(screen.getByText('Important note')).toBeDefined()
    expect(screen.getByText(/1 note found/)).toBeDefined()
  })

  it('shows no results message', async () => {
    mockSearchMessageNotes.mockResolvedValue([])
    render(<NoteSearchPanel {...defaultProps} />)

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: 'nothing' } })
      vi.advanceTimersByTime(300)
    })

    await waitFor(() => {
      expect(screen.getByText(/No notes found for/)).toBeDefined()
    })
  })

  it('handles search error gracefully', async () => {
    mockSearchMessageNotes.mockRejectedValue(new Error('fail'))
    render(<NoteSearchPanel {...defaultProps} />)

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: 'err' } })
      vi.advanceTimersByTime(300)
    })

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /err/ })).toBeNull()
    })
  })

  it('clears results when query is cleared', async () => {
    mockSearchMessageNotes.mockResolvedValue([
      { sessionId: 's1', messageId: 'm1', content: 'Note', createdAt: Date.now(), updatedAt: Date.now() },
    ])
    render(<NoteSearchPanel {...defaultProps} />)

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: 'note' } })
      vi.advanceTimersByTime(300)
    })
    await waitFor(() => expect(screen.getByText('Note')).toBeDefined())

    fireEvent.click(screen.getByLabelText('Clear search'))
    expect(screen.getByText('Type to search notes across all conversations')).toBeDefined()
  })

  it('navigates to note and closes on result click', async () => {
    const onNavigate = vi.fn()
    const onClose = vi.fn()
    mockSearchMessageNotes.mockResolvedValue([
      { sessionId: 'sess-abc', messageId: 'msg-123', content: 'Click me', createdAt: Date.now(), updatedAt: Date.now() },
    ])
    render(<NoteSearchPanel {...defaultProps} onNavigateToNote={onNavigate} onClose={onClose} />)

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: 'click' } })
      vi.advanceTimersByTime(300)
    })

    await waitFor(() => {
      fireEvent.click(screen.getByText('Click me'))
    })
    expect(onNavigate).toHaveBeenCalledWith('sess-abc', 'msg-123')
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on Escape key', () => {
    const onClose = vi.fn()
    render(<NoteSearchPanel {...defaultProps} onClose={onClose} />)
    fireEvent.keyDown(screen.getByPlaceholderText(/Search notes/), { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on backdrop click', () => {
    const onClose = vi.fn()
    render(<NoteSearchPanel {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getByText('Esc'))
    expect(onClose).toHaveBeenCalled()
  })

  it('does not search when query is whitespace only', async () => {
    render(<NoteSearchPanel {...defaultProps} />)
    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: '   ' } })
      vi.advanceTimersByTime(300)
    })
    expect(mockSearchMessageNotes).not.toHaveBeenCalled()
  })

  it('shows note count with plural form', async () => {
    mockSearchMessageNotes.mockResolvedValue([
      { sessionId: 's1', messageId: 'm1', content: 'N1', createdAt: Date.now(), updatedAt: Date.now() },
      { sessionId: 's1', messageId: 'm2', content: 'N2', createdAt: Date.now(), updatedAt: Date.now() },
    ])
    render(<NoteSearchPanel {...defaultProps} />)

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText(/Search notes/), { target: { value: 'test' } })
      vi.advanceTimersByTime(300)
    })

    await waitFor(() => {
      expect(screen.getByText(/2 notes found/)).toBeDefined()
    })
  })
})
