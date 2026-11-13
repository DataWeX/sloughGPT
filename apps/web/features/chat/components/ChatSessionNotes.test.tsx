import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionNotes } from './ChatSessionNotes'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

describe('ChatSessionNotes', () => {
  it('renders empty state', () => {
    render(<ChatSessionNotes sessionId="s1" />)
    expect(screen.getByText(/No notes yet/)).toBeInTheDocument()
  })

  it('renders title', () => {
    render(<ChatSessionNotes sessionId="s1" />)
    expect(screen.getByText('Session Notes')).toBeInTheDocument()
  })

  it('opens new note form', () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    expect(screen.getByPlaceholderText('Add a note...')).toBeInTheDocument()
  })

  it('adds a note', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'My note' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    expect(screen.getByText('My note')).toBeInTheDocument()
  })

  it('persists notes to localStorage', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'Persisted note' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    const stored = JSON.parse(localStorage.getItem('chat-session-notes') || '{}')
    expect(stored.s1).toHaveLength(1)
    expect(stored.s1[0].content).toBe('Persisted note')
  })

  it('deletes a note', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'Delete me' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    fireEvent.click(screen.getByTitle('Delete'))
    expect(screen.queryByText('Delete me')).not.toBeInTheDocument()
  })

  it('edits a note', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'Original' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    fireEvent.click(screen.getByTitle('Edit'))
    const textarea = screen.getByRole('textbox')
    fireEvent.change(textarea, { target: { value: 'Edited' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Save').closest('div')!.querySelector('button')!)
    })
    expect(screen.getByText('Edited')).toBeInTheDocument()
  })

  it('cancels edit on Escape', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'Test' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    fireEvent.click(screen.getByTitle('Edit'))
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' })
    expect(screen.getByText('Test')).toBeInTheDocument()
  })

  it('shows note count', () => {
    render(<ChatSessionNotes sessionId="s1" />)
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('formats date', async () => {
    render(<ChatSessionNotes sessionId="s1" />)
    fireEvent.click(screen.getByText('+ New'))
    fireEvent.change(screen.getByPlaceholderText('Add a note...'), {
      target: { value: 'Dated note' },
    })
    await act(async () => {
      fireEvent.click(screen.getByText('Save'))
    })
    expect(screen.getByText('Dated note')).toBeInTheDocument()
  })

  it('loads notes for specific session', () => {
    const stored = {
      s1: [
        {
          id: 'n1',
          content: 'Existing note',
          createdAt: Date.now(),
          updatedAt: Date.now(),
        },
      ],
    }
    localStorage.setItem('chat-session-notes', JSON.stringify(stored))
    render(<ChatSessionNotes sessionId="s1" />)
    expect(screen.getByText('Existing note')).toBeInTheDocument()
  })

  it('shows empty state for different session', () => {
    const stored = {
      s1: [
        {
          id: 'n1',
          content: 'Other session note',
          createdAt: Date.now(),
          updatedAt: Date.now(),
        },
      ],
    }
    localStorage.setItem('chat-session-notes', JSON.stringify(stored))
    render(<ChatSessionNotes sessionId="s2" />)
    expect(screen.getByText(/No notes yet/)).toBeInTheDocument()
  })
})