import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { MessageNotes } from './MessageNotes'
import { chatDB } from '@/lib/db'

afterEach(cleanup)

vi.mock('@/lib/db', () => ({
  chatDB: {
    getMessageNotes: vi.fn().mockResolvedValue([]),
    saveMessageNote: vi.fn().mockResolvedValue(undefined),
    removeMessageNote: vi.fn().mockResolvedValue(undefined),
  },
}))

describe('MessageNotes', () => {
  it('renders add note button when no note exists', () => {
    render(<MessageNotes sessionId="s1" messageId="m1" />)
    expect(screen.getByText('Add note')).toBeInTheDocument()
  })

  it('opens editor when add note clicked', () => {
    render(<MessageNotes sessionId="s1" messageId="m1" />)
    fireEvent.click(screen.getByText('Add note'))
    expect(screen.getByPlaceholderText('Add a note...')).toBeInTheDocument()
  })

  it('renders existing note', async () => {
    vi.mocked(chatDB.getMessageNotes).mockResolvedValue([{
      id: 's1:m1',
      sessionId: 's1',
      messageId: 'm1',
      content: 'My note',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }])
    render(<MessageNotes sessionId="s1" messageId="m1" />)
    expect(await screen.findByText('My note')).toBeInTheDocument()
  })

  it('does not render when sessionId is null', () => {
    const { container } = render(<MessageNotes sessionId={null} messageId="m1" />)
    expect(container.firstChild).toBeNull()
  })

  it('saves note on Ctrl+Enter', async () => {
    render(<MessageNotes sessionId="s1" messageId="m1" />)
    fireEvent.click(screen.getByText('Add note'))
    const textarea = screen.getByPlaceholderText('Add a note...')
    fireEvent.change(textarea, { target: { value: 'New note' } })
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true })
    expect(chatDB.saveMessageNote).toHaveBeenCalledWith(expect.objectContaining({
      content: 'New note',
      messageId: 'm1',
    }))
  })

  it('cancels on Escape', () => {
    render(<MessageNotes sessionId="s1" messageId="m1" />)
    fireEvent.click(screen.getByText('Add note'))
    const textarea = screen.getByPlaceholderText('Add a note...')
    fireEvent.keyDown(textarea, { key: 'Escape' })
    expect(screen.getByText('Add note')).toBeInTheDocument()
  })
})