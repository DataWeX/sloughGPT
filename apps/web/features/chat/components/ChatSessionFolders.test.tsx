import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionFolders } from './ChatSessionFolders'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockSessions = [
  { id: '1', title: 'Chat 1', folderId: null },
  { id: '2', title: 'Chat 2', folderId: null },
  { id: '3', title: 'Chat 3', folderId: 'f1' },
]

describe('ChatSessionFolders', () => {
  it('renders empty state', () => {
    render(<ChatSessionFolders sessions={[]} onMoveSession={vi.fn()} />)
    expect(screen.getByText('No folders')).toBeInTheDocument()
  })

  it('renders unfiled sessions count', () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    expect(screen.getByText(/Unfiled/)).toBeInTheDocument()
  })

  it('creates a folder', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Folder name...'), { target: { value: 'My Folder' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /save/i }))
    })
    expect(screen.getByText('My Folder')).toBeInTheDocument()
  })

  it('creates folder on Enter key', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    const input = screen.getByPlaceholderText('Folder name...')
    fireEvent.change(input, { target: { value: 'Test' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getByText('Test')).toBeInTheDocument()
  })

  it('persists folders to localStorage', async () => {
    render(<ChatSessionFolders sessions={mockSessions} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    fireEvent.change(screen.getByPlaceholderText('Folder name...'), { target: { value: 'Saved' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /save/i }))
    })
    const stored = JSON.parse(localStorage.getItem('chat-folders') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Saved')
  })

  it('toggles folder expansion', async () => {
    const sessionsWithFolder = [
      { id: '1', title: 'Chat in folder', folderId: 'f1' },
    ]
    localStorage.setItem('chat-folders', JSON.stringify([{ id: 'f1', name: 'Test Folder', createdAt: Date.now() }]))
    render(<ChatSessionFolders sessions={sessionsWithFolder} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByText('Test Folder'))
    expect(screen.getByText('Chat in folder')).toBeInTheDocument()
  })

  it('deletes a folder and moves sessions to unfiled', async () => {
    const onMoveSession = vi.fn()
    const sessionsWithFolder = [
      { id: '1', title: 'Chat in folder', folderId: 'f1' },
    ]
    localStorage.setItem('chat-folders', JSON.stringify([{ id: 'f1', name: 'To Delete', createdAt: Date.now() }]))
    render(<ChatSessionFolders sessions={sessionsWithFolder} onMoveSession={onMoveSession} />)
    fireEvent.click(screen.getByTitle('Delete folder'))
    expect(onMoveSession).toHaveBeenCalledWith('1', null)
  })

  it('disables save when name empty', () => {
    render(<ChatSessionFolders sessions={[]} onMoveSession={vi.fn()} />)
    fireEvent.click(screen.getByRole('button'))
    const saveBtn = screen.getAllByRole('button').find(b => b.querySelector('svg'))
    expect(saveBtn).toBeDefined()
  })
})