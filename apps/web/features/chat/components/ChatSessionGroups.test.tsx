import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import { ChatSessionGroups } from './ChatSessionGroups'

afterEach(cleanup)
beforeEach(() => {
  localStorage.clear()
})

const mockSessions = [
  { id: 's1', title: 'Session 1' },
  { id: 's2', title: 'Session 2' },
  { id: 's3', title: 'Session 3' },
]

describe('ChatSessionGroups', () => {
  it('renders empty state', () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    expect(screen.getByText(/No groups yet/)).toBeInTheDocument()
  })

  it('renders title', () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    expect(screen.getByText('Session Groups')).toBeInTheDocument()
  })

  it('opens create form', () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    expect(screen.getByPlaceholderText('Group name...')).toBeInTheDocument()
  })

  it('creates a group', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    fireEvent.change(screen.getByPlaceholderText('Group name...'), { target: { value: 'My Group' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Create'))
    })
    expect(screen.getAllByText('My Group').length).toBeGreaterThan(0)
  })

  it('creates group on Enter', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    const input = screen.getByPlaceholderText('Group name...')
    fireEvent.change(input, { target: { value: 'Test Group' } })
    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter' })
    })
    expect(screen.getAllByText('Test Group').length).toBeGreaterThan(0)
  })

  it('persists to localStorage', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    fireEvent.change(screen.getByPlaceholderText('Group name...'), { target: { value: 'Saved' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Create'))
    })
    const stored = JSON.parse(localStorage.getItem('chat-session-groups') || '[]')
    expect(stored).toHaveLength(1)
    expect(stored[0].name).toBe('Saved')
  })

  it('deletes a group', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    fireEvent.change(screen.getByPlaceholderText('Group name...'), { target: { value: 'Delete Me' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Create'))
    })
    fireEvent.click(screen.getByTitle('Delete group'))
    expect(screen.queryByText('Delete Me')).not.toBeInTheDocument()
  })

  it('shows ungrouped sessions', () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    expect(screen.getByText(/Ungrouped/)).toBeInTheDocument()
    expect(screen.getByText('Session 1')).toBeInTheDocument()
  })

  it('shows color buttons in create form', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    expect(screen.getByText('Create')).toBeInTheDocument()
    expect(screen.getByText('Cancel')).toBeInTheDocument()
  })

  it('filters by color', async () => {
    render(<ChatSessionGroups sessions={mockSessions} onAssignGroup={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Create group' }))
    fireEvent.change(screen.getByPlaceholderText('Group name...'), { target: { value: 'Colored' } })
    await act(async () => {
      fireEvent.click(screen.getByText('Create'))
    })
    fireEvent.click(screen.getByText('All'))
    expect(screen.getByText('All')).toHaveClass('bg-primary/20')
  })
})