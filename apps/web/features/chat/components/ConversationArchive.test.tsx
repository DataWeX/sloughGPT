import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { ConversationArchive } from './ConversationArchive'

afterEach(cleanup)

const mockConversations = [
  { id: '1', title: 'Chat about React', messageCount: 12, lastActivity: Date.now(), archived: false },
  { id: '2', title: 'Old discussion', messageCount: 5, lastActivity: Date.now() - 86400000, archived: true, archivedAt: Date.now() - 43200000 },
  { id: '3', title: 'Recent chat', messageCount: 3, lastActivity: Date.now(), archived: false },
]

describe('ConversationArchive', () => {
  it('renders stats', () => {
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('(2 active, 1 archived)')).toBeInTheDocument()
  })

  it('shows all conversations by default', () => {
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('Chat about React')).toBeInTheDocument()
    expect(screen.getByText('Old discussion')).toBeInTheDocument()
    expect(screen.getByText('Recent chat')).toBeInTheDocument()
  })

  it('filters active conversations', () => {
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    fireEvent.click(screen.getByText('active'))
    expect(screen.getByText('Chat about React')).toBeInTheDocument()
    expect(screen.queryByText('Old discussion')).not.toBeInTheDocument()
    expect(screen.getByText('Recent chat')).toBeInTheDocument()
  })

  it('filters archived conversations', () => {
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    fireEvent.click(screen.getByText('archived'))
    expect(screen.queryByText('Chat about React')).not.toBeInTheDocument()
    expect(screen.getByText('Old discussion')).toBeInTheDocument()
    expect(screen.queryByText('Recent chat')).not.toBeInTheDocument()
  })

  it('calls onArchive when archive button clicked', () => {
    const onArchive = vi.fn()
    render(<ConversationArchive conversations={mockConversations} onArchive={onArchive} onRestore={vi.fn()} onDelete={vi.fn()} />)
    fireEvent.click(screen.getAllByTitle('Archive')[0])
    expect(onArchive).toHaveBeenCalledWith('1')
  })

  it('calls onRestore when restore button clicked', () => {
    const onRestore = vi.fn()
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={onRestore} onDelete={vi.fn()} />)
    fireEvent.click(screen.getByTitle('Restore'))
    expect(onRestore).toHaveBeenCalledWith('2')
  })

  it('requires double-click to delete', () => {
    const onDelete = vi.fn()
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={onDelete} />)
    const deleteBtn = screen.getAllByTitle('Delete')[0]
    fireEvent.click(deleteBtn)
    expect(onDelete).not.toHaveBeenCalled()
    fireEvent.click(deleteBtn)
    expect(onDelete).toHaveBeenCalledWith('1')
  })

  it('shows empty state', () => {
    render(<ConversationArchive conversations={[]} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText('No conversations')).toBeInTheDocument()
  })

  it('shows archived date for archived conversations', () => {
    render(<ConversationArchive conversations={mockConversations} onArchive={vi.fn()} onRestore={vi.fn()} onDelete={vi.fn()} />)
    expect(screen.getByText(/Archived/)).toBeInTheDocument()
  })
})