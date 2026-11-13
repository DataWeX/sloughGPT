import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { NoteEditor } from './NoteEditor'
import type { Note } from './types'

afterEach(() => cleanup())

const makeNote = (overrides: Partial<Note> = {}): Note => ({
  id: 'n1', title: 'My Note', body: 'Note body', created_at: '', updated_at: '',
  tags: ['tag1', 'tag2'], status: 'open', sprint: 'sprint-1', gh: 'gh-123', ...overrides,
})

const defaultProps = {
  onClose: vi.fn(),
  onUpdate: vi.fn(),
  onDelete: vi.fn(),
}

describe('NoteEditor', () => {
  it('returns null when note is null', () => {
    const { container } = render(<NoteEditor note={null} {...defaultProps} />)
    expect(container.innerHTML).toBe('')
  })
  it('renders Edit Note heading', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getAllByText('Edit Note').length).toBeGreaterThanOrEqual(1)
  })
  it('populates title field', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getByDisplayValue('My Note')).toBeDefined()
  })
  it('populates body field', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getByDisplayValue('Note body')).toBeDefined()
  })
  it('populates status select', () => {
    render(<NoteEditor note={makeNote({ status: 'wip' })} {...defaultProps} />)
    expect(screen.getByDisplayValue('WIP')).toBeDefined()
  })
  it('populates tags field', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getByDisplayValue('tag1, tag2')).toBeDefined()
  })
  it('populates sprint field', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getByDisplayValue('sprint-1')).toBeDefined()
  })
  it('populates gh field', () => {
    render(<NoteEditor note={makeNote()} {...defaultProps} />)
    expect(screen.getByDisplayValue('gh-123')).toBeDefined()
  })
  it('calls onDelete with note id', () => {
    const onDelete = vi.fn()
    render(<NoteEditor note={makeNote()} {...defaultProps} onDelete={onDelete} />)
    fireEvent.click(screen.getAllByText('Delete')[0])
    expect(onDelete).toHaveBeenCalledWith('n1')
  })
  it('calls onClose on Cancel', () => {
    const onClose = vi.fn()
    render(<NoteEditor note={makeNote()} {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getAllByText('Cancel')[0])
    expect(onClose).toHaveBeenCalled()
  })
  it('calls onSave with edited data', () => {
    const onUpdate = vi.fn()
    render(<NoteEditor note={makeNote()} {...defaultProps} onUpdate={onUpdate} />)
    fireEvent.change(screen.getByDisplayValue('My Note'), { target: { value: 'Updated Title' } })
    fireEvent.click(screen.getAllByText('Save')[0])
    expect(onUpdate).toHaveBeenCalledWith('n1', expect.objectContaining({ title: 'Updated Title' }))
  })
})
