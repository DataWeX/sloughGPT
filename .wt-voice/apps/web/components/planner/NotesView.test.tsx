import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { NotesView } from './NotesView'
import type { Note } from './types'

afterEach(() => cleanup())

const makeNote = (overrides: Partial<Note> = {}): Note => ({
  id: 'n1', title: 'Test Note', body: 'Some content', created_at: '', updated_at: '',
  tags: ['python', 'backend'], status: 'open', sprint: 's1', gh: '', ...overrides,
})

const defaultProps = { onRefresh: vi.fn() }

describe('NotesView', () => {
  it('renders notes', () => {
    render(<NotesView notes={[makeNote()]} {...defaultProps} />)
    expect(screen.getAllByText('Test Note').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Some content').length).toBeGreaterThanOrEqual(1)
  })
  it('shows No notes found when empty', () => {
    render(<NotesView notes={[]} {...defaultProps} />)
    expect(screen.getAllByText('No notes found').length).toBeGreaterThanOrEqual(1)
  })
  it('renders note tags', () => {
    render(<NotesView notes={[makeNote({ tags: ['a', 'b'] })]} {...defaultProps} />)
    expect(screen.getAllByText('a').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('b').length).toBeGreaterThanOrEqual(1)
  })
  it('shows +N for more than 3 tags', () => {
    render(<NotesView notes={[makeNote({ tags: ['a', 'b', 'c', 'd'] })]} {...defaultProps} />)
    expect(screen.getAllByText('+1').length).toBeGreaterThanOrEqual(1)
  })
  it('renders status badge', () => {
    render(<NotesView notes={[makeNote({ status: 'done' })]} {...defaultProps} />)
    expect(screen.getAllByText('done').length).toBeGreaterThanOrEqual(1)
  })
  it('filters by search query', () => {
    render(<NotesView notes={[makeNote({ title: 'Alpha' }), makeNote({ id: 'n2', title: 'Beta' })]} {...defaultProps} />)
    fireEvent.change(screen.getByPlaceholderText('Search notes...'), { target: { value: 'Alpha' } })
    expect(screen.getAllByText('Alpha').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText('Beta')).toBeNull()
  })
  it('searches in body', () => {
    render(<NotesView notes={[makeNote({ body: 'unique body text' })]} {...defaultProps} />)
    fireEvent.change(screen.getByPlaceholderText('Search notes...'), { target: { value: 'unique' } })
    expect(screen.getAllByText('unique body text').length).toBeGreaterThanOrEqual(1)
  })
  it('searches in tags', () => {
    render(<NotesView notes={[makeNote({ tags: ['searchable-tag'] })]} {...defaultProps} />)
    fireEvent.change(screen.getByPlaceholderText('Search notes...'), { target: { value: 'searchable' } })
    expect(screen.getAllByText('searchable-tag').length).toBeGreaterThanOrEqual(1)
  })
  it('filters by status', () => {
    render(<NotesView notes={[makeNote({ status: 'open' }), makeNote({ id: 'n2', status: 'done' })]} {...defaultProps} />)
    fireEvent.click(screen.getAllByText('done')[0])
    expect(screen.getAllByText('done').length).toBeGreaterThanOrEqual(1)
  })
  it('shows All button', () => {
    render(<NotesView notes={[makeNote()]} {...defaultProps} />)
    expect(screen.getAllByText('All').length).toBeGreaterThanOrEqual(1)
  })
})
