import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { CardEditor } from './CardEditor'
import type { Card } from './types'

afterEach(() => cleanup())

const makeCard = (overrides: Partial<Card> = {}): Card => ({
  id: 'c1', title: 'My Card', description: 'Card desc', column: 'todo',
  priority: 'high', tags: ['a', 'b'], due_date: '2026-09-10',
  assignee: 'alice', sprint: 's1', gh: 'gh-1', notes: [],
  created_at: '', updated_at: '', root_hash: '', ...overrides,
})

const defaultProps = {
  onClose: vi.fn(),
  onUpdate: vi.fn(),
  onDelete: vi.fn(),
}

beforeEach(() => {
  global.fetch = vi.fn(() => Promise.resolve({ ok: false })) as any
})

describe('CardEditor', () => {
  it('returns null when card is null', () => {
    const { container } = render(<CardEditor card={null} {...defaultProps} />)
    expect(container.innerHTML).toBe('')
  })
  it('renders Edit Card heading', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getAllByText('Edit Card').length).toBeGreaterThanOrEqual(1)
  })
  it('populates title field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('My Card')).toBeDefined()
  })
  it('populates description field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('Card desc')).toBeDefined()
  })
  it('populates column select', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('To Do')).toBeDefined()
  })
  it('populates priority select', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('High')).toBeDefined()
  })
  it('populates tags field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('a, b')).toBeDefined()
  })
  it('populates assignee field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('alice')).toBeDefined()
  })
  it('populates due date field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('2026-09-10')).toBeDefined()
  })
  it('populates sprint field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('s1')).toBeDefined()
  })
  it('populates gh field', () => {
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    expect(screen.getByDisplayValue('gh-1')).toBeDefined()
  })
  it('calls onDelete with card id', () => {
    const onDelete = vi.fn()
    render(<CardEditor card={makeCard()} {...defaultProps} onDelete={onDelete} />)
    fireEvent.click(screen.getAllByText('Delete')[0])
    expect(onDelete).toHaveBeenCalledWith('c1')
  })
  it('calls onClose on Cancel', () => {
    const onClose = vi.fn()
    render(<CardEditor card={makeCard()} {...defaultProps} onClose={onClose} />)
    fireEvent.click(screen.getAllByText('Cancel')[0])
    expect(onClose).toHaveBeenCalled()
  })
  it('calls onSave with edited data', () => {
    const onUpdate = vi.fn()
    render(<CardEditor card={makeCard()} {...defaultProps} onUpdate={onUpdate} />)
    fireEvent.change(screen.getByDisplayValue('My Card'), { target: { value: 'Updated Card' } })
    fireEvent.click(screen.getAllByText('Save')[0])
    expect(onUpdate).toHaveBeenCalledWith('c1', expect.objectContaining({ title: 'Updated Card' }))
  })
  it('loads hash tree on mount', async () => {
    const hashTree = { root: { root: 'abc123def456789012345678901234567890', tray: 'todo' }, notes: [], history: [], commits: [] }
    global.fetch = vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(hashTree) })) as any
    render(<CardEditor card={makeCard()} {...defaultProps} />)
    await screen.findByText('Hash Tree')
    expect(screen.getAllByText('Hash Tree').length).toBeGreaterThanOrEqual(1)
  })
})
