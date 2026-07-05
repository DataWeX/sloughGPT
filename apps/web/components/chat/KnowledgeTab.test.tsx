import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import React from 'react'

const mockKnowledgeItems: Array<{ id: string; content: string; source?: string }> = []
const mockKnowledgeController = {
  list: vi.fn().mockResolvedValue(mockKnowledgeItems),
  batchIngest: vi.fn().mockResolvedValue({}),
  delete: vi.fn().mockResolvedValue({}),
}

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: mockKnowledgeController,
}))

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, variant, size, className, ...rest }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
}))

vi.mock('@/components/ui', () => ({
  IconX: () => <span data-testid="icon-x">x</span>,
}))

import { KnowledgeTab } from './KnowledgeTab'

const STORAGE_KEY = 'man_injected_knowledge'

describe('KnowledgeTab', () => {
  const onOpenConversationViewer = vi.fn()
  const onOpenSettings = vi.fn()
  const onOpenShortcuts = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  afterEach(cleanup)

  it('shows empty state', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText(/No knowledge stored/)).toBeDefined()
  })

  it('shows 0 snippets count', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('shows 1 snippet count', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: 'fact one', timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('1 snippet')).toBeDefined()
  })

  it('loads knowledge from localStorage', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: 'fact one', timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('fact one')).toBeDefined()
  })

  it('shows add form when + Add clicked', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    fireEvent.click(screen.getByText('+ Add'))
    expect(screen.getByPlaceholderText(/Enter a fact/)).toBeDefined()
  })

  it('adds knowledge item', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    fireEvent.click(screen.getByText('+ Add'))
    const textarea = screen.getByPlaceholderText(/Enter a fact/)
    fireEvent.change(textarea, { target: { value: 'my fact' } })
    fireEvent.click(screen.getByText('Save'))
    expect(screen.getByText('my fact')).toBeDefined()
    expect(screen.getByText('1 snippet')).toBeDefined()
  })

  it('cancels add knowledge', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    fireEvent.click(screen.getByText('+ Add'))
    fireEvent.change(screen.getByPlaceholderText(/Enter a fact/), { target: { value: 'will cancel' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('will cancel')).toBeNull()
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('removes knowledge item', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: 'remove me', timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('remove me')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Remove knowledge'))
    expect(screen.queryByText('remove me')).toBeNull()
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('edits knowledge item', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: 'edit me', timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('edit me')).toBeDefined()
    fireEvent.click(screen.getByLabelText('Edit knowledge'))
    const editArea = screen.getByLabelText('Edit knowledge snippet')
    fireEvent.change(editArea, { target: { value: 'edited content' } })
    fireEvent.click(screen.getByText('Save'))
    expect(screen.getByText('edited content')).toBeDefined()
    expect(screen.queryByText('edit me')).toBeNull()
  })

  it('cancels edit', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: 'original', timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    fireEvent.click(screen.getByLabelText('Edit knowledge'))
    fireEvent.change(screen.getByLabelText('Edit knowledge snippet'), { target: { value: 'changed' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.getByText('original')).toBeDefined()
    expect(screen.queryByText('changed')).toBeNull()
  })

  it('clears all knowledge', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([
      { id: 'k1', content: 'fact one', timestamp: Date.now() },
      { id: 'k2', content: 'fact two', timestamp: Date.now() },
    ]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    expect(screen.getByText('2 snippets')).toBeDefined()
    fireEvent.click(screen.getByText('Clear all'))
    expect(screen.getByText('0 snippets')).toBeDefined()
    expect(screen.getByText(/No knowledge stored/)).toBeDefined()
  })

  it('does not add empty knowledge', () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    fireEvent.click(screen.getByText('+ Add'))
    fireEvent.click(screen.getByText('Save'))
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('truncates long content at 200 chars', () => {
    const longContent = 'a'.repeat(250)
    localStorage.setItem(STORAGE_KEY, JSON.stringify([{ id: 'k1', content: longContent, timestamp: Date.now() }]))
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    const expected = 'a'.repeat(200) + '...'
    expect(screen.getByText(expected)).toBeDefined()
  })
})
