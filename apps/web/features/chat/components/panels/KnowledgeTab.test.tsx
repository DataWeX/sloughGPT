import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act, waitFor } from '@testing-library/react'
import React from 'react'

const { mockKnowledgeItems, mockKnowledgeController, mockChatDB } = vi.hoisted(() => {
  const mockKnowledgeItems = new Map<string, { id: string; content: string; timestamp: number }>()
  const mockKnowledgeController = {
    list: vi.fn().mockResolvedValue([]),
    batchIngest: vi.fn().mockResolvedValue({}),
    delete: vi.fn().mockResolvedValue({}),
  }
  const mockChatDB = {
    getKnowledge: vi.fn(async () => [...mockKnowledgeItems.values()]),
    addKnowledge: vi.fn(async (item: any) => { mockKnowledgeItems.set(item.id, item) }),
    updateKnowledge: vi.fn(async (id: string, updates: any) => {
      const item = mockKnowledgeItems.get(id)
      if (item) Object.assign(item, updates)
    }),
    deleteKnowledge: vi.fn(async (id: string) => { mockKnowledgeItems.delete(id) }),
    clearKnowledge: vi.fn(async () => { mockKnowledgeItems.clear() }),
    importKnowledge: vi.fn(async (items: any[]) => {
      mockKnowledgeItems.clear()
      for (const item of items) mockKnowledgeItems.set(item.id, item)
    }),
  }
  return { mockKnowledgeItems, mockKnowledgeController, mockChatDB }
})

vi.mock('@/lib/knowledge-controller', () => ({
  knowledgeController: mockKnowledgeController,
}))

vi.mock('@/lib/db', () => ({
  chatDB: mockChatDB,
  KnowledgeItem: {},
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, variant, size, className, ...rest }: any) => (
    <button onClick={onClick} className={className} data-variant={variant} data-size={size} {...rest}>{children}</button>
  ),
  IconX: () => <span data-testid="icon-x">x</span>,
  IconEdit: () => <span data-testid="icon-edit">edit</span>,
}))

import { KnowledgeTab } from './KnowledgeTab'

describe('KnowledgeTab', () => {
  const onOpenConversationViewer = vi.fn()
  const onOpenSettings = vi.fn()
  const onOpenShortcuts = vi.fn()

  beforeEach(async () => {
    vi.clearAllMocks()
    mockKnowledgeItems.clear()
    mockChatDB.getKnowledge.mockImplementation(async () => [...mockKnowledgeItems.values()])
    mockChatDB.addKnowledge.mockImplementation(async (item: any) => { mockKnowledgeItems.set(item.id, item) })
    mockChatDB.clearKnowledge.mockImplementation(async () => { mockKnowledgeItems.clear() })
    mockChatDB.importKnowledge.mockImplementation(async (items: any[]) => {
      mockKnowledgeItems.clear()
      for (const item of items) mockKnowledgeItems.set(item.id, item)
    })
    mockKnowledgeController.list.mockResolvedValue([])
  })

  afterEach(cleanup)

  it('shows empty state', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText(/No knowledge stored/)).toBeDefined()
    })
  })

  it('shows 0 snippets count', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('0 snippets')).toBeDefined()
    })
  })

  it('shows 1 snippet count', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: 'fact one', timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('1 snippet')).toBeDefined()
    })
  })

  it('loads knowledge from chatDB', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: 'fact one', timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('fact one')).toBeDefined()
    })
  })

  it('shows add form when + Add clicked', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => expect(screen.getByText('0 snippets')).toBeDefined())
    fireEvent.click(screen.getByText('+ Add'))
    expect(screen.getByPlaceholderText(/Enter a fact/)).toBeDefined()
  })

  it('adds knowledge item', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => expect(screen.getByText('0 snippets')).toBeDefined())
    fireEvent.click(screen.getByText('+ Add'))
    const textarea = screen.getByPlaceholderText(/Enter a fact/)
    fireEvent.change(textarea, { target: { value: 'my fact' } })
    await act(async () => { fireEvent.click(screen.getByText('Save')) })
    await waitFor(() => {
      expect(screen.getByText('my fact')).toBeDefined()
      expect(screen.getByText('1 snippet')).toBeDefined()
    })
  })

  it('cancels add knowledge', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => expect(screen.getByText('0 snippets')).toBeDefined())
    fireEvent.click(screen.getByText('+ Add'))
    fireEvent.change(screen.getByPlaceholderText(/Enter a fact/), { target: { value: 'will cancel' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('will cancel')).toBeNull()
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('removes knowledge item', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: 'remove me', timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('remove me')).toBeDefined()
    })
    await act(async () => { fireEvent.click(screen.getByLabelText('Remove knowledge')) })
    await waitFor(() => {
      expect(screen.queryByText('remove me')).toBeNull()
      expect(screen.getByText('0 snippets')).toBeDefined()
    })
  })

  it('edits knowledge item', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: 'edit me', timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('edit me')).toBeDefined()
    })
    fireEvent.click(screen.getByLabelText('Edit knowledge'))
    const editArea = screen.getByLabelText('Edit knowledge snippet')
    fireEvent.change(editArea, { target: { value: 'edited content' } })
    await act(async () => { fireEvent.click(screen.getByText('Save')) })
    await waitFor(() => {
      expect(screen.getByText('edited content')).toBeDefined()
      expect(screen.queryByText('edit me')).toBeNull()
    })
  })

  it('cancels edit', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: 'original', timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('original')).toBeDefined()
    })
    fireEvent.click(screen.getByLabelText('Edit knowledge'))
    fireEvent.change(screen.getByLabelText('Edit knowledge snippet'), { target: { value: 'changed' } })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.getByText('original')).toBeDefined()
    expect(screen.queryByText('changed')).toBeNull()
  })

  it('clears all knowledge', async () => {
    mockChatDB.getKnowledge.mockResolvedValueOnce([
      { id: 'k1', content: 'fact one', timestamp: Date.now() },
      { id: 'k2', content: 'fact two', timestamp: Date.now() },
    ])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      expect(screen.getByText('2 snippets')).toBeDefined()
    })
    await act(async () => { fireEvent.click(screen.getByText('Clear all')) })
    await waitFor(() => {
      expect(screen.getByText('0 snippets')).toBeDefined()
      expect(screen.getByText(/No knowledge stored/)).toBeDefined()
    })
  })

  it('does not add empty knowledge', async () => {
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => expect(screen.getByText('0 snippets')).toBeDefined())
    fireEvent.click(screen.getByText('+ Add'))
    fireEvent.click(screen.getByText('Save'))
    expect(screen.getByText('0 snippets')).toBeDefined()
  })

  it('truncates long content at 200 chars', async () => {
    const longContent = 'a'.repeat(250)
    mockChatDB.getKnowledge.mockResolvedValueOnce([{ id: 'k1', content: longContent, timestamp: Date.now() }])
    render(<KnowledgeTab onOpenConversationViewer={onOpenConversationViewer} onOpenSettings={onOpenSettings} onOpenShortcuts={onOpenShortcuts} />)
    await waitFor(() => {
      const expected = 'a'.repeat(200) + '...'
      expect(screen.getByText(expected)).toBeDefined()
    })
  })
})
