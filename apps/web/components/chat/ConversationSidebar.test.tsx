import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('next/link', () => ({
  default: ({ children, href, onClick }: { children: React.ReactNode; href: string; onClick?: () => void }) => (
    <a href={href} onClick={onClick}>{children}</a>
  ),
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => { const C = () => <span data-testid={`icon-${name}`}>{name}</span>; C.displayName = `Icon${name}`; return C }
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Button: ({ children, onClick, variant, size, className }: any) => (
      <button onClick={onClick} className={className} data-variant={variant} data-size={size}>{children}</button>
    ),
    IconPlus: iconMock('plus'),
    IconStar: ({ filled }: any) => <span data-testid="icon-star" data-filled={String(filled)}>star</span>,
    IconPin: iconMock('pin'),
    IconChat: iconMock('chat'),
    IconChevronRight: iconMock('chevron-right'),
    IconX: iconMock('x'),
    IconSearch: iconMock('search'),
    IconFolder: iconMock('folder'),
    IconSort: iconMock('sort'),
    IconCheck: iconMock('check'),
    AlertDialog: ({ open, onOpenChange, children }: any) => open ? <div data-testid="alert-dialog">{children}</div> : null,
    AlertDialogContent: ({ children }: any) => <div>{children}</div>,
    AlertDialogHeader: ({ children }: any) => <div>{children}</div>,
    AlertDialogTitle: ({ children }: any) => <div>{children}</div>,
    AlertDialogDescription: ({ children }: any) => <div>{children}</div>,
    AlertDialogFooter: ({ children }: any) => <div>{children}</div>,
    AlertDialogCancel: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    AlertDialogAction: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  }
})

import { ConversationSidebar } from './ConversationSidebar'
import type { Conversation } from '@/lib/session-controller'

const createConv = (id: string, overrides: Partial<Conversation> = {}): Conversation => ({
  id,
  name: `Conversation ${id}`,
  session_id: id,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  starred: false,
  pinned: false,
  message_count: 1,
  messages: [{ id: 'm1', role: 'user', content: 'Hello', timestamp: Date.now() }],
  ...overrides,
})

describe('ConversationSidebar', () => {
  const onLoadConversation = vi.fn()
  const onNewChat = vi.fn()
  const onDeleteConversation = vi.fn()
  const onClose = vi.fn()

  const defaultProps = {
    conversations: [],
    currentConversationId: undefined,
    onLoadConversation,
    onNewChat,
    onDeleteConversation,
    open: false,
    onClose,
  }

  beforeEach(() => { vi.clearAllMocks(); localStorage.clear() })
  afterEach(cleanup)

  it('renders desktop sidebar', () => {
    render(<ConversationSidebar {...defaultProps} />)
    const aside = document.querySelector('aside')
    expect(aside).toBeDefined()
    expect(aside!.className).toContain('hidden lg:flex')
  })

  it('shows empty state message', () => {
    render(<ConversationSidebar {...defaultProps} />)
    expect(screen.getByText(/No conversations yet/)).toBeDefined()
  })

  it('shows search input when conversations exist', () => {
    const conversations = [createConv('1'), createConv('2')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByLabelText('Search conversations')).toBeDefined()
  })

  it('shows New button', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByText('New')).toBeDefined()
  })

  it('calls onNewChat when New button clicked', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByText('New'))
    expect(onNewChat).toHaveBeenCalledTimes(1)
  })

  it('calls onLoadConversation when conversation clicked', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByText('Conversation 1'))
    expect(onLoadConversation).toHaveBeenCalledWith('1')
  })

  it('calls onClose when conversation selected', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByText('Conversation 1'))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows starred section for starred conversations', () => {
    const conversations = [
      createConv('1', { starred: true, name: 'Starred Chat' }),
      createConv('2', { name: 'Regular Chat' }),
    ]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByText('Starred')).toBeDefined()
    expect(screen.getByText('Today')).toBeDefined()
    expect(screen.getByText('Starred Chat')).toBeDefined()
    expect(screen.getByText('Regular Chat')).toBeDefined()
  })

  it('highlights active conversation', () => {
    const conversations = [createConv('1')]
    render(
      <ConversationSidebar
        {...defaultProps}
        conversations={conversations}
        currentConversationId="1"
      />
    )
    const convEl = screen.getByText('Conversation 1').closest('[role="button"]')
    expect(convEl?.className).toContain('bg-primary/10')
  })

  it('filters conversations by search', () => {
    const conversations = [createConv('1', { name: 'Apple pie' }), createConv('2', { name: 'Banana bread' })]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const searchInput = screen.getByLabelText('Search conversations')
    fireEvent.change(searchInput, { target: { value: 'Apple' } })
    expect(screen.getByText('Apple pie')).toBeDefined()
    expect(screen.queryByText('Banana bread')).toBeNull()
  })

  it('shows no-match message on search', () => {
    const conversations = [createConv('1', { name: 'Apple' })]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const searchInput = screen.getByLabelText('Search conversations')
    fireEvent.change(searchInput, { target: { value: 'zzz' } })
    expect(screen.getByText(/No conversations match/)).toBeDefined()
  })

  it('calls onDeleteConversation when delete button clicked', () => {
    const conversations = [createConv('1')]
    render(
      <ConversationSidebar
        {...defaultProps}
        conversations={conversations}
        onDeleteConversation={onDeleteConversation}
      />
    )
    const deleteBtn = screen.getByLabelText('Delete Conversation 1')
    fireEvent.click(deleteBtn)
    expect(screen.getByTestId('alert-dialog')).toBeDefined()
    const deleteAction = screen.getByText('Delete')
    fireEvent.click(deleteAction)
    expect(onDeleteConversation).toHaveBeenCalledWith('1')
  })

  it('does not call onDeleteConversation when dialog cancelled', () => {
    const conversations = [createConv('1')]
    render(
      <ConversationSidebar
        {...defaultProps}
        conversations={conversations}
        onDeleteConversation={onDeleteConversation}
      />
    )
    const deleteBtn = screen.getByLabelText('Delete Conversation 1')
    fireEvent.click(deleteBtn)
    expect(screen.getByTestId('alert-dialog')).toBeDefined()
    const cancelBtn = screen.getByText('Cancel')
    fireEvent.click(cancelBtn)
    expect(onDeleteConversation).not.toHaveBeenCalled()
  })

  it('shows mobile drawer when open is true', () => {
    render(<ConversationSidebar {...defaultProps} open={true} />)
    const overlay = document.querySelector('.fixed.inset-0')
    expect(overlay).toBeDefined()
  })

  it('closes mobile drawer when overlay clicked', () => {
    render(<ConversationSidebar {...defaultProps} open={true} />)
    const overlay = document.querySelector('.absolute.inset-0')
    fireEvent.click(overlay!)
    expect(onClose).toHaveBeenCalled()
  })

  it('shows close button in mobile drawer', () => {
    render(<ConversationSidebar {...defaultProps} open={true} />)
    expect(screen.getByLabelText('Close sidebar')).toBeDefined()
  })

  it('closes mobile drawer when close button clicked', () => {
    render(<ConversationSidebar {...defaultProps} open={true} />)
    fireEvent.click(screen.getByLabelText('Close sidebar'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows conversation metadata (msg count, date)', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByText(/1 msgs/)).toBeDefined()
  })

  it('shows pin icon for pinned conversations', () => {
    const conversations = [createConv('1', { pinned: true, name: 'Pinned Chat' })]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByTestId('icon-pin')).toBeDefined()
  })

  it('shows star icon for starred conversations', () => {
    const conversations = [createConv('1', { starred: true, name: 'Starred Chat' })]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    expect(screen.getByTestId('icon-star')).toBeDefined()
  })

  it('renders View all conversations link', () => {
    render(<ConversationSidebar {...defaultProps} />)
    const link = screen.getByText('View all conversations')
    expect(link).toBeDefined()
  })

  it('calls onClose when view all conversations clicked', () => {
    render(<ConversationSidebar {...defaultProps} />)
    const link = screen.getByText('View all conversations')
    fireEvent.click(link)
    expect(onClose).toHaveBeenCalled()
  })

  it('handles Enter key on conversation row', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const convEl = screen.getByText('Conversation 1').closest('[role="button"]')
    expect(convEl).not.toBeNull()
    fireEvent.keyDown(convEl!, { key: 'Enter' })
    expect(onLoadConversation).toHaveBeenCalledWith('1')
  })

  it('displays truncated message preview', () => {
    const longText = 'A'.repeat(50)
    const conversations = [createConv('1', {
      messages: [{ id: 'm1', role: 'user', content: longText, timestamp: Date.now() }],
    })]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const expected = 'A'.repeat(36) + '…'
    expect(screen.getByText(expected)).toBeDefined()
  })

  it('hides message preview when last message has no content', () => {
    const conversations = [createConv('1', {
      messages: [{ id: 'm1', role: 'user', content: '', timestamp: Date.now() }],
    })]
    const { container } = render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const lineClamp = container.querySelector('.line-clamp-1')
    expect(lineClamp).toBeNull()
  })

  it('double-clicking conversation name opens rename input', () => {
    const conversations = [createConv('1')]
    const { container } = render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const nameEl = screen.getByText('Conversation 1')
    fireEvent.doubleClick(nameEl)
    const input = container.querySelector('input[aria-label="Rename conversation"]')
    expect(input).not.toBeNull()
  })

  it('saves rename on Enter', () => {
    const onRename = vi.fn()
    const conversations = [createConv('1')]
    const { container } = render(
      <ConversationSidebar
        {...defaultProps}
        conversations={conversations}
        onRenameConversation={onRename}
      />
    )
    fireEvent.doubleClick(screen.getByText('Conversation 1'))
    const input = container.querySelector('input[aria-label="Rename conversation"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'Renamed' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onRename).toHaveBeenCalledWith('1', 'Renamed')
  })

  it('cancels rename on Escape', () => {
    const onRename = vi.fn()
    const conversations = [createConv('1', { name: 'Original' })]
    const { container } = render(
      <ConversationSidebar
        {...defaultProps}
        conversations={conversations}
        onRenameConversation={onRename}
      />
    )
    fireEvent.doubleClick(screen.getByText('Original'))
    const input = container.querySelector('input[aria-label="Rename conversation"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'Changed' } })
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(onRename).not.toHaveBeenCalled()
    expect(screen.getByText('Original')).toBeDefined()
  })

  it('sort button opens sort dropdown', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const sortBtn = screen.getByLabelText('Sort conversations')
    fireEvent.click(sortBtn)
    expect(screen.getByText('Last updated')).toBeDefined()
    expect(screen.getByText('Name')).toBeDefined()
    expect(screen.getByText('Message count')).toBeDefined()
  })

  it('sort by name orders alphabetically', () => {
    const conversations = [
      createConv('1', { name: 'Zebra', updated_at: '2026-01-01' }),
      createConv('2', { name: 'Apple', updated_at: '2026-01-02' }),
      createConv('3', { name: 'Mango', updated_at: '2026-01-03' }),
    ]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByLabelText('Sort conversations'))
    fireEvent.click(screen.getByText('Name'))
    const items = screen.getAllByText(/^(Apple|Mango|Zebra)$/)
    expect(items[0]).toHaveProperty('textContent', 'Apple')
    expect(items[1]).toHaveProperty('textContent', 'Mango')
    expect(items[2]).toHaveProperty('textContent', 'Zebra')
  })

  it('sort by message count orders descending', () => {
    const conversations = [
      createConv('1', { name: 'Few', message_count: 2, updated_at: '2026-01-01' }),
      createConv('2', { name: 'Many', message_count: 20, updated_at: '2026-01-02' }),
      createConv('3', { name: 'Mid', message_count: 10, updated_at: '2026-01-03' }),
    ]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByLabelText('Sort conversations'))
    fireEvent.click(screen.getByText('Message count'))
    const items = screen.getAllByText(/^(Few|Many|Mid)$/)
    expect(items[0]).toHaveProperty('textContent', 'Many')
    expect(items[1]).toHaveProperty('textContent', 'Mid')
    expect(items[2]).toHaveProperty('textContent', 'Few')
  })

  it('sort icon shows primary color when non-default sort active', () => {
    const conversations = [createConv('1')]
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const sortBtn = screen.getByLabelText('Sort conversations')
    expect(sortBtn.className).not.toContain('text-primary')
    fireEvent.click(sortBtn)
    fireEvent.click(screen.getByText('Name'))
    const updatedBtn = screen.getByLabelText('Sort conversations')
    expect(updatedBtn.className).toContain('text-primary')
  })

  it('persists sort preference across remounts', () => {
    localStorage.clear()
    const conversations = [
      createConv('1', { name: 'Zebra', updated_at: '2026-01-01' }),
      createConv('2', { name: 'Apple', updated_at: '2026-01-02' }),
    ]
    // First render: change sort to name
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    fireEvent.click(screen.getByLabelText('Sort conversations'))
    fireEvent.click(screen.getByText('Name'))
    expect(localStorage.getItem('sloughgpt:sidebar-sort')).toBe('name')
    cleanup()

    // Second render: sort should persist from localStorage as 'name' (non-default)
    render(<ConversationSidebar {...defaultProps} conversations={conversations} />)
    const sortBtn = screen.getByLabelText('Sort conversations')
    expect(sortBtn.className).toContain('text-primary')
    // Items should appear in name order
    const items = screen.getAllByText(/^(Apple|Zebra)$/)
    expect(items[0]).toHaveProperty('textContent', 'Apple')
    expect(items[1]).toHaveProperty('textContent', 'Zebra')
  })
})
