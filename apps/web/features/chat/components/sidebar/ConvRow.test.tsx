import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ConvRow } from './ConvRow'
import type { Conversation } from '@/lib/session-controller'

vi.mock('@/lib/conversations-utils', () => ({
  formatDate: (d: string) => d ? new Date(d).toLocaleDateString() : '',
  truncateMessage: (c: string, max: number) => c.length > max ? c.slice(0, max) + '…' : c,
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
  IconPin: (props: any) => <span data-testid="icon-pin" {...props} />,
  IconStar: (props: any) => <span data-testid="icon-star" {...props} />,
  IconDot: (props: any) => <span data-testid="icon-dot" {...props} />,
  IconDotOutline: (props: any) => <span data-testid="icon-dot-outline" {...props} />,
  IconDownload: (props: any) => <span data-testid="icon-download" {...props} />,
  IconDocument: (props: any) => <span data-testid="icon-document" {...props} />,
  IconCopy: (props: any) => <span data-testid="icon-copy" {...props} />,
  IconFolder: (props: any) => <span data-testid="icon-folder" {...props} />,
  IconX: (props: any) => <span data-testid="icon-x" {...props} />,
  IconMore: (props: any) => <span data-testid="icon-more" {...props} />,
  // Structural stand-ins: the row only cares that one affordance renders and
  // that the actions land in the menu, not how the menu positions itself.
  DropdownMenu: ({ children }: any) => <>{children}</>,
  DropdownMenuTrigger: ({ children }: any) => children,
  DropdownMenuContent: ({ children }: any) => <div role="menu">{children}</div>,
  DropdownMenuItem: ({ children, onSelect, disabled, destructive }: any) => (
    <div role="menuitem" aria-disabled={disabled || undefined} data-destructive={destructive ? '' : undefined} onClick={onSelect}>
      {children}
    </div>
  ),
  DropdownMenuSeparator: () => <hr data-testid="menu-separator" />,
}))

const createConv = (id: string, overrides: Partial<Conversation> = {}): Conversation => ({
  id,
  name: `Test Conversation ${id}`,
  session_id: id,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-02T00:00:00Z',
  starred: false,
  pinned: false,
  message_count: 5,
  messages: [{ id: 'm1', role: 'user', content: 'Hello world', timestamp: Date.now() }],
  ...overrides,
})

describe('ConvRow', () => {
  const onSelect = vi.fn()
  const onDelete = vi.fn()
  const onStar = vi.fn()
  const onPin = vi.fn()
  const onRename = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })

  it('renders without crashing', () => {
    render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} />)
    expect(screen.getByText('Test Conversation 1')).toBeInTheDocument()
  })

  it('calls onSelect when clicked', () => {
    render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} />)
    fireEvent.click(screen.getByText('Test Conversation 1'))
    expect(onSelect).toHaveBeenCalledWith('1')
  })

  it('shows message count from messages array', () => {
    const msgs = Array.from({ length: 8 }, (_, i) => ({ id: `m${i}`, role: 'user' as const, content: `msg ${i}`, timestamp: Date.now() }))
    render(<ConvRow conversation={createConv('1', { messages: msgs })} isActive={false} onSelect={onSelect} />)
    expect(screen.getByText('8')).toBeInTheDocument()
  })

  it('displays truncated message preview', () => {
    render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} />)
    expect(screen.getByText('Hello world')).toBeInTheDocument()
  })

  it('highlights active conversation', () => {
    const { container } = render(<ConvRow conversation={createConv('1')} isActive={true} onSelect={onSelect} />)
    const row = container.querySelector('[role="button"]')
    expect(row?.className).toContain('bg-primary/10')
  })

  it('double-click enters rename mode', () => {
    const { container } = render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} onRename={onRename} />)
    fireEvent.doubleClick(screen.getByText('Test Conversation 1'))
    const input = container.querySelector('input[aria-label="Rename conversation"]') as HTMLInputElement
    expect(input).not.toBeNull()
    expect(input.value).toBe('Test Conversation 1')
  })

  it('saves rename on Enter', () => {
    const { container } = render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} onRename={onRename} />)
    fireEvent.doubleClick(screen.getByText('Test Conversation 1'))
    const input = container.querySelector('input[aria-label="Rename conversation"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'Renamed' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onRename).toHaveBeenCalledWith('1', 'Renamed')
  })

  it('cancels rename on Escape', () => {
    const { container } = render(<ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} onRename={onRename} />)
    fireEvent.doubleClick(screen.getByText('Test Conversation 1'))
    const input = container.querySelector('input[aria-label="Rename conversation"]') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'Changed' } })
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(onRename).not.toHaveBeenCalled()
  })

  it('keeps one overflow affordance instead of spending the row width on inline buttons', () => {
    const { container } = render(
      <ConvRow
        conversation={createConv('1')}
        isActive={false}
        onSelect={onSelect}
        onDelete={onDelete}
        onStar={onStar}
        onPin={onPin}
      />,
    )
    expect(screen.getByRole('button', { name: 'Actions for Test Conversation 1' })).toBeInTheDocument()

    // The eight inline controls that used to claim 212px of a 244px row are gone.
    const inline = container.querySelectorAll(
      'button[aria-label="Pin"], button[aria-label="Unpin"], button[aria-label="Star"], button[aria-label="Unstar"], button[aria-label^="Duplicate"], button[aria-label="Archive"], button[aria-label^="Delete"]',
    )
    expect(inline.length).toBe(0)

    // …and every one of them is reachable from the menu.
    const items = screen.getAllByRole('menuitem').map((n) => n.textContent)
    expect(items).toContain('Pin')
    expect(items).toContain('Star')
    expect(items).toContain('Delete conversation')
  })

  it('does not select the conversation when the actions menu is opened', () => {
    render(
      <ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} onDelete={onDelete} onPin={onPin} />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Actions for Test Conversation 1' }))
    expect(onSelect).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText('Test Conversation 1'))
    expect(onSelect).toHaveBeenCalledWith('1')
  })

  it('runs the chosen action with this conversation id', () => {
    render(
      <ConvRow
        conversation={createConv('1')}
        isActive={false}
        onSelect={onSelect}
        onDelete={onDelete}
        onStar={onStar}
        onPin={onPin}
      />,
    )
    fireEvent.click(screen.getByRole('menuitem', { name: /Delete/ }))
    expect(onDelete).toHaveBeenCalledTimes(1)
    expect(onDelete.mock.calls[0][1]).toBe('1')

    fireEvent.click(screen.getByRole('menuitem', { name: /Star/ }))
    expect(onStar).toHaveBeenCalledTimes(1)
    expect(onStar.mock.calls[0][1]).toBe('1')
    expect(onStar.mock.calls[0][2]).toBe(true)
  })

  it('hides the actions menu while the title is being renamed', () => {
    render(
      <ConvRow conversation={createConv('1')} isActive={false} onSelect={onSelect} onRename={onRename} onPin={onPin} />,
    )
    expect(screen.getByRole('button', { name: 'Actions for Test Conversation 1' })).toBeInTheDocument()
    fireEvent.doubleClick(screen.getByText('Test Conversation 1'))
    expect(screen.queryByRole('button', { name: 'Actions for Test Conversation 1' })).toBeNull()
  })
})
