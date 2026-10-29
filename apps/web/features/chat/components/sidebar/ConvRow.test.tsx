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
})
