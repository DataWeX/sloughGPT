import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockList = vi.fn()

vi.mock('@/lib/session-controller', () => ({
  sessionController: { list: (...args: any[]) => mockList(...args) },
}))

vi.mock('@/lib/conversations-utils', () => ({
  truncateMessage: (s: string) => s?.slice(0, 60) || 'Empty conversation',
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Button: ({ children, onClick, ...props }: any) => <button onClick={onClick} {...props}>{children}</button>,
  IconSearch: (p: any) => <svg {...p} />,
  IconX: (p: any) => <svg {...p} />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
    Separator: () => <hr />,
    Tooltip: ({ children }: any) => <>{children}</>,
    TooltipTrigger: ({ children }: any) => <>{children}</>,
    TooltipContent: ({ children }: any) => <>{children}</>,
    Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
    Avatar: ({ children }: any) => <div>{children}</div>,
    AvatarFallback: ({ children }: any) => <div>{children}</div>,
    ScrollArea: ({ children }: any) => <div>{children}</div>,
    Table: ({ children }: any) => <table>{children}</table>,
    TableBody: ({ children }: any) => <tbody>{children}</tbody>,
    TableRow: ({ children }: any) => <tr>{children}</tr>,
    TableCell: ({ children }: any) => <td>{children}</td>,
    TableHead: ({ children }: any) => <th>{children}</th>,
    TableHeader: ({ children }: any) => <thead>{children}</thead>,
    Collapsible: ({ children }: any) => <div>{children}</div>,
    CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    CollapsibleContent: ({ children }: any) => <div>{children}</div>,
    Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    ToggleGroup: ({ children }: any) => <div>{children}</div>,
    ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Command: ({ children }: any) => <div>{children}</div>,
    CommandInput: ({ ...props }: any) => <input {...props} />,
    CommandList: ({ children }: any) => <div>{children}</div>,
    CommandEmpty: ({ children }: any) => <div>{children}</div>,
    CommandGroup: ({ children }: any) => <div>{children}</div>,
    CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

import { ConversationPicker } from './ConversationPicker'

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  onSelect: vi.fn(),
  currentConversationId: 'conv-1',
}

beforeEach(() => {
  vi.clearAllMocks()
  mockList.mockResolvedValue([
    { id: 'conv-1', name: 'Current Chat', created_at: '2026-01-01', updated_at: '2026-01-10', messages: [] },
    { id: 'conv-2', name: 'Other Chat', created_at: '2026-01-02', updated_at: '2026-01-11', messages: [{ id: '1', role: 'user', content: 'hi' }] },
    { id: 'conv-3', name: 'Third Chat', created_at: '2026-01-03', updated_at: '2026-01-12', pinned: true, starred: true, messages: [{ id: '1', role: 'user', content: 'hey' }, { id: '2', role: 'assistant', content: 'yo' }] },
  ])
})

afterEach(cleanup)

describe('ConversationPicker', () => {
  it('renders nothing when closed', () => {
    render(<ConversationPicker {...defaultProps} open={false} />)
    expect(screen.queryByPlaceholderText('Search conversations...')).toBeNull()
  })

  it('renders when open and fetches conversations', async () => {
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search conversations...')).toBeDefined()
    })
    expect(mockList).toHaveBeenCalled()
  })

  it('filters out current conversation', async () => {
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText('Other Chat')).toBeDefined()
    })
    expect(screen.queryByText('Current Chat')).toBeNull()
  })

  it('searches conversations by name', async () => {
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText('Other Chat')).toBeDefined()
    })

    fireEvent.change(screen.getByPlaceholderText('Search conversations...'), { target: { value: 'Third' } })
    expect(screen.getByText('Third Chat')).toBeDefined()
    expect(screen.queryByText('Other Chat')).toBeNull()
  })

  it('shows empty state when no results', async () => {
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText('Other Chat')).toBeDefined()
    })

    fireEvent.change(screen.getByPlaceholderText('Search conversations...'), { target: { value: 'zzz' } })
    expect(screen.getByText('No conversations match')).toBeDefined()
  })

  it('selects a conversation and closes', async () => {
    const onSelect = vi.fn()
    const onClose = vi.fn()
    render(<ConversationPicker {...defaultProps} onSelect={onSelect} onClose={onClose} />)
    await waitFor(() => {
      expect(screen.getByText('Other Chat')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Other Chat'))
    expect(onSelect).toHaveBeenCalledWith('conv-2')
    expect(onClose).toHaveBeenCalled()
  })

  it('closes on backdrop click', async () => {
    const onClose = vi.fn()
    render(<ConversationPicker {...defaultProps} onClose={onClose} />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search conversations...')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Esc'))
    expect(onClose).toHaveBeenCalled()
  })

  it('shows loading spinner while fetching', async () => {
    mockList.mockReturnValue(new Promise(() => {}))
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search conversations...')).toBeDefined()
    })
    expect(document.querySelector('.animate-spin')).toBeDefined()
  })

  it('handles list error gracefully', async () => {
    mockList.mockRejectedValue(new Error('network'))
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search conversations...')).toBeDefined()
    })
    expect(screen.getByText('No other conversations')).toBeDefined()
  })

  it('shows message count for each conversation', async () => {
    render(<ConversationPicker {...defaultProps} />)
    await waitFor(() => {
      expect(screen.getByText('Other Chat')).toBeDefined()
    })
    expect(screen.getByText(/1 message/)).toBeDefined()
    expect(screen.getByText(/2 messages/)).toBeDefined()
  })
})
