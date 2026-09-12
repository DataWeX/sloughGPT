import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mockSearchAllSessions = vi.hoisted(() => vi.fn())

vi.mock('@/lib/db', () => ({
  chatDB: { searchAllSessions: mockSearchAllSessions },
}))

vi.mock('@/lib/session-controller', () => ({
  sessionController: { search: vi.fn().mockRejectedValue(new Error('no server')) },
}))

vi.mock('@sloughgpt/strui', () => {
  const iconMock = (name: string) => {
    const C = () => <span data-testid={`icon-${name}`}>{name}</span>
    C.displayName = `Icon${name}`
    return C
  }
  return {
    cn: vi.fn((...args: any[]) => args.join(' ')),
    Input: (props: any) => <input data-testid="search-input" {...props} />,
    Button: ({ children, onClick, variant, size, ...rest }: any) => (
      <button onClick={onClick} data-variant={variant} data-size={size} {...rest}>{children}</button>
    ),
    IconSearch: iconMock('search'),
    IconX: iconMock('x'),
    IconMessage: iconMock('message'),
  
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
}
})

import { ConversationSearch } from './ConversationSearch'

describe('ConversationSearch', () => {
  const onClose = vi.fn()
  const onNavigate = vi.fn()

  beforeEach(() => { vi.clearAllMocks() })
  afterEach(cleanup)

  it('returns null when not open', () => {
    const { container } = render(
      <ConversationSearch open={false} onClose={onClose} onNavigate={onNavigate} />
    )
    expect(container.innerHTML).toBe('')
  })

  it('renders search dialog when open', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByLabelText('Search all conversations')).toBeDefined()
    expect(screen.getByTestId('search-input')).toBeDefined()
  })

  it('shows placeholder when query is empty', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByText('Type to search across all conversations')).toBeDefined()
  })

  it('shows Esc hint', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    expect(screen.getByText('Esc')).toBeDefined()
  })

  it('shows loading spinner while searching', () => {
    mockSearchAllSessions.mockReturnValue(new Promise(() => {}))
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeDefined()
  })

  it('shows no results message', async () => {
    mockSearchAllSessions.mockResolvedValue([])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText(/No results for/)).toBeDefined()
    })
  })

  it('renders search results', async () => {
    mockSearchAllSessions.mockResolvedValue([
      {
        session: { id: 's1', name: 'Test Chat', updated_at: 0, message_count: 2 },
        matches: [{ content: 'hello world' }],
      },
    ])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText('1 conversation found')).toBeDefined()
      expect(screen.getByText('Test Chat')).toBeDefined()
      expect(screen.getByText('1 match')).toBeDefined()
    })
  })

  it('navigates and closes on result click', async () => {
    mockSearchAllSessions.mockResolvedValue([
      {
        session: { id: 's1', name: 'Test Chat', updated_at: 0, message_count: 2 },
        matches: [{ content: 'hello world' }],
      },
    ])
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    await waitFor(() => {
      expect(screen.getByText('Test Chat')).toBeDefined()
    })
    fireEvent.click(screen.getByText('Test Chat'))
    expect(onNavigate).toHaveBeenCalledWith('s1')
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape key', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onClose when backdrop clicked', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    fireEvent.click(screen.getByLabelText('Search all conversations').parentElement!)
    expect(onClose).toHaveBeenCalled()
  })

  it('shows clear button when query exists', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input')
    fireEvent.change(input, { target: { value: 'hello' } })
    expect(screen.getByLabelText('Clear')).toBeDefined()
  })

  it('clears query on clear button click', () => {
    render(<ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />)
    const input = screen.getByTestId('search-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hello' } })
    fireEvent.click(screen.getByLabelText('Clear'))
    expect(input.value).toBe('')
  })

  it('resets on close', () => {
    const { rerender } = render(
      <ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />
    )
    const input = screen.getByTestId('search-input') as HTMLInputElement
    fireEvent.change(input, { target: { value: 'hello' } })
    rerender(<ConversationSearch open={false} onClose={onClose} onNavigate={onNavigate} />)
    const { container } = render(
      <ConversationSearch open={true} onClose={onClose} onNavigate={onNavigate} />
    )
    const newInput = container.querySelector('[data-testid="search-input"]') as HTMLInputElement
    expect(newInput.value).toBe('')
  })
})
