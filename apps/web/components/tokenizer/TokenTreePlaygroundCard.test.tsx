import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockTokenize: vi.fn(),
  mockEncode: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/tokenizer-controller', () => ({
  tokenizerController: { tokenize: mocks.mockTokenize },
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: { encode: mocks.mockEncode },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: typeof mocks.mockAddToast }) => unknown) =>
    selector({ addToast: mocks.mockAddToast }),
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Textarea: (props: any) => <textarea {...props} />,
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  IconRefresh: () => <span />,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
    Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
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

import { TokenTreePlaygroundCard } from './TokenTreePlaygroundCard'

describe('TokenTreePlaygroundCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders with default sample text and compare button', () => {
    render(<TokenTreePlaygroundCard />)
    expect(screen.getByText('Token Tree Playground')).toBeDefined()
    expect(screen.getByLabelText('Text to compare')).toHaveValue('the quick brown fox jumps over the lazy dog')
    expect(screen.getByRole('button', { name: /Compare tokenizers/i })).toBeDefined()
  })

  it('compares token counts and shows a compactness verdict', async () => {
    mocks.mockTokenize.mockResolvedValueOnce({
      tokens: ['the</w>', 'quick</w>', 'brown</w>', 'fox</w>'],
      ids: [1, 2, 3, 4],
    })
    mocks.mockEncode.mockResolvedValueOnce({
      tokens: ['the quick</w>', 'brown fox</w>'],
      ids: [10, 11],
    })
    render(<TokenTreePlaygroundCard />)

    fireEvent.change(screen.getByLabelText('Text to compare'), { target: { value: 'the quick brown fox' } })
    fireEvent.click(screen.getByRole('button', { name: /Compare tokenizers/i }))

    await waitFor(() => expect(mocks.mockTokenize).toHaveBeenCalledWith('the quick brown fox'))
    expect(mocks.mockEncode).toHaveBeenCalledWith('the quick brown fox')
    expect(screen.getByText('Token tree is 100% more compact')).toBeDefined()
    expect(screen.getByText('Base BPE — 4 tokens')).toBeDefined()
    expect(screen.getByText('Token tree — 2 tokens')).toBeDefined()
  })

  it('reports when the base tokenizer is more compact', async () => {
    mocks.mockTokenize.mockResolvedValueOnce({ tokens: ['a</w>'], ids: [1] })
    mocks.mockEncode.mockResolvedValueOnce({
      tokens: ['a</w>', 'b</w>', 'c</w>', 'd</w>'],
      ids: [2, 3, 4, 5],
    })
    render(<TokenTreePlaygroundCard />)
    fireEvent.click(screen.getByRole('button', { name: /Compare tokenizers/i }))

    await waitFor(() => expect(screen.getByText('Base tokenizer is 300% more compact')).toBeDefined())
  })

  it('shows "same token count" when both split identically', async () => {
    mocks.mockTokenize.mockResolvedValueOnce({ tokens: ['a</w>'], ids: [1] })
    mocks.mockEncode.mockResolvedValueOnce({ tokens: ['a</w>'], ids: [1] })
    render(<TokenTreePlaygroundCard />)
    fireEvent.click(screen.getByRole('button', { name: /Compare tokenizers/i }))

    await waitFor(() =>
      expect(screen.getByText('Both tokenizers produce the same token count')).toBeDefined(),
    )
  })

  it('still renders the available side when only one tokenizer succeeds', async () => {
    mocks.mockTokenize.mockResolvedValueOnce({ tokens: ['a</w>'], ids: [1] })
    mocks.mockEncode.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreePlaygroundCard />)
    fireEvent.click(screen.getByRole('button', { name: /Compare tokenizers/i }))

    await waitFor(() => expect(screen.getByText('Base BPE — 1 tokens')).toBeDefined())
    expect(screen.getByText('Token tree — unavailable')).toBeDefined()
    expect(screen.getByText('Token tree not trained.')).toBeDefined()
    expect(mocks.mockAddToast).not.toHaveBeenCalled()
  })

  it('shows an error toast when both tokenizers fail', async () => {
    mocks.mockTokenize.mockRejectedValueOnce(new Error('boom'))
    mocks.mockEncode.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreePlaygroundCard />)
    fireEvent.click(screen.getByRole('button', { name: /Compare tokenizers/i }))

    await waitFor(() =>
      expect(mocks.mockAddToast).toHaveBeenCalledWith('Both tokenizers failed. Is the tokenizer trained?', 'error'),
    )
  })

  it('does nothing when text is empty', async () => {
    render(<TokenTreePlaygroundCard />)
    fireEvent.change(screen.getByLabelText('Text to compare'), { target: { value: '   ' } })
    const button = screen.getByRole('button', { name: /Compare tokenizers/i })
    expect(button).toHaveProperty('disabled', true)
    fireEvent.click(button)

    expect(mocks.mockTokenize).not.toHaveBeenCalled()
    expect(mocks.mockEncode).not.toHaveBeenCalled()
  })
})
