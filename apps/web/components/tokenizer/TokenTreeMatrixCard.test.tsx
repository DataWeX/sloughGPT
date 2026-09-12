import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetMatrixSummary: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getMatrixSummary: mocks.mockGetMatrixSummary,
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ActionCard: ({ title, actions, children, ...p }: any) => <div data-testid="action-card" {...p}>{title}{actions}{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  Skeleton: ({ className }: { className?: string }) => <div data-testid="skeleton" className={className} />,
  IconRefresh: () => <svg data-testid="icon-refresh" />,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
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

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ variant, message }: { variant: string; message: string }) => (
    <div data-variant={variant}>{message}</div>
  ),
}))

import { TokenTreeMatrixCard } from './TokenTreeMatrixCard'

const SUMMARY = {
  matrix: [128, 16],
  norm_min: 0.5,
  norm_mean: 0.8,
  norm_max: 1,
  dead_tokens: 2,
  live_tokens: 126,
  most_energetic: [
    ['quick</w>', 12, 1],
    ['brown</w>', 20, 0.98],
  ],
  least_energetic: [['the</w>', 3, 0.55]],
}

describe('TokenTreeMatrixCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the card title', () => {
    render(<TokenTreeMatrixCard />)
    expect(screen.getByText('Embedding Matrix Overview')).toBeDefined()
  })

  it('fetches the matrix summary on mount', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledWith(8))
    expect(screen.getByText('128 x 16 matrix')).toBeDefined()
    expect(screen.getByText('norm 0.500–1.000')).toBeDefined()
    expect(screen.getByText('mean 0.800')).toBeDefined()
    expect(screen.getByText('126 live / 2 dead')).toBeDefined()
  })

  it('shows most and least energetic tokens', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(screen.getByText('Most energetic')).toBeDefined())
    expect(screen.getByText(/quick/)).toBeDefined()
    expect(screen.getByText(/brown/)).toBeDefined()
    expect(screen.getByText('Least energetic')).toBeDefined()
    expect(screen.getAllByText(/the/).length).toBeGreaterThan(0)
    expect(screen.getByText('1.0000')).toBeDefined()
    expect(screen.getByText('0.5500')).toBeDefined()
  })

  it('shows the disabled state when matrix is null', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue({ ...SUMMARY, matrix: null })
    render(<TokenTreeMatrixCard />)
    await waitFor(() =>
      expect(screen.getByText(/Embeddings are disabled for this tree/)).toBeDefined(),
    )
  })

  it('refreshes when the refresh button is clicked', async () => {
    mocks.mockGetMatrixSummary.mockResolvedValue(SUMMARY)
    render(<TokenTreeMatrixCard />)
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: /Refresh matrix overview/ }))
    await waitFor(() => expect(mocks.mockGetMatrixSummary).toHaveBeenCalledTimes(2))
  })

  it('shows an error box when the fetch fails', async () => {
    mocks.mockGetMatrixSummary.mockRejectedValue(new Error('boom'))
    render(<TokenTreeMatrixCard />)
    await waitFor(() =>
      expect(screen.getByText(/Could not load the embedding matrix overview/)).toBeDefined(),
    )
  })

  it('shows a skeleton while loading', async () => {
    let resolvePromise: any
    mocks.mockGetMatrixSummary.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreeMatrixCard />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0)
    resolvePromise(SUMMARY)
    await waitFor(() => expect(screen.getByText('128 x 16 matrix')).toBeDefined())
  })
})
