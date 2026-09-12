import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockGetEmbedding: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: {
    getEmbedding: mocks.mockGetEmbedding,
  },
}))

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Chip: ({ label }: { label: string }) => <span>{label}</span>,
  cn: (...args: any[]) => args.filter(Boolean).join(' '),

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

vi.mock('@/components/composed/StatusBanner', () => ({
  StatusBanner: ({ variant, message }: { variant: string; message: string }) => (
    <div data-variant={variant}>{message}</div>
  ),
}))

import { TokenTreeEmbeddingsCard } from './TokenTreeEmbeddingsCard'

const EMBED = {
  token: 'the',
  id: 3,
  dim: 8,
  norm: 1,
  top: [
    [0, 0.9],
    [1, -0.8],
  ],
  embedding_points: 200,
  compression_ratio: 4,
}

describe('TokenTreeEmbeddingsCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the token input with a default', () => {
    render(<TokenTreeEmbeddingsCard />)
    expect(screen.getByLabelText('Token to inspect')).toBeDefined()
    expect((screen.getByLabelText('Token to inspect') as HTMLInputElement).value).toBe('quick')
  })

  it('inspects a token and shows the embedding summary', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)

    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'the' } })
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() => expect(mocks.mockGetEmbedding).toHaveBeenCalledWith('the', 8))

    expect(screen.getByText(/"the" · id 3/)).toBeDefined()
    expect(screen.getByText('Dim 8')).toBeDefined()
    expect(screen.getByText('L2 norm 1.0000')).toBeDefined()
    expect(screen.getByText('200 points')).toBeDefined()
    expect(screen.getByText('4x compressed')).toBeDefined()
  })

  it('disables the button for a blank token', () => {
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: '   ' } })
    expect((screen.getByRole('button', { name: /^Inspect$/ }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('inspects on Enter key', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.change(screen.getByLabelText('Token to inspect'), { target: { value: 'the' } })
    fireEvent.keyDown(screen.getByLabelText('Token to inspect'), { key: 'Enter' })
    await waitFor(() => expect(mocks.mockGetEmbedding).toHaveBeenCalledWith('the', 8))
  })

  it('shows an error box when the token is unknown or embeddings are disabled', async () => {
    mocks.mockGetEmbedding.mockRejectedValue(new Error('Token not in vocabulary'))
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() =>
      expect(screen.getByText(/Token not in the vocabulary, or embeddings are disabled/)).toBeDefined(),
    )
  })

  it('shows top dimension values with sign', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    await waitFor(() => expect(screen.getByText('+0.9000')).toBeDefined())
    expect(screen.getByText('-0.8000')).toBeDefined()
  })

  it('shows inspecting... text while loading', async () => {
    let resolvePromise: any
    mocks.mockGetEmbedding.mockImplementation(() => new Promise(r => { resolvePromise = r }))
    render(<TokenTreeEmbeddingsCard />)
    fireEvent.click(screen.getByRole('button', { name: /^Inspect$/ }))
    expect(screen.getByText('Inspecting...')).toBeDefined()
    resolvePromise(EMBED)
    await waitFor(() => expect(screen.getByText('Inspect')).toBeDefined())
  })

  it('allows changing the token value', async () => {
    mocks.mockGetEmbedding.mockResolvedValue(EMBED)
    render(<TokenTreeEmbeddingsCard />)
    const input = screen.getByLabelText('Token to inspect')
    fireEvent.change(input, { target: { value: 'hello' } })
    expect((input as HTMLInputElement).value).toBe('hello')
  })

  it('renders CardTitle', () => {
    render(<TokenTreeEmbeddingsCard />)
    expect(screen.getByText('Token Embedding Explorer')).toBeDefined()
  })
})
