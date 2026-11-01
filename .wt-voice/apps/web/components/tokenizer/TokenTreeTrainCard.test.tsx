import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react'
import React from 'react'

const mocks = vi.hoisted(() => ({
  mockTrain: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/token-tree-controller', () => ({
  tokenTreeController: { train: mocks.mockTrain },
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
  Input: (props: any) => <input {...props} />,
  Button: ({ children, onClick, disabled }: any) => (
    <button onClick={onClick} disabled={disabled}>{children}</button>
  ),
  IconRefresh: () => <span />,
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

import { TokenTreeTrainCard } from './TokenTreeTrainCard'

describe('TokenTreeTrainCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(cleanup)

  it('renders the card with default parameters', () => {
    render(<TokenTreeTrainCard />)
    expect(screen.getByText('Train Token Tree')).toBeDefined()
    expect(screen.getByLabelText('Token tree vocab size')).toHaveValue(512)
    expect(screen.getByLabelText('Token tree embed dim')).toHaveValue(16)
    expect(screen.getByLabelText('Token tree min frequency')).toHaveValue(2)
    expect(screen.getByRole('button', { name: /Train token tree/i })).toBeDefined()
  })

  it('trains with the pasted corpus and parameters', async () => {
    mocks.mockTrain.mockResolvedValueOnce({
      status: 'trained',
      vocab_size: 64,
      embed_dim: 8,
      embedding_points: 64,
      embedding_compression_ratio: 2.5,
    })
    const onTrained = vi.fn()
    render(<TokenTreeTrainCard onTrained={onTrained} />)

    fireEvent.change(screen.getByLabelText('Token tree training corpus'), {
      target: { value: 'line one\n\nline two\n  ' },
    })
    fireEvent.change(screen.getByLabelText('Token tree vocab size'), { target: { value: '64' } })
    fireEvent.change(screen.getByLabelText('Token tree embed dim'), { target: { value: '8' } })
    fireEvent.change(screen.getByLabelText('Token tree min frequency'), { target: { value: '1' } })
    fireEvent.click(screen.getByRole('button', { name: /Train token tree/i }))

    await waitFor(() => expect(mocks.mockTrain).toHaveBeenCalledWith({
      texts: ['line one', 'line two'],
      vocab_size: 64,
      embed_dim: 8,
      min_frequency: 1,
    }))
    expect(onTrained).toHaveBeenCalled()
  })

  it('omits texts when the corpus is empty', async () => {
    mocks.mockTrain.mockResolvedValueOnce({
      status: 'trained',
      vocab_size: 512,
      embed_dim: 16,
      embedding_points: 512,
      embedding_compression_ratio: 3.0,
    })
    render(<TokenTreeTrainCard />)
    fireEvent.click(screen.getByRole('button', { name: /Train token tree/i }))

    await waitFor(() => expect(mocks.mockTrain).toHaveBeenCalledWith({
      vocab_size: 512,
      embed_dim: 16,
      min_frequency: 2,
    }))
  })

  it('shows the result banner after a successful train', async () => {
    mocks.mockTrain.mockResolvedValueOnce({
      status: 'trained',
      vocab_size: 512,
      embed_dim: 16,
      embedding_points: 512,
      embedding_compression_ratio: 2.75,
    })
    render(<TokenTreeTrainCard />)
    fireEvent.click(screen.getByRole('button', { name: /Train token tree/i }))

    await waitFor(() => expect(screen.getByText(/Trained: vocab 512/)).toBeDefined())
    expect(screen.getByText(/compression 2.75x/)).toBeDefined()
  })

  it('shows a failure banner and error toast when training fails', async () => {
    mocks.mockTrain.mockRejectedValueOnce(new Error('boom'))
    render(<TokenTreeTrainCard />)
    fireEvent.click(screen.getByRole('button', { name: /Train token tree/i }))

    await waitFor(() => expect(mocks.mockAddToast).toHaveBeenCalledWith('boom', 'error'))
    expect(screen.getByText(/Training failed/)).toBeDefined()
  })

  it('disables the button while training', async () => {
    let resolveTrain!: (v: { status: string }) => void
    mocks.mockTrain.mockReturnValue(new Promise(r => { resolveTrain = r }))
    render(<TokenTreeTrainCard />)
    fireEvent.click(screen.getByRole('button', { name: /Train token tree/i }))

    const button = await waitFor(() => screen.getByRole('button', { name: /Training\.\.\./i }))
    expect((button as HTMLButtonElement).disabled).toBe(true)

    resolveTrain({ status: 'trained' })
    await waitFor(() => expect(screen.getByRole('button', { name: /Train token tree/i })).toBeDefined())
  })
})
