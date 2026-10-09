/// <reference types="vitest" />
import { render, screen, cleanup } from '@testing-library/react'
import { describe, it, expect, afterEach, vi } from 'vitest'
import { TokenTreeStatsCard, type TokenTreeStats } from './TokenTreeStatsCard'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h3 data-testid="card-title" {...props}>{children}</h3>,
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div data-testid="card-content" {...props}>{children}</div>,

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

afterEach(() => cleanup())

const mockStats: TokenTreeStats = {
  trained: true,
  vocab_size: 1024,
  num_merges: 256,
  embedding_points: 512,
  num_base_tokens: 256,
  embedding_compression_ratio: 1.5,
  embed_dim: 64,
}

describe('TokenTreeStatsCard', () => {
  it('renders the card title', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Token Tree Stats')).toBeDefined()
  })

  it('renders vocab size', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('1,024')).toBeDefined()
  })

  it('renders trained status as Yes when true', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Yes')).toBeDefined()
  })

  it('renders trained status as No when false', () => {
    render(<TokenTreeStatsCard stats={{ ...mockStats, trained: false }} />)
    expect(screen.getByText('No')).toBeDefined()
  })

  it('renders all stat labels', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('Merges')).toBeDefined()
    expect(screen.getByText('Embeddings')).toBeDefined()
    expect(screen.getByText('Compression')).toBeDefined()
    expect(screen.getByText('Embed Dim')).toBeDefined()
  })

  it('renders compression ratio with two decimals', () => {
    render(<TokenTreeStatsCard stats={mockStats} />)
    expect(screen.getByText('1.50')).toBeDefined()
  })
})
