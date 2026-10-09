// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,

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

import { BenchmarkChartCard } from './BenchmarkChartCard'

afterEach(() => { cleanup() })

describe('BenchmarkChartCard', () => {
  it('renders empty state', () => {
    render(<BenchmarkChartCard history={[]} />)
    expect(screen.getByText('Metrics Over Time')).toBeTruthy()
    expect(screen.getByText('No benchmark history yet.')).toBeTruthy()
  })

  it('renders chart with data', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50, latency: 200, memory: 512 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'gpt2', throughput: 55, latency: 180, memory: 512 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Metrics Over Time')).toBeTruthy()
    expect(screen.getByText(/2 runs/)).toBeTruthy()
  })

  it('shows multiple models', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
      { timestamp: '2024-01-01T11:00:00Z', model: 'llama', throughput: 80 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText(/2 models/)).toBeTruthy()
    expect(screen.getByText('gpt2')).toBeTruthy()
    expect(screen.getByText('llama')).toBeTruthy()
  })

  it('hides metrics with no data', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Throughput')).toBeTruthy()
    expect(screen.queryByText('Latency')).toBeNull()
    expect(screen.queryByText('Memory')).toBeNull()
  })

  it('renders recent runs section', () => {
    const history = [
      { timestamp: '2024-01-01T10:00:00Z', model: 'gpt2', throughput: 50 },
    ]
    render(<BenchmarkChartCard history={history} />)
    expect(screen.getByText('Recent runs')).toBeTruthy()
  })
})
