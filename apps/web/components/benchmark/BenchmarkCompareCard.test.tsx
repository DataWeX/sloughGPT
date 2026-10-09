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

import { BenchmarkCompareCard } from './BenchmarkCompareCard'

afterEach(() => { cleanup() })

describe('BenchmarkCompareCard', () => {
  it('renders empty state', () => {
    render(<BenchmarkCompareCard results={[]} />)
    expect(screen.getByText('Model Comparison')).toBeTruthy()
    expect(screen.getByText('Run benchmarks on multiple models to compare.')).toBeTruthy()
  })

  it('renders comparison with results', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { model: 'gpt2', throughput_tokens_per_sec: 50, memory_mb: 512, inference_time_ms: 200, total_tokens: 1000 }],
      ['llama', { model: 'llama', throughput_tokens_per_sec: 80, memory_mb: 1024, inference_time_ms: 150, total_tokens: 2000 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('Model Comparison')).toBeTruthy()
    expect(screen.getByText('2 models')).toBeTruthy()
  })

  it('marks best values with star', () => {
    const results: [string, Record<string, unknown>][] = [
      ['slow', { throughput_tokens_per_sec: 30 }],
      ['fast', { throughput_tokens_per_sec: 100 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText(/fast ★/)).toBeTruthy()
  })

  it('shows model summary cards', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { total_tokens: 1000, inference_count: 5 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('gpt2')).toBeTruthy()
    expect(screen.getByText('1000 tokens · 5 runs')).toBeTruthy()
  })

  it('hides metrics with no values', () => {
    const results: [string, Record<string, unknown>][] = [
      ['gpt2', { throughput_tokens_per_sec: 50 }],
    ]
    render(<BenchmarkCompareCard results={results} />)
    expect(screen.getByText('Throughput')).toBeTruthy()
    expect(screen.queryByText('Memory')).toBeNull()
  })
})
