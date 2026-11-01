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

import { SentimentTrendCard } from './SentimentTrendCard'

afterEach(() => { cleanup() })

describe('SentimentTrendCard', () => {
  it('renders empty state', () => {
    render(<SentimentTrendCard history={[]} />)
    expect(screen.getByText('Sentiment Trend')).toBeTruthy()
    expect(screen.getByText('No feedback history yet.')).toBeTruthy()
  })

  it('renders stats with positive feedback', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_down' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText('Sentiment Trend')).toBeTruthy()
    expect(screen.getByText(/66\.7% positive/)).toBeTruthy()
  })

  it('shows improving trend when recent is better', () => {
    const now = Date.now()
    const history = [
      { timestamp: new Date(now - 10 * 86400000).toISOString(), rating: 'thumbs_down' as const },
      { timestamp: new Date(now - 8 * 86400000).toISOString(), rating: 'thumbs_down' as const },
      { timestamp: new Date(now - 1 * 86400000).toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date(now - 1 * 86400000).toISOString(), rating: 'thumbs_up' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText(/improving/)).toBeTruthy()
  })

  it('shows thumbs up/down icons', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const },
      { timestamp: new Date().toISOString(), rating: 'thumbs_down' as const },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText('👍')).toBeTruthy()
    expect(screen.getByText('👎')).toBeTruthy()
  })

  it('shows quality score when present', () => {
    const history = [
      { timestamp: new Date().toISOString(), rating: 'thumbs_up' as const, quality_score: 0.85 },
    ]
    render(<SentimentTrendCard history={history} />)
    expect(screen.getByText(/Score: 85%/)).toBeTruthy()
  })
})
