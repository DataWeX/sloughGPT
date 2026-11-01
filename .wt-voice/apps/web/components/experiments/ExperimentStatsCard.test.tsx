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

import { ExperimentStatsCard } from './ExperimentStatsCard'

afterEach(() => cleanup())

describe('ExperimentStatsCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<ExperimentStatsCard />)
    expect(container.firstChild).toBeNull()
  })

  it('shows metrics', () => {
    render(<ExperimentStatsCard metrics={[
      { metric: 'accuracy', value: 0.95, step: 1, timestamp: '2026-09-09' },
      { metric: 'loss', value: 0.05, step: 1, timestamp: '2026-09-09' },
    ]} />)
    expect(screen.getByText('Experiment Data')).toBeTruthy()
    expect(screen.getByText('Metrics')).toBeTruthy()
    expect(screen.getByText('0.9500')).toBeTruthy()
    expect(screen.getByText('0.0500')).toBeTruthy()
  })

  it('shows params', () => {
    render(<ExperimentStatsCard params={[
      { param: 'lr', value: '0.001', timestamp: '2026-09-09' },
      { param: 'batch_size', value: '32', timestamp: '2026-09-09' },
    ]} />)
    expect(screen.getByText('Parameters')).toBeTruthy()
    expect(screen.getByText('lr')).toBeTruthy()
    expect(screen.getByText('0.001')).toBeTruthy()
    expect(screen.getByText('batch_size')).toBeTruthy()
    expect(screen.getByText('32')).toBeTruthy()
  })

  it('shows status', () => {
    render(<ExperimentStatsCard status={{ status: 'completed', completed_at: '2026-09-09T12:00:00Z' }} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText(/Completed/)).toBeTruthy()
  })

  it('shows metric entry counts', () => {
    render(<ExperimentStatsCard metrics={[
      { metric: 'acc', value: 0.9, step: 1, timestamp: '' },
      { metric: 'acc', value: 0.95, step: 2, timestamp: '' },
      { metric: 'acc', value: 0.97, step: 3, timestamp: '' },
    ]} />)
    expect(screen.getByText('3 entries')).toBeTruthy()
  })
})
