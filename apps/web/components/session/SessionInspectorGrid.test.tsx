// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
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

import { SessionInspectorGrid } from './SessionInspectorGrid'

describe('SessionInspectorGrid', () => {
  afterEach(() => cleanup())

  it('renders nothing when stats is undefined', () => {
    const { container } = render(<SessionInspectorGrid />)
    expect(container.innerHTML).toBe('')
  })

  it('renders stat labels when stats are provided', () => {
    render(<SessionInspectorGrid stats={{ messages: 10, knowledgeFacts: 5, feedback: 3, elapsedMs: 120 }} />)
    expect(screen.getByText('Messages')).toBeTruthy()
    expect(screen.getByText('Knowledge Facts')).toBeTruthy()
    expect(screen.getByText('Feedback')).toBeTruthy()
    expect(screen.getByText('Inspect Time')).toBeTruthy()
  })

  it('renders stat values', () => {
    render(<SessionInspectorGrid stats={{ messages: 10, elapsedMs: 120 }} />)
    expect(screen.getByText('10')).toBeTruthy()
    expect(screen.getByText('120ms')).toBeTruthy()
  })

  it('renders Workspace and Modes & Traits cards', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} />)
    expect(screen.getByText('Workspace')).toBeTruthy()
    expect(screen.getByText('Modes & Traits')).toBeTruthy()
  })

  it('renders working memory items', () => {
    render(<SessionInspectorGrid stats={{ workingMemory: ['item1', 'item2'] }} />)
    expect(screen.getByText('item1')).toBeTruthy()
    expect(screen.getByText('item2')).toBeTruthy()
  })

  it('renders modes', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} modes={{ tone: 'friendly', style: 'concise' }} />)
    expect(screen.getByText('friendly')).toBeTruthy()
    expect(screen.getByText('concise')).toBeTruthy()
  })

  it('renders traits as JSON', () => {
    render(<SessionInspectorGrid stats={{ messages: 1 }} traits={{ creativity: 0.8 }} />)
    expect(screen.getByText(/"creativity": 0.8/)).toBeTruthy()
  })
})
