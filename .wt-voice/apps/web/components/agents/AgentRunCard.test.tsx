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
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,

Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
    Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
    Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
    ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
    Tabs: ({ children }: any) => <div>{children}</div>,
    TabsList: ({ children }: any) => <div>{children}</div>,
    TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    TabsContent: ({ children }: any) => <div>{children}</div>,
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

import { AgentRunCard } from './AgentRunCard'

afterEach(() => cleanup())

const mockRuns = [
  {
    id: 'run-1',
    status: 'completed' as const,
    task: 'Summarize the document',
    result: 'The document covers AI safety...',
    started_at: '2026-09-10T10:00:00Z',
    completed_at: '2026-09-10T10:02:00Z',
  },
  {
    id: 'run-2',
    status: 'failed' as const,
    task: 'Translate text to French',
    started_at: '2026-09-10T09:50:00Z',
    completed_at: '2026-09-10T09:50:05Z',
  },
  {
    id: 'run-3',
    status: 'running' as const,
    task: 'Generate code review',
    started_at: '2026-09-10T10:05:00Z',
  },
]

describe('AgentRunCard', () => {
  it('renders title', () => {
    render(<AgentRunCard runs={[]} />)
    expect(screen.getByText('Recent Runs')).toBeTruthy()
  })

  it('shows empty state', () => {
    render(<AgentRunCard runs={[]} />)
    expect(screen.getByText('No runs yet.')).toBeTruthy()
  })

  it('renders run tasks', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getByText('Summarize the document')).toBeTruthy()
    expect(screen.getByText('Translate text to French')).toBeTruthy()
    expect(screen.getByText('Generate code review')).toBeTruthy()
  })

  it('renders run statuses', () => {
    render(<AgentRunCard runs={mockRuns} />)
    const badges = screen.getAllByText('completed')
    expect(badges.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('failed')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
  })

  it('renders result preview', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getByText('The document covers AI safety...')).toBeTruthy()
  })

  it('renders timestamps', () => {
    render(<AgentRunCard runs={mockRuns} />)
    expect(screen.getAllByText(/Started/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Completed/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders status dots', () => {
    const { container } = render(<AgentRunCard runs={mockRuns} />)
    const dots = container.querySelectorAll('.rounded-full')
    expect(dots.length).toBe(3)
  })
})
