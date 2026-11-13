// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
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

import { AuditTrailListCard } from './AuditTrailListCard'

afterEach(() => cleanup())

const mockActivities = [
  { type: 'training', action: 'LoRA trained', detail: 'Used 50 pairs', status: 'completed', timestamp: '2026-09-09T10:00:00Z', user: 'alice' },
  { type: 'audit', action: 'Login', detail: 'From 192.168.1.1', status: 'success', timestamp: '2026-09-09T11:00:00Z', user: 'bob' },
  { type: 'system', action: 'Backup', detail: '', status: 'running', timestamp: '2026-09-09T12:00:00Z', user: '' },
]

describe('AuditTrailListCard', () => {
  it('shows empty state', () => {
    render(<AuditTrailListCard activities={[]} />)
    expect(screen.getByText('Events')).toBeTruthy()
    expect(screen.getByText('No events recorded.')).toBeTruthy()
  })

  it('shows activity list', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('LoRA trained')).toBeTruthy()
    expect(screen.getByText('Login')).toBeTruthy()
    expect(screen.getByText('Backup')).toBeTruthy()
  })

  it('shows type badges', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('training')).toBeTruthy()
    expect(screen.getByText('audit')).toBeTruthy()
    expect(screen.getByText('system')).toBeTruthy()
  })

  it('shows status badges', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('success')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
  })

  it('expands and collapses on click', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    const eventEl = screen.getByTestId('audit-event-0')
    // Before expand: no expanded detail panel
    expect(screen.queryByText('Time:')).toBeNull()
    fireEvent.click(eventEl)
    // After expand: detail panel visible with grid labels
    expect(screen.queryAllByText('Time:').length).toBeGreaterThan(0)
    fireEvent.click(eventEl)
    // After collapse: detail panel removed
    expect(screen.queryByText('Time:')).toBeNull()
  })
})
