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
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,

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

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '1m ago',
}))

import { AdminAuditLogCard } from './AdminAuditLogCard'

afterEach(() => { cleanup() })

const baseLogs = [
  { event_type: 'auth.login', timestamp: new Date().toISOString(), user: 'admin', resource: 'session', detail: 'Successful login' },
  { event_type: 'model.loaded', timestamp: new Date().toISOString(), resource: 'gpt-4' },
]

describe('AdminAuditLogCard', () => {
  it('renders title', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Audit Logs')).toBeTruthy()
  })

  it('renders filter input', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByLabelText('Filter audit logs')).toBeTruthy()
  })

  it('renders history mode toggle button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Session')).toBeTruthy()
  })

  it('shows persisted mode text', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="persisted" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Persisted')).toBeTruthy()
  })

  it('renders load older button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Load Older')).toBeTruthy()
  })

  it('shows loading text when loadingMore', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={true} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('renders refresh button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByLabelText('Refresh logs')).toBeTruthy()
  })

  it('shows log entries when provided', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('auth.login')).toBeTruthy()
    expect(screen.getByText('model.loaded')).toBeTruthy()
  })

  it('shows user when present', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('@admin')).toBeTruthy()
  })

  it('shows resource when present', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('on session')).toBeTruthy()
  })

  it('shows no logs message when empty', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('No logs found.')).toBeTruthy()
  })

  it('calls onRefresh when refresh button clicked', () => {
    const onRefresh = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={onRefresh} />)
    fireEvent.click(screen.getByLabelText('Refresh logs'))
    expect(onRefresh).toHaveBeenCalledOnce()
  })

  it('calls onToggleHistory when mode button clicked', () => {
    const onToggleHistory = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={onToggleHistory} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    fireEvent.click(screen.getByText('Session'))
    expect(onToggleHistory).toHaveBeenCalledOnce()
  })

  it('calls onLoadOlder when load older button clicked', () => {
    const onLoadOlder = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={onLoadOlder} onRefresh={vi.fn()} />)
    fireEvent.click(screen.getByText('Load Older'))
    expect(onLoadOlder).toHaveBeenCalledOnce()
  })

  it('calls onFilterChange when filter input changes', () => {
    const onFilterChange = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={onFilterChange} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('Filter audit logs'), { target: { value: 'auth' } })
    expect(onFilterChange).toHaveBeenCalledWith('auth')
  })
})
