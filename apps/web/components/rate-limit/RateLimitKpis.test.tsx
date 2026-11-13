// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  KpiGrid: ({ children, columns, ...props }: any) => <div data-testid="kpi-grid" data-columns={columns}>{children}</div>,
  StatCard: ({ label, value, icon }: any) => (
    <div data-testid="stat-card">
      {icon && <span data-testid="stat-icon">{icon}</span>}
      <span data-testid="stat-label">{label}</span>
      <span data-testid="stat-value">{value}</span>
    </div>
  ),

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

import { RateLimitKpis } from './RateLimitKpis'
import type { RateLimitStatus } from './RateLimitKpis'

afterEach(() => cleanup())

const activeStatus: RateLimitStatus = {
  enabled: true,
  requests_per_minute: 120,
  burst_size: 50,
}

const inactiveStatus: RateLimitStatus = {
  enabled: false,
  requests_per_minute: 0,
  burst_size: 0,
}

describe('RateLimitKpis', () => {
  it('renders 3 stat cards', () => {
    render(<RateLimitKpis />)
    const cards = screen.getAllByTestId('stat-card')
    expect(cards.length).toBe(3)
  })

  it('shows Active when enabled', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('shows Inactive when disabled', () => {
    render(<RateLimitKpis status={inactiveStatus} />)
    expect(screen.getByText('Inactive')).toBeTruthy()
  })

  it('displays requests per minute', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('120')).toBeTruthy()
  })

  it('displays burst size', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('50')).toBeTruthy()
  })

  it('shows dash placeholders when no status', () => {
    render(<RateLimitKpis />)
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThanOrEqual(2)
  })

  it('renders with columns prop', () => {
    render(<RateLimitKpis status={activeStatus} />)
    const grid = screen.getByTestId('kpi-grid')
    expect(grid.getAttribute('data-columns')).toBe('3')
  })
})
