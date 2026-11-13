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
  timeAgo: () => '2m ago',
}))

import { ThreatLogCard } from './ThreatLogCard'

afterEach(() => { cleanup() })

const makeLog = (overrides: Partial<{ event_type: string; timestamp: string; detail: string }> = {}) => ({
  event_type: 'user.login',
  timestamp: new Date().toISOString(),
  ...overrides,
})

describe('ThreatLogCard', () => {
  it('renders with no logs', () => {
    render(<ThreatLogCard logs={[]} />)
    expect(screen.getByText('Threat Log')).toBeTruthy()
    expect(screen.getByText('No events recorded yet.')).toBeTruthy()
  })

  it('renders logs with severity badges', () => {
    const logs = [
      makeLog({ event_type: 'user.login' }),
      makeLog({ event_type: 'dataset.delete' }),
      makeLog({ event_type: 'model.load' }),
    ]
    render(<ThreatLogCard logs={logs} />)
    expect(screen.getByText('Threat Log')).toBeTruthy()
    expect(screen.getByText(/high/)).toBeTruthy()
    expect(screen.getByText(/low/)).toBeTruthy()
  })

  it('detects brute force anomaly', () => {
    const logs = Array.from({ length: 5 }, () =>
      makeLog({ event_type: 'auth.fail', detail: 'invalid password' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).not.toHaveProperty('disabled', true)
    expect(btn.textContent).toContain('1')
  })

  it('detects bulk delete anomaly', () => {
    const logs = Array.from({ length: 6 }, () =>
      makeLog({ event_type: 'dataset.delete' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).not.toHaveProperty('disabled', true)
    expect(btn.textContent).toContain('1')
  })

  it('toggles to anomalies view', () => {
    const logs = Array.from({ length: 5 }, () =>
      makeLog({ event_type: 'auth.fail', detail: 'fail' })
    )
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    fireEvent.click(btn)
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('expands to show all events when more than 10', () => {
    const logs = Array.from({ length: 15 }, (_, i) =>
      makeLog({ event_type: `event.${i}` })
    )
    render(<ThreatLogCard logs={logs} />)
    expect(screen.getByText(/Show all 15 events/)).toBeTruthy()
    fireEvent.click(screen.getByText(/Show all 15 events/))
    expect(screen.getByText(/Show less/)).toBeTruthy()
  })

  it('disables anomaly button when no anomalies', () => {
    const logs = [makeLog({ event_type: 'model.load' })]
    render(<ThreatLogCard logs={logs} />)
    const btn = screen.getByRole('button', { name: /Anomalies/ })
    expect(btn).toHaveProperty('disabled', true)
  })
})
