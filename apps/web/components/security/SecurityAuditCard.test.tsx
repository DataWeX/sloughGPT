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

import { SecurityAuditCard } from './SecurityAuditCard'

afterEach(() => { cleanup() })

describe('SecurityAuditCard', () => {
  it('renders with empty data', () => {
    render(<SecurityAuditCard logs={[]} keys={[]} />)
    expect(screen.getByText('Security Audit')).toBeTruthy()
    expect(screen.getByText('Security score')).toBeTruthy()
  })

  it('shows good score with no issues', () => {
    render(<SecurityAuditCard logs={[]} keys={[]} />)
    expect(screen.getByText('Good')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
  })

  it('flags wildcard keys as critical', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/wildcard scope/i)).toBeTruthy()
    expect(screen.getByText(/Good|Fair|Poor/)).toBeTruthy()
  })

  it('flags auth failures', () => {
    const logs = Array.from({ length: 5 }, () => ({
      event_type: 'auth.fail',
      timestamp: new Date().toISOString(),
    }))
    render(<SecurityAuditCard logs={logs} keys={[]} />)
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('flags old keys', () => {
    const oldTimestamp = Math.floor(Date.now() / 1000) - 100 * 24 * 60 * 60
    const keys = [
      { id: '1', name: 'old-key', key_hash: 'a', scopes: ['read'], created_at: oldTimestamp, revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/Old API keys/)).toBeTruthy()
  })

  it('flags many active keys', () => {
    const keys = Array.from({ length: 6 }, (_, i) => ({
      id: String(i),
      name: `key-${i}`,
      key_hash: `hash-${i}`,
      scopes: ['read'],
      created_at: 1700000000,
      revoked: false,
    }))
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/Many active API keys/)).toBeTruthy()
  })

  it('calculates correct score with multiple issues', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    const logs = Array.from({ length: 5 }, () => ({
      event_type: 'auth.fail',
      timestamp: new Date().toISOString(),
    }))
    render(<SecurityAuditCard logs={logs} keys={keys} />)
    expect(screen.getByText(/Fair/)).toBeTruthy()
    expect(screen.getByText(/wildcard scope/i)).toBeTruthy()
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('shows no issues message when everything is fine', () => {
    const keys = [
      { id: '1', name: 'good', key_hash: 'a', scopes: ['read'], created_at: Math.floor(Date.now() / 1000), revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText('Security posture looks good')).toBeTruthy()
  })
})
