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

import { PermissionMatrixCard } from './PermissionMatrixCard'

afterEach(() => { cleanup() })

describe('PermissionMatrixCard', () => {
  it('shows empty state when no keys', () => {
    render(<PermissionMatrixCard keys={[]} />)
    expect(screen.getByText('No active API keys.')).toBeTruthy()
  })

  it('renders matrix with active keys', () => {
    const keys = [
      { id: '1', name: 'ci-pipeline', key_hash: 'abc', scopes: ['read', 'write'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'admin-key', key_hash: 'def', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('Permission Matrix')).toBeTruthy()
    expect(screen.getByText('ci-pipeline')).toBeTruthy()
    expect(screen.getByText('admin-key')).toBeTruthy()
  })

  it('hides revoked keys', () => {
    const keys = [
      { id: '1', name: 'active', key_hash: 'a', scopes: ['read'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'revoked', key_hash: 'b', scopes: ['read'], created_at: 1700000000, revoked: true },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('active')).toBeTruthy()
    expect(screen.queryByText('revoked')).toBeNull()
  })

  it('shows wildcard warning', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('* Wildcard scope active')).toBeTruthy()
  })

  it('shows scope counts in footer', () => {
    const keys = [
      { id: '1', name: 'key1', key_hash: 'a', scopes: ['read', 'write'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'key2', key_hash: 'b', scopes: ['read'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('2/2')).toBeTruthy()
    expect(screen.getByText('1/2')).toBeTruthy()
  })
})
