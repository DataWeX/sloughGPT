// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
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

import { ApiKeyListCard } from './ApiKeyListCard'

afterEach(() => cleanup())

const mockKeys = [
  { id: 'k-1', name: 'Prod Key', key_hash: 'sk-abc...xyz', scopes: ['*'], created_at: Date.now() / 1000, revoked: false },
  { id: 'k-2', name: 'Dev Key', key_hash: 'sk-def...uvw', scopes: ['read', 'write'], created_at: Date.now() / 1000, expires_at: Date.now() / 1000 + 86400, revoked: false },
  { id: 'k-3', name: 'Old Key', key_hash: 'sk-old...key', scopes: [], created_at: Date.now() / 1000 - 864000, revoked: true },
]

describe('ApiKeyListCard', () => {
  it('shows empty state', () => {
    render(<ApiKeyListCard keys={[]} />)
    expect(screen.getByText('No API keys.')).toBeTruthy()
  })

  it('shows active keys', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('(2 active)')).toBeTruthy()
    expect(screen.getByText('Prod Key')).toBeTruthy()
    expect(screen.getByText('Dev Key')).toBeTruthy()
  })

  it('shows revoked keys', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('Old Key')).toBeTruthy()
    const badges = screen.getAllByText('Revoked')
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })

  it('shows key hashes', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('sk-abc...xyz')).toBeTruthy()
    expect(screen.getByText('sk-def...uvw')).toBeTruthy()
  })

  it('shows scopes', () => {
    render(<ApiKeyListCard keys={mockKeys} />)
    expect(screen.getByText('scopes: read, write')).toBeTruthy()
  })

  it('calls onRotate', () => {
    const onRotate = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRotate={onRotate} />)
    const rotateBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Rotate')
    fireEvent.click(rotateBtns[0])
    expect(onRotate).toHaveBeenCalledWith('k-1')
  })

  it('shows revoke confirmation', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRevoke={onRevoke} />)
    const revokeBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Revoke')
    fireEvent.click(revokeBtns[0])
    expect(screen.getByText('Confirm')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('calls onRevoke after confirm', () => {
    const onRevoke = vi.fn()
    render(<ApiKeyListCard keys={mockKeys} onRevoke={onRevoke} />)
    const revokeBtns = screen.getAllByRole('button').filter(b => b.textContent === 'Revoke')
    fireEvent.click(revokeBtns[0])
    fireEvent.click(screen.getByText('Confirm'))
    expect(onRevoke).toHaveBeenCalledWith('k-1')
  })
})
