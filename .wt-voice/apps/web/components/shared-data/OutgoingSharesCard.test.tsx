// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button data-testid="button" {...props}>{children}</button>,

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

vi.mock('lucide-react', () => ({
  ArrowRight: (props: any) => <span data-testid="icon-arrow-right" {...props} />,
  Database: (props: any) => <span data-testid="icon-database" {...props} />,
  Key: (props: any) => <span data-testid="icon-key" {...props} />,
}))

import { OutgoingSharesCard } from './OutgoingSharesCard'

const resolveName = (_type: string, id: string) => `resolved-${id.slice(0, 4)}`
const resolveWorkspace = (id: string) => `workspace-${id.slice(0, 4)}`

const shares = [
  {
    id: 'share-1',
    resource_type: 'dataset',
    resource_id: 'ds-1234',
    source_workspace_id: 'ws-self',
    target_workspace_id: 'ws-abcd',
    permission: 'read',
    shared_by: 'alice',
    shared_at: '2025-01-15T10:00:00Z',
  },
]

describe('OutgoingSharesCard', () => {
  afterEach(() => cleanup())

  it('renders title with share count', () => {
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText('Shared by Me (1)')).toBeTruthy()
  })

  it('renders empty message when no shares', () => {
    render(<OutgoingSharesCard shares={[]} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText('No data shared from this workspace')).toBeTruthy()
  })

  it('renders resolved resource name', () => {
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText('resolved-ds-1')).toBeTruthy()
  })

  it('renders resolved workspace name', () => {
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText('workspace-ws-a')).toBeTruthy()
  })

  it('renders Revoke button', () => {
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText('Revoke')).toBeTruthy()
  })

  it('calls onRevoke with share id when Revoke is clicked', () => {
    const onRevoke = vi.fn()
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={onRevoke} />)
    fireEvent.click(screen.getByText('Revoke'))
    expect(onRevoke).toHaveBeenCalledWith('share-1')
  })

  it('renders permission', () => {
    render(<OutgoingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} onRevoke={vi.fn()} />)
    expect(screen.getByText(/· read/)).toBeTruthy()
  })
})
