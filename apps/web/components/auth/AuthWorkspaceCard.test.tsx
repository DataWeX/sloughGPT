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

import { AuthWorkspaceCard } from './AuthWorkspaceCard'

afterEach(() => cleanup())

const mockWorkspaces = [
  { id: 'ws-1', name: 'Project Alpha', tenant_id: 't-1', description: 'Main workspace', role: 'owner' },
  { id: 'ws-2', name: 'Research', tenant_id: 't-2', description: 'R&D projects', role: 'member' },
  { id: 'ws-3', name: 'Admin', tenant_id: 't-3', description: '', role: 'admin' },
]

describe('AuthWorkspaceCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<AuthWorkspaceCard workspaces={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows workspace list', () => {
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} />)
    expect(screen.getByText('Workspaces')).toBeTruthy()
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('Project Alpha')).toBeTruthy()
    expect(screen.getByText('Research')).toBeTruthy()
    expect(screen.getByText('Admin')).toBeTruthy()
  })

  it('shows role badges', () => {
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} />)
    expect(screen.getByText('owner')).toBeTruthy()
    expect(screen.getByText('member')).toBeTruthy()
    expect(screen.getByText('admin')).toBeTruthy()
  })

  it('shows descriptions', () => {
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} />)
    expect(screen.getByText('Main workspace')).toBeTruthy()
    expect(screen.getByText('R&D projects')).toBeTruthy()
  })

  it('shows tenant IDs', () => {
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} />)
    expect(screen.getByText('t-1')).toBeTruthy()
    expect(screen.getByText('t-2')).toBeTruthy()
  })

  it('calls onSelect when clicked', () => {
    const onSelect = vi.fn()
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} onSelect={onSelect} />)
    fireEvent.click(screen.getByTestId('workspace-ws-1'))
    expect(onSelect).toHaveBeenCalledWith('ws-1')
  })

  it('highlights active workspace', () => {
    render(<AuthWorkspaceCard workspaces={mockWorkspaces} activeWorkspace="ws-2" />)
    const ws2 = screen.getByTestId('workspace-ws-2')
    expect(ws2.className).toContain('border-primary/50')
  })
})
