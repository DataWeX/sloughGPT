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

import { WorkspaceSelector } from './WorkspaceSelector'

afterEach(() => { cleanup() })

const workspaces = [
  { id: 'ws-1', name: 'Acme Corp', member_count: 5 },
  { id: 'ws-2', name: 'Beta Inc', member_count: 12 },
]

describe('WorkspaceSelector', () => {
  it('renders title', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('Select Workspace')).toBeDefined()
  })

  it('renders all workspace buttons', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText(/Acme Corp/)).toBeDefined()
    expect(screen.getByText(/Beta Inc/)).toBeDefined()
  })

  it('displays member counts', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('(5)')).toBeDefined()
    expect(screen.getByText('(12)')).toBeDefined()
  })

  it('calls onSelect when workspace is clicked', () => {
    const onSelect = vi.fn()
    render(<WorkspaceSelector workspaces={workspaces} selectedId={null} onSelect={onSelect} />)
    screen.getByText(/Acme Corp/).click()
    expect(onSelect).toHaveBeenCalledWith('ws-1')
  })

  it('highlights the selected workspace', () => {
    render(<WorkspaceSelector workspaces={workspaces} selectedId="ws-2" onSelect={() => {}} />)
    const btn = screen.getByText(/Beta Inc/)
    expect(btn.className).toContain('bg-primary')
  })

  it('renders empty list without error', () => {
    render(<WorkspaceSelector workspaces={[]} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('Select Workspace')).toBeDefined()
  })
})
