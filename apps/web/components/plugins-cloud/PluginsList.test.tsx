// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Skeleton: (props: any) => <div data-testid="skeleton" {...props} />,

    Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
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

import { PluginsList } from './PluginsList'
import type { PluginInfo } from './PluginsList'

afterEach(() => cleanup())

const mockPlugins: PluginInfo[] = [
  { name: 'my-plugin', version: '1.0.0', description: 'A plugin', author: 'Alice', enabled: true },
  { name: 'other-plugin', version: '2.1.0', description: 'Another', author: 'Bob', enabled: false },
]

describe('PluginsList', () => {
  it('renders the card title', () => {
    render(<PluginsList />)
    expect(screen.getAllByText('Installed Plugins').length).toBeGreaterThanOrEqual(1)
  })

  it('renders the card description', () => {
    render(<PluginsList />)
    expect(screen.getAllByText('Manage loaded plugins').length).toBeGreaterThanOrEqual(1)
  })

  it('shows empty state when no plugins', () => {
    render(<PluginsList />)
    expect(screen.getByText(/No plugins installed/)).toBeTruthy()
  })

  it('shows skeleton when loading', () => {
    render(<PluginsList loading />)
    expect(screen.getByTestId('skeleton')).toBeTruthy()
  })

  it('renders plugins when provided', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('my-plugin')).toBeTruthy()
    expect(screen.getByText('other-plugin')).toBeTruthy()
  })

  it('renders plugin versions and authors', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('v1.0.0')).toBeTruthy()
    expect(screen.getByText('v2.1.0')).toBeTruthy()
    expect(screen.getByText('by Alice')).toBeTruthy()
    expect(screen.getByText('by Bob')).toBeTruthy()
  })

  it('renders correct badge text', () => {
    render(<PluginsList plugins={mockPlugins} />)
    expect(screen.getByText('Enabled')).toBeTruthy()
    expect(screen.getByText('Disabled')).toBeTruthy()
  })
})
