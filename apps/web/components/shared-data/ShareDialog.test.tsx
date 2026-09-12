// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Button: ({ children, ...props }: any) => <button data-testid="button" {...props}>{children}</button>,
  Input: (props: any) => <input data-testid="input" {...props} />,

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

import { ShareDialog } from './ShareDialog'

const defaultProps = {
  shareType: 'dataset',
  shareResourceId: '',
  shareTargetWs: '',
  sharePermission: 'read',
  datasets: [{ id: 'ds-1', name: 'My Dataset' }],
  workspaces: [{ id: 'ws-1', name: 'Workspace A' }],
  onShareTypeChange: vi.fn(),
  onResourceIdChange: vi.fn(),
  onTargetWsChange: vi.fn(),
  onPermissionChange: vi.fn(),
  onShare: vi.fn(),
  onCancel: vi.fn(),
}

describe('ShareDialog', () => {
  afterEach(() => cleanup())

  it('renders title "Share Data with Another Workspace"', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('Share Data with Another Workspace')).toBeTruthy()
  })

  it('renders Share and Cancel buttons', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('Share')).toBeTruthy()
    expect(screen.getByText('Cancel')).toBeTruthy()
  })

  it('renders dataset select with options', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('My Dataset')).toBeTruthy()
    expect(screen.getByText('Select dataset...')).toBeTruthy()
  })

  it('renders workspace select with options', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('Workspace A')).toBeTruthy()
    expect(screen.getByText('Select workspace...')).toBeTruthy()
  })

  it('renders Input when shareType is not dataset', () => {
    render(<ShareDialog {...defaultProps} shareType="knowledge" />)
    expect(screen.getByTestId('input')).toBeTruthy()
  })

  it('renders permission options', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('Read only')).toBeTruthy()
    expect(screen.getByText('Admin')).toBeTruthy()
  })

  it('renders type select with all resource types', () => {
    render(<ShareDialog {...defaultProps} />)
    expect(screen.getByText('Dataset')).toBeTruthy()
    expect(screen.getByText('Knowledge')).toBeTruthy()
    expect(screen.getByText('API Key')).toBeTruthy()
  })
})
