/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { RolePermissionMatrix } from './RolePermissionMatrix'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,

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

afterEach(() => cleanup())

const mockRoles = {
  admin: { name: 'Admin', permissions: ['model.create', 'user.view'] },
  user: { name: 'User', permissions: ['model.view'] },
}

const mockAllPermissions = {
  model: ['model.create', 'model.view'],
  user: ['user.view'],
}

describe('RolePermissionMatrix', () => {
  it('renders title "Role Permissions"', () => {
    render(<RolePermissionMatrix roles={mockRoles} allPermissions={mockAllPermissions} />)
    expect(screen.getByText('Role Permissions')).toBeDefined()
  })

  it('renders role headers', () => {
    render(<RolePermissionMatrix roles={mockRoles} allPermissions={mockAllPermissions} />)
    expect(screen.getByText('admin')).toBeDefined()
    expect(screen.getByText('user')).toBeDefined()
  })

  it('renders permission names', () => {
    render(<RolePermissionMatrix roles={mockRoles} allPermissions={mockAllPermissions} />)
    expect(screen.getByText('model.create')).toBeDefined()
    expect(screen.getByText('model.view')).toBeDefined()
    expect(screen.getByText('user.view')).toBeDefined()
  })

  it('renders category labels', () => {
    render(<RolePermissionMatrix roles={mockRoles} allPermissions={mockAllPermissions} />)
    expect(screen.getByText('Models')).toBeDefined()
    expect(screen.getByText('Users')).toBeDefined()
  })

  it('renders checkmarks for granted permissions', () => {
    render(<RolePermissionMatrix roles={mockRoles} allPermissions={mockAllPermissions} />)
    const checkmarks = screen.getAllByText('✓')
    expect(checkmarks.length).toBeGreaterThanOrEqual(1)
  })
})
