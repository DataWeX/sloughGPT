/// <reference types="vitest" />
import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { RolePermissionMatrix } from './RolePermissionMatrix'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
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
