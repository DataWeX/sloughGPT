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
