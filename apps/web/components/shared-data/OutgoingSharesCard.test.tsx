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
