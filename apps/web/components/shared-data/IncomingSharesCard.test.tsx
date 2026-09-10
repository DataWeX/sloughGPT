// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
}))

vi.mock('lucide-react', () => ({
  ArrowLeft: (props: any) => <span data-testid="icon-arrow-left" {...props} />,
  Database: (props: any) => <span data-testid="icon-database" {...props} />,
  Key: (props: any) => <span data-testid="icon-key" {...props} />,
}))

import { IncomingSharesCard } from './IncomingSharesCard'

const resolveName = (_type: string, id: string) => `resolved-${id.slice(0, 4)}`
const resolveWorkspace = (id: string) => `workspace-${id.slice(0, 4)}`

const shares = [
  {
    id: 'share-1',
    resource_type: 'dataset',
    resource_id: 'ds-1234',
    source_workspace_id: 'ws-abcd',
    target_workspace_id: 'ws-self',
    permission: 'read',
    shared_by: 'alice',
    shared_at: '2025-01-15T10:00:00Z',
  },
]

describe('IncomingSharesCard', () => {
  afterEach(() => cleanup())

  it('renders title with share count', () => {
    render(<IncomingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText('Shared With Me (1)')).toBeTruthy()
  })

  it('renders empty message when no shares', () => {
    render(<IncomingSharesCard shares={[]} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText('No data shared with this workspace')).toBeTruthy()
  })

  it('renders resource type badge', () => {
    render(<IncomingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText('dataset')).toBeTruthy()
  })

  it('renders resolved resource name', () => {
    render(<IncomingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText('resolved-ds-1')).toBeTruthy()
  })

  it('renders resolved workspace name', () => {
    render(<IncomingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText('workspace-ws-a')).toBeTruthy()
  })

  it('renders permission and date', () => {
    render(<IncomingSharesCard shares={shares} resolveName={resolveName} resolveWorkspace={resolveWorkspace} />)
    expect(screen.getByText(/· read/)).toBeTruthy()
    expect(screen.getByText(new Date('2025-01-15T10:00:00Z').toLocaleDateString())).toBeTruthy()
  })
})
