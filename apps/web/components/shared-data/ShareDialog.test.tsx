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
