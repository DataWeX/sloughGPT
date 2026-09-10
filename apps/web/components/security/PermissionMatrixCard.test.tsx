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
}))

import { PermissionMatrixCard } from './PermissionMatrixCard'

afterEach(() => { cleanup() })

describe('PermissionMatrixCard', () => {
  it('shows empty state when no keys', () => {
    render(<PermissionMatrixCard keys={[]} />)
    expect(screen.getByText('No active API keys.')).toBeTruthy()
  })

  it('renders matrix with active keys', () => {
    const keys = [
      { id: '1', name: 'ci-pipeline', key_hash: 'abc', scopes: ['read', 'write'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'admin-key', key_hash: 'def', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('Permission Matrix')).toBeTruthy()
    expect(screen.getByText('ci-pipeline')).toBeTruthy()
    expect(screen.getByText('admin-key')).toBeTruthy()
  })

  it('hides revoked keys', () => {
    const keys = [
      { id: '1', name: 'active', key_hash: 'a', scopes: ['read'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'revoked', key_hash: 'b', scopes: ['read'], created_at: 1700000000, revoked: true },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('active')).toBeTruthy()
    expect(screen.queryByText('revoked')).toBeNull()
  })

  it('shows wildcard warning', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('* Wildcard scope active')).toBeTruthy()
  })

  it('shows scope counts in footer', () => {
    const keys = [
      { id: '1', name: 'key1', key_hash: 'a', scopes: ['read', 'write'], created_at: 1700000000, revoked: false },
      { id: '2', name: 'key2', key_hash: 'b', scopes: ['read'], created_at: 1700000000, revoked: false },
    ]
    render(<PermissionMatrixCard keys={keys} />)
    expect(screen.getByText('2/2')).toBeTruthy()
    expect(screen.getByText('1/2')).toBeTruthy()
  })
})
