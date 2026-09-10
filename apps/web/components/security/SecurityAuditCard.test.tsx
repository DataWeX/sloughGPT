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

import { SecurityAuditCard } from './SecurityAuditCard'

afterEach(() => { cleanup() })

describe('SecurityAuditCard', () => {
  it('renders with empty data', () => {
    render(<SecurityAuditCard logs={[]} keys={[]} />)
    expect(screen.getByText('Security Audit')).toBeTruthy()
    expect(screen.getByText('Security score')).toBeTruthy()
  })

  it('shows good score with no issues', () => {
    render(<SecurityAuditCard logs={[]} keys={[]} />)
    expect(screen.getByText('Good')).toBeTruthy()
    expect(screen.getByText('100')).toBeTruthy()
  })

  it('flags wildcard keys as critical', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/wildcard scope/i)).toBeTruthy()
    expect(screen.getByText(/Good|Fair|Poor/)).toBeTruthy()
  })

  it('flags auth failures', () => {
    const logs = Array.from({ length: 5 }, () => ({
      event_type: 'auth.fail',
      timestamp: new Date().toISOString(),
    }))
    render(<SecurityAuditCard logs={logs} keys={[]} />)
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('flags old keys', () => {
    const oldTimestamp = Math.floor(Date.now() / 1000) - 100 * 24 * 60 * 60
    const keys = [
      { id: '1', name: 'old-key', key_hash: 'a', scopes: ['read'], created_at: oldTimestamp, revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/Old API keys/)).toBeTruthy()
  })

  it('flags many active keys', () => {
    const keys = Array.from({ length: 6 }, (_, i) => ({
      id: String(i),
      name: `key-${i}`,
      key_hash: `hash-${i}`,
      scopes: ['read'],
      created_at: 1700000000,
      revoked: false,
    }))
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText(/Many active API keys/)).toBeTruthy()
  })

  it('calculates correct score with multiple issues', () => {
    const keys = [
      { id: '1', name: 'wild', key_hash: 'a', scopes: ['*'], created_at: 1700000000, revoked: false },
    ]
    const logs = Array.from({ length: 5 }, () => ({
      event_type: 'auth.fail',
      timestamp: new Date().toISOString(),
    }))
    render(<SecurityAuditCard logs={logs} keys={keys} />)
    expect(screen.getByText(/Fair/)).toBeTruthy()
    expect(screen.getByText(/wildcard scope/i)).toBeTruthy()
    expect(screen.getByText(/brute force/i)).toBeTruthy()
  })

  it('shows no issues message when everything is fine', () => {
    const keys = [
      { id: '1', name: 'good', key_hash: 'a', scopes: ['read'], created_at: Math.floor(Date.now() / 1000), revoked: false },
    ]
    render(<SecurityAuditCard logs={[]} keys={keys} />)
    expect(screen.getByText('Security posture looks good')).toBeTruthy()
  })
})
