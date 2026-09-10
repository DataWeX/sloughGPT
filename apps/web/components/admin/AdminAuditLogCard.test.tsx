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
  Input: (props: any) => <input {...props} />,
}))

vi.mock('@/lib/time-ago', () => ({
  timeAgo: () => '1m ago',
}))

import { AdminAuditLogCard } from './AdminAuditLogCard'

afterEach(() => { cleanup() })

const baseLogs = [
  { event_type: 'auth.login', timestamp: new Date().toISOString(), user: 'admin', resource: 'session', detail: 'Successful login' },
  { event_type: 'model.loaded', timestamp: new Date().toISOString(), resource: 'gpt-4' },
]

describe('AdminAuditLogCard', () => {
  it('renders title', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Audit Logs')).toBeTruthy()
  })

  it('renders filter input', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByLabelText('Filter audit logs')).toBeTruthy()
  })

  it('renders history mode toggle button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Session')).toBeTruthy()
  })

  it('shows persisted mode text', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="persisted" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Persisted')).toBeTruthy()
  })

  it('renders load older button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Load Older')).toBeTruthy()
  })

  it('shows loading text when loadingMore', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={true} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('Loading...')).toBeTruthy()
  })

  it('renders refresh button', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByLabelText('Refresh logs')).toBeTruthy()
  })

  it('shows log entries when provided', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('auth.login')).toBeTruthy()
    expect(screen.getByText('model.loaded')).toBeTruthy()
  })

  it('shows user when present', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('@admin')).toBeTruthy()
  })

  it('shows resource when present', () => {
    render(<AdminAuditLogCard logs={baseLogs} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('on session')).toBeTruthy()
  })

  it('shows no logs message when empty', () => {
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    expect(screen.getByText('No logs found.')).toBeTruthy()
  })

  it('calls onRefresh when refresh button clicked', () => {
    const onRefresh = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={onRefresh} />)
    fireEvent.click(screen.getByLabelText('Refresh logs'))
    expect(onRefresh).toHaveBeenCalledOnce()
  })

  it('calls onToggleHistory when mode button clicked', () => {
    const onToggleHistory = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={onToggleHistory} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    fireEvent.click(screen.getByText('Session'))
    expect(onToggleHistory).toHaveBeenCalledOnce()
  })

  it('calls onLoadOlder when load older button clicked', () => {
    const onLoadOlder = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={vi.fn()} onToggleHistory={vi.fn()} onLoadOlder={onLoadOlder} onRefresh={vi.fn()} />)
    fireEvent.click(screen.getByText('Load Older'))
    expect(onLoadOlder).toHaveBeenCalledOnce()
  })

  it('calls onFilterChange when filter input changes', () => {
    const onFilterChange = vi.fn()
    render(<AdminAuditLogCard logs={[]} filter="" historyMode="session" loadingMore={false} onFilterChange={onFilterChange} onToggleHistory={vi.fn()} onLoadOlder={vi.fn()} onRefresh={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('Filter audit logs'), { target: { value: 'auth' } })
    expect(onFilterChange).toHaveBeenCalledWith('auth')
  })
})
