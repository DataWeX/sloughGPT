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
}))

import { AuditTrailListCard } from './AuditTrailListCard'

afterEach(() => cleanup())

const mockActivities = [
  { type: 'training', action: 'LoRA trained', detail: 'Used 50 pairs', status: 'completed', timestamp: '2026-09-09T10:00:00Z', user: 'alice' },
  { type: 'audit', action: 'Login', detail: 'From 192.168.1.1', status: 'success', timestamp: '2026-09-09T11:00:00Z', user: 'bob' },
  { type: 'system', action: 'Backup', detail: '', status: 'running', timestamp: '2026-09-09T12:00:00Z', user: '' },
]

describe('AuditTrailListCard', () => {
  it('shows empty state', () => {
    render(<AuditTrailListCard activities={[]} />)
    expect(screen.getByText('Events')).toBeTruthy()
    expect(screen.getByText('No events recorded.')).toBeTruthy()
  })

  it('shows activity list', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('(3)')).toBeTruthy()
    expect(screen.getByText('LoRA trained')).toBeTruthy()
    expect(screen.getByText('Login')).toBeTruthy()
    expect(screen.getByText('Backup')).toBeTruthy()
  })

  it('shows type badges', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('training')).toBeTruthy()
    expect(screen.getByText('audit')).toBeTruthy()
    expect(screen.getByText('system')).toBeTruthy()
  })

  it('shows status badges', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    expect(screen.getByText('completed')).toBeTruthy()
    expect(screen.getByText('success')).toBeTruthy()
    expect(screen.getByText('running')).toBeTruthy()
  })

  it('expands and collapses on click', () => {
    render(<AuditTrailListCard activities={mockActivities} />)
    const eventEl = screen.getByTestId('audit-event-0')
    // Before expand: no expanded detail panel
    expect(screen.queryByText('Time:')).toBeNull()
    fireEvent.click(eventEl)
    // After expand: detail panel visible with grid labels
    expect(screen.queryAllByText('Time:').length).toBeGreaterThan(0)
    fireEvent.click(eventEl)
    // After collapse: detail panel removed
    expect(screen.queryByText('Time:')).toBeNull()
  })
})
