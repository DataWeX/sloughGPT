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

import { AuditTrailStatsCard } from './AuditTrailStatsCard'

afterEach(() => cleanup())

describe('AuditTrailStatsCard', () => {
  it('renders nothing when empty', () => {
    const { container } = render(<AuditTrailStatsCard activities={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows stats', () => {
    render(<AuditTrailStatsCard activities={[
      { type: 'training', action: 'a', status: 'completed', timestamp: new Date().toISOString() },
      { type: 'audit', action: 'b', status: 'success', timestamp: new Date().toISOString() },
    ]} />)
    expect(screen.getByText('Overview')).toBeTruthy()
    expect(screen.getByText('Total')).toBeTruthy()
  })

  it('shows type breakdown', () => {
    render(<AuditTrailStatsCard activities={[
      { type: 'training', action: 'a', status: '', timestamp: new Date().toISOString() },
      { type: 'training', action: 'b', status: '', timestamp: new Date().toISOString() },
      { type: 'audit', action: 'c', status: '', timestamp: new Date().toISOString() },
    ]} />)
    expect(screen.getByText('training')).toBeTruthy()
    expect(screen.getByText('audit')).toBeTruthy()
  })
})
