// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  KpiGrid: ({ children, columns, ...props }: any) => <div data-testid="kpi-grid" data-columns={columns}>{children}</div>,
  StatCard: ({ label, value, icon }: any) => (
    <div data-testid="stat-card">
      {icon && <span data-testid="stat-icon">{icon}</span>}
      <span data-testid="stat-label">{label}</span>
      <span data-testid="stat-value">{value}</span>
    </div>
  ),
}))

import { RateLimitKpis } from './RateLimitKpis'
import type { RateLimitStatus } from './RateLimitKpis'

afterEach(() => cleanup())

const activeStatus: RateLimitStatus = {
  enabled: true,
  requests_per_minute: 120,
  burst_size: 50,
}

const inactiveStatus: RateLimitStatus = {
  enabled: false,
  requests_per_minute: 0,
  burst_size: 0,
}

describe('RateLimitKpis', () => {
  it('renders 3 stat cards', () => {
    render(<RateLimitKpis />)
    const cards = screen.getAllByTestId('stat-card')
    expect(cards.length).toBe(3)
  })

  it('shows Active when enabled', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('Active')).toBeTruthy()
  })

  it('shows Inactive when disabled', () => {
    render(<RateLimitKpis status={inactiveStatus} />)
    expect(screen.getByText('Inactive')).toBeTruthy()
  })

  it('displays requests per minute', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('120')).toBeTruthy()
  })

  it('displays burst size', () => {
    render(<RateLimitKpis status={activeStatus} />)
    expect(screen.getByText('50')).toBeTruthy()
  })

  it('shows dash placeholders when no status', () => {
    render(<RateLimitKpis />)
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThanOrEqual(2)
  })

  it('renders with columns prop', () => {
    render(<RateLimitKpis status={activeStatus} />)
    const grid = screen.getByTestId('kpi-grid')
    expect(grid.getAttribute('data-columns')).toBe('3')
  })
})
