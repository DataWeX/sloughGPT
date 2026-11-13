// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ErrorInsightsCard } from './ErrorInsightsCard'

afterEach(() => { cleanup() })

const baseGrouped = [
  { message: 'timeout connecting to model', count: 5, lastSeen: '2025-01-01T00:00:00Z' },
  { message: 'auth token expired', count: 3, lastSeen: '2025-01-01T00:00:00Z' },
  { message: 'crash during inference', count: 2, lastSeen: '2025-01-01T00:00:00Z' },
]

const baseRecent = [
  { message: 'timeout connecting to model', timestamp: new Date(Date.now() - 60000).toISOString() },
  { message: 'auth token expired', timestamp: new Date(Date.now() - 120000).toISOString() },
]

describe('ErrorInsightsCard', () => {
  it('renders empty state for empty data', () => {
    const { container } = render(<ErrorInsightsCard grouped={[]} recent={[]} total={0} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders card', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={baseRecent} total={10} />)
    expect(screen.getAllByTestId('error-insights').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Error Insights').length).toBeGreaterThanOrEqual(1)
  })

  it('shows total count', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={baseRecent} total={10} />)
    expect(screen.getAllByText('10').length).toBeGreaterThanOrEqual(1)
  })

  it('shows unique error count', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={baseRecent} total={10} />)
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1)
  })

  it('shows top error multiplier', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={baseRecent} total={10} />)
    expect(screen.getAllByText('5x').length).toBeGreaterThanOrEqual(1)
  })

  it('shows severity breakdown', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={baseRecent} total={10} />)
    expect(screen.getAllByText('Warning').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Security').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Critical').length).toBeGreaterThanOrEqual(1)
  })

  it('handles grouped-only data (empty recent)', () => {
    render(<ErrorInsightsCard grouped={baseGrouped} recent={[]} total={10} />)
    expect(screen.getAllByTestId('error-insights').length).toBeGreaterThanOrEqual(1)
  })

  it('handles single error', () => {
    const single = [{ message: 'something broke', count: 1, lastSeen: '2025-01-01T00:00:00Z' }]
    render(<ErrorInsightsCard grouped={single} recent={[{ message: 'something broke', timestamp: new Date().toISOString() }]} total={1} />)
    expect(screen.getAllByTestId('error-insights').length).toBeGreaterThanOrEqual(1)
  })
})
