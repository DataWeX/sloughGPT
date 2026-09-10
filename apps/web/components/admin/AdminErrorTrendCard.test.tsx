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

import { AdminErrorTrendCard } from './AdminErrorTrendCard'

afterEach(() => { cleanup() })

const sampleTrends = [
  { hour: '10', count: 5 },
  { hour: '11', count: 12 },
  { hour: '12', count: 3 },
]

describe('AdminErrorTrendCard', () => {
  it('renders title', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={0} lastHourCount={0} topError={null} />)
    expect(screen.getByText('Hourly Trend')).toBeTruthy()
  })

  it('shows total errors', () => {
    render(<AdminErrorTrendCard trends={[]} total={42} groupedCount={10} lastHourCount={5} topError={null} />)
    expect(screen.getByText('42')).toBeTruthy()
    expect(screen.getByTestId('total-errors')).toBeTruthy()
  })

  it('shows grouped count', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={7} lastHourCount={0} topError={null} />)
    expect(screen.getByTestId('grouped-count').textContent).toBe('7')
  })

  it('shows last hour count', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={0} lastHourCount={3} topError={null} />)
    expect(screen.getByTestId('last-hour-count').textContent).toBe('3')
  })

  it('shows top error when provided', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={0} lastHourCount={0} topError="TimeoutError" />)
    expect(screen.getByText('TimeoutError')).toBeTruthy()
  })

  it('shows dash when topError is null', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={0} lastHourCount={0} topError={null} />)
    expect(screen.getByText('—')).toBeTruthy()
  })

  it('renders bar chart when trends provided', () => {
    render(<AdminErrorTrendCard trends={sampleTrends} total={20} groupedCount={3} lastHourCount={99} topError={null} />)
    expect(screen.getByTestId('bar-chart')).toBeTruthy()
    expect(screen.getByText('10')).toBeTruthy()
    expect(screen.getByText('11')).toBeTruthy()
    expect(screen.getByText('12')).toBeTruthy()
  })

  it('shows no trend data when empty', () => {
    render(<AdminErrorTrendCard trends={[]} total={0} groupedCount={0} lastHourCount={0} topError={null} />)
    expect(screen.getByText('No trend data.')).toBeTruthy()
  })

  it('renders bar heights proportional to max', () => {
    const { container } = render(
      <AdminErrorTrendCard trends={sampleTrends} total={20} groupedCount={3} lastHourCount={12} topError={null} />
    )
    const bars = container.querySelectorAll('[data-testid="bar-chart"] > div > div:first-child')
    expect(bars.length).toBe(3)
  })

  it('displays KPI grid with all stats', () => {
    render(<AdminErrorTrendCard trends={sampleTrends} total={100} groupedCount={25} lastHourCount={8} topError="IndexError" />)
    expect(screen.getByText('Total Errors')).toBeTruthy()
    expect(screen.getByText('Grouped')).toBeTruthy()
    expect(screen.getByText('Last Hour')).toBeTruthy()
    expect(screen.getByText('Top Error')).toBeTruthy()
    expect(screen.getByTestId('total-errors').textContent).toBe('100')
    expect(screen.getByTestId('grouped-count').textContent).toBe('25')
    expect(screen.getByTestId('last-hour-count').textContent).toBe('8')
    expect(screen.getByText('IndexError')).toBeTruthy()
  })
})
