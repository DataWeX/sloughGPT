// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}))

vi.mock('@/lib/download-utils', () => ({
  downloadBlob: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-01-01',
}))

import { LossChart } from './LossChart'

describe('LossChart', () => {
  afterEach(cleanup)

  it('renders nothing when data is empty', () => {
    const { container } = render(<LossChart data={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders nothing when both data and rewardData are empty', () => {
    const { container } = render(<LossChart data={[]} rewardData={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders chart with train data', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} />)
    expect(screen.getByTestId('responsive-container')).toBeDefined()
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('renders chart with eval data', () => {
    render(<LossChart data={[{ step: 1, value: 0.4, type: 'eval' }]} />)
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('renders composed chart when reward data is present', () => {
    render(
      <LossChart
        data={[{ step: 1, value: 0.5, type: 'train' }]}
        rewardData={[{ step: 1, value: 1.0 }]}
      />,
    )
    expect(screen.getByTestId('composed-chart')).toBeDefined()
  })

  it('renders legend when showLegend is true', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} showLegend={true} />)
    expect(screen.getByTestId('legend')).toBeDefined()
  })

  it('hides legend when showLegend is false', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} showLegend={false} />)
    expect(screen.queryByTestId('legend')).toBeNull()
  })

  it('sets custom height on ResponsiveContainer', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} height={400} />)
    expect(screen.getByTestId('responsive-container')).toBeDefined()
  })
})
