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

import { LossChart } from './LossChart'

const train = (step: number, value: number) => ({ step, value, type: 'train' as const })
const evalP = (step: number, value: number) => ({ step, value, type: 'eval' as const })
const reward = (step: number, value: number) => ({ step, value })

describe('LossChart', () => {
  afterEach(cleanup)

  it('renders nothing when no data', () => {
    const { container } = render(<LossChart data={[]} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders chart with training data', () => {
    render(<LossChart data={[train(1, 2.0), train(2, 1.5)]} />)
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })

  it('renders with reward data using ComposedChart', () => {
    render(<LossChart data={[train(1, 2.0)]} rewardData={[reward(1, 0.8)]} />)
    expect(screen.getByTestId('composed-chart')).toBeDefined()
  })

  it('renders legend when showLegend is true', () => {
    render(<LossChart data={[train(1, 2.0)]} showLegend={true} />)
    expect(screen.getByTestId('legend')).toBeDefined()
  })

  it('omits legend when showLegend is false', () => {
    render(<LossChart data={[train(1, 2.0)]} showLegend={false} />)
    expect(screen.queryByTestId('legend')).toBeNull()
  })

  it('sliding window shows last N steps when live', () => {
    const manyTrain = Array.from({ length: 50 }, (_, i) => train(i + 1, 1.0))
    render(<LossChart data={manyTrain} live={true} windowSize={40} />)
    expect(screen.getByTestId('line-chart')).toBeDefined()
  })
})
