// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('@/lib/download-utils', () => ({
  downloadBlob: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-01-01',
}))

import { LossChart } from './LossChart'

const trainAt = (step: number, value: number) => ({ step, value, type: 'train' as const })

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

  it('renders an svg with the train line for train data', () => {
    render(<LossChart data={[trainAt(1, 0.5)]} />)
    expect(screen.getByTestId('chart-svg')).toBeInTheDocument()
    expect(screen.getByTestId('train-line')).toBeInTheDocument()
    expect(screen.queryByTestId('eval-line')).toBeNull()
  })

  it('renders the eval line for eval data', () => {
    render(<LossChart data={[{ step: 1, value: 0.4, type: 'eval' }]} />)
    expect(screen.getByTestId('eval-line')).toBeInTheDocument()
    expect(screen.queryByTestId('train-line')).toBeNull()
  })

  it('renders reward axis and dashed reward line when reward data is present', () => {
    render(
      <LossChart
        data={[trainAt(1, 0.5)]}
        rewardData={[{ step: 1, value: 1.0 }]}
      />,
    )
    expect(screen.getByTestId('reward-axis')).toBeInTheDocument()
    expect(screen.getByTestId('reward-line')).toBeInTheDocument()
    expect(screen.queryByTestId('reward-axis')).not.toBeNull()
  })

  it('omits the reward axis without reward data', () => {
    render(<LossChart data={[trainAt(1, 0.5)]} />)
    expect(screen.queryByTestId('reward-axis')).toBeNull()
  })

  it('renders legend when showLegend is true', () => {
    render(<LossChart data={[trainAt(1, 0.5)]} showLegend={true} />)
    expect(screen.getByTestId('legend')).toBeInTheDocument()
    expect(screen.getByTestId('legend').textContent).toContain('Train loss')
  })

  it('hides legend when showLegend is false', () => {
    render(<LossChart data={[trainAt(1, 0.5)]} showLegend={false} />)
    expect(screen.queryByTestId('legend')).toBeNull()
  })

  it('sets custom height on the svg', () => {
    render(<LossChart data={[trainAt(1, 0.5)]} height={400} />)
    expect(screen.getByTestId('chart-svg').getAttribute('height')).toBe('400')
  })

  // Regression guard for the irregular category axis (ticks at 2,5,6,9,11,14,...)
  it('draws evenly spaced x-axis ticks for irregular step intervals', () => {
    const steps = [2, 5, 6, 9, 11, 14, 19, 20, 25, 30]
    const { container } = render(
      <LossChart data={steps.map((s, i) => trainAt(s, 4 + i * 0.1))} />,
    )
    const labels = [...container.querySelectorAll('[data-testid="x-axis"] text')]
      .map(n => n.textContent ?? '')
      .filter(t => /^\d+$/.test(t))
      .map(Number)

    expect(labels.length).toBeGreaterThanOrEqual(3)
    const deltas = labels.slice(1).map((v, i) => v - labels[i])
    deltas.forEach(d => expect(d).toBeCloseTo(deltas[0], 6))
    expect(deltas[0]).toBeGreaterThan(0)
  })

  it('shows a Step-labelled tooltip on hover', () => {
    render(<LossChart data={[trainAt(1, 0.5), trainAt(5, 0.4)]} />)
    fireEvent.mouseMove(screen.getByTestId('hover-overlay'), { clientX: 100 })
    const tooltip = screen.getByTestId('chart-tooltip')
    expect(tooltip.textContent).toMatch(/^Step \d+/)
    expect(tooltip.textContent).toContain('Train loss')
  })

  it('hides the tooltip on mouse leave', () => {
    render(<LossChart data={[trainAt(1, 0.5), trainAt(5, 0.4)]} />)
    const overlay = screen.getByTestId('hover-overlay')
    fireEvent.mouseMove(overlay, { clientX: 100 })
    expect(screen.getByTestId('chart-tooltip')).toBeInTheDocument()
    fireEvent.mouseLeave(overlay)
    expect(screen.queryByTestId('chart-tooltip')).toBeNull()
  })
})
