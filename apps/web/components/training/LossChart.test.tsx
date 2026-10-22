// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  Line: ({ stroke }: any) => <div data-testid="line" data-stroke={stroke} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: ({ content }: any) => <div data-testid="tooltip">{content}</div>,
  Legend: () => <div data-testid="legend" />,
  Brush: () => <div data-testid="brush" />,
}))

vi.mock('@/lib/download-utils', () => ({
  downloadBlob: vi.fn(),
}))

vi.mock('@/lib/format-bytes', () => ({
  todayDateString: () => '2026-01-01',
}))

import { LossChart } from './LossChart'
import { downloadBlob } from '@/lib/download-utils'

const sixTrain = [1, 2, 3, 4, 5, 6].map(step => ({ step, value: 1 / step, type: 'train' as const }))

describe('LossChart', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders empty state when data is empty', () => {
    render(<LossChart data={[]} />)
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('empty')
    expect(screen.getByText('No loss data yet')).toBeDefined()
  })

  it('renders empty state when both data and rewardData are empty', () => {
    render(<LossChart data={[]} rewardData={[]} />)
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('empty')
    expect(screen.queryByTestId('line-chart')).toBeNull()
  })

  it('renders chart with train data', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} />)
    expect(screen.getByTestId('responsive-container')).toBeDefined()
    expect(screen.getByTestId('line-chart')).toBeDefined()
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('ready')
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

  it('sets custom height on the chart region', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} height={400} />)
    expect(screen.getByRole('img').style.height).toBe('400px')
  })

  it('names the chart region for assistive tech', () => {
    render(
      <LossChart
        data={[{ step: 1, value: 0.5, type: 'train' }]}
        rewardData={[{ step: 1, value: 1.0 }]}
      />,
    )
    expect(screen.getByRole('img').getAttribute('aria-label')).toBe('Loss and reward chart')
  })

  it('tokenized series colors: train=chart-1, eval=chart-2, reward=chart-3', () => {
    render(
      <LossChart
        data={[
          { step: 1, value: 0.5, type: 'train' },
          { step: 1, value: 0.4, type: 'eval' },
        ]}
        rewardData={[{ step: 1, value: 1.0 }]}
      />,
    )
    const strokes = screen.getAllByTestId('line').map(l => l.getAttribute('data-stroke'))
    expect(strokes).toEqual(['rgb(var(--chart-1))', 'rgb(var(--chart-2))', 'rgb(var(--chart-3))'])
  })

  it('loading state withholds the chart', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} loading />)
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('loading')
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull()
    expect(screen.queryByTestId('line-chart')).toBeNull()
  })

  it('error state renders alert and fires onRetry', () => {
    const onRetry = vi.fn()
    render(
      <LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} error="metrics fetch failed" onRetry={onRetry} />,
    )
    expect(screen.getByTestId('chart-frame').getAttribute('data-state')).toBe('error')
    expect(screen.getByRole('alert').textContent).toContain('metrics fetch failed')
    fireEvent.click(screen.getByLabelText('Retry'))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('shows export action only with more than five points', () => {
    render(<LossChart data={[{ step: 1, value: 0.5, type: 'train' }]} />)
    expect(screen.queryByLabelText('Export chart data as CSV')).toBeNull()
    cleanup()
    render(<LossChart data={sixTrain} />)
    expect(screen.getByLabelText('Export chart data as CSV')).toBeDefined()
  })

  it('export action prefers onExportData over CSV download (regression: was never called)', () => {
    const onExportData = vi.fn()
    render(<LossChart data={sixTrain} onExportData={onExportData} />)
    fireEvent.click(screen.getByLabelText('Export chart data as CSV'))
    expect(onExportData).toHaveBeenCalledTimes(1)
    expect(downloadBlob).not.toHaveBeenCalled()
  })

  it('export action falls back to CSV download when no handler given', () => {
    render(<LossChart data={sixTrain} />)
    fireEvent.click(screen.getByLabelText('Export chart data as CSV'))
    expect(downloadBlob).toHaveBeenCalledTimes(1)
  })

  it('hides the detail toggle for short series (< 12 steps)', () => {
    render(<LossChart data={sixTrain} />)
    expect(screen.queryByLabelText('Toggle detailed chart view')).toBeNull()
    expect(screen.queryByTestId('brush')).toBeNull()
  })

  it('detail toggle progressively discloses the range brush', () => {
    const long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map(step => ({
      step,
      value: 1 / step,
      type: 'train' as const,
    }))
    render(<LossChart data={long} />)
    const toggle = screen.getByLabelText('Toggle detailed chart view')
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    expect(screen.queryByTestId('brush')).toBeNull()

    fireEvent.click(toggle)
    expect(screen.getByTestId('brush')).toBeDefined()
    expect(screen.getByLabelText('Toggle detailed chart view').getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(screen.getByLabelText('Toggle detailed chart view'))
    expect(screen.queryByTestId('brush')).toBeNull()
  })

  it('detail toggle can be disabled by the caller', () => {
    const long = Array.from({ length: 20 }, (_, i) => ({
      step: i + 1,
      value: 1 / (i + 1),
      type: 'train' as const,
    }))
    render(<LossChart data={long} detailToggle={false} />)
    expect(screen.queryByLabelText('Toggle detailed chart view')).toBeNull()
  })
})
